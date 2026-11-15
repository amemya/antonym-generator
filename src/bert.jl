using Downloads
using Flux
using JSON3
using NNlib
using Statistics
using Zygote

"""
    BertConfig

Configuration parameters for a BERT Transformer backbone.
"""
Base.@kwdef struct BertConfig
    vocab_size::Int
    hidden_size::Int
    num_hidden_layers::Int
    num_attention_heads::Int
    intermediate_size::Int
    max_position_embeddings::Int
    type_vocab_size::Int = 2
    hidden_dropout_prob::Float32 = 0.1f0
    attention_probs_dropout_prob::Float32 = 0.1f0
    layer_norm_eps::Float32 = 1.0f-12
    pad_token_id::Int = 1
    cls_token_id::Int = 3
    sep_token_id::Int = 4
end

"""
    BertEmbeddings

BERT token, position, and token-type embeddings followed by LayerNorm and Dropout.
"""
struct BertEmbeddings{WE, PE, TE, LN, DO}
    word_embeddings::WE
    position_embeddings::PE
    token_type_embeddings::TE
    norm::LN
    dropout::DO
end
Flux.@layer BertEmbeddings

function BertEmbeddings(cfg::BertConfig)
    we = Flux.Embedding(cfg.vocab_size => cfg.hidden_size)
    pe = Flux.Embedding(cfg.max_position_embeddings => cfg.hidden_size)
    te = Flux.Embedding(max(cfg.type_vocab_size, 2) => cfg.hidden_size)
    ln = Flux.LayerNorm(cfg.hidden_size; eps = cfg.layer_norm_eps)
    do_layer = Flux.Dropout(cfg.hidden_dropout_prob)
    return BertEmbeddings(we, pe, te, ln, do_layer)
end

function (e::BertEmbeddings)(token_ids::AbstractMatrix{<:Integer}, token_type_ids::AbstractMatrix{<:Integer})
    S = size(token_ids, 1)
    H = size(e.word_embeddings.weight, 1)
    we = e.word_embeddings(token_ids)
    te = e.token_type_embeddings(token_type_ids)
    pe = reshape(e.position_embeddings.weight[:, 1:S], H, S, 1)
    x = we .+ te .+ pe
    return e.dropout(e.norm(x))
end

"""
    BertSelfAttention

Multi-head self-attention block with output projection, residual connection, and LayerNorm.
"""
struct BertSelfAttention{Q, K, V, O, DO, LN}
    query::Q
    key::K
    value::V
    out_proj::O
    attn_dropout::DO
    resid_dropout::DO
    norm::LN
    num_heads::Int
    head_dim::Int
end
Flux.@layer BertSelfAttention

function BertSelfAttention(cfg::BertConfig)
    H = cfg.hidden_size
    n_heads = cfg.num_attention_heads
    H % n_heads == 0 || error("hidden_size ($H) must be divisible by num_attention_heads ($n_heads)")
    head_dim = div(H, n_heads)
    return BertSelfAttention(
        Flux.Dense(H => H),
        Flux.Dense(H => H),
        Flux.Dense(H => H),
        Flux.Dense(H => H),
        Flux.Dropout(cfg.attention_probs_dropout_prob),
        Flux.Dropout(cfg.hidden_dropout_prob),
        Flux.LayerNorm(H; eps = cfg.layer_norm_eps),
        n_heads,
        head_dim,
    )
end

function (sa::BertSelfAttention)(x::AbstractArray{Float32, 3}, pad_mask::AbstractMatrix{Float32})
    H, S, B = size(x)
    nh = sa.num_heads
    dh = sa.head_dim

    q4 = permutedims(reshape(sa.query(x), dh, nh, S, B), (1, 3, 2, 4))
    kt4 = permutedims(reshape(sa.key(x), dh, nh, S, B), (3, 1, 2, 4))
    v4 = permutedims(reshape(sa.value(x), dh, nh, S, B), (1, 3, 2, 4))

    q = reshape(q4, dh, S, nh * B)
    kt = reshape(kt4, S, dh, nh * B)
    v = reshape(v4, dh, S, nh * B)

    scale = 1.0f0 / sqrt(Float32(dh))
    # scores[k_pos, q_pos, head_batch]
    scores = NNlib.batched_mul(kt, q) .* scale
    scores4 = reshape(scores, S, S, nh, B)
    key_bias = Zygote.ChainRulesCore.ignore_derivatives() do
        reshape((1.0f0 .- pad_mask) .* (-1.0f4), S, 1, 1, B)
    end
    masked_scores = reshape(scores4 .+ key_bias, S, S, nh * B)

    attn_probs = sa.attn_dropout(NNlib.softmax(masked_scores; dims = 1))
    context = NNlib.batched_mul(v, attn_probs) # (dh, S, nh * B)

    context4 = permutedims(reshape(context, dh, S, nh, B), (1, 3, 2, 4))
    merged = reshape(context4, H, S, B)
    projected = sa.resid_dropout(sa.out_proj(merged))
    return sa.norm(x .+ projected)
end

"""
    BertLayer

Single BERT encoder layer: multi-head self-attention + GELU feed-forward network + residual LayerNorm.
"""
struct BertLayer{A, I, O, DO, LN}
    attention::A
    intermediate::I
    output::O
    dropout::DO
    norm::LN
end
Flux.@layer BertLayer

function BertLayer(cfg::BertConfig)
    H = cfg.hidden_size
    I = cfg.intermediate_size
    return BertLayer(
        BertSelfAttention(cfg),
        Flux.Dense(H => I, NNlib.gelu),
        Flux.Dense(I => H),
        Flux.Dropout(cfg.hidden_dropout_prob),
        Flux.LayerNorm(H; eps = cfg.layer_norm_eps),
    )
end

function (layer::BertLayer)(x::AbstractArray{Float32, 3}, pad_mask::AbstractMatrix{Float32})
    attn_out = layer.attention(x, pad_mask)
    ffn = layer.output(layer.intermediate(attn_out))
    return layer.norm(attn_out .+ layer.dropout(ffn))
end

"""
    BertBackbone

Full BERT encoder stack returning sequence hidden states and pooled representations.
"""
struct BertBackbone{E, L}
    embeddings::E
    layers::Vector{L}
    config::BertConfig
end
Flux.@layer BertBackbone

function BertBackbone(cfg::BertConfig)
    emb = BertEmbeddings(cfg)
    layers = [BertLayer(cfg) for _ in 1:cfg.num_hidden_layers]
    return BertBackbone(emb, layers, cfg)
end

function mean_pool(hidden::AbstractArray{Float32, 3}, pad_mask::AbstractMatrix{Float32})
    S, B = size(pad_mask)
    m = reshape(pad_mask, 1, S, B)
    summed = dropdims(sum(hidden .* m; dims = 2); dims = 2)
    denom = max.(sum(pad_mask; dims = 1), 1.0f0)
    return summed ./ denom
end

function char_embedding_pool(
    bb::BertBackbone,
    token_ids::AbstractMatrix{<:Integer},
    pad_mask::AbstractMatrix{Float32},
)
    raw_emb = bb.embeddings.word_embeddings(token_ids)
    cfg = bb.config
    eff_mask = Zygote.ChainRulesCore.ignore_derivatives() do
        char_mask = Float32.((token_ids .!= cfg.pad_token_id) .& (token_ids .!= cfg.cls_token_id) .& (token_ids .!= cfg.sep_token_id))
        has_chars = sum(char_mask; dims = 1) .> 0.5f0
        return ifelse.(has_chars, char_mask, pad_mask)
    end
    return mean_pool(raw_emb, eff_mask)
end

function (bb::BertBackbone)(
    token_ids::AbstractMatrix{<:Integer},
    token_type_ids::AbstractMatrix{<:Integer},
    pad_mask::AbstractMatrix{Float32},
)
    h = bb.embeddings(token_ids, token_type_ids)
    for layer in bb.layers
        h = layer(h, pad_mask)
    end
    pooled = mean_pool(h, pad_mask)
    return h, pooled
end

"""
    load_safetensors(path::AbstractString) -> Dict{String, Array{Float32}}

Parse a HuggingFace `.safetensors` binary file in pure Julia and return `Dict(name => Array{Float32})`.
Returned arrays preserve the row-major flattened byte order reshaped cleanly when mapped by callers.
"""
function load_safetensors(path::AbstractString)
    bytes = read(path)
    length(bytes) >= 8 || error("Invalid safetensors file (too small): $path")
    header_len = Int(reinterpret(UInt64, @view bytes[1:8])[1])
    data_start = 8 + header_len
    header_str = String(@view bytes[9:data_start])
    meta = JSON3.read(header_str)

    tensors = Dict{String, Tuple{Vector{Int}, Vector{Float32}}}()
    for (k_sym, info) in pairs(meta)
        k = String(k_sym)
        k == "__metadata__" && continue
        dtype = String(info[:dtype])
        shape = Int[Int(d) for d in info[:shape]]
        offsets = info[:data_offsets]
        s_off = Int(offsets[1]) + data_start + 1
        e_off = Int(offsets[2]) + data_start

        raw_slice = @view bytes[s_off:e_off]
        vals = if dtype == "F32"
            collect(reinterpret(Float32, raw_slice))
        elseif dtype == "F16"
            Float32.(reinterpret(Float16, raw_slice))
        else
            error("Unsupported safetensors dtype '$dtype' for tensor '$k'")
        end
        tensors[k] = (shape, vals)
    end
    return tensors
end

function _load_dense_weight!(dense::Flux.Dense, tensors, w_key::String, b_key::String)
    w_shape, w_vals = tensors[w_key]
    length(w_shape) == 2 || error("Expected 2D weight for $w_key, got $w_shape")
    out_dim, in_dim = w_shape[1], w_shape[2]
    # PyTorch row-major [out_dim, in_dim] -> Julia reshape(in_dim, out_dim) -> permutedims(2, 1)
    copyto!(dense.weight, permutedims(reshape(w_vals, in_dim, out_dim), (2, 1)))
    if haskey(tensors, b_key)
        _, b_vals = tensors[b_key]
        copyto!(dense.bias, b_vals)
    end
end

function _load_layernorm!(ln::Flux.LayerNorm, tensors, w_key::String, b_key::String)
    _, w_vals = tensors[w_key]
    _, b_vals = tensors[b_key]
    copyto!(ln.diag.scale, w_vals)
    copyto!(ln.diag.bias, b_vals)
end

function _resolve_hf_key(tensors, suffix::String)
    if haskey(tensors, "bert." * suffix)
        return "bert." * suffix
    elseif haskey(tensors, suffix)
        return suffix
    else
        error("Tensor '$suffix' (or 'bert.$suffix') not found in safetensors weights.")
    end
end

function _load_bert_weights!(bb::BertBackbone, tensors)
    # Embeddings
    we_key = _resolve_hf_key(tensors, "embeddings.word_embeddings.weight")
    we_shape, we_vals = tensors[we_key]
    copyto!(bb.embeddings.word_embeddings.weight, reshape(we_vals, we_shape[2], we_shape[1]))

    pe_key = _resolve_hf_key(tensors, "embeddings.position_embeddings.weight")
    pe_shape, pe_vals = tensors[pe_key]
    copyto!(bb.embeddings.position_embeddings.weight, reshape(pe_vals, pe_shape[2], pe_shape[1]))

    te_key = _resolve_hf_key(tensors, "embeddings.token_type_embeddings.weight")
    te_shape, te_vals = tensors[te_key]
    te_mat = reshape(te_vals, te_shape[2], te_shape[1])
    if te_shape[1] == 1
        # Broadcast single token_type embedding across both segment 1 and segment 2
        bb.embeddings.token_type_embeddings.weight[:, 1] .= @view te_mat[:, 1]
        bb.embeddings.token_type_embeddings.weight[:, 2] .= @view te_mat[:, 1]
    else
        n_copy = min(size(bb.embeddings.token_type_embeddings.weight, 2), te_shape[1])
        bb.embeddings.token_type_embeddings.weight[:, 1:n_copy] .= @view te_mat[:, 1:n_copy]
    end

    _load_layernorm!(
        bb.embeddings.norm,
        tensors,
        _resolve_hf_key(tensors, "embeddings.LayerNorm.weight"),
        _resolve_hf_key(tensors, "embeddings.LayerNorm.bias"),
    )

    for (idx, layer) in enumerate(bb.layers)
        p = "encoder.layer.$(idx - 1)"
        _load_dense_weight!(layer.attention.query, tensors, _resolve_hf_key(tensors, "$p.attention.self.query.weight"), _resolve_hf_key(tensors, "$p.attention.self.query.bias"))
        _load_dense_weight!(layer.attention.key, tensors, _resolve_hf_key(tensors, "$p.attention.self.key.weight"), _resolve_hf_key(tensors, "$p.attention.self.key.bias"))
        _load_dense_weight!(layer.attention.value, tensors, _resolve_hf_key(tensors, "$p.attention.self.value.weight"), _resolve_hf_key(tensors, "$p.attention.self.value.bias"))
        _load_dense_weight!(layer.attention.out_proj, tensors, _resolve_hf_key(tensors, "$p.attention.output.dense.weight"), _resolve_hf_key(tensors, "$p.attention.output.dense.bias"))
        _load_layernorm!(layer.attention.norm, tensors, _resolve_hf_key(tensors, "$p.attention.output.LayerNorm.weight"), _resolve_hf_key(tensors, "$p.attention.output.LayerNorm.bias"))

        _load_dense_weight!(layer.intermediate, tensors, _resolve_hf_key(tensors, "$p.intermediate.dense.weight"), _resolve_hf_key(tensors, "$p.intermediate.dense.bias"))
        _load_dense_weight!(layer.output, tensors, _resolve_hf_key(tensors, "$p.output.dense.weight"), _resolve_hf_key(tensors, "$p.output.dense.bias"))
        _load_layernorm!(layer.norm, tensors, _resolve_hf_key(tensors, "$p.output.LayerNorm.weight"), _resolve_hf_key(tensors, "$p.output.LayerNorm.bias"))
    end
    return bb
end

function _ensure_pretrained_files(repo_or_dir::AbstractString, cache_dir::AbstractString)
    if isdir(repo_or_dir)
        return String(repo_or_dir)
    end
    local_dir = joinpath(cache_dir, replace(repo_or_dir, "/" => "__"))
    mkpath(local_dir)
    base_url = "https://huggingface.co/$repo_or_dir/resolve/main"
    for fname in ("config.json", "vocab.txt", "model.safetensors")
        dest = joinpath(local_dir, fname)
        if !isfile(dest)
            url = "$base_url/$fname"
            @info "Downloading HuggingFace model artifact..." repo=repo_or_dir file=fname
            Downloads.download(url, dest)
        end
    end
    return local_dir
end

"""
    load_pretrained_bert(
        repo_or_dir::AbstractString;
        cache_dir::AbstractString = "data/pretrained",
        num_layers::Union{Int, Nothing} = nothing,
    ) -> Tuple{BertBackbone, CharBertTokenizer, BertConfig}

Load a pretrained HuggingFace BERT model (`config.json`, `vocab.txt`, `model.safetensors`)
from either a local directory or a HuggingFace repo identifier.
"""
function load_pretrained_bert(
    repo_or_dir::AbstractString;
    cache_dir::AbstractString = "data/pretrained",
    num_layers::Union{Int, Nothing} = nothing,
)
    model_dir = _ensure_pretrained_files(repo_or_dir, cache_dir)
    cfg_json = JSON3.read(read(joinpath(model_dir, "config.json"), String))
    vocab_size = Int(cfg_json[:vocab_size])
    tok = load_vocab(joinpath(model_dir, "vocab.txt"); max_vocab_size = vocab_size)

    n_layers = Int(cfg_json[:num_hidden_layers])
    if num_layers !== nothing
        n_layers = min(n_layers, num_layers)
    end

    cfg = BertConfig(
        vocab_size = Int(cfg_json[:vocab_size]),
        hidden_size = Int(cfg_json[:hidden_size]),
        num_hidden_layers = n_layers,
        num_attention_heads = Int(cfg_json[:num_attention_heads]),
        intermediate_size = Int(cfg_json[:intermediate_size]),
        max_position_embeddings = Int(cfg_json[:max_position_embeddings]),
        type_vocab_size = Int(get(cfg_json, :type_vocab_size, 2)),
        hidden_dropout_prob = Float32(get(cfg_json, :hidden_dropout_prob, 0.1)),
        attention_probs_dropout_prob = Float32(get(cfg_json, :attention_probs_dropout_prob, 0.1)),
        layer_norm_eps = Float32(get(cfg_json, :layer_norm_eps, 1e-12)),
        pad_token_id = tok.pad_id,
        cls_token_id = tok.cls_id,
        sep_token_id = tok.sep_id,
    )

    bb = BertBackbone(cfg)
    tensors = load_safetensors(joinpath(model_dir, "model.safetensors"))
    _load_bert_weights!(bb, tensors)
    return bb, tok, cfg
end
