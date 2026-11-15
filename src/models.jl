using Flux
using LinearAlgebra
using NNlib
using Zygote

"""
    l2_normalize(x::AbstractMatrix{Float32}) -> AbstractMatrix{Float32}

L2-normalize columns of `x`.
"""
function l2_normalize(x::AbstractMatrix{Float32})
    return x ./ sqrt.(sum(abs2, x; dims = 1) .+ 1.0f-12)
end

"""
    BiEncoder

Dual-projection Bi-Encoder wrapping a shared `BertBackbone` with asymmetric `query_head`
and `candidate_head` plus direct linear skip connections from `[mean_pool; char_pool]`.
"""
struct BiEncoder
    backbone::BertBackbone
    query_head::Flux.Chain
    query_skip::Flux.Dense
    candidate_head::Flux.Chain
    candidate_skip::Flux.Dense
end
Flux.@layer BiEncoder

function BiEncoder(backbone::BertBackbone; proj_dim::Int = 256, dropout::Float32 = 0.1f0)
    H = backbone.config.hidden_size
    in_dim = 2 * H
    q_head = Flux.Chain(
        Flux.Dense(in_dim => H, NNlib.gelu),
        Flux.Dropout(dropout),
        Flux.Dense(H => proj_dim),
    )
    q_skip = Flux.Dense(in_dim => proj_dim)
    k_head = Flux.Chain(
        Flux.Dense(in_dim => H, NNlib.gelu),
        Flux.Dropout(dropout),
        Flux.Dense(H => proj_dim),
    )
    k_skip = Flux.Dense(in_dim => proj_dim)
    return BiEncoder(backbone, q_head, q_skip, k_head, k_skip)
end

function _biencoder_repr(
    model::BiEncoder,
    token_ids::AbstractMatrix{<:Integer},
    token_type_ids::AbstractMatrix{<:Integer},
    pad_mask::AbstractMatrix{Float32},
)
    _, pooled = model.backbone(token_ids, token_type_ids, pad_mask)
    char_p = char_embedding_pool(model.backbone, token_ids, pad_mask)
    return vcat(pooled, char_p)
end

"""
    encode_query(model::BiEncoder, token_ids, token_type_ids, pad_mask) -> Matrix{Float32}

Encode query words into L2-normalized `(proj_dim, B)` query vectors.
"""
function encode_query(
    model::BiEncoder,
    token_ids::AbstractMatrix{<:Integer},
    token_type_ids::AbstractMatrix{<:Integer},
    pad_mask::AbstractMatrix{Float32},
)
    repr = _biencoder_repr(model, token_ids, token_type_ids, pad_mask)
    z = model.query_head(repr) .+ model.query_skip(repr)
    return l2_normalize(z)
end

"""
    encode_candidate(model::BiEncoder, token_ids, token_type_ids, pad_mask) -> Matrix{Float32}

Encode candidate words into L2-normalized `(proj_dim, B)` candidate vectors.
"""
function encode_candidate(
    model::BiEncoder,
    token_ids::AbstractMatrix{<:Integer},
    token_type_ids::AbstractMatrix{<:Integer},
    pad_mask::AbstractMatrix{Float32},
)
    repr = _biencoder_repr(model, token_ids, token_type_ids, pad_mask)
    z = model.candidate_head(repr) .+ model.candidate_skip(repr)
    return l2_normalize(z)
end

"""
    multi_positive_infonce_loss(
        q_emb::AbstractMatrix{Float32},
        k_emb::AbstractMatrix{Float32},
        pos_mask::AbstractMatrix{Float32},
        hard_neg_mask::AbstractMatrix{Float32};
        temperature::Float32 = 0.07f0,
        hard_neg_weight::Float32 = 3.0f0,
    ) -> Float32

Compute Multi-positive InfoNCE loss over query embeddings `q_emb` (`(D, B)`) and candidate
embeddings `k_emb` (`(D, M)`), allowing multiple ground-truth antonyms per query via `pos_mask`
(`(B, M)`) and upweighting explicit synonym hard negatives via `hard_neg_mask` (`(B, M)`).
"""
function multi_positive_infonce_loss(
    q_emb::AbstractMatrix{Float32},
    k_emb::AbstractMatrix{Float32},
    pos_mask::AbstractMatrix{Float32},
    hard_neg_mask::AbstractMatrix{Float32};
    temperature::Float32 = 0.07f0,
    hard_neg_weight::Float32 = 3.0f0,
)
    S = (q_emb' * k_emb) ./ temperature
    log_w, pos_bias, valid, denom_valid = Zygote.ChainRulesCore.ignore_derivatives() do
        neg_only = hard_neg_mask .* (1.0f0 .- pos_mask)
        lw = log.(1.0f0 .+ (hard_neg_weight - 1.0f0) .* neg_only)
        pb = (1.0f0 .- pos_mask) .* (-1.0f4)
        v = Float32.(sum(pos_mask; dims = 2) .> 0.5f0)
        dv = max(sum(v), 1.0f0)
        return lw, pb, v, dv
    end

    pos_logits = S .+ pos_bias
    log_num = NNlib.logsumexp(pos_logits; dims = 2)
    log_denom = NNlib.logsumexp(S .+ log_w; dims = 2)

    row_loss = (log_denom .- log_num) .* valid
    return sum(row_loss) / denom_valid
end

"""
    CrossEncoder

Pairwise Cross-Encoder taking `[CLS] w1 [SEP] w2 [SEP]` through a `BertBackbone` and
combining `[cls_pooled; mean_pooled; pair_features]` to output 3-class logits:
`1: Unrelated`, `2: Synonym`, `3: Antonym`.
"""
struct CrossEncoder
    backbone::BertBackbone
    classifier::Flux.Chain
    pair_feat_dim::Int
end
Flux.@layer CrossEncoder

function CrossEncoder(backbone::BertBackbone; pair_feat_dim::Int = 10, dropout::Float32 = 0.1f0)
    H = backbone.config.hidden_size
    in_dim = 2 * H + pair_feat_dim
    clf = Flux.Chain(
        Flux.Dense(in_dim => H, NNlib.gelu),
        Flux.Dropout(dropout),
        Flux.Dense(H => 3),
    )
    return CrossEncoder(backbone, clf, pair_feat_dim)
end

function (m::CrossEncoder)(
    token_ids::AbstractMatrix{<:Integer},
    token_type_ids::AbstractMatrix{<:Integer},
    pad_mask::AbstractMatrix{Float32},
    pair_features::AbstractMatrix{Float32},
)
    hidden, mean_p = m.backbone(token_ids, token_type_ids, pad_mask)
    cls_p = hidden[:, 1, :]
    feat = vcat(cls_p, mean_p, pair_features)
    return m.classifier(feat)
end

function (m::CrossEncoder)(
    token_ids::AbstractMatrix{<:Integer},
    token_type_ids::AbstractMatrix{<:Integer},
    pad_mask::AbstractMatrix{Float32},
)
    B = size(token_ids, 2)
    zeros_pf = Zygote.ChainRulesCore.ignore_derivatives() do
        z = zeros(Float32, m.pair_feat_dim, B)
        Flux.get_device(pad_mask)(z)
    end
    return m(token_ids, token_type_ids, pad_mask, zeros_pf)
end

"""
    cross_encoder_loss(
        logits::AbstractMatrix{Float32},
        labels::AbstractVector{<:Integer};
        class_weights::Union{Nothing, AbstractVector{Float32}} = nothing,
    ) -> Float32

Compute 3-class cross-entropy loss for 1-based labels (`1: Unrelated`, `2: Synonym`, `3: Antonym`).
"""
function cross_encoder_loss(
    logits::AbstractMatrix{Float32},
    labels::AbstractVector{<:Integer};
    class_weights::Union{Nothing, AbstractVector{Float32}} = nothing,
)
    y_onehot, sample_w = Zygote.ChainRulesCore.ignore_derivatives() do
        dev = Flux.get_device(logits)
        cpu_labels = Array(labels)
        oh = Float32.(Flux.onehotbatch(cpu_labels, 1:3))
        if class_weights === nothing
            return dev(oh), nothing
        else
            w = Float32[class_weights[l] for l in cpu_labels]
            return dev(oh), dev(reshape(w, 1, length(w)))
        end
    end

    if sample_w === nothing
        return Flux.Losses.logitcrossentropy(logits, y_onehot)
    else
        log_probs = NNlib.logsoftmax(logits; dims = 1)
        per_sample = -sum(y_onehot .* log_probs; dims = 1)
        return sum(per_sample .* sample_w) / max(sum(sample_w), 1.0f0)
    end
end
