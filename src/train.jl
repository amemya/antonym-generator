using DataFrames
using Flux
using JLD2
using JSON3
using LinearAlgebra
using Optimisers
using Random
using Statistics
using Zygote

function _build_rel_maps(df::DataFrame)
    ant_map = Dict{String, Set{String}}()
    syn_map = Dict{String, Set{String}}()
    ant_pairs = Tuple{String, String}[]
    for row in eachrow(df)
        w1 = String(row.w1)
        w2 = String(row.w2)
        lbl = Int(row.label)
        if lbl == LABEL_ANTONYM
            push!(get!(ant_map, w1, Set{String}()), w2)
            push!(ant_pairs, (w1, w2))
        elseif lbl == LABEL_SYNONYM
            push!(get!(syn_map, w1, Set{String}()), w2)
        end
    end
    return ant_pairs, ant_map, syn_map
end

function _prepare_bi_batch(
    batch_pairs::AbstractVector{Tuple{String, String}},
    ant_map::Dict{String, Set{String}},
    syn_map::Dict{String, Set{String}},
    rng::AbstractRNG;
    max_hard_negs::Int = 32,
)
    B = length(batch_pairs)
    queries = String[p[1] for p in batch_pairs]
    cands = String[p[2] for p in batch_pairs]
    cand_set = Set{String}(cands)

    # Append explicit synonym hard negatives for queries in the batch
    extra_cands = String[]
    for q in queries
        length(extra_cands) >= max_hard_negs && break
        syns = get(syn_map, q, nothing)
        syns === nothing && continue
        syn_list = collect(syns)
        s = syn_list[rand(rng, 1:length(syn_list))]
        if !(s in cand_set)
            push!(cand_set, s)
            push!(extra_cands, s)
        end
    end
    append!(cands, extra_cands)
    M = length(cands)

    pos_mask = zeros(Float32, B, M)
    hard_neg_mask = zeros(Float32, B, M)
    @inbounds for i in 1:B
        q = queries[i]
        ants = get(ant_map, q, nothing)
        syns = get(syn_map, q, nothing)
        for j in 1:M
            c = cands[j]
            if ants !== nothing && (c in ants)
                pos_mask[i, j] = 1.0f0
            elseif (syns !== nothing && (c in syns)) || c == q
                hard_neg_mask[i, j] = 1.0f0
            end
        end
    end
    return queries, cands, pos_mask, hard_neg_mask
end

"""
    train_bi_encoder!(
        model::BiEncoder,
        tok::CharBertTokenizer,
        train_df::DataFrame,
        val_df::DataFrame,
        vocab_words::AbstractVector{<:AbstractString};
        epochs::Int = 5,
        batch_size::Int = 64,
        lr::Float32 = 3.0f-4,
        freeze_backbone::Bool = false,
        backbone_lr_ratio::Float32 = 0.2f0,
        temperature::Float32 = 0.07f0,
        hard_neg_weight::Float32 = 3.0f0,
        max_len::Int = 16,
        max_train_pairs::Union{Int, Nothing} = nothing,
        device_cfg::DeviceConfig = DeviceConfig(:cpu, false),
        seed::Int = 42,
    )

Train `BiEncoder` with Multi-positive InfoNCE loss using in-batch negatives and explicit
synonym hard negatives. Updates `model` with the best validation checkpoint and returns `(model, history)`.
"""
function train_bi_encoder!(
    model::BiEncoder,
    tok::CharBertTokenizer,
    train_df::DataFrame,
    val_df::DataFrame,
    vocab_words::AbstractVector{<:AbstractString};
    epochs::Int = 5,
    batch_size::Int = 64,
    lr::Float32 = 3.0f-4,
    freeze_backbone::Bool = false,
    backbone_lr_ratio::Float32 = 0.2f0,
    temperature::Float32 = 0.07f0,
    hard_neg_weight::Float32 = 3.0f0,
    max_len::Int = 16,
    max_train_pairs::Union{Int, Nothing} = nothing,
    device_cfg::DeviceConfig = DeviceConfig(:cpu, false),
    seed::Int = 42,
)
    rng = MersenneTwister(seed)
    train_pairs, train_ant_map, train_syn_map = _build_rel_maps(train_df)
    val_pairs, val_ant_map, val_syn_map = _build_rel_maps(val_df)

    isempty(train_pairs) && error("No antonym pairs in train_df.")
    if max_train_pairs !== nothing
        if length(train_pairs) > max_train_pairs
            train_pairs = train_pairs[shuffle(rng, 1:length(train_pairs))[1:max_train_pairs]]
        end
        val_cap = max(div(max_train_pairs, 2), 256)
        if length(val_pairs) > val_cap
            val_pairs = val_pairs[shuffle(MersenneTwister(seed + 1), 1:length(val_pairs))[1:val_cap]]
        end
    end

    model_dev = to_device(device_cfg, model)
    opt_state = Optimisers.setup(Optimisers.AdamW(lr, (0.9f0, 0.999f0), 1.0f-4), model_dev)
    if freeze_backbone
        Optimisers.freeze!(opt_state.backbone)
    else
        Optimisers.adjust!(opt_state.backbone, lr * backbone_lr_ratio)
    end

    best_val_loss = Inf32
    best_state = Flux.state(cpu_device(model_dev))
    train_losses = Float32[]
    val_losses = Float32[]

    for epoch in 1:epochs
        Flux.trainmode!(model_dev)
        perm = shuffle(rng, 1:length(train_pairs))
        epoch_loss = 0.0
        n_batches = 0

        for start_idx in 1:batch_size:length(perm)
            end_idx = min(start_idx + batch_size - 1, length(perm))
            batch_pairs = train_pairs[perm[start_idx:end_idx]]
            queries, cands, pos_mask, hard_neg_mask = _prepare_bi_batch(
                batch_pairs,
                train_ant_map,
                train_syn_map,
                rng,
            )

            q_ids, q_tids, q_mask = encode_batch_single(tok, queries; max_len = max_len, dynamic_pad = true)
            k_ids, k_tids, k_mask = encode_batch_single(tok, cands; max_len = max_len, dynamic_pad = true)

            q_ids_d = to_device(device_cfg, q_ids)
            q_tids_d = to_device(device_cfg, q_tids)
            q_mask_d = to_device(device_cfg, q_mask)
            k_ids_d = to_device(device_cfg, k_ids)
            k_tids_d = to_device(device_cfg, k_tids)
            k_mask_d = to_device(device_cfg, k_mask)
            pos_d = to_device(device_cfg, pos_mask)
            hneg_d = to_device(device_cfg, hard_neg_mask)

            loss, grads = Zygote.withgradient(model_dev) do m
                q_emb = encode_query(m, q_ids_d, q_tids_d, q_mask_d)
                k_emb = encode_candidate(m, k_ids_d, k_tids_d, k_mask_d)
                multi_positive_infonce_loss(
                    q_emb,
                    k_emb,
                    pos_d,
                    hneg_d;
                    temperature = temperature,
                    hard_neg_weight = hard_neg_weight,
                )
            end
            opt_state, model_dev = Optimisers.update!(opt_state, model_dev, grads[1])
            epoch_loss += Float64(loss)
            n_batches += 1
        end

        mean_train = Float32(epoch_loss / max(n_batches, 1))
        push!(train_losses, mean_train)

        # Validation pass
        mean_val = mean_train
        if !isempty(val_pairs)
            Flux.testmode!(model_dev)
            v_loss = 0.0
            v_batches = 0
            for start_idx in 1:batch_size:length(val_pairs)
                end_idx = min(start_idx + batch_size - 1, length(val_pairs))
                batch_pairs = val_pairs[start_idx:end_idx]
                queries, cands, pos_mask, hard_neg_mask = _prepare_bi_batch(
                    batch_pairs,
                    val_ant_map,
                    val_syn_map,
                    MersenneTwister(seed + v_batches),
                )
                q_ids, q_tids, q_mask = encode_batch_single(tok, queries; max_len = max_len, dynamic_pad = true)
                k_ids, k_tids, k_mask = encode_batch_single(tok, cands; max_len = max_len, dynamic_pad = true)
                q_emb = encode_query(model_dev, to_device(device_cfg, q_ids), to_device(device_cfg, q_tids), to_device(device_cfg, q_mask))
                k_emb = encode_candidate(model_dev, to_device(device_cfg, k_ids), to_device(device_cfg, k_tids), to_device(device_cfg, k_mask))
                l = multi_positive_infonce_loss(
                    q_emb,
                    k_emb,
                    to_device(device_cfg, pos_mask),
                    to_device(device_cfg, hard_neg_mask);
                    temperature = temperature,
                    hard_neg_weight = hard_neg_weight,
                )
                v_loss += Float64(l)
                v_batches += 1
            end
            mean_val = Float32(v_loss / max(v_batches, 1))
        end
        push!(val_losses, mean_val)

        if mean_val <= best_val_loss
            best_val_loss = mean_val
            best_state = Flux.state(cpu_device(model_dev))
        end

        @info "Stage 1 Bi-Encoder Epoch $epoch/$epochs" train_loss=round(mean_train; digits=4) val_loss=round(mean_val; digits=4)
    end

    Flux.loadmodel!(model, best_state)
    Flux.testmode!(model)
    return model, (train_losses = train_losses, val_losses = val_losses, best_val_loss = best_val_loss)
end

function _encode_word_dict(
    bi_model::BiEncoder,
    tok::CharBertTokenizer,
    words::AbstractVector{<:AbstractString};
    batch_size::Int = 256,
    max_len::Int = 16,
    device_cfg::DeviceConfig = DeviceConfig(:cpu, false),
)
    bi_dev = to_device(device_cfg, bi_model)
    Flux.testmode!(bi_dev)

    proj_dim = size(bi_dev.candidate_skip.weight, 1)
    n = length(words)
    q_mat = Matrix{Float32}(undef, proj_dim, n)
    k_mat = Matrix{Float32}(undef, proj_dim, n)

    for start_idx in 1:batch_size:n
        end_idx = min(start_idx + batch_size - 1, n)
        bw = @view words[start_idx:end_idx]
        ids, tids, mask = encode_batch_single(tok, bw; max_len = max_len, dynamic_pad = true)
        ids_d = to_device(device_cfg, ids)
        tids_d = to_device(device_cfg, tids)
        mask_d = to_device(device_cfg, mask)
        q_mat[:, start_idx:end_idx] = cpu_device(encode_query(bi_dev, ids_d, tids_d, mask_d))
        k_mat[:, start_idx:end_idx] = cpu_device(encode_candidate(bi_dev, ids_d, tids_d, mask_d))
    end
    return q_mat, k_mat
end

"""
    mine_hard_negatives(
        bi_model::BiEncoder,
        tok::CharBertTokenizer,
        train_df::DataFrame,
        vocab_words::AbstractVector{<:AbstractString};
        top_k::Int = 30,
        max_per_query::Int = 3,
        max_queries::Union{Int, Nothing} = nothing,
        batch_size::Int = 256,
        max_len::Int = 16,
        device_cfg::DeviceConfig = DeviceConfig(:cpu, false),
    ) -> DataFrame

Run Stage 1 retrieval over training queries and mine high-scoring false-positive candidates
(prioritizing ground-truth synonyms ranked high by Stage 1) for Stage 2 Cross-Encoder training.
"""
function mine_hard_negatives(
    bi_model::BiEncoder,
    tok::CharBertTokenizer,
    train_df::DataFrame,
    vocab_words::AbstractVector{<:AbstractString};
    top_k::Int = 30,
    max_per_query::Int = 3,
    max_queries::Union{Int, Nothing} = nothing,
    batch_size::Int = 256,
    max_len::Int = 16,
    device_cfg::DeviceConfig = DeviceConfig(:cpu, false),
)
    _, ant_map, syn_map = _build_rel_maps(train_df)
    queries = sort(collect(keys(ant_map)))
    if max_queries !== nothing && length(queries) > max_queries
        queries = queries[1:max_queries]
    end
    isempty(queries) && return DataFrame(w1 = String[], w2 = String[], label = Int[])

    cand_pool = if max_queries !== nothing && length(vocab_words) > 5000
        sort(collect(union(Set{String}(String.(train_df.w1)), Set{String}(String.(train_df.w2)))))
    else
        String[String(w) for w in vocab_words]
    end

    _, k_vocab = _encode_word_dict(bi_model, tok, cand_pool; batch_size = batch_size, max_len = max_len, device_cfg = device_cfg)
    q_embs, _ = _encode_word_dict(bi_model, tok, queries; batch_size = batch_size, max_len = max_len, device_cfg = device_cfg)

    w1_out = String[]
    w2_out = String[]
    lbl_out = Int[]

    for start_idx in 1:batch_size:length(queries)
        end_idx = min(start_idx + batch_size - 1, length(queries))
        scores_blk = @view(q_embs[:, start_idx:end_idx])' * k_vocab # (b, V)
        for local_i in 1:(end_idx - start_idx + 1)
            q = queries[start_idx + local_i - 1]
            ants = ant_map[q]
            syns = get(syn_map, q, Set{String}())
            row_scores = @view scores_blk[local_i, :]
            top_indices = partialsortperm(row_scores, 1:min(top_k, length(cand_pool)); rev = true)

            added = 0
            # Prioritize mined synonyms first, then top false-positive unrelated words
            for idx in top_indices
                c = cand_pool[idx]
                (c == q || (c in ants)) && continue
                if c in syns
                    push!(w1_out, q); push!(w2_out, c); push!(lbl_out, LABEL_SYNONYM)
                    added += 1
                    added >= max_per_query && break
                end
            end
            if added < max_per_query
                for idx in top_indices
                    c = cand_pool[idx]
                    (c == q || (c in ants) || (c in syns)) && continue
                    push!(w1_out, q); push!(w2_out, c); push!(lbl_out, LABEL_UNRELATED)
                    added += 1
                    added >= max_per_query && break
                end
            end
        end
    end

    return DataFrame(w1 = w1_out, w2 = w2_out, label = lbl_out)
end

function _build_pair_features_batch(
    w1s::AbstractVector{<:AbstractString},
    w2s::AbstractVector{<:AbstractString},
    word_to_idx::Dict{String, Int},
    q_embs::Matrix{Float32},
    k_embs::Matrix{Float32},
)
    surf = extract_batch_surface_features(w1s, w2s)
    B = length(w1s)
    s1 = Matrix{Float32}(undef, 2, B)
    @inbounds for b in 1:B
        i1 = word_to_idx[String(w1s[b])]
        i2 = word_to_idx[String(w2s[b])]
        s1[1, b] = dot(@view(q_embs[:, i1]), @view(k_embs[:, i2]))
        s1[2, b] = dot(@view(k_embs[:, i1]), @view(k_embs[:, i2]))
    end
    return vcat(surf, s1)
end

"""
    train_cross_encoder!(
        model::CrossEncoder,
        bi_model::BiEncoder,
        tok::CharBertTokenizer,
        train_df::DataFrame,
        val_df::DataFrame;
        hard_neg_df::Union{Nothing, DataFrame} = nothing,
        epochs::Int = 4,
        batch_size::Int = 64,
        lr::Float32 = 2.0f-4,
        freeze_backbone::Bool = false,
        backbone_lr_ratio::Float32 = 0.2f0,
        max_len::Int = 32,
        max_train_pairs::Union{Int, Nothing} = nothing,
        device_cfg::DeviceConfig = DeviceConfig(:cpu, false),
        seed::Int = 42,
    )

Train the 3-class `CrossEncoder` (`Unrelated`, `Synonym`, `Antonym`) combining transformer
pair encoding, lexical surface features, and Stage 1 Bi-Encoder scores.
Updates `model` with the best validation checkpoint and returns `(model, history)`.
"""
function train_cross_encoder!(
    model::CrossEncoder,
    bi_model::BiEncoder,
    tok::CharBertTokenizer,
    train_df::DataFrame,
    val_df::DataFrame;
    hard_neg_df::Union{Nothing, DataFrame} = nothing,
    epochs::Int = 4,
    batch_size::Int = 64,
    lr::Float32 = 2.0f-4,
    freeze_backbone::Bool = false,
    backbone_lr_ratio::Float32 = 0.2f0,
    max_len::Int = 32,
    max_train_pairs::Union{Int, Nothing} = nothing,
    device_cfg::DeviceConfig = DeviceConfig(:cpu, false),
    seed::Int = 42,
)
    rng = MersenneTwister(seed)
    full_train = if hard_neg_df !== nothing && nrow(hard_neg_df) > 0
        vcat(train_df, hard_neg_df)
    else
        copy(train_df)
    end

    val_df_eval = val_df
    if max_train_pairs !== nothing
        if nrow(full_train) > max_train_pairs
            idx_sub = shuffle(rng, 1:nrow(full_train))[1:max_train_pairs]
            full_train = full_train[idx_sub, :]
        end
        val_cap = max(div(max_train_pairs, 2), 256)
        if nrow(val_df_eval) > val_cap
            idx_val = shuffle(MersenneTwister(seed + 1), 1:nrow(val_df_eval))[1:val_cap]
            val_df_eval = val_df_eval[idx_val, :]
        end
    end

    # Precompute Stage 1 representations for all words in train + val once
    all_words_set = Set{String}()
    for df in (full_train, val_df_eval)
        for row in eachrow(df)
            push!(all_words_set, String(row.w1))
            push!(all_words_set, String(row.w2))
        end
    end
    all_words = sort(collect(all_words_set))
    word_to_idx = Dict{String, Int}(w => i for (i, w) in enumerate(all_words))
    q_embs, k_embs = _encode_word_dict(bi_model, tok, all_words; batch_size = 256, max_len = 16, device_cfg = device_cfg)

    # Compute capped inverse-frequency class weights for 3 classes (1..3)
    counts = zeros(Int, 3)
    for l in full_train.label
        counts[Int(l) + 1] += 1
    end
    total = sum(counts)
    raw_w = Float32[c > 0 ? min(Float32(total / (3.0 * c)), 3.0f0) : 1.0f0 for c in counts]
    class_weights = raw_w ./ Float32(mean(raw_w))

    model_dev = to_device(device_cfg, model)
    opt_state = Optimisers.setup(Optimisers.AdamW(lr, (0.9f0, 0.999f0), 1.0f-4), model_dev)
    if freeze_backbone
        Optimisers.freeze!(opt_state.backbone)
    else
        Optimisers.adjust!(opt_state.backbone, lr * backbone_lr_ratio)
    end

    train_w1 = String[String(x) for x in full_train.w1]
    train_w2 = String[String(x) for x in full_train.w2]
    train_y = Int[Int(x) + 1 for x in full_train.label]

    val_w1 = String[String(x) for x in val_df_eval.w1]
    val_w2 = String[String(x) for x in val_df_eval.w2]
    val_y = Int[Int(x) + 1 for x in val_df_eval.label]

    best_val_loss = Inf32
    best_state = Flux.state(cpu_device(model_dev))
    train_losses = Float32[]
    val_losses = Float32[]
    val_accs = Float32[]

    for epoch in 1:epochs
        Flux.trainmode!(model_dev)
        perm = shuffle(rng, 1:length(train_y))
        epoch_loss = 0.0
        n_batches = 0

        for start_idx in 1:batch_size:length(perm)
            end_idx = min(start_idx + batch_size - 1, length(perm))
            idx = perm[start_idx:end_idx]
            bw1 = @view train_w1[idx]
            bw2 = @view train_w2[idx]
            by = @view train_y[idx]

            ids, tids, mask = encode_batch_pair(tok, bw1, bw2; max_len = max_len, dynamic_pad = true)
            pf = _build_pair_features_batch(bw1, bw2, word_to_idx, q_embs, k_embs)

            ids_d = to_device(device_cfg, ids)
            tids_d = to_device(device_cfg, tids)
            mask_d = to_device(device_cfg, mask)
            pf_d = to_device(device_cfg, pf)

            loss, grads = Zygote.withgradient(model_dev) do m
                logits = m(ids_d, tids_d, mask_d, pf_d)
                cross_encoder_loss(logits, by; class_weights = class_weights)
            end
            opt_state, model_dev = Optimisers.update!(opt_state, model_dev, grads[1])
            epoch_loss += Float64(loss)
            n_batches += 1
        end

        mean_train = Float32(epoch_loss / max(n_batches, 1))
        push!(train_losses, mean_train)

        # Validation
        mean_val = mean_train
        val_acc = 0.0f0
        if !isempty(val_y)
            Flux.testmode!(model_dev)
            v_loss = 0.0
            v_batches = 0
            correct = 0
            for start_idx in 1:batch_size:length(val_y)
                end_idx = min(start_idx + batch_size - 1, length(val_y))
                bw1 = @view val_w1[start_idx:end_idx]
                bw2 = @view val_w2[start_idx:end_idx]
                by = @view val_y[start_idx:end_idx]

                ids, tids, mask = encode_batch_pair(tok, bw1, bw2; max_len = max_len, dynamic_pad = true)
                pf = _build_pair_features_batch(bw1, bw2, word_to_idx, q_embs, k_embs)
                logits = model_dev(
                    to_device(device_cfg, ids),
                    to_device(device_cfg, tids),
                    to_device(device_cfg, mask),
                    to_device(device_cfg, pf),
                )
                l = cross_encoder_loss(logits, by; class_weights = class_weights)
                v_loss += Float64(l)
                v_batches += 1
                logits_cpu = cpu_device(logits)
                for j in 1:length(by)
                    pred = argmax(@view logits_cpu[:, j])
                    pred == by[j] && (correct += 1)
                end
            end
            mean_val = Float32(v_loss / max(v_batches, 1))
            val_acc = Float32(correct / length(val_y))
        end
        push!(val_losses, mean_val)
        push!(val_accs, val_acc)

        if mean_val <= best_val_loss
            best_val_loss = mean_val
            best_state = Flux.state(cpu_device(model_dev))
        end

        @info "Stage 2 Cross-Encoder Epoch $epoch/$epochs" train_loss=round(mean_train; digits=4) val_loss=round(mean_val; digits=4) val_acc=round(val_acc; digits=4)
    end

    Flux.loadmodel!(model, best_state)
    Flux.testmode!(model)
    return model, (train_losses = train_losses, val_losses = val_losses, val_accs = val_accs, best_val_loss = best_val_loss)
end

"""
    save_artifacts(
        artifacts_dir::AbstractString,
        pipeline::AntonymPipeline,
        metrics::Dict,
    )

Save model weights (`bi_encoder.jld2`, `cross_encoder.jld2`), `vocab_index.jld2`, and `metrics.json`
in CPU-portable format so checkpoints trained on Windows GPU load cleanly on Mac CPU and vice versa.
"""
function save_artifacts(artifacts_dir::AbstractString, pipeline::AntonymPipeline, metrics::Dict)
    mkpath(artifacts_dir)

    bi_cpu = cpu_device(pipeline.bi_encoder)
    ce_cpu = cpu_device(pipeline.cross_encoder)

    jldsave(
        joinpath(artifacts_dir, "bi_encoder.jld2");
        state = Flux.state(bi_cpu),
        config = bi_cpu.backbone.config,
        proj_dim = size(bi_cpu.candidate_skip.weight, 1),
    )
    jldsave(
        joinpath(artifacts_dir, "cross_encoder.jld2");
        state = Flux.state(ce_cpu),
        config = ce_cpu.backbone.config,
        pair_feat_dim = ce_cpu.pair_feat_dim,
    )
    jldsave(
        joinpath(artifacts_dir, "vocab_index.jld2");
        vocab_words = pipeline.vocab_words,
        reading_map = pipeline.reading_map,
        vocab_k_embs = pipeline.vocab_k_embs,
        tokenizer_vocab = pipeline.tokenizer.vocab,
    )
    open(joinpath(artifacts_dir, "metrics.json"), "w") do io
        JSON3.pretty(io, metrics)
    end
    return artifacts_dir
end

"""
    load_artifacts(artifacts_dir::AbstractString; device_cfg::DeviceConfig = DeviceConfig(:cpu, false)) -> AntonymPipeline

Load saved `BiEncoder`, `CrossEncoder`, tokenizer vocabulary, and precomputed Stage 1 index from `artifacts_dir`.
"""
function load_artifacts(artifacts_dir::AbstractString; device_cfg::DeviceConfig = DeviceConfig(:cpu, false))
    bi_data = JLD2.load(joinpath(artifacts_dir, "bi_encoder.jld2"))
    ce_data = JLD2.load(joinpath(artifacts_dir, "cross_encoder.jld2"))
    idx_data = JLD2.load(joinpath(artifacts_dir, "vocab_index.jld2"))

    tok_vocab = Vector{String}(idx_data["tokenizer_vocab"])
    tok_to_id = Dict{String, Int}(t => i for (i, t) in enumerate(tok_vocab))
    tok = CharBertTokenizer(
        tok_vocab,
        tok_to_id,
        get(tok_to_id, "[PAD]", 1),
        get(tok_to_id, "[UNK]", 2),
        get(tok_to_id, "[CLS]", 3),
        get(tok_to_id, "[SEP]", 4),
        get(tok_to_id, "[MASK]", 5),
    )

    bi_cfg = bi_data["config"]
    bi_model = BiEncoder(BertBackbone(bi_cfg); proj_dim = Int(bi_data["proj_dim"]))
    Flux.loadmodel!(bi_model, bi_data["state"])
    Flux.testmode!(bi_model)

    ce_cfg = ce_data["config"]
    ce_model = CrossEncoder(BertBackbone(ce_cfg); pair_feat_dim = Int(ce_data["pair_feat_dim"]))
    Flux.loadmodel!(ce_model, ce_data["state"])
    Flux.testmode!(ce_model)

    return AntonymPipeline(
        tok,
        bi_model,
        ce_model,
        Vector{String}(idx_data["vocab_words"]),
        Dict{String, String}(idx_data["reading_map"]),
        Matrix{Float32}(idx_data["vocab_k_embs"]),
        device_cfg,
    )
end
