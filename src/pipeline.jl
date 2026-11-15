using Flux
using LinearAlgebra
using Unicode

"""
    AntonymPipeline

End-to-end 2-stage antonym retrieval and reranking pipeline combining a `BiEncoder`
candidate retriever, a `CrossEncoder` 3-class reranker, and lexical post-processing.
"""
mutable struct AntonymPipeline
    tokenizer::CharBertTokenizer
    bi_encoder::BiEncoder
    cross_encoder::CrossEncoder
    vocab_words::Vector{String}
    reading_map::Dict{String, String}
    vocab_k_embs::Matrix{Float32}
    device_cfg::DeviceConfig
end

function AntonymPipeline(
    tokenizer::CharBertTokenizer,
    bi_encoder::BiEncoder,
    cross_encoder::CrossEncoder,
    vocab_words::Vector{String},
    reading_map::Dict{String, String};
    device_cfg::DeviceConfig = DeviceConfig(:cpu, false),
)
    return AntonymPipeline(
        tokenizer,
        to_device(device_cfg, bi_encoder),
        to_device(device_cfg, cross_encoder),
        vocab_words,
        reading_map,
        Matrix{Float32}(undef, 0, 0),
        device_cfg,
    )
end

"""
    build_vocab_index!(pipeline::AntonymPipeline; batch_size::Int = 256, max_len::Int = 16)

Precompute normalized candidate embeddings `K_vocab` for all words in `pipeline.vocab_words`.
"""
function build_vocab_index!(pipeline::AntonymPipeline; batch_size::Int = 256, max_len::Int = 16)
    V = length(pipeline.vocab_words)
    V > 0 || error("Cannot build index for empty vocabulary.")
    bi_dev = to_device(pipeline.device_cfg, pipeline.bi_encoder)
    Flux.testmode!(bi_dev)

    proj_dim = size(bi_dev.candidate_skip.weight, 1)
    k_mat = Matrix{Float32}(undef, proj_dim, V)

    for start_idx in 1:batch_size:V
        end_idx = min(start_idx + batch_size - 1, V)
        batch_words = @view pipeline.vocab_words[start_idx:end_idx]
        ids, tids, mask = encode_batch_single(pipeline.tokenizer, batch_words; max_len = max_len, dynamic_pad = true)
        ids_d = to_device(pipeline.device_cfg, ids)
        tids_d = to_device(pipeline.device_cfg, tids)
        mask_d = to_device(pipeline.device_cfg, mask)
        k_batch = encode_candidate(bi_dev, ids_d, tids_d, mask_d)
        k_mat[:, start_idx:end_idx] = cpu_device(k_batch)
    end

    pipeline.vocab_k_embs = k_mat
    return pipeline
end

function _lookup_reading(reading_map::AbstractDict{String, String}, word::AbstractString)
    norm = Unicode.normalize(String(word), :NFKC)
    return get(reading_map, norm, to_hiragana(norm))
end

"""
    postprocess_candidates(
        query_word::AbstractString,
        candidates::AbstractVector{<:AbstractString},
        scores::AbstractVector{<:Real},
        syn_probs::AbstractVector{<:Real},
        reading_map::AbstractDict{String, String};
        stage1_scores::AbstractVector{<:Real} = zeros(Float32, length(candidates)),
        top_n::Int = 10,
    )

Filter self-matches and reading-identical orthographic variants of `query_word`, deduplicate
candidates sharing the same normalized reading, and return the top `top_n` candidates.
"""
function postprocess_candidates(
    query_word::AbstractString,
    candidates::AbstractVector{<:AbstractString},
    scores::AbstractVector{<:Real},
    syn_probs::AbstractVector{<:Real},
    reading_map::AbstractDict{String, String};
    stage1_scores::AbstractVector{<:Real} = zeros(Float32, length(candidates)),
    top_n::Int = 10,
)
    n = length(candidates)
    length(scores) == n && length(syn_probs) == n && length(stage1_scores) == n ||
        error("Candidate and score vector lengths must match.")

    q_norm = Unicode.normalize(String(query_word), :NFKC)
    q_reading = _lookup_reading(reading_map, q_norm)

    order = sortperm(scores; rev = true)
    seen_words = Set{String}([q_norm])
    seen_readings = Set{String}([q_reading])

    out = NamedTuple{(:word, :reading, :antonym_prob, :synonym_prob, :stage1_score), Tuple{String, String, Float32, Float32, Float32}}[]
    for idx in order
        cand = Unicode.normalize(String(candidates[idx]), :NFKC)
        cand in seen_words && continue
        c_reading = _lookup_reading(reading_map, cand)
        c_reading in seen_readings && continue

        push!(seen_words, cand)
        push!(seen_readings, c_reading)
        push!(out, (
            word = cand,
            reading = c_reading,
            antonym_prob = Float32(scores[idx]),
            synonym_prob = Float32(syn_probs[idx]),
            stage1_score = Float32(stage1_scores[idx]),
        ))
        length(out) >= top_n && break
    end
    return out
end

"""
    retrieve_stage1(pipeline::AntonymPipeline, word::AbstractString; top_k::Int = 200, max_len::Int = 16)

Retrieve top `top_k` candidate words and their `(q1' * k2, k1' * k2)` Stage 1 scores for `word`.
"""
function retrieve_stage1(pipeline::AntonymPipeline, word::AbstractString; top_k::Int = 200, max_len::Int = 16)
    isempty(pipeline.vocab_k_embs) && build_vocab_index!(pipeline)

    bi_dev = to_device(pipeline.device_cfg, pipeline.bi_encoder)
    Flux.testmode!(bi_dev)

    ids, tids, mask = encode_batch_single(pipeline.tokenizer, [String(word)]; max_len = max_len, dynamic_pad = true)
    ids_d = to_device(pipeline.device_cfg, ids)
    tids_d = to_device(pipeline.device_cfg, tids)
    mask_d = to_device(pipeline.device_cfg, mask)

    q1 = vec(cpu_device(encode_query(bi_dev, ids_d, tids_d, mask_d)))
    k1 = vec(cpu_device(encode_candidate(bi_dev, ids_d, tids_d, mask_d)))

    q_scores = vec(q1' * pipeline.vocab_k_embs)
    k_scores = vec(k1' * pipeline.vocab_k_embs)

    q_norm = Unicode.normalize(String(word), :NFKC)
    order = sortperm(q_scores; rev = true)

    cands = String[]
    s_qk = Float32[]
    s_kk = Float32[]
    sizehint!(cands, top_k)
    for idx in order
        w = pipeline.vocab_words[idx]
        w == q_norm && continue
        push!(cands, w)
        push!(s_qk, Float32(q_scores[idx]))
        push!(s_kk, Float32(k_scores[idx]))
        length(cands) >= top_k && break
    end
    return cands, s_qk, s_kk
end

"""
    generate_antonyms(
        pipeline::AntonymPipeline,
        word::AbstractString;
        top_k_stage1::Int = 200,
        top_n::Int = 10,
        batch_size::Int = 128,
        max_len_pair::Int = 32,
    )

Run the full 2-stage pipeline (Stage 1 Bi-Encoder retrieval + Stage 2 Cross-Encoder reranking +
Stage 3 post-processing) for `word`.
"""
function generate_antonyms(
    pipeline::AntonymPipeline,
    word::AbstractString;
    top_k_stage1::Int = 200,
    top_n::Int = 10,
    batch_size::Int = 128,
    max_len_single::Int = 16,
    max_len_pair::Int = 32,
)
    cands, s_qk, s_kk = retrieve_stage1(pipeline, word; top_k = top_k_stage1, max_len = max_len_single)
    n_cands = length(cands)
    n_cands == 0 && return NamedTuple{(:word, :reading, :antonym_prob, :synonym_prob, :stage1_score), Tuple{String, String, Float32, Float32, Float32}}[]

    ce_dev = to_device(pipeline.device_cfg, pipeline.cross_encoder)
    Flux.testmode!(ce_dev)

    ant_probs = Vector{Float32}(undef, n_cands)
    syn_probs = Vector{Float32}(undef, n_cands)
    q_str = String(word)

    for start_idx in 1:batch_size:n_cands
        end_idx = min(start_idx + batch_size - 1, n_cands)
        m = end_idx - start_idx + 1
        w1s = fill(q_str, m)
        w2s = @view cands[start_idx:end_idx]

        ids, tids, mask = encode_batch_pair(pipeline.tokenizer, w1s, w2s; max_len = max_len_pair, dynamic_pad = true)
        surf = extract_batch_surface_features(w1s, w2s)
        s1_feats = vcat(reshape(s_qk[start_idx:end_idx], 1, m), reshape(s_kk[start_idx:end_idx], 1, m))
        pair_feats = vcat(surf, s1_feats)

        logits = ce_dev(
            to_device(pipeline.device_cfg, ids),
            to_device(pipeline.device_cfg, tids),
            to_device(pipeline.device_cfg, mask),
            to_device(pipeline.device_cfg, pair_feats),
        )
        probs = Flux.softmax(cpu_device(logits); dims = 1)
        syn_probs[start_idx:end_idx] = vec(probs[2, :])
        ant_probs[start_idx:end_idx] = vec(probs[3, :])
    end

    return postprocess_candidates(
        q_str,
        cands,
        ant_probs,
        syn_probs,
        pipeline.reading_map;
        stage1_scores = s_qk,
        top_n = top_n,
    )
end
