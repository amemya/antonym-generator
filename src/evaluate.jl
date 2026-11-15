using DataFrames
using Statistics

function _group_ground_truth(eval_df::DataFrame)
    gt_ant = Dict{String, Set{String}}()
    gt_syn = Dict{String, Set{String}}()
    for row in eachrow(eval_df)
        w1 = String(row.w1)
        w2 = String(row.w2)
        lbl = Int(row.label)
        if lbl == LABEL_ANTONYM
            push!(get!(gt_ant, w1, Set{String}()), w2)
        elseif lbl == LABEL_SYNONYM
            push!(get!(gt_syn, w1, Set{String}()), w2)
        end
    end
    return gt_ant, gt_syn
end

"""
    evaluate_stage1(
        pipeline::AntonymPipeline,
        eval_df::DataFrame;
        ks::AbstractVector{Int} = [50, 100, 200],
        max_queries::Union{Int, Nothing} = nothing,
    )

Evaluate Stage 1 (`BiEncoder`) candidate retrieval `Recall@K` for each `K in ks`.
"""
function evaluate_stage1(
    pipeline::AntonymPipeline,
    eval_df::DataFrame;
    ks::AbstractVector{Int} = [50, 100, 200],
    max_queries::Union{Int, Nothing} = nothing,
)
    isempty(pipeline.vocab_k_embs) && build_vocab_index!(pipeline)
    gt_ant, _ = _group_ground_truth(eval_df)
    queries = sort(collect(keys(gt_ant)))
    if max_queries !== nothing && length(queries) > max_queries
        queries = queries[1:max_queries]
    end
    n_q = length(queries)
    n_q > 0 || error("No antonym queries found in evaluation DataFrame.")

    max_k = maximum(ks)
    hits = Dict(k => 0 for k in ks)

    for q in queries
        ants = gt_ant[q]
        cands, _, _ = retrieve_stage1(pipeline, q; top_k = max_k)
        for k in ks
            upper = min(k, length(cands))
            if any(w -> w in ants, @view cands[1:upper])
                hits[k] += 1
            end
        end
    end

    recalls = Dict("recall_at_$k" => hits[k] / n_q for k in ks)
    return (num_queries = n_q, recalls = recalls)
end

"""
    evaluate_pipeline(
        pipeline::AntonymPipeline,
        eval_df::DataFrame;
        top_k_stage1::Int = 200,
        ks::AbstractVector{Int} = [1, 5, 10],
        max_queries::Union{Int, Nothing} = nothing,
    )

Evaluate the full 2-stage pipeline (`Recall@K`, `MRR`, and `Synonym Error Rate@5`) on `eval_df`.
"""
function evaluate_pipeline(
    pipeline::AntonymPipeline,
    eval_df::DataFrame;
    top_k_stage1::Int = 200,
    ks::AbstractVector{Int} = [1, 5, 10],
    max_queries::Union{Int, Nothing} = nothing,
)
    isempty(pipeline.vocab_k_embs) && build_vocab_index!(pipeline)
    gt_ant, gt_syn = _group_ground_truth(eval_df)
    queries = sort(collect(keys(gt_ant)))
    if max_queries !== nothing && length(queries) > max_queries
        queries = queries[1:max_queries]
    end
    n_q = length(queries)
    n_q > 0 || error("No antonym queries found in evaluation DataFrame.")

    hits = Dict(k => 0 for k in ks)
    rr_sum = 0.0
    syn_err_count = 0
    top5_slots = 0

    for q in queries
        ants = gt_ant[q]
        syns = get(gt_syn, q, Set{String}())
        preds = generate_antonyms(pipeline, q; top_k_stage1 = top_k_stage1, top_n = top_k_stage1)
        pred_words = [p.word for p in preds]

        for k in ks
            upper = min(k, length(pred_words))
            if any(w -> w in ants, @view pred_words[1:upper])
                hits[k] += 1
            end
        end

        first_rank = findfirst(w -> w in ants, pred_words)
        if first_rank !== nothing
            rr_sum += 1.0 / first_rank
        end

        n5 = min(5, length(pred_words))
        top5_slots += n5
        for i in 1:n5
            if pred_words[i] in syns
                syn_err_count += 1
            end
        end
    end

    recalls = Dict("recall_at_$k" => hits[k] / n_q for k in ks)
    mrr = rr_sum / n_q
    syn_err_rate_5 = top5_slots > 0 ? syn_err_count / top5_slots : 0.0

    return (
        num_queries = n_q,
        recalls = recalls,
        mrr = mrr,
        synonym_error_rate_at_5 = syn_err_rate_5,
    )
end
