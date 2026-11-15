#!/usr/bin/env julia

using AntonymGenerator
using DataFrames

function parse_args(args::Vector{String})
    opts = Dict{String, Any}(
        "device" => "auto",
        "pretrained" => "bluolightning/bert-tiny-japanese-char",
        "epochs_stage1" => 5,
        "epochs_stage2" => 4,
        "batch_size" => 64,
        "lr_stage1" => 3.0f-4,
        "lr_stage2" => 2.0f-4,
        "proj_dim" => 256,
        "freeze_backbone" => false,
        "max_train_pairs" => nothing,
        "max_eval_queries" => nothing,
        "num_layers" => nothing,
        "top_k_stage1" => 200,
        "data_dir" => nothing,
        "artifacts_dir" => nothing,
    )

    i = 1
    while i <= length(args)
        arg = args[i]
        if startswith(arg, "--")
            if occursin('=', arg)
                k, v = split(arg[3:end], '='; limit = 2)
                key = replace(k, '-' => '_')
                _assign_opt!(opts, key, String(v))
                i += 1
            else
                key = replace(arg[3:end], '-' => '_')
                if key == "freeze_backbone"
                    opts[key] = true
                    i += 1
                else
                    i + 1 <= length(args) || error("Missing value for flag $arg")
                    _assign_opt!(opts, key, args[i + 1])
                    i += 2
                end
            end
        else
            error("Unexpected positional argument: $arg")
        end
    end
    return opts
end

function _assign_opt!(opts::Dict{String, Any}, key::String, val::String)
    haskey(opts, key) || error("Unknown CLI option: --$(replace(key, '_' => '-'))")
    if key in ("epochs_stage1", "epochs_stage2", "batch_size", "proj_dim", "top_k_stage1")
        opts[key] = parse(Int, val)
    elseif key in ("max_train_pairs", "max_eval_queries", "num_layers")
        opts[key] = val == "nothing" ? nothing : parse(Int, val)
    elseif key in ("lr_stage1", "lr_stage2")
        opts[key] = parse(Float32, val)
    elseif key == "freeze_backbone"
        opts[key] = lowercase(val) in ("true", "1", "yes")
    else
        opts[key] = val
    end
end

function main()
    opts = parse_args(copy(ARGS))
    project_root = dirname(@__DIR__)
    data_dir = something(opts["data_dir"], joinpath(project_root, "data", "processed"))
    artifacts_dir = something(opts["artifacts_dir"], joinpath(project_root, "artifacts"))
    pretrained_cache = joinpath(project_root, "data", "pretrained")

    device_cfg = resolve_device(Symbol(opts["device"]))
    pretrained = String(opts["pretrained"])

    if !device_cfg.use_gpu && occursin("bert-base", pretrained)
        @warn "Full 12-layer BERT (`$pretrained`) is heavy on CPU. For full-scale training, run on the Windows 20GB VRAM GPU machine with `--device cuda`."
    end

    if !isfile(joinpath(data_dir, "train.csv"))
        @info "Processed dataset not found; building dataset first..."
        download_and_build_dataset(; raw_dir = joinpath(project_root, "data", "raw"), out_dir = data_dir)
    end

    splits = load_dataset_splits(data_dir)
    vocab_words = Vector{String}(splits.vocab.word)
    reading_map = Dict{String, String}(String(row.word) => String(row.reading) for row in eachrow(splits.vocab))

    @info "Loading pretrained BERT backbone for Stage 1 Bi-Encoder..." repo=pretrained
    bi_backbone, tok, cfg = load_pretrained_bert(pretrained; cache_dir = pretrained_cache, num_layers = opts["num_layers"])
    @info "Loading pretrained BERT backbone for Stage 2 Cross-Encoder..." repo=pretrained
    ce_backbone, _, _ = load_pretrained_bert(pretrained; cache_dir = pretrained_cache, num_layers = opts["num_layers"])

    bi_model = BiEncoder(bi_backbone; proj_dim = opts["proj_dim"])
    ce_model = CrossEncoder(ce_backbone; pair_feat_dim = 10)

    println("\n=== Training Stage 1: Bi-Encoder (Multi-positive InfoNCE) ===")
    bi_model, bi_hist = train_bi_encoder!(
        bi_model,
        tok,
        splits.train,
        splits.val,
        vocab_words;
        epochs = opts["epochs_stage1"],
        batch_size = opts["batch_size"],
        lr = opts["lr_stage1"],
        freeze_backbone = opts["freeze_backbone"],
        max_train_pairs = opts["max_train_pairs"],
        device_cfg = device_cfg,
    )

    println("\n=== Mining Stage 1 Hard Negatives ===")
    hard_neg_df = mine_hard_negatives(
        bi_model,
        tok,
        splits.train,
        vocab_words;
        top_k = 30,
        max_per_query = 3,
        max_queries = opts["max_train_pairs"],
        batch_size = max(opts["batch_size"], 128),
        device_cfg = device_cfg,
    )
    n_syn_hn = count(==(LABEL_SYNONYM), hard_neg_df.label)
    n_unrel_hn = count(==(LABEL_UNRELATED), hard_neg_df.label)
    println("Mined hard negatives: total=", nrow(hard_neg_df), " (synonym=", n_syn_hn, ", unrelated=", n_unrel_hn, ")")

    println("\n=== Training Stage 2: Cross-Encoder (3-class Classification) ===")
    ce_model, ce_hist = train_cross_encoder!(
        ce_model,
        bi_model,
        tok,
        splits.train,
        splits.val;
        hard_neg_df = hard_neg_df,
        epochs = opts["epochs_stage2"],
        batch_size = opts["batch_size"],
        lr = opts["lr_stage2"],
        freeze_backbone = opts["freeze_backbone"],
        max_train_pairs = opts["max_train_pairs"],
        device_cfg = device_cfg,
    )

    println("\n=== Building Stage 1 Vocabulary Index & Evaluating on Test Split ===")
    pipeline = AntonymPipeline(tok, bi_model, ce_model, vocab_words, reading_map; device_cfg = device_cfg)
    build_vocab_index!(pipeline; batch_size = max(opts["batch_size"], 128))

    s1_eval = evaluate_stage1(pipeline, splits.test; ks = [50, 100, 200], max_queries = opts["max_eval_queries"])
    full_eval = evaluate_pipeline(
        pipeline,
        splits.test;
        top_k_stage1 = opts["top_k_stage1"],
        ks = [1, 5, 10],
        max_queries = opts["max_eval_queries"],
    )

    metrics = Dict{String, Any}(
        "pretrained" => pretrained,
        "device" => String(device_cfg.mode),
        "stage1_train_losses" => bi_hist.train_losses,
        "stage1_val_losses" => bi_hist.val_losses,
        "stage2_train_losses" => ce_hist.train_losses,
        "stage2_val_losses" => ce_hist.val_losses,
        "stage2_val_accs" => ce_hist.val_accs,
        "test_stage1" => Dict("num_queries" => s1_eval.num_queries, "recalls" => s1_eval.recalls),
        "test_pipeline" => Dict(
            "num_queries" => full_eval.num_queries,
            "recalls" => full_eval.recalls,
            "mrr" => full_eval.mrr,
            "synonym_error_rate_at_5" => full_eval.synonym_error_rate_at_5,
        ),
    )
    save_artifacts(artifacts_dir, pipeline, metrics)

    println("\n=== Evaluation Summary (Test Split, queries = $(full_eval.num_queries)) ===")
    println("Stage 1 Recall@50     : ", round(s1_eval.recalls["recall_at_50"]; digits = 4))
    println("Stage 1 Recall@100    : ", round(s1_eval.recalls["recall_at_100"]; digits = 4))
    println("Stage 1 Recall@200    : ", round(s1_eval.recalls["recall_at_200"]; digits = 4))
    println("Pipeline Recall@1     : ", round(full_eval.recalls["recall_at_1"]; digits = 4))
    println("Pipeline Recall@5     : ", round(full_eval.recalls["recall_at_5"]; digits = 4))
    println("Pipeline Recall@10    : ", round(full_eval.recalls["recall_at_10"]; digits = 4))
    println("Pipeline MRR          : ", round(full_eval.mrr; digits = 4))
    println("Synonym Error Rate@5  : ", round(full_eval.synonym_error_rate_at_5; digits = 4))
    println("Saved artifacts to    : ", artifacts_dir)
end

main()
