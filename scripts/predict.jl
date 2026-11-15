#!/usr/bin/env julia

using AntonymGenerator

function parse_predict_args(args::Vector{String})
    project_root = dirname(@__DIR__)
    opts = Dict{String, Any}(
        "artifacts_dir" => joinpath(project_root, "artifacts"),
        "top_n" => 10,
        "top_k_stage1" => 200,
        "device" => "auto",
    )
    queries = String[]

    i = 1
    while i <= length(args)
        arg = args[i]
        if startswith(arg, "--")
            if occursin('=', arg)
                k, v = split(arg[3:end], '='; limit = 2)
                key = replace(k, '-' => '_')
                _assign_predict_opt!(opts, key, String(v))
                i += 1
            else
                key = replace(arg[3:end], '-' => '_')
                i + 1 <= length(args) || error("Missing value for flag $arg")
                _assign_predict_opt!(opts, key, args[i + 1])
                i += 2
            end
        else
            push!(queries, arg)
            i += 1
        end
    end

    if isempty(queries)
        queries = ["高い", "可能", "増加", "暑い"]
    end
    return opts, queries
end

function _assign_predict_opt!(opts::Dict{String, Any}, key::String, val::String)
    haskey(opts, key) || error("Unknown CLI option: --$(replace(key, '_' => '-'))")
    if key in ("top_n", "top_k_stage1")
        opts[key] = parse(Int, val)
    else
        opts[key] = val
    end
end

function main()
    opts, queries = parse_predict_args(copy(ARGS))
    artifacts_dir = String(opts["artifacts_dir"])
    isdir(artifacts_dir) || error("Artifacts directory not found: $artifacts_dir. Run scripts/train_and_eval.jl first.")

    device_cfg = resolve_device(Symbol(opts["device"]))
    pipeline = load_artifacts(artifacts_dir; device_cfg = device_cfg)

    for q in queries
        preds = generate_antonyms(
            pipeline,
            q;
            top_k_stage1 = opts["top_k_stage1"],
            top_n = opts["top_n"],
        )
        println("\n=== Query: $q ===")
        println("Rank | Word       | Reading    | P(Antonym) | P(Synonym) | Stage1")
        println("-----+------------+------------+------------+------------+--------")
        for (rank, p) in enumerate(preds)
            println(
                lpad(rank, 4), " | ",
                rpad(p.word, 10), " | ",
                rpad(p.reading, 10), " | ",
                lpad(round(p.antonym_prob; digits = 4), 10), " | ",
                lpad(round(p.synonym_prob; digits = 4), 10), " | ",
                lpad(round(p.stage1_score; digits = 4), 6),
            )
        end
    end
end

main()
