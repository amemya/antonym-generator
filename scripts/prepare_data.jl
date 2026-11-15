#!/usr/bin/env julia

using AntonymGenerator

function main()
    project_root = dirname(@__DIR__)
    raw_dir = joinpath(project_root, "data", "raw")
    out_dir = joinpath(project_root, "data", "processed")

    stats = download_and_build_dataset(; raw_dir = raw_dir, out_dir = out_dir, seed = 42)

    println("=== Dataset Preparation Complete ===")
    println("Vocabulary size        : ", stats.vocab_size)
    println("Connected components   : ", stats.num_components)
    println("Antonym graph words    : train=", stats.train_antonym_words,
            ", val=", stats.val_antonym_words,
            ", test=", stats.test_antonym_words)
    println("Disjoint overlap check : train∩val=", stats.overlap_train_val,
            ", train∩test=", stats.overlap_train_test,
            ", val∩test=", stats.overlap_val_test)
    println("Train pairs            : ", stats.train_counts)
    println("Val pairs              : ", stats.val_counts)
    println("Test pairs             : ", stats.test_counts)
    println("Saved CSVs to          : ", out_dir)
end

main()
