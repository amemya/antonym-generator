using AntonymGenerator
using DataFrames
using Flux
using JSON3
using LinearAlgebra
using Optimisers
using Test
using Zygote

function write_synthetic_safetensors(path::AbstractString, tensors::Vector{Pair{String, Tuple{Vector{Int}, Vector{Float32}}}})
    header_dict = Dict{String, Any}()
    offset = 0
    buf = IOBuffer()
    for (name, (shape, vals)) in tensors
        nbytes = length(vals) * sizeof(Float32)
        header_dict[name] = Dict(
            "dtype" => "F32",
            "shape" => shape,
            "data_offsets" => [offset, offset + nbytes],
        )
        write(buf, reinterpret(UInt8, vals))
        offset += nbytes
    end
    hdr_bytes = Vector{UInt8}(JSON3.write(header_dict))
    # Pad header to 8-byte alignment with spaces
    rem8 = mod(length(hdr_bytes), 8)
    if rem8 != 0
        append!(hdr_bytes, fill(UInt8(' '), 8 - rem8))
    end
    hdr_len = UInt64(length(hdr_bytes))

    open(path, "w") do io
        write(io, reinterpret(UInt8, [hdr_len]))
        write(io, hdr_bytes)
        write(io, take!(buf))
    end
    return path
end

function build_tiny_test_bert_dir(dir::AbstractString; type_vocab_size::Int = 1)
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", " ", "高", "い", "低", "可", "能", "不", "増", "加", "減", "少", "ひ", "く", "暑", "寒"]
    # Write with CRLF + preserve space token at index 6
    open(joinpath(dir, "vocab.txt"), "w") do io
        for tok in vocab
            write(io, tok, "\r\n")
        end
    end

    V = length(vocab)
    H = 16
    I = 32
    max_pos = 32
    cfg = Dict(
        "vocab_size" => V,
        "hidden_size" => H,
        "num_hidden_layers" => 2,
        "num_attention_heads" => 2,
        "intermediate_size" => I,
        "max_position_embeddings" => max_pos,
        "type_vocab_size" => type_vocab_size,
        "hidden_dropout_prob" => 0.0,
        "attention_probs_dropout_prob" => 0.0,
        "layer_norm_eps" => 1e-12,
    )
    open(joinpath(dir, "config.json"), "w") do io
        JSON3.write(io, cfg)
    end

    rand_f32(n) = Float32.(0.05 .* randn(n))
    ones_f32(n) = ones(Float32, n)
    zeros_f32(n) = zeros(Float32, n)

    tensors = Pair{String, Tuple{Vector{Int}, Vector{Float32}}}[
        "bert.embeddings.word_embeddings.weight" => ([V, H], rand_f32(V * H)),
        "bert.embeddings.position_embeddings.weight" => ([max_pos, H], rand_f32(max_pos * H)),
        "bert.embeddings.token_type_embeddings.weight" => ([type_vocab_size, H], rand_f32(type_vocab_size * H)),
        "bert.embeddings.LayerNorm.weight" => ([H], ones_f32(H)),
        "bert.embeddings.LayerNorm.bias" => ([H], zeros_f32(H)),
    ]
    for l in 0:1
        p = "bert.encoder.layer.$l"
        for sub in ("query", "key", "value")
            push!(tensors, "$p.attention.self.$sub.weight" => ([H, H], rand_f32(H * H)))
            push!(tensors, "$p.attention.self.$sub.bias" => ([H], zeros_f32(H)))
        end
        push!(tensors, "$p.attention.output.dense.weight" => ([H, H], rand_f32(H * H)))
        push!(tensors, "$p.attention.output.dense.bias" => ([H], zeros_f32(H)))
        push!(tensors, "$p.attention.output.LayerNorm.weight" => ([H], ones_f32(H)))
        push!(tensors, "$p.attention.output.LayerNorm.bias" => ([H], zeros_f32(H)))
        push!(tensors, "$p.intermediate.dense.weight" => ([I, H], rand_f32(I * H)))
        push!(tensors, "$p.intermediate.dense.bias" => ([I], zeros_f32(I)))
        push!(tensors, "$p.output.dense.weight" => ([H, I], rand_f32(H * I)))
        push!(tensors, "$p.output.dense.bias" => ([H], zeros_f32(H)))
        push!(tensors, "$p.output.LayerNorm.weight" => ([H], ones_f32(H)))
        push!(tensors, "$p.output.LayerNorm.bias" => ([H], zeros_f32(H)))
    end

    write_synthetic_safetensors(joinpath(dir, "model.safetensors"), tensors)
    return dir
end

@testset "AntonymGenerator.jl" begin
    @testset "Tokenizer & Surface Features" begin
        mktempdir() do tmp
            build_tiny_test_bert_dir(tmp)
            tok = load_vocab(joinpath(tmp, "vocab.txt"))
            @test length(tok.vocab) == 20
            @test tok.vocab[6] == " "
            @test tok.token_to_id["高"] == 7
            @test tok.pad_id == 1
            @test tok.cls_id == 3
            @test tok.sep_id == 4

            ids, tids, mask = encode_single(tok, "高い"; max_len = 8)
            @test length(ids) == 8
            @test ids[1:4] == [tok.cls_id, tok.token_to_id["高"], tok.token_to_id["い"], tok.sep_id]
            @test all(==(tok.pad_id), ids[5:8])
            @test mask == Float32[1, 1, 1, 1, 0, 0, 0, 0]
            @test all(==(1), tids)

            bids, btids, bmask = encode_batch_single(tok, ["高い", "不可能"]; max_len = 8)
            @test size(bids) == (8, 2)
            @test size(btids) == (8, 2)
            @test size(bmask) == (8, 2)
            @test bids[1:5, 2] == [tok.cls_id, tok.token_to_id["不"], tok.token_to_id["可"], tok.token_to_id["能"], tok.sep_id]

            dbids, _, _ = encode_batch_single(tok, ["高い", "不可能"]; max_len = 16, dynamic_pad = true)
            @test size(dbids) == (5, 2)

            pids, ptids, pmask = encode_batch_pair(tok, ["可能"], ["不可能"]; max_len = 12)
            @test size(pids) == (12, 1)
            dpids, _, _ = encode_batch_pair(tok, ["可能"], ["不可能"]; max_len = 32, dynamic_pad = true)
            @test size(dpids) == (8, 1)

            tok_trunc = load_vocab(joinpath(tmp, "vocab.txt"); max_vocab_size = 10)
            @test length(tok_trunc.vocab) == 10
            tids_trunc, _, _ = encode_single(tok_trunc, "寒"; max_len = 6)
            @test tids_trunc[2] == tok_trunc.unk_id
            @test pids[1:8, 1] == [
                tok.cls_id, tok.token_to_id["可"], tok.token_to_id["能"], tok.sep_id,
                tok.token_to_id["不"], tok.token_to_id["可"], tok.token_to_id["能"], tok.sep_id,
            ]
            @test ptids[1:8, 1] == [1, 1, 1, 1, 2, 2, 2, 2]
            @test pmask[1:8, 1] == ones(Float32, 8)
            @test pmask[9:12, 1] == zeros(Float32, 4)

            f_neg = extract_pair_surface_features("可能", "不可能")
            @test length(f_neg) == 8
            @test f_neg[4] == 1.0f0 # Negation prefix asymmetry

            f_kanji = extract_pair_surface_features("増加", "減少")
            @test f_kanji[2] == 1.0f0 # Equal length
            @test f_kanji[5] == 1.0f0 # Opposing Kanji (増 / 減)

            f_adj = extract_pair_surface_features("高い", "低い")
            @test f_adj[5] == 1.0f0 # Opposing Kanji (高 / 低)
            @test f_adj[6] == 1.0f0 # Shared suffix い
        end
    end

    @testset "BertBackbone & Safetensors" begin
        mktempdir() do tmp
            build_tiny_test_bert_dir(tmp; type_vocab_size = 1)
            bb, tok, cfg = load_pretrained_bert(tmp)
            @test cfg.hidden_size == 16
            @test cfg.num_hidden_layers == 2
            @test size(bb.embeddings.token_type_embeddings.weight) == (16, 2)
            @test bb.embeddings.token_type_embeddings.weight[:, 1] == bb.embeddings.token_type_embeddings.weight[:, 2]

            ids, tids, mask = encode_batch_pair(tok, ["高い", "増加"], ["低い", "減少"]; max_len = 10)
            Flux.testmode!(bb)
            hidden, pooled = bb(ids, tids, mask)
            @test size(hidden) == (16, 10, 2)
            @test size(pooled) == (16, 2)
            @test all(isfinite, hidden)

            loss_val, grads = Zygote.withgradient(bb) do m
                h, p = m(ids, tids, mask)
                sum(abs2, p) + 0.1f0 * sum(abs2, h)
            end
            @test isfinite(loss_val)
            @test grads[1] !== nothing
            @test grads[1].embeddings.word_embeddings.weight !== nothing
            @test grads[1].layers[1].attention.query.weight !== nothing
        end
    end

    @testset "BiEncoder & Multi-positive InfoNCE" begin
        mktempdir() do tmp
            build_tiny_test_bert_dir(tmp)
            bb, tok, _ = load_pretrained_bert(tmp)
            bi = BiEncoder(bb; proj_dim = 12)
            Flux.testmode!(bi)

            q_ids, q_tids, q_mask = encode_batch_single(tok, ["高い", "増加"]; max_len = 8)
            k_ids, k_tids, k_mask = encode_batch_single(tok, ["低い", "減少", "高い"]; max_len = 8)

            q_emb = encode_query(bi, q_ids, q_tids, q_mask)
            k_emb = encode_candidate(bi, k_ids, k_tids, k_mask)
            @test size(q_emb) == (12, 2)
            @test size(k_emb) == (12, 3)
            @test isapprox(norm(q_emb[:, 1]), 1.0f0; atol = 1f-5)
            @test isapprox(norm(k_emb[:, 2]), 1.0f0; atol = 1f-5)

            pos_mask = Float32[
                1 0 0;
                0 1 0
            ]
            hard_neg_mask = Float32[
                0 0 1;
                0 0 0
            ]
            # Multiple positives on row 1
            pos_mask_multi = Float32[
                1 0 1;
                0 1 0
            ]
            loss_multi = multi_positive_infonce_loss(q_emb, k_emb, pos_mask_multi, hard_neg_mask)
            @test isfinite(loss_multi) && loss_multi >= 0.0f0

            opt_state = Optimisers.setup(Optimisers.Adam(1f-3), bi)
            loss0 = multi_positive_infonce_loss(
                encode_query(bi, q_ids, q_tids, q_mask),
                encode_candidate(bi, k_ids, k_tids, k_mask),
                pos_mask,
                hard_neg_mask,
            )
            @test isfinite(loss0)
            for _ in 1:5
                _, grads = Zygote.withgradient(bi) do m
                    qe = encode_query(m, q_ids, q_tids, q_mask)
                    ke = encode_candidate(m, k_ids, k_tids, k_mask)
                    multi_positive_infonce_loss(qe, ke, pos_mask, hard_neg_mask)
                end
                @test grads[1] !== nothing
                opt_state, bi = Optimisers.update!(opt_state, bi, grads[1])
            end
            loss1 = multi_positive_infonce_loss(
                encode_query(bi, q_ids, q_tids, q_mask),
                encode_candidate(bi, k_ids, k_tids, k_mask),
                pos_mask,
                hard_neg_mask,
            )
            @test loss1 < loss0
        end
    end

    @testset "CrossEncoder & 3-Class Loss" begin
        mktempdir() do tmp
            build_tiny_test_bert_dir(tmp)
            bb, tok, _ = load_pretrained_bert(tmp)
            ce = CrossEncoder(bb; pair_feat_dim = 10)
            Flux.testmode!(ce)

            w1s = ["高い", "可能", "増加"]
            w2s = ["低い", "不可能", "増える"]
            ids, tids, mask = encode_batch_pair(tok, w1s, w2s; max_len = 12)
            surf = extract_batch_surface_features(w1s, w2s)
            pf = vcat(surf, randn(Float32, 2, 3))
            labels = [3, 3, 2] # 1-based classes: Antonym, Antonym, Synonym
            cw = Float32[1.0, 1.5, 2.0]

            logits = ce(ids, tids, mask, pf)
            @test size(logits) == (3, 3)

            loss_ce, grads = Zygote.withgradient(ce) do m
                l = m(ids, tids, mask, pf)
                cross_encoder_loss(l, labels; class_weights = cw)
            end
            @test isfinite(loss_ce) && loss_ce >= 0.0f0
            @test grads[1] !== nothing
        end
    end

    @testset "Connected-Components Disjoint Split" begin
        antonym_pairs = [
            ("高い", "低い"), ("高い", "安い"),
            ("暑い", "寒い"), ("熱い", "冷たい"),
            ("増加", "減少"), ("拡大", "縮小"),
            ("開始", "終了"), ("成功", "失敗"),
            ("明るい", "暗い"), ("重い", "軽い"),
        ]
        synonym_pairs = [
            ("高い", "高価"), ("低い", "低級"),
            ("暑い", "猛暑"), ("増加", "増大"),
            ("開始", "着手"), ("成功", "達成"),
        ]
        readings = Dict(
            "高い" => Set(["たかい"]),
            "低い" => Set(["ひくい"]),
        )

        train_df, val_df, test_df, vocab_df, stats = build_disjoint_splits(
            antonym_pairs,
            synonym_pairs,
            readings;
            all_vocab = ["机", "椅子", "空", "海"],
            seed = 42,
        )

        @test stats.overlap_train_val == 0
        @test stats.overlap_train_test == 0
        @test stats.overlap_val_test == 0

        train_ant_words = Set(vcat(train_df[train_df.label .== LABEL_ANTONYM, :w1], train_df[train_df.label .== LABEL_ANTONYM, :w2]))
        val_ant_words = Set(vcat(val_df[val_df.label .== LABEL_ANTONYM, :w1], val_df[val_df.label .== LABEL_ANTONYM, :w2]))
        test_ant_words = Set(vcat(test_df[test_df.label .== LABEL_ANTONYM, :w1], test_df[test_df.label .== LABEL_ANTONYM, :w2]))

        @test isempty(intersect(train_ant_words, val_ant_words))
        @test isempty(intersect(train_ant_words, test_ant_words))
        @test isempty(intersect(val_ant_words, test_ant_words))

        # No val/test antonym words leaked into train_df
        all_train_words = Set(vcat(train_df.w1, train_df.w2))
        @test isempty(intersect(all_train_words, union(val_ant_words, test_ant_words)))
        @test nrow(vocab_df) >= length(union(train_ant_words, val_ant_words, test_ant_words))
    end

    @testset "PostProcessor, Evaluation & Artifacts Round-Trip" begin
        reading_map = Dict(
            "高い" => "たかい",
            "たかい" => "たかい",
            "低い" => "ひくい",
            "ひくい" => "ひくい",
            "安い" => "やすい",
            "高価" => "こうか",
        )
        cands = ["高い", "たかい", "低い", "ひくい", "安い", "高価"]
        scores = Float32[0.99, 0.95, 0.92, 0.90, 0.85, 0.20]
        syns = Float32[0.01, 0.02, 0.05, 0.06, 0.10, 0.75]

        res = postprocess_candidates("高い", cands, scores, syns, reading_map; top_n = 5)
        res_words = [r.word for r in res]
        # Excludes query itself and reading-identical variant "たかい"
        @test !("高い" in res_words)
        @test !("たかい" in res_words)
        # Deduplicates "低い" vs "ひくい" keeping highest-scoring "低い"
        @test res_words == ["低い", "安い", "高価"]

        mktempdir() do tmp
            build_tiny_test_bert_dir(tmp)
            bb1, tok, _ = load_pretrained_bert(tmp)
            bb2, _, _ = load_pretrained_bert(tmp)
            bi = BiEncoder(bb1; proj_dim = 12)
            ce = CrossEncoder(bb2; pair_feat_dim = 10)
            vocab_words = ["高い", "低い", "ひくい", "安い", "可能", "不可能", "増加", "減少", "高価"]
            pipeline = AntonymPipeline(tok, bi, ce, vocab_words, reading_map)
            build_vocab_index!(pipeline; batch_size = 4)

            eval_df = DataFrame(
                w1 = ["高い", "高い", "可能"],
                w2 = ["低い", "高価", "不可能"],
                label = [LABEL_ANTONYM, LABEL_SYNONYM, LABEL_ANTONYM],
            )
            s1_res = evaluate_stage1(pipeline, eval_df; ks = [2, 5])
            @test s1_res.num_queries == 2
            @test 0.0 <= s1_res.recalls["recall_at_2"] <= 1.0

            full_res = evaluate_pipeline(pipeline, eval_df; top_k_stage1 = 6, ks = [1, 5])
            @test full_res.num_queries == 2
            @test 0.0 <= full_res.mrr <= 1.0
            @test 0.0 <= full_res.synonym_error_rate_at_5 <= 1.0

            art_dir = joinpath(tmp, "artifacts")
            save_artifacts(art_dir, pipeline, Dict("mrr" => full_res.mrr))
            loaded = load_artifacts(art_dir)
            p1 = generate_antonyms(pipeline, "高い"; top_k_stage1 = 6, top_n = 3)
            p2 = generate_antonyms(loaded, "高い"; top_k_stage1 = 6, top_n = 3)
            @test [x.word for x in p1] == [x.word for x in p2]
            @test all(isapprox(p1[i].antonym_prob, p2[i].antonym_prob; atol = 1f-5) for i in 1:length(p1))
        end
    end
end
