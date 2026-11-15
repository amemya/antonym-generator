using CSV
using DataFrames
using Downloads
using Random
using Unicode

const WNJA_URL = "https://github.com/bond-lab/wnja/releases/download/v2.1/wnja-2.1.tar.xz"
const SUDACHI_SYNONYMS_URL = "https://raw.githubusercontent.com/WorksApplications/SudachiDict/develop/src/main/text/synonyms.txt"

const LABEL_UNRELATED = 0
const LABEL_SYNONYM = 1
const LABEL_ANTONYM = 2

const JA_WORD_RE = r"^[\p{Han}\p{Hiragana}\p{Katakana}ー々]+$"
const MORPH_NEG_PREFIXES = ("不", "非", "無", "未")

"""
    is_clean_ja_word(w::AbstractString) -> Bool

Return `true` if `w` is a clean Japanese word (1..10 chars, purely Kanji/Hiragana/Katakana/ー/々).
"""
function is_clean_ja_word(w::AbstractString)
    len = length(w)
    return 1 <= len <= 10 && occursin(JA_WORD_RE, w)
end

"""
    to_hiragana(s::AbstractString) -> String

Normalize `s` to NFKC and convert full-width Katakana (`ァ`..`ヶ`) to Hiragana.
"""
function to_hiragana(s::AbstractString)
    norm = Unicode.normalize(String(s), :NFKC)
    io = IOBuffer()
    for c in norm
        if '\u30a1' <= c <= '\u30f6'
            write(io, Char(Int(c) - 0x60))
        else
            write(io, c)
        end
    end
    return String(take!(io))
end

function _ensure_raw_datasets(raw_dir::AbstractString)
    mkpath(raw_dir)
    tar_path = joinpath(raw_dir, "wnja-2.1.tar.xz")
    xml_path = joinpath(raw_dir, "wnja", "wnja-2.1.xml")
    sudachi_path = joinpath(raw_dir, "sudachi_synonyms.txt")

    if !isfile(xml_path)
        if !isfile(tar_path)
            @info "Downloading Japanese WordNet v2.1..." url=WNJA_URL dest=tar_path
            Downloads.download(WNJA_URL, tar_path)
        end
        @info "Extracting wnja/wnja-2.1.xml..."
        run(`tar -xJf $tar_path -C $raw_dir wnja/wnja-2.1.xml`)
    end

    if !isfile(sudachi_path)
        @info "Downloading Sudachi Synonym Dictionary..." url=SUDACHI_SYNONYMS_URL dest=sudachi_path
        Downloads.download(SUDACHI_SYNONYMS_URL, sudachi_path)
    end

    return xml_path, sudachi_path
end

function _parse_wnja_xml_detailed(xml_path::AbstractString)
    re_wf = r"writtenForm=\"([^\"]+)\""
    re_syn = r"synset=\"([^\"]+)\""
    re_cnt = r">(\d+)</Count>"
    re_id = r"id=\"([^\"]+)\""
    re_tgt = r"target=\"([^\"]+)\""

    # Pass 1: collect candidate clean words and count occurrences in definitions/examples
    corpus_freq = Dict{String, Int}()
    def_lines = String[]
    for line in eachline(xml_path)
        if occursin("<Lemma ", line) || (occursin("<Form ", line) && !occursin("script=", line))
            m = match(re_wf, line)
            if m !== nothing
                w = Unicode.normalize(String(m.captures[1]), :NFKC)
                is_clean_ja_word(w) && (corpus_freq[w] = 0)
            end
        elseif occursin("<Definition>", line) || occursin("<Example>", line)
            push!(def_lines, line)
        end
    end
    for line in def_lines
        chars = collect(line)
        n = length(chars)
        @inbounds for i in 1:n
            c1 = chars[i]
            (c1 == '<' || c1 == '>' || isspace(c1)) && continue
            for j in i:min(n, i + 9)
                w = String(@view chars[i:j])
                haskey(corpus_freq, w) && (corpus_freq[w] += 1)
            end
        end
    end

    # Pass 2: parse entries including canonical <Form> orthographic variants and JSemCor counts
    synset_raw = Dict{String, Vector{Tuple{String, Int}}}()
    antonym_links = Tuple{String, String}[]
    reading_map = Dict{String, Set{String}}()
    all_words = Set{String}()
    word_max_count = Dict{String, Int}()

    cur_forms = String[]
    cur_hira = String[]
    forms_finalized = false
    cur_synset_id = ""
    in_sense = false
    sense_synset = ""
    sense_count = 0

    for line in eachline(xml_path)
        if occursin("<LexicalEntry ", line)
            empty!(cur_forms)
            empty!(cur_hira)
            forms_finalized = false
            in_sense = false
        elseif occursin("<Lemma ", line) || (occursin("<Form ", line) && !occursin("script=", line))
            m = match(re_wf, line)
            if m !== nothing
                w = Unicode.normalize(String(m.captures[1]), :NFKC)
                if is_clean_ja_word(w) && !(w in cur_forms)
                    push!(cur_forms, w)
                end
            end
        elseif !isempty(cur_forms) && occursin("<Form ", line) && occursin("script=\"hira\"", line)
            m = match(re_wf, line)
            if m !== nothing
                hira = to_hiragana(m.captures[1])
                !isempty(hira) && !(hira in cur_hira) && push!(cur_hira, hira)
            end
        elseif !isempty(cur_forms) && occursin("<Sense ", line)
            if !forms_finalized
                sorted_f = sort(cur_forms; by = w -> (-get(corpus_freq, w, 0), length(w), w))
                kept_f = String[sorted_f[1]]
                best_cf = get(corpus_freq, sorted_f[1], 0)
                for k in 2:length(sorted_f)
                    cf = get(corpus_freq, sorted_f[k], 0)
                    if cf >= 10 && cf >= 0.2 * best_cf
                        push!(kept_f, sorted_f[k])
                    end
                end
                for w in cur_forms
                    push!(all_words, w)
                    !isempty(cur_hira) && union!(get!(reading_map, w, Set{String}()), cur_hira)
                end
                cur_forms = kept_f
                forms_finalized = true
            end
            m = match(re_syn, line)
            if m !== nothing
                s = String(m.captures[1])
                if occursin("/>", line)
                    for w in cur_forms
                        push!(get!(synset_raw, s, Tuple{String, Int}[]), (w, 0))
                        word_max_count[w] = max(get(word_max_count, w, 0), 0)
                    end
                    in_sense = false
                else
                    in_sense = true
                    sense_synset = s
                    sense_count = 0
                end
            end
        elseif in_sense && occursin("<Count ", line)
            m = match(re_cnt, line)
            m !== nothing && (sense_count += parse(Int, m.captures[1]))
        elseif in_sense && occursin("</Sense>", line)
            for w in cur_forms
                push!(get!(synset_raw, sense_synset, Tuple{String, Int}[]), (w, sense_count))
                word_max_count[w] = max(get(word_max_count, w, 0), sense_count)
            end
            in_sense = false
        elseif occursin("</LexicalEntry>", line)
            if !forms_finalized && !isempty(cur_forms)
                for w in cur_forms
                    push!(all_words, w)
                    !isempty(cur_hira) && union!(get!(reading_map, w, Set{String}()), cur_hira)
                end
            end
            in_sense = false
        elseif occursin("<Synset ", line)
            m = match(re_id, line)
            cur_synset_id = m !== nothing ? String(m.captures[1]) : ""
        elseif !isempty(cur_synset_id) && occursin("relType=\"antonym\"", line)
            m = match(re_tgt, line)
            m !== nothing && push!(antonym_links, (cur_synset_id, String(m.captures[1])))
        elseif occursin("</Synset>", line)
            cur_synset_id = ""
        end
    end

    synset_clean = Dict{String, Vector{Tuple{String, Int}}}()
    sizehint!(synset_clean, length(synset_raw))
    for (sid, entries) in synset_raw
        best_by_word = Dict{String, Int}()
        for (w, cnt) in entries
            best_by_word[w] = max(get(best_by_word, w, 0), cnt)
        end
        synset_clean[sid] = Tuple{String, Int}[(w, cnt) for (w, cnt) in best_by_word]
    end

    return synset_clean, antonym_links, reading_map, all_words, word_max_count, corpus_freq
end

function _top_synset_words(
    synset_clean::Dict{String, Vector{Tuple{String, Int}}},
    corpus_freq::Dict{String, Int};
    max_lemmas_per_synset::Int = 6,
)
    synset_words = Dict{String, Vector{String}}()
    sizehint!(synset_words, length(synset_clean))
    for (sid, entries) in synset_clean
        sorted = sort(entries; by = kv -> (-kv[2], -get(corpus_freq, kv[1], 0), length(kv[1]), kv[1]))
        n_keep = min(length(sorted), max_lemmas_per_synset)
        synset_words[sid] = String[sorted[i][1] for i in 1:n_keep]
    end
    return synset_words
end

function _parse_wnja_xml(xml_path::AbstractString; max_lemmas_per_synset::Int = 6)
    synset_clean, antonym_links, reading_map, all_words, _, corpus_freq = _parse_wnja_xml_detailed(xml_path)
    synset_words = _top_synset_words(synset_clean, corpus_freq; max_lemmas_per_synset = max_lemmas_per_synset)
    return synset_words, antonym_links, reading_map, all_words
end

function _prune_wnja_synset_side(
    e1::Vector{Tuple{String, Int}},
    e2::Vector{Tuple{String, Int}},
    word_max_count::Dict{String, Int},
    corpus_freq::Dict{String, Int},
)
    isempty(e1) && return String[]
    needs_struct_rescue(w, c) = c == 0 &&
        (get(corpus_freq, w, 0) >= 1 || length(e1) == 1) &&
        any(is_structural_antonym_pair(w, w2) for (w2, _) in e2)

    filtered = filter(e1) do (w, c)
        c > 0 && return true
        length(e1) == 1 && return true
        needs_struct_rescue(w, c) && return true
        return get(word_max_count, w, 0) < 2 && get(corpus_freq, w, 0) > 0
    end
    if isempty(filtered)
        min_mc = minimum(get(word_max_count, x[1], 0) for x in e1)
        filtered = filter(x -> get(word_max_count, x[1], 0) == min_mc, e1)
    end

    has_pos = any(x -> x[2] > 0, filtered)
    if has_pos
        filtered = filter(x -> x[2] > 0 || needs_struct_rescue(x[1], x[2]), filtered)
    else
        max_cf = maximum(get(corpus_freq, x[1], 0) for x in filtered)
        if max_cf >= 15
            filtered = filter(x -> get(corpus_freq, x[1], 0) >= max(5, div(max_cf, 5)) || needs_struct_rescue(x[1], x[2]), filtered)
        end
    end

    sorted = sort(filtered; by = x -> (
        -(any(is_structural_antonym_pair(x[1], w2) for (w2, _) in e2) ? 1 : 0),
        has_katakana(x[1]) ? 1 : 0,
        -(10 * x[2] + min(get(corpus_freq, x[1], 0), 50)),
        -get(corpus_freq, x[1], 0),
        length(x[1]),
        x[1],
    ))
    has_struct = any(any(is_structural_antonym_pair(x[1], w2) for (w2, _) in e2) for x in sorted)
    cap = has_struct ? 4 : (has_pos ? 3 : 2)
    return String[sorted[i][1] for i in 1:min(length(sorted), cap)]
end

function _build_wnja_antonym_pairs(
    synset_clean::Dict{String, Vector{Tuple{String, Int}}},
    antonym_links::Vector{Tuple{String, String}},
    word_max_count::Dict{String, Int},
    corpus_freq::Dict{String, Int},
    synonym_pairs::Set{Tuple{String, String}},
)
    antonym_pairs = Set{Tuple{String, String}}()
    empty_entries = Tuple{String, Int}[]
    for (s1, s2) in antonym_links
        e1 = get(synset_clean, s1, empty_entries)
        e2 = get(synset_clean, s2, empty_entries)
        w1s = _prune_wnja_synset_side(e1, e2, word_max_count, corpus_freq)
        w2s = _prune_wnja_synset_side(e2, e1, word_max_count, corpus_freq)
        (isempty(w1s) || isempty(w2s)) && continue

        for w1 in w1s, w2 in w2s
            w1 == w2 && continue
            (w1, w2) in synonym_pairs && continue

            w1_has_struct = any(is_structural_antonym_pair(w1, y) for y in w2s)
            w2_has_struct = any(is_structural_antonym_pair(x, w2) for x in w1s)
            if (w1_has_struct || w2_has_struct) && !is_structural_antonym_pair(w1, w2)
                continue
            end

            if !is_structural_antonym_pair(w1, w2)
                w1_shape = word_shape_class(w1)
                w2_shape = word_shape_class(w2)
                w1_has_shape = any(w1_shape == word_shape_class(y) for y in w2s)
                w2_has_shape = any(word_shape_class(x) == w2_shape for x in w1s)
                if (w1_has_shape || w2_has_shape) && w1_shape != w2_shape
                    continue
                end
            end

            push!(antonym_pairs, (w1, w2))
            push!(antonym_pairs, (w2, w1))
        end
    end
    return antonym_pairs
end

function _parse_sudachi_synonyms(sudachi_path::AbstractString; max_partners_per_word::Int = 5)
    # group_id -> lexeme_id -> Vector{Tuple{String, Bool}} (word, is_standard_headword)
    groups = Dict{String, Dict{String, Vector{Tuple{String, Bool}}}}()
    all_words = Set{String}()

    for raw_line in eachline(sudachi_path)
        line = strip(raw_line)
        isempty(line) && continue
        parts = split(line, ',')
        length(parts) >= 9 || continue

        group_id = String(parts[1])
        exp_flag = String(parts[3])
        exp_flag == "2" && continue  # Deprecated / do not expand
        lex_id = String(parts[4])
        is_std = (String(parts[5]) == "0" && String(parts[7]) == "0")
        word = Unicode.normalize(String(parts[9]), :NFKC)
        is_clean_ja_word(word) || continue

        push!(all_words, word)
        g = get!(groups, group_id, Dict{String, Vector{Tuple{String, Bool}}}())
        push!(get!(g, lex_id, Tuple{String, Bool}[]), (word, is_std))
    end

    synonym_pairs = Set{Tuple{String, String}}()
    variant_clusters = Dict{String, Set{String}}()

    for (_, lex_map) in groups
        # Identical (group_id, lexeme_id) are orthographic/reading variants; only use pure kana forms as readings
        for (_, variant_entries) in lex_map
            length(variant_entries) >= 2 || continue
            kana_variants = String[w for (w, _) in variant_entries if all(c -> !is_kanji(c), w)]
            isempty(kana_variants) && continue
            canon = sort(kana_variants; by = v -> (length(v), v))[1]
            canon_hira = to_hiragana(canon)
            for (v, _) in variant_entries
                push!(get!(variant_clusters, v, Set{String}()), canon_hira)
            end
        end

        lex_ids = sort(collect(keys(lex_map)))
        length(lex_ids) >= 2 || continue
        reps = String[]
        for lid in lex_ids
            # Prioritize standard headword (parts[5] == "0" && parts[7] == "0") per lexeme
            cands = lex_map[lid]
            std_idx = findfirst(x -> x[2], cands)
            push!(reps, std_idx === nothing ? cands[1][1] : cands[std_idx][1])
        end
        n_rep = length(reps)
        for i in 1:n_rep
            w1 = reps[i]
            added = 0
            for j in 1:n_rep
                i == j && continue
                w2 = reps[j]
                w1 == w2 && continue
                push!(synonym_pairs, (w1, w2))
                added += 1
                added >= max_partners_per_word && break
            end
        end
    end

    return synonym_pairs, variant_clusters, all_words
end

function _connected_components(nodes::Vector{String}, edges::Vector{Tuple{String, String}})
    adj = Dict{String, Vector{String}}()
    for u in nodes
        adj[u] = String[]
    end
    for (u, v) in edges
        u == v && continue
        push!(get!(adj, u, String[]), v)
        push!(get!(adj, v, String[]), u)
    end

    visited = Set{String}()
    comps = Vector{Vector{String}}()
    for start in sort(nodes)
        start in visited && continue
        comp = String[]
        queue = String[start]
        push!(visited, start)
        head = 1
        while head <= length(queue)
            cur = queue[head]
            head += 1
            push!(comp, cur)
            for nxt in adj[cur]
                if !(nxt in visited)
                    push!(visited, nxt)
                    push!(queue, nxt)
                end
            end
        end
        sort!(comp)
        push!(comps, comp)
    end
    return comps
end

"""
    build_disjoint_splits(
        antonym_pairs::AbstractVector{Tuple{String, String}},
        synonym_pairs::AbstractVector{Tuple{String, String}},
        reading_map::Dict{String, Set{String}};
        all_vocab::AbstractVector{String} = String[],
        seed::Int = 42,
        train_ratio::Float64 = 0.80,
        val_ratio::Float64 = 0.10,
    )

Split antonym pairs into `train`, `val`, and `test` using connected-components partitioning
so zero antonym-graph words overlap across splits. Assigns synonym pairs and sampled unrelated
pairs without leaking `val`/`test` words into `train`.
Returns `(train_df, val_df, test_df, vocab_df, stats)`.
"""
function build_disjoint_splits(
    antonym_pairs::AbstractVector{Tuple{String, String}},
    synonym_pairs::AbstractVector{Tuple{String, String}},
    reading_map::Dict{String, Set{String}};
    all_vocab::AbstractVector{String} = String[],
    seed::Int = 42,
    train_ratio::Float64 = 0.80,
    val_ratio::Float64 = 0.10,
)
    rng = MersenneTwister(seed)

    # Deduplicate directed antonym pairs and collect nodes
    ant_set = Set{Tuple{String, String}}()
    ant_nodes_set = Set{String}()
    for (a, b) in antonym_pairs
        a == b && continue
        push!(ant_set, (a, b))
        push!(ant_set, (b, a))
        push!(ant_nodes_set, a)
        push!(ant_nodes_set, b)
    end

    syn_set = Set{Tuple{String, String}}()
    for (a, b) in synonym_pairs
        a == b && continue
        (a, b) in ant_set && continue
        push!(syn_set, (a, b))
        push!(syn_set, (b, a))
    end

    ant_nodes = sort(collect(ant_nodes_set))
    ant_edges = collect(ant_set)
    comps = _connected_components(ant_nodes, ant_edges)

    # Count directed antonym pairs per component
    node_to_comp = Dict{String, Int}()
    for (cid, comp) in enumerate(comps)
        for w in comp
            node_to_comp[w] = cid
        end
    end
    comp_edge_counts = zeros(Int, length(comps))
    for (a, _) in ant_set
        comp_edge_counts[node_to_comp[a]] += 1
    end

    total_ant_edges = max(sum(comp_edge_counts), 1)
    test_ratio = max(1.0 - train_ratio - val_ratio, 0.05)

    # Shuffle components deterministically, keeping largest components prioritized in deficit order
    perm = shuffle(rng, collect(1:length(comps)))
    sort!(perm; by = cid -> -comp_edge_counts[cid])

    # Assign larger components using deficit-based greedy balancing
    targets = Dict(:train => train_ratio * total_ant_edges, :val => val_ratio * total_ant_edges, :test => test_ratio * total_ant_edges)
    counts = Dict(:train => 0, :val => 0, :test => 0)
    comp_split = Vector{Symbol}(undef, length(comps))

    for cid in perm
        # Pick split with maximum relative deficit `(target - current) / target`
        best_sp = :train
        best_score = -Inf
        for sp in (:train, :val, :test)
            t = max(targets[sp], 1.0)
            score = (targets[sp] - counts[sp]) / t
            if score > best_score
                best_score = score
                best_sp = sp
            end
        end
        comp_split[cid] = best_sp
        counts[best_sp] += comp_edge_counts[cid]
    end

    word_split = Dict{String, String}()
    for (cid, comp) in enumerate(comps)
        sp_str = String(comp_split[cid])
        for w in comp
            word_split[w] = sp_str
        end
    end

    vocab_set = Set{String}(all_vocab)
    union!(vocab_set, ant_nodes_set)
    for (a, b) in syn_set
        push!(vocab_set, a)
        push!(vocab_set, b)
    end
    vocab_list = sort(collect(vocab_set))
    for w in vocab_list
        if !haskey(word_split, w)
            word_split[w] = "extra"
        end
    end

    # Partition antonym pairs strictly by component split
    split_ant = Dict("train" => Tuple{String, String}[], "val" => Tuple{String, String}[], "test" => Tuple{String, String}[])
    for (a, b) in sort(collect(ant_set))
        sp = word_split[a]
        @assert sp == word_split[b] "Connected component split invariant violated for ($a, $b)"
        push!(split_ant[sp], (a, b))
    end

    # Partition synonym pairs without leaking val/test words into train
    val_or_test_words = Set{String}(w for (w, sp) in word_split if sp == "val" || sp == "test")
    train_ant_words = Set{String}(w for (w, sp) in word_split if sp == "train")
    val_ant_words = Set{String}(w for (w, sp) in word_split if sp == "val")
    test_ant_words = Set{String}(w for (w, sp) in word_split if sp == "test")

    split_syn = Dict("train" => Tuple{String, String}[], "val" => Tuple{String, String}[], "test" => Tuple{String, String}[])
    for (a, b) in sort(collect(syn_set))
        if !(a in val_or_test_words) && !(b in val_or_test_words)
            # Keep train synonyms either involving train antonym words or up to a balanced cap
            if (a in train_ant_words) || (b in train_ant_words)
                push!(split_syn["train"], (a, b))
            end
        elseif (a in val_ant_words) && !(b in train_ant_words) && !(b in test_ant_words)
            push!(split_syn["val"], (a, b))
        elseif (a in test_ant_words) && !(b in train_ant_words) && !(b in val_ant_words)
            push!(split_syn["test"], (a, b))
        end
    end

    # Add extra clean train synonyms from background vocabulary so synonym classifier sees diverse examples
    extra_train_syn = Tuple{String, String}[]
    for (a, b) in sort(collect(syn_set))
        if !(a in val_or_test_words) && !(b in val_or_test_words) && !(a in train_ant_words) && !(b in train_ant_words)
            push!(extra_train_syn, (a, b))
        end
    end
    shuffle!(rng, extra_train_syn)
    max_extra_syn = min(length(extra_train_syn), max(length(split_ant["train"]), 2000))
    append!(split_syn["train"], extra_train_syn[1:max_extra_syn])

    # Cap train synonyms to at most 2x train antonyms for stable balance
    shuffle!(rng, split_syn["train"])
    if length(split_syn["train"]) > 2 * max(length(split_ant["train"]), 500)
        resize!(split_syn["train"], 2 * max(length(split_ant["train"]), 500))
    end

    # Retain all antonym-graph words and split synonym words; cap extra background words at 3000
    core_vocab = copy(ant_nodes_set)
    for sp in ("train", "val", "test"), (a, b) in split_syn[sp]
        push!(core_vocab, a)
        push!(core_vocab, b)
    end
    extra_candidates = sort(String[w for w in vocab_list if !(w in core_vocab)])
    shuffle!(rng, extra_candidates)
    if length(extra_candidates) > 3000
        resize!(extra_candidates, 3000)
    end
    vocab_list = sort(vcat(collect(core_vocab), extra_candidates))

    # Sample Unrelated pairs per split
    known_rel = union(ant_set, syn_set)
    train_pool = String[w for w in vocab_list if !(w in val_or_test_words)]
    val_pool = isempty(val_ant_words) ? train_pool : sort(collect(val_ant_words))
    test_pool = isempty(test_ant_words) ? train_pool : sort(collect(test_ant_words))

    function sample_unrelated(
        queries::Vector{String},
        pool::Vector{String},
        target_n::Int;
        ant_pairs::Vector{Tuple{String, String}} = Tuple{String, String}[],
    )
        out = Tuple{String, String}[]
        (isempty(queries) || isempty(pool)) && return out
        seen = Set{Tuple{String, String}}()

        # First guarantee balanced query-side and candidate-side Unrelated negatives for every antonym pair
        for (a, b) in ant_pairs
            for _ in 1:20
                rand_b = pool[rand(rng, 1:length(pool))]
                rand_b == a && continue
                pair_q = (a, rand_b)
                if !(pair_q in known_rel || pair_q in seen)
                    push!(seen, pair_q)
                    push!(out, pair_q)
                    break
                end
            end
            for _ in 1:20
                rand_a = queries[rand(rng, 1:length(queries))]
                rand_a == b && continue
                pair_k = (rand_a, b)
                if !(pair_k in known_rel || pair_k in seen)
                    push!(seen, pair_k)
                    push!(out, pair_k)
                    break
                end
            end
        end

        attempts = 0
        max_attempts = max(target_n * 20, 2000)
        while length(out) < target_n && attempts < max_attempts
            attempts += 1
            w1 = queries[rand(rng, 1:length(queries))]
            w2 = pool[rand(rng, 1:length(pool))]
            w1 == w2 && continue
            pair = (w1, w2)
            (pair in known_rel || pair in seen) && continue
            push!(seen, pair)
            push!(out, pair)
        end
        return out
    end

    split_unrel = Dict(
        "train" => sample_unrelated(
            isempty(train_ant_words) ? train_pool : sort(collect(train_ant_words)),
            train_pool,
            max(length(split_ant["train"]) * 2, 100);
            ant_pairs = split_ant["train"],
        ),
        "val" => sample_unrelated(
            val_pool,
            [w for w in vocab_list if !(w in train_ant_words) && !(w in test_ant_words)],
            max(length(split_ant["val"]) * 2, 20);
            ant_pairs = split_ant["val"],
        ),
        "test" => sample_unrelated(
            test_pool,
            [w for w in vocab_list if !(w in train_ant_words) && !(w in val_ant_words)],
            max(length(split_ant["test"]) * 2, 20);
            ant_pairs = split_ant["test"],
        ),
    )

    function to_df(sp::String)
        w1s = String[]
        w2s = String[]
        labels = Int[]
        for (a, b) in split_ant[sp]
            push!(w1s, a); push!(w2s, b); push!(labels, LABEL_ANTONYM)
        end
        for (a, b) in split_syn[sp]
            push!(w1s, a); push!(w2s, b); push!(labels, LABEL_SYNONYM)
        end
        for (a, b) in split_unrel[sp]
            push!(w1s, a); push!(w2s, b); push!(labels, LABEL_UNRELATED)
        end
        df = DataFrame(w1 = w1s, w2 = w2s, label = labels)
        return df[shuffle(rng, 1:nrow(df)), :]
    end

    train_df = to_df("train")
    val_df = to_df("val")
    test_df = to_df("test")

    clean_reading_map = Dict{String, String}()
    for (w, rset) in reading_map
        isempty(rset) && continue
        kana_r = String[r for r in rset if !any(is_kanji, r)]
        if !isempty(kana_r)
            clean_reading_map[w] = first(sort(kana_r))
        else
            clean_reading_map[w] = first(sort(collect(rset)))
        end
    end

    neg_kana_prefixes = (("不", "ふ"), ("非", "ひ"), ("無", "む"), ("未", "み"))
    readings_col = String[]
    splits_col = String[]
    for w in vocab_list
        r = if haskey(clean_reading_map, w)
            clean_reading_map[w]
        else
            inferred = ""
            if length(w) >= 2
                for (pref, pkana) in neg_kana_prefixes
                    if startswith(w, pref)
                        base = String(w[nextind(w, 1):end])
                        if haskey(clean_reading_map, base)
                            inferred = pkana * clean_reading_map[base]
                            break
                        end
                    end
                end
                if isempty(inferred)
                    c_end = last(w)
                    if '\u3041' <= c_end <= '\u3096'
                        stem = String(w[1:prevind(w, lastindex(w))])
                        if haskey(clean_reading_map, stem)
                            inferred = clean_reading_map[stem] * string(c_end)
                        end
                    end
                end
            end
            isempty(inferred) ? to_hiragana(w) : inferred
        end
        push!(readings_col, r)
        push!(splits_col, word_split[w])
    end
    vocab_df = DataFrame(word = vocab_list, reading = readings_col, split = splits_col)

    stats = (
        vocab_size = length(vocab_list),
        num_components = length(comps),
        train_antonym_words = length(train_ant_words),
        val_antonym_words = length(val_ant_words),
        test_antonym_words = length(test_ant_words),
        overlap_train_val = length(intersect(train_ant_words, val_ant_words)),
        overlap_train_test = length(intersect(train_ant_words, test_ant_words)),
        overlap_val_test = length(intersect(val_ant_words, test_ant_words)),
        train_counts = (antonym = length(split_ant["train"]), synonym = length(split_syn["train"]), unrelated = length(split_unrel["train"])),
        val_counts = (antonym = length(split_ant["val"]), synonym = length(split_syn["val"]), unrelated = length(split_unrel["val"])),
        test_counts = (antonym = length(split_ant["test"]), synonym = length(split_syn["test"]), unrelated = length(split_unrel["test"])),
    )

    return train_df, val_df, test_df, vocab_df, stats
end

"""
    download_and_build_dataset(; raw_dir = "data/raw", out_dir = "data/processed", seed = 42)

Download Japanese WordNet v2.1 and Sudachi Synonyms, construct antonym/synonym/unrelated pairs,
apply connected-components disjoint splitting, and save `train.csv`, `val.csv`, `test.csv`, `vocab.csv`.
"""
function download_and_build_dataset(; raw_dir::AbstractString = "data/raw", out_dir::AbstractString = "data/processed", seed::Int = 42)
    xml_path, sudachi_path = _ensure_raw_datasets(raw_dir)
    mkpath(out_dir)

    @info "Parsing Japanese WordNet XML..." path=xml_path
    synset_clean, antonym_links, reading_map, wn_words, word_max_count, corpus_freq = _parse_wnja_xml_detailed(xml_path)
    synset_words = _top_synset_words(synset_clean, corpus_freq; max_lemmas_per_synset = 6)

    @info "Parsing Sudachi Synonym Dictionary..." path=sudachi_path
    sudachi_syn_pairs, variant_clusters, sudachi_words = _parse_sudachi_synonyms(sudachi_path)

    # Merge reading / variant map
    for (w, vset) in variant_clusters
        dest = get!(reading_map, w, Set{String}())
        union!(dest, vset)
    end

    all_vocab_set = union(wn_words, sudachi_words)

    # Build WordNet synonym pairs (same synset)
    synonym_pairs = Set{Tuple{String, String}}(sudachi_syn_pairs)
    for (_, words) in synset_words
        n = length(words)
        n >= 2 || continue
        for i in 1:n, j in 1:n
            i == j && continue
            push!(synonym_pairs, (words[i], words[j]))
        end
    end

    # Build pruned & shape/structure-aligned WordNet antonym pairs
    antonym_pairs = _build_wnja_antonym_pairs(
        synset_clean,
        antonym_links,
        word_max_count,
        corpus_freq,
        synonym_pairs,
    )

    # Add morphological negation antonym pairs when both base (>= 2 chars) and prefix*base exist in vocab
    for w in all_vocab_set
        length(w) >= 2 || continue
        for pref in MORPH_NEG_PREFIXES
            neg_w = pref * w
            if neg_w in all_vocab_set && !((w, neg_w) in synonym_pairs)
                push!(antonym_pairs, (w, neg_w))
                push!(antonym_pairs, (neg_w, w))
            end
        end
    end

    # Remove any antonym pair that also appears in synonym_pairs
    filter!(p -> !(p in synonym_pairs), antonym_pairs)

    all_vocab = sort(collect(all_vocab_set))
    train_df, val_df, test_df, vocab_df, stats = build_disjoint_splits(
        collect(antonym_pairs),
        collect(synonym_pairs),
        reading_map;
        all_vocab = all_vocab,
        seed = seed,
    )

    CSV.write(joinpath(out_dir, "train.csv"), train_df)
    CSV.write(joinpath(out_dir, "val.csv"), val_df)
    CSV.write(joinpath(out_dir, "test.csv"), test_df)
    CSV.write(joinpath(out_dir, "vocab.csv"), vocab_df)

    @info "Dataset built and saved." out_dir stats
    return stats
end

"""
    load_dataset_splits(out_dir = "data/processed") -> NamedTuple

Load `train.csv`, `val.csv`, `test.csv`, and `vocab.csv` from `out_dir`.
"""
function load_dataset_splits(out_dir::AbstractString = "data/processed")
    train_df = CSV.read(joinpath(out_dir, "train.csv"), DataFrame; stringtype = String)
    val_df = CSV.read(joinpath(out_dir, "val.csv"), DataFrame; stringtype = String)
    test_df = CSV.read(joinpath(out_dir, "test.csv"), DataFrame; stringtype = String)
    vocab_df = CSV.read(joinpath(out_dir, "vocab.csv"), DataFrame; stringtype = String)
    return (train = train_df, val = val_df, test = test_df, vocab = vocab_df)
end
