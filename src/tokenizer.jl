using Unicode

"""
    CharBertTokenizer

Character-level Japanese BERT tokenizer compatible with HuggingFace `vocab.txt` files.
"""
struct CharBertTokenizer
    vocab::Vector{String}
    token_to_id::Dict{String, Int}
    pad_id::Int
    unk_id::Int
    cls_id::Int
    sep_id::Int
    mask_id::Int
end

"""
    load_vocab(vocab_path::AbstractString; max_vocab_size::Union{Int, Nothing} = nothing) -> CharBertTokenizer

Load a `vocab.txt` file (handling LF and CRLF line endings while preserving whitespace tokens)
and construct a `CharBertTokenizer`. If `max_vocab_size` is provided, truncates the vocabulary
to `max_vocab_size` entries so out-of-range tokens map to `[UNK]`.
"""
function load_vocab(vocab_path::AbstractString; max_vocab_size::Union{Int, Nothing} = nothing)
    lines = readlines(vocab_path)
    while !isempty(lines) && isempty(rstrip(lines[end], '\r'))
        pop!(lines)
    end
    vocab = String[String(rstrip(line, '\r')) for line in lines]
    if max_vocab_size !== nothing && length(vocab) > max_vocab_size
        resize!(vocab, max_vocab_size)
    end
    token_to_id = Dict{String, Int}()
    sizehint!(token_to_id, length(vocab))
    for (idx, tok) in enumerate(vocab)
        # Keep first index if duplicate token exists
        if !haskey(token_to_id, tok)
            token_to_id[tok] = idx
        end
    end
    pad_id = get(token_to_id, "[PAD]", 1)
    unk_id = get(token_to_id, "[UNK]", 2)
    cls_id = get(token_to_id, "[CLS]", 3)
    sep_id = get(token_to_id, "[SEP]", 4)
    mask_id = get(token_to_id, "[MASK]", 5)
    return CharBertTokenizer(vocab, token_to_id, pad_id, unk_id, cls_id, sep_id, mask_id)
end

function _word_to_token_ids(tok::CharBertTokenizer, word::AbstractString)
    norm = Unicode.normalize(String(word), :NFKC)
    ids = Int[]
    sizehint!(ids, length(norm))
    for ch in norm
        s = string(ch)
        push!(ids, get(tok.token_to_id, s, tok.unk_id))
    end
    return ids
end

"""
    encode_single(tok::CharBertTokenizer, word::AbstractString; max_len::Int = 16)

Encode a single word as `[CLS] c1 c2 ... [SEP]` padded to `max_len`.
Returns `(token_ids, token_type_ids, pad_mask)`.
"""
function encode_single(tok::CharBertTokenizer, word::AbstractString; max_len::Int = 16)
    max_len >= 3 || error("max_len must be >= 3, got $max_len")
    char_ids = _word_to_token_ids(tok, word)
    n_keep = min(length(char_ids), max_len - 2)
    ids = fill(tok.pad_id, max_len)
    type_ids = ones(Int, max_len)
    mask = zeros(Float32, max_len)

    ids[1] = tok.cls_id
    mask[1] = 1.0f0
    for i in 1:n_keep
        ids[i + 1] = char_ids[i]
        mask[i + 1] = 1.0f0
    end
    ids[n_keep + 2] = tok.sep_id
    mask[n_keep + 2] = 1.0f0
    return ids, type_ids, mask
end

"""
    encode_batch_single(tok::CharBertTokenizer, words::AbstractVector{<:AbstractString}; max_len::Int = 16, dynamic_pad::Bool = false)

Encode a batch of single words into `(token_ids, token_type_ids, pad_mask)` matrices of shape `(seq_len, batch_size)`.
"""
function encode_batch_single(
    tok::CharBertTokenizer,
    words::AbstractVector{<:AbstractString};
    max_len::Int = 16,
    dynamic_pad::Bool = false,
)
    B = length(words)
    ids = fill(tok.pad_id, max_len, B)
    type_ids = ones(Int, max_len, B)
    mask = zeros(Float32, max_len, B)
    max_used = 3
    @inbounds for b in 1:B
        char_ids = _word_to_token_ids(tok, words[b])
        n_keep = min(length(char_ids), max_len - 2)
        ids[1, b] = tok.cls_id
        mask[1, b] = 1.0f0
        for i in 1:n_keep
            ids[i + 1, b] = char_ids[i]
            mask[i + 1, b] = 1.0f0
        end
        ids[n_keep + 2, b] = tok.sep_id
        mask[n_keep + 2, b] = 1.0f0
        max_used = max(max_used, n_keep + 2)
    end
    if dynamic_pad && max_used < max_len
        return ids[1:max_used, :], type_ids[1:max_used, :], mask[1:max_used, :]
    end
    return ids, type_ids, mask
end

"""
    encode_batch_pair(tok::CharBertTokenizer, w1s::AbstractVector{<:AbstractString}, w2s::AbstractVector{<:AbstractString}; max_len::Int = 32, dynamic_pad::Bool = false)

Encode word pairs as `[CLS] chars(w1) [SEP] chars(w2) [SEP]` with `token_type_ids` `1` for segment 1 and `2` for segment 2.
Returns `(token_ids, token_type_ids, pad_mask)` of shape `(seq_len, batch_size)`.
"""
function encode_batch_pair(
    tok::CharBertTokenizer,
    w1s::AbstractVector{<:AbstractString},
    w2s::AbstractVector{<:AbstractString};
    max_len::Int = 32,
    dynamic_pad::Bool = false,
)
    length(w1s) == length(w2s) || error("Batch length mismatch: $(length(w1s)) vs $(length(w2s))")
    max_len >= 5 || error("max_len must be >= 5, got $max_len")
    B = length(w1s)
    ids = fill(tok.pad_id, max_len, B)
    type_ids = ones(Int, max_len, B)
    mask = zeros(Float32, max_len, B)

    budget = max_len - 3
    max_used = 5
    @inbounds for b in 1:B
        c1 = _word_to_token_ids(tok, w1s[b])
        c2 = _word_to_token_ids(tok, w2s[b])
        # Symmetrically trim longer word if total length exceeds budget
        while length(c1) + length(c2) > budget
            if length(c1) >= length(c2)
                pop!(c1)
            else
                pop!(c2)
            end
        end
        pos = 1
        ids[pos, b] = tok.cls_id
        type_ids[pos, b] = 1
        mask[pos, b] = 1.0f0
        pos += 1
        for id in c1
            ids[pos, b] = id
            type_ids[pos, b] = 1
            mask[pos, b] = 1.0f0
            pos += 1
        end
        ids[pos, b] = tok.sep_id
        type_ids[pos, b] = 1
        mask[pos, b] = 1.0f0
        pos += 1
        for id in c2
            ids[pos, b] = id
            type_ids[pos, b] = 2
            mask[pos, b] = 1.0f0
            pos += 1
        end
        ids[pos, b] = tok.sep_id
        type_ids[pos, b] = 2
        mask[pos, b] = 1.0f0
        max_used = max(max_used, pos)
    end
    if dynamic_pad && max_used < max_len
        return ids[1:max_used, :], type_ids[1:max_used, :], mask[1:max_used, :]
    end
    return ids, type_ids, mask
end

const NEGATION_PREFIXES = ("不", "非", "無", "未", "反", "脱", "逆", "異")

const OPPOSING_KANJI_PAIRS = let
    raw_pairs = [
        ('高', '低'), ('高', '安'), ('大', '小'), ('多', '少'), ('増', '減'),
        ('上', '下'), ('前', '後'), ('左', '右'), ('内', '外'), ('入', '出'),
        ('開', '閉'), ('始', '終'), ('強', '弱'), ('長', '短'), ('広', '狭'),
        ('深', '浅'), ('重', '軽'), ('新', '旧'), ('明', '暗'), ('暑', '寒'),
        ('暖', '冷'), ('温', '冷'), ('良', '悪'), ('善', '悪'), ('正', '誤'),
        ('正', '負'), ('有', '無'), ('生', '死'), ('勝', '敗'), ('攻', '守'),
        ('売', '買'), ('貸', '借'), ('与', '受'), ('送', '受'), ('進', '退'),
        ('昇', '降'), ('浮', '沈'), ('動', '静'), ('早', '遅'), ('速', '遅'),
        ('遠', '近'), ('硬', '軟'), ('固', '柔'), ('乾', '湿'), ('晴', '曇'),
        ('晴', '雨'), ('男', '女'), ('父', '母'), ('天', '地'), ('昼', '夜'),
        ('朝', '夕'), ('公', '私'), ('主', '従'), ('個', '全'), ('自', '他'),
        ('賛', '反'), ('加', '減'), ('黒', '白'), ('表', '裏'), ('陰', '陽'),
        ('貴', '賤'), ('富', '貧'), ('栄', '衰'), ('興', '亡'), ('存', '亡'),
        ('満', '欠'), ('盈', '虚'), ('実', '虚'), ('真', '偽'), ('純', '雑'),
        ('粗', '密'), ('濃', '淡'), ('厚', '薄'), ('太', '細'), ('肥', '痩'),
        ('鋭', '鈍'), ('利', '害'), ('得', '失'), ('損', '得'), ('吉', '凶'),
        ('福', '禍'), ('喜', '悲'), ('笑', '泣'), ('愛', '憎'), ('好', '嫌'),
        ('親', '疎'), ('合', '離'), ('集', '散'), ('和', '戦'), ('同', '異'),
        ('常', '変'), ('順', '逆'), ('正', '邪'), ('優', '劣'), ('勝', '劣'),
        ('問', '答'), ('疑', '信'), ('起', '寝'), ('覚', '眠'), ('着', '脱'),
        ('乗', '降'), ('登', '降'), ('往', '復'), ('来', '去'), ('送', '迎'),
        ('押', '引'), ('吸', '吐'), ('伸', '縮'), ('膨', '収'), ('緩', '急'),
        ('繁', '簡'), ('単', '複'), ('簡', '難'), ('易', '難'), ('全', '部'),
        ('完', '不'), ('既', '未'), ('永', '瞬'), ('古', '今'), ('昔', '今'),
        ('東', '西'), ('南', '北'), ('縦', '横'), ('表', '内'), ('本', '末'),
        ('源', '末'), ('先', '後'), ('初', '終'), ('首', '尾'), ('頭', '尾'),
        ('熱', '冷'), ('寒', '暖'), ('寒', '温'), ('涼', '暖'), ('涼', '暑'),
        ('始', '了'), ('開', '終'), ('成', '敗'), ('成', '失'), ('攻', '防'),
        ('美', '醜'), ('賢', '愚'), ('苦', '楽'), ('寛', '厳'), ('清', '濁'),
        ('乾', '濡'), ('干', '満'), ('賛', '否'), ('濃', '薄'),
    ]
    s = Set{Tuple{Char, Char}}()
    for (a, b) in raw_pairs
        push!(s, (a, b))
        push!(s, (b, a))
    end
    s
end

@inline is_kanji(c::Char) = ('\u4e00' <= c <= '\u9fff') || ('\u3400' <= c <= '\u4dbf') || ('\uf900' <= c <= '\ufaff')
@inline has_katakana(w::AbstractString) = any(c -> ('\u30a1' <= c <= '\u30fc'), w)

"""
    word_shape_class(w::AbstractString) -> Symbol

Classify the surface morphological shape of a Japanese word into one of
`:katakana`, `:single_kanji`, `:kanji_compound`, `:i_adj`, `:kanji_okuri`, or `:hiragana`.
"""
function word_shape_class(w::AbstractString)::Symbol
    s = Unicode.normalize(String(w), :NFKC)
    isempty(s) && return :other
    has_katakana(s) && return :katakana
    chars = collect(s)
    if all(c -> is_kanji(c) || c == '々', chars)
        return length(chars) == 1 ? :single_kanji : :kanji_compound
    elseif is_kanji(chars[1]) && chars[end] == 'い'
        return :i_adj
    elseif any(is_kanji, chars)
        return :kanji_okuri
    else
        return :hiragana
    end
end

"""
    has_opposing_kanji(w1::AbstractString, w2::AbstractString; require_same_shape::Bool = false) -> Bool

Return `true` if `w1` and `w2` contain an opposing Kanji pair from `OPPOSING_KANJI_PAIRS`.
"""
function has_opposing_kanji(w1::AbstractString, w2::AbstractString; require_same_shape::Bool = false)::Bool
    if require_same_shape && word_shape_class(w1) != word_shape_class(w2)
        return false
    end
    for a in w1
        is_kanji(a) || continue
        for b in w2
            is_kanji(b) || continue
            if a != b && (a, b) in OPPOSING_KANJI_PAIRS
                return true
            end
        end
    end
    return false
end

"""
    is_negation_pair(w1::AbstractString, w2::AbstractString) -> Bool

Return `true` if one word is formed by prepending a negation prefix in `NEGATION_PREFIXES` to the other.
"""
function is_negation_pair(w1::AbstractString, w2::AbstractString)::Bool
    s1 = Unicode.normalize(String(w1), :NFKC)
    s2 = Unicode.normalize(String(w2), :NFKC)
    for p in NEGATION_PREFIXES
        if s2 == p * s1 || s1 == p * s2
            return true
        end
    end
    return false
end

"""
    is_structural_antonym_pair(w1::AbstractString, w2::AbstractString) -> Bool

Return `true` if `(w1, w2)` is either a negation-prefix pair or a same-shape opposing-Kanji pair.
"""
is_structural_antonym_pair(w1::AbstractString, w2::AbstractString)::Bool =
    has_opposing_kanji(w1, w2; require_same_shape = true) || is_negation_pair(w1, w2)

"""
    extract_pair_surface_features(w1::AbstractString, w2::AbstractString) -> Vector{Float32}

Extract 8 surface/morphological features for a Japanese word pair `(w1, w2)`:
1. Normalized character length difference
2. Exact length match indicator
3. Shared Kanji Jaccard ratio
4. Negation prefix asymmetry (`不`, `非`, `無`, `未`, `反`, `脱`, `逆`, `異`)
5. Opposing Kanji pair presence indicator
6. Shared final character indicator (e.g. matching okurigana/suffix)
7. Full character Jaccard similarity
8. Word shape class compatibility indicator (`word_shape_class(w1) == word_shape_class(w2)` or exact negation prefix)
"""
function extract_pair_surface_features(w1::AbstractString, w2::AbstractString)
    s1 = Unicode.normalize(String(w1), :NFKC)
    s2 = Unicode.normalize(String(w2), :NFKC)
    c1 = collect(s1)
    c2 = collect(s2)
    l1 = length(c1)
    l2 = length(c2)

    len_diff = Float32(abs(l1 - l2) / max(l1, l2, 1))
    len_eq = l1 == l2 ? 1.0f0 : 0.0f0

    k1 = Set(ch for ch in c1 if is_kanji(ch))
    k2 = Set(ch for ch in c2 if is_kanji(ch))
    k_union = length(union(k1, k2))
    kanji_shared = k_union > 0 ? Float32(length(intersect(k1, k2)) / k_union) : 0.0f0

    neg_asym = 0.0f0
    for p in NEGATION_PREFIXES
        if s2 == p * s1 || s1 == p * s2
            neg_asym = 1.0f0
            break
        elseif (startswith(s1, p) && !startswith(s2, p) || startswith(s2, p) && !startswith(s1, p)) &&
               !isempty(intersect(k1, k2))
            neg_asym = max(neg_asym, 0.5f0)
        end
    end

    opposing_kanji = 0.0f0
    for a in k1, b in k2
        if a != b && (a, b) in OPPOSING_KANJI_PAIRS
            opposing_kanji = 1.0f0
            break
        end
    end

    shared_suffix = (l1 >= 2 && l2 >= 2 && c1[end] == c2[end]) ? 1.0f0 : 0.0f0

    set1 = Set(c1)
    set2 = Set(c2)
    u_len = length(union(set1, set2))
    char_jaccard = u_len > 0 ? Float32(length(intersect(set1, set2)) / u_len) : 0.0f0

    shape_compat = (word_shape_class(s1) == word_shape_class(s2) || neg_asym == 1.0f0) ? 1.0f0 : 0.0f0

    return Float32[
        len_diff,
        len_eq,
        kanji_shared,
        neg_asym,
        opposing_kanji,
        shared_suffix,
        char_jaccard,
        shape_compat,
    ]
end

"""
    extract_batch_surface_features(w1s::AbstractVector{<:AbstractString}, w2s::AbstractVector{<:AbstractString}) -> Matrix{Float32}

Extract `(8, batch_size)` surface feature matrix for a batch of pairs.
"""
function extract_batch_surface_features(
    w1s::AbstractVector{<:AbstractString},
    w2s::AbstractVector{<:AbstractString},
)
    B = length(w1s)
    feats = Matrix{Float32}(undef, 8, B)
    @inbounds for b in 1:B
        feats[:, b] = extract_pair_surface_features(w1s[b], w2s[b])
    end
    return feats
end
