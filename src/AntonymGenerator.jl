module AntonymGenerator

export DeviceConfig, resolve_device, to_device, cpu_device, reclaim_device_memory!
export CharBertTokenizer, load_vocab, encode_single, encode_batch_single, encode_batch_pair
export extract_pair_surface_features, extract_batch_surface_features
export BertConfig, BertEmbeddings, BertSelfAttention, BertLayer, BertBackbone
export load_safetensors, load_pretrained_bert, mean_pool, char_embedding_pool
export BiEncoder, CrossEncoder, encode_query, encode_candidate, encode_query_and_candidate
export multi_positive_infonce_loss, cross_encoder_loss, l2_normalize
export LABEL_UNRELATED, LABEL_SYNONYM, LABEL_ANTONYM
export is_clean_ja_word, to_hiragana, build_disjoint_splits, download_and_build_dataset, load_dataset_splits
export AntonymPipeline, build_vocab_index!, retrieve_stage1, generate_antonyms, postprocess_candidates
export evaluate_stage1, evaluate_pipeline
export train_bi_encoder!, mine_hard_negatives, train_cross_encoder!, save_artifacts, load_artifacts

include("device.jl")
include("tokenizer.jl")
include("bert.jl")
include("models.jl")
include("data.jl")
include("pipeline.jl")
include("evaluate.jl")
include("train.jl")

end
