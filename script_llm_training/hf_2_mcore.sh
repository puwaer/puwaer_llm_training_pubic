CUDA_VISIBLE_DEVICES=0 \
swift export \
    --model ./puwaer_llm_training_pubic/base_model/Qwen3-14B \
    --to_mcore true \
    --torch_dtype bfloat16 \
    --output_dir ./puwaer_llm_training_pubic/base_model/Qwen3-14B-mcore

CUDA_VISIBLE_DEVICES=0 \
swift export \
    --mcore_model output_path/Qwen3-14B-mcore \
    --to_hf true \
    --torch_dtype bfloat16 \
    --output_dir ./puwaer_llm_training_pubic/base_model/Qwen3-14B-mcore-hf