export nproc_per_node=8
export WANDB_PROJECT="test-14b"
export OMP_NUM_THREADS=4

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=$nproc_per_node \
swift sft \
    --model ./puwaer_llm_training_pubic/base_model/Qwen3-14B \
    --model_type qwen3 \
    --train_type full \
    --dataset dataset_path/test_data.jsonl \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --learning_rate 1e-5 \
    --gradient_accumulation_steps 4 \
    --eval_steps 2000 \
    --save_steps 2000 \
    --save_total_limit 10 \
    --logging_steps 5 \
    --max_length 4096 \
    --output_dir ./output_model \
    --system 'You are a helpful assistant.' \
    --warmup_ratio 0.1 \
    --lr_scheduler_type cosine \
    --dataloader_num_workers 4 \
    --model_author swift \
    --model_name swift-robot \
    --report_to wandb \
    --run_name "qwen3_14b_test_model" \
    --attn_impl flash_attn \
    --deepspeed zero1 
