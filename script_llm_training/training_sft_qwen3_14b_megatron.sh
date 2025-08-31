nproc_per_node=8
export WANDB_PROJECT="test-14b"
export OMP_NUM_THREADS=4

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=$nproc_per_node \
megatron sft \
    --load ./puwaer_llm_training_pubic/base_model/Qwen3-14B-mcore \
    --dataset dataset_path/test_data.jsonl \
    --tensor_model_parallel_size 1 \
    --pipeline_model_parallel_size 2 \
    --context_parallel_size 1 \
    --sequence_parallel false \
    --context_parallel_size 1 \
    --streaming true \
    --train_iters 620 \
    --packing true \
    --packing_cache /root/dataDisk/packing_cache \
    --micro_batch_size 8 \
    --global_batch_size 128 \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1 \
    --finetune true \
    --cross_entropy_loss_fusion true \
    --lr 1e-5 \
    --min_lr 1e-5 \
    --lr_warmup_iters 10 \
    --max_epochs 1 \
    --lr_decay_style linear \
    --save ./output_model/Qwen3-14B-mcore \
    --eval_interval 500 \
    --save_interval 500 \
    --eval_iters 5 \
    --max_length 4096 \
    --use_flash_attn true \
    --num_workers 8 \
    --no_save_optim true \
    --no_save_rng true \
    --dataset_num_proc 4 \
    --model_author swift \
    --model_name swift-robot \
    --wandb_project test-14b \
    --wandb_exp_name sft_qwen3_14b_test_8gpu_mega \
    --wandb_save_dir ./wandb_logs \