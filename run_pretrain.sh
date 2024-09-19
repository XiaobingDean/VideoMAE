
OUTPUT_DIR='output/ssv2_videomae_pretrain_base_patch16_224_frame_16x2_tube_mask_ratio_0.9_e800'
DATA_PATH='NViST/data'

OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=2 \
        --master_port 12320 --nnodes=1 \
        --node_rank=0 --master_addr=127.0.0.1 \
        run_mae_pretraining.py \
        --data_path ${DATA_PATH} \
        --mask_type tube \
        --mask_ratio 0.9 \
        --model pretrain_videomae_base_patch16_224 \
        --decoder_depth 4 \
        --batch_size 32 \
        --num_frames 16 \
        --sampling_rate 2 \
        --opt adamw \
        --opt_betas 0.9 0.95 \
        --warmup_epochs 40 \
        --save_ckpt_freq 200 \
        --epochs 100001 \
        --log_dir ${OUTPUT_DIR} \
        --output_dir ${OUTPUT_DIR}
