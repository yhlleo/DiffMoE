CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nnodes=1 --nproc_per_node=4  --master_port=29520 sample_ddp_feature.py --image-size 256 \
    --per-proc-batch-size 125 --num-fid-samples 50000 --cfg-scale 1.0 --num-sampling-steps 250 --sample-dir /m2v_intern/shiminglei/DiffMoE/samples \
    --ckpt /m2v_intern/shiminglei/DiT_MoE_Dynamic/Done-Exps/P0-SOTA-Dense/0182-2025_DiT-XL-2-Flow_Dense_1data_0noise_global_seed0_usesittransport/checkpoints/0700000.pt

CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nnodes=1 --nproc_per_node=4  --master_port=29519 sample_ddp_feature.py --image-size 256 \
    --per-proc-batch-size 125 --num-fid-samples 50000 --cfg-scale 1.0 --num-sampling-steps 250 --sample-dir /m2v_intern/shiminglei/DiffMoE/samples \
    --ckpt /m2v_intern/shiminglei/DiT_MoE_Dynamic/Done-Exps/P0-SOTA-Dense/0182-2024_Dense-DiT-L-Flow/checkpoints/0700000.pt
