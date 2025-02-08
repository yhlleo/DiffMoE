# example train
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
    --nnodes=1 \
    --nproc_per_node=4 \
    --master_port=12345 \
    train.py \
    --config ../config/000_DiffMoE_S_E16_Flow.yaml

# example cache
torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_port=12345 \
    main_cache.py \
    --img_size 128 --vae_path stabilityai/sd-vae-ft-mse --vae_embed_dim 16 \
    --batch_size 256 \
    --data_path ${IMAGENET_PATH} --cached_path cache/
