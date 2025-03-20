# example train
CUDA_VISIBLE_DEVICES=0 torchrun \
    --nnodes=1 \
    --nproc_per_node=1 \
    --master_port=12345 \
    train.py \
    --config config/000_DiffMoE_S_E16_Flow.yaml
