CUDA_VISIBLE_DEVICES=0 python -u main.py \
    --mode train \
    --task joint \
    --debug 0 \
    --dataset ccks \
    --conf conf/ccks_joint.json \
    --use_cpu 0 \
    --epochs 20 \
    --accumulate_step 1 \
    --batch_size 64
    
CUDA_VISIBLE_DEVICES=0 python -u main.py \
    --mode test \
    --task joint \
    --debug 0 \
    --dataset ccks \
    --conf conf/ccks_joint.json \
    --use_cpu 0 \
    --epochs 20 \
    --accumulate_step 1 \
    --batch_size 64