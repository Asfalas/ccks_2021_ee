cd sentence_evt_ext/
pwd
CUDA_VISIBLE_DEVICES=0 python -u main.py \
    --mode train \
    --task joint \
    --debug 1 \
    --dataset duee \
    --conf conf/duee_joint.json \
    --use_cpu 1 \
    --epochs 10 \
    --accumulate_step 1 \
    --batch_size 1