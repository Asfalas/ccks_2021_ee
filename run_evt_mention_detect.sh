cd sentence_evt_ext/

CUDA_VISIBLE_DEVICES=0 python -u main.py \
    --mode train \
    --debug 1 \
    --dataset duee \
    --conf conf/duee_evt_men.json \
    --use_cpu 1 \
    --epochs 20 \
    --accumulate_step 1 \
    --batch_size 32