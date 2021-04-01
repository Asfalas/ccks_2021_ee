cd sentence_evt_ext/
pwd
CUDA_VISIBLE_DEVICES=0 python -u main.py \
    --mode train \
    --debug 0 \
    --dataset duee \
    --conf conf/duee_evt_men.json \
    --use_cpu 0 \
    --epochs 20 \
    --accumulate_step 1 \
    --batch_size 24