cd sentence_evt_ext/
pwd
CUDA_VISIBLE_DEVICES=0 python -u main.py \
    --mode train \
    --task men_cls \
    --debug 1 \
    --dataset duee \
    --conf conf/duee_men_cls.json \
    --use_cpu 0 \
    --epochs 1 \
    --accumulate_step 1 \
    --batch_size 2