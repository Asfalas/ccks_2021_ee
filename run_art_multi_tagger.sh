cd article_evt_ext/
pwd
CUDA_VISIBLE_DEVICES=0 python -u main.py \
    --mode train \
    --task multi_tagger \
    --debug 1 \
    --dataset duee_fin \
    --conf conf/duee_fin_multi_tagger.json \
    --use_cpu 1 \
    --epochs 20 \
    --accumulate_step 2 \
    --batch_size 2