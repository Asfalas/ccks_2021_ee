cd article_evt_ext/
pwd
CUDA_VISIBLE_DEVICES=0 python -u main.py \
    --mode train \
    --task joint \
    --debug 1 \
    --dataset duee_fin \
    --conf conf/duee_fin_joint.json \
    --use_cpu 1 \
    --epochs 1 \
    --accumulate_step 1 \
    --batch_size 1