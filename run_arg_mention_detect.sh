cd sentence_evt_ext/
pwd
CUDA_VISIBLE_DEVICES=0 python -u main.py \
    --mode test \
    --task arg_men_detect \
    --debug 0 \
    --dataset duee \
    --conf conf/duee_arg_men.json \
    --use_cpu 0 \
    --epochs 10 \
    --accumulate_step 1 \
    --batch_size 16