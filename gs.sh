for a in $(seq 1 1 4 | awk '{print $1/5.0}')
do   
    export CUDA_VISIBLE_DEVICES=3
    python train.py PETA --gpus 3 --a $a
done