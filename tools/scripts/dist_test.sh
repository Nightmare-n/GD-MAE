#!/usr/bin/env bash

while true
do
    PORT=$(( ((RANDOM<<15)|RANDOM) % 49152 + 10000 ))
    status="$(nc -z 127.0.0.1 $PORT < /dev/null &>/dev/null; echo $?)"
    if [ "${status}" != "0" ]; then
        break;
    fi
done
echo $PORT

export CUDA_VISIBLE_DEVICES=0,1,2,3

NGPUS=4

CFG_NAME=kitti_models/graph_rcnn_po
TAG_NAME=default

CKPT=../data/ckpts/graph_rcnn_po.pth

python -m torch.distributed.launch --nproc_per_node=${NGPUS} --master_port $PORT test.py --launcher pytorch --cfg_file cfgs/$CFG_NAME.yaml --workers 4 --extra_tag $TAG_NAME --ckpt $CKPT
