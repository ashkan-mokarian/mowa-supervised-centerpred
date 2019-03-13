#!/usr/bin/env bash

# Settings
num_epochs=1000
gpu_vis_devices="1"
virtual_environment="/home/ashkan/.virtualenvs/mowa/bin/activate"

# script assumes data, output, mowa, folders already exist
if [[ ${virtual_environment} != "" ]]; then
    source ${virtual_environment}
fi

# if $projectdir/data does not already exist, then it means it needs data consolidate
# for now since one has to manually distribute train/test data, lets skip this part

projectdir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
cd $projectdir
# must be some better workaround for python modules without adding to pythonpath, relative pathing when in parent dir?
export PYTHONPATH=$PYTHONPATH:$projectdir/mowa

# Run training
echo Starting training
CUDA_VISIBLE_DEVICES=${gpu_vis_devices} \
    python mowa/train.py $num_epochs \
    2>&1 | tee ./output/train.log
#
#echo Starting evaluation
## Run evaluation afterwards
#CUDA_VISIBLE_DEVICES=${gpu_vis_devices} \
#python mowa/evaluate.py \
#    >> ./output/eval_log.txt

echo FINISHED!!!
