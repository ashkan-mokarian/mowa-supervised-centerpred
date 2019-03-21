#!/usr/bin/env bash
data_dir="/home/ashkan/workspace/myCode/MoWA/mowa-supervised-centerpred/data"
projectdir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
experiments_dir="$projectdir/experiments"
ID="$(date +"%y%m%d_%H%M%S")"

experiment_name="MoWA-$1-$ID"

cd $experiments_dir
mkdir $experiment_name

experiment_dir="$experiments_dir/$experiment_name"
cd $experiment_dir

mkdir mowa
mkdir output
ln -s $data_dir data
cp -a $projectdir/mowa/. ./mowa/
cp $projectdir/*.sh ./
cp $projectdir/params.json ./
cp $projectdir/model_description.txt ./

echo Go and run it manually, first check parameters in run_experiment.sh

echo FINISH!!!
exit 0