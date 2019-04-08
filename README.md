## Supervised Nuclei Center Point Prediction for C-Elegans

### working on:
- linux 18.04
- python 3.6
- tensorflow 1.12

### To run:
(all scripts should be run at project root dir)
1. run consolidate with `python mowa/consolidate_data.py $dataset_dir ./data` which creates $root/data directory. 
requires 
the 
`imagesAsMhdRawAligned`, 
`groundTruthInstanceSeg`
 (containing the .ano.curated.aligned.tiff and .ano.curated.aligned.txt files), and a `universe.txt` file
 
2. `python mowa/train.py` to run or with a `-d` flag to run with some debug settings, e.g. 13 epochs, model 
checkpointing every 2 epoch, etc.