#!/bin/bash
# Author: Jianfei Guo
# https://github.com/PJLab-ADG/neuralsim/blob/main/dataio/autonomous_driving/waymo/download_waymo.sh

# NOTE: Before proceeding, you need to fill out the Waymo terms of use and complete `gcloud auth login`.

lst=$1 # dataio/autonomous_driving/waymo/waymo_static_32.lst
dest=$2 # /data1/waymo/training/

mkdir -p $dest

total_files=$(cat $lst | wc -l)
counter=0
# Read filenames from the .lst file and process them one by one
while IFS= read -r filename; do
    counter=$((counter + 1))
    echo "[${counter}/${total_files}] Dowloading $filename ..."

    # can be in training
    source=gs://waymo_open_dataset_v_1_4_2/individual_files/training
    gsutil cp -n ${source}/${filename}.tfrecord ${dest}

    # or can be in validation
    source=gs://waymo_open_dataset_v_1_4_2/individual_files/validation
    gsutil cp -n ${source}/${filename}.tfrecord ${dest}
done < $lst