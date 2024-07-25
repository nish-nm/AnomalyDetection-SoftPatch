#!/bin/bash

# Define the datapath (you might want to adjust this if using Docker volumes)
datapath=/app/MVTec
datasets=('carpet')
dataset_flags=($(for dataset in "${datasets[@]}"; do echo '-d '$dataset; done))

python main.py --dataset mvtec --data_path $datapath "${dataset_flags[@]}" --noise 0.1 --save_segmentation_images
