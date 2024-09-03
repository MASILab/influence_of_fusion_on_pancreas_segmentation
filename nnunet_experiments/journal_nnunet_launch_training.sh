export nnUNet_raw="/home/remedilw/data/nnUNet/nnUNet_raw"
export nnUNet_preprocessed="/home/remedilw/data/nnUNet/nnUNet_preprocessed"
export nnUNet_results="/home/remedilw/data/nnUNet/nnUNet_results"




dataset="$1"
fold="$2"

nnUNetv2_train $dataset 3d_fullres $fold

