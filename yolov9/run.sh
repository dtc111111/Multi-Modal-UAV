#!/bin/bash

# Check if the correct number of arguments are provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <source> <output>"
    exit 1
fi

# Assign the input arguments to variables
SOURCE=$1
OUTPUT=$2

# Run the Python script with the specified arguments
python3 detect.py --source "$SOURCE" --img 640 --device 0 --weights './yolov9-e.pt' --name "$OUTPUT" --num_kf 5 --kf_int 50 --save-crop
