#!/usr/bin/env bash

#take in model name argument
model_name=$1

#apply 
python make-docker.py --input_model_name ${model_name}

chmod +x build_and_push.sh

./build_and_push.sh ${model_name}
