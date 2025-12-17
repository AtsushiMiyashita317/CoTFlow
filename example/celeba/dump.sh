#!/bin/bash

dump_features \
    --autoencoder ./pretrained/autoencoder \
    --predictors \
        attr=./pretrained/predictor_attr \
        bbox=./pretrained/predictor_bbox \
        landmark=./pretrained/predictor_landmark \
    --dump_h5 ./dump/features.h5 \
    --batch_size 32
