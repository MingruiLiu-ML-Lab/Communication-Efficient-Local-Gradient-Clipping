#!/bin/bash
python ../main_lstm.py --dataroot ../data/ptbdataset --dataset PennTreebank --model LSTM --epochs 250 --batch-size 3 --dropouti 0.4 --dropouth 0.25 --eta0 30 --clipping-option local --clipping-param 7.5 --momentum 0.0 --world-size 8 --rank 0 --gpu-id 0 --init-method file://<path-to-your-folder>/sharedfile --communication-interval 2 --reproducible --seed 2020 --log-folder ../logs &
python ../main_lstm.py --dataroot ../data/ptbdataset --dataset PennTreebank --model LSTM --epochs 250 --batch-size 3 --dropouti 0.4 --dropouth 0.25 --eta0 30 --clipping-option local --clipping-param 7.5 --momentum 0.0 --world-size 8 --rank 1 --gpu-id 1 --init-method file://<path-to-your-folder>/sharedfile --communication-interval 2 --reproducible --seed 2020 --log-folder ../logs &
python ../main_lstm.py --dataroot ../data/ptbdataset --dataset PennTreebank --model LSTM --epochs 250 --batch-size 3 --dropouti 0.4 --dropouth 0.25 --eta0 30 --clipping-option local --clipping-param 7.5 --momentum 0.0 --world-size 8 --rank 2 --gpu-id 2 --init-method file://<path-to-your-folder>/sharedfile --communication-interval 2 --reproducible --seed 2020 --log-folder ../logs &
python ../main_lstm.py --dataroot ../data/ptbdataset --dataset PennTreebank --model LSTM --epochs 250 --batch-size 3 --dropouti 0.4 --dropouth 0.25 --eta0 30 --clipping-option local --clipping-param 7.5 --momentum 0.0 --world-size 8 --rank 3 --gpu-id 3 --init-method file://<path-to-your-folder>/sharedfile --communication-interval 2 --reproducible --seed 2020 --log-folder ../logs