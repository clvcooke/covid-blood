#!/bin/bash
python fine_tuning_classifier.py --epochs 150 --fold_number 0 --gpu_number 0 &
python fine_tuning_classifier.py --epochs 150 --fold_number 1 --gpu_number 0 &
python fine_tuning_classifier.py --epochs 150 --fold_number 2 --gpu_number 2 &
python fine_tuning_classifier.py --epochs 150 --fold_number 3 --gpu_number 2 &
wait
python fine_tuning_classifier.py --epochs 150 --fold_number 4 --gpu_number 0 &
python fine_tuning_classifier.py --epochs 150 --fold_number 5 --gpu_number 2
