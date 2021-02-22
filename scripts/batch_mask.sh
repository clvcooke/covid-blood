#!/bin/bash
folds=(0 1 2 3 4 5)
seeds=(0 1 2)
for fold in "${folds[@]}"
do
	python fine_tuning_classifier.py --epochs 100 --gpu_number 1 --cell_mask nuc --fold_number $fold --random_seed 12&
	python fine_tuning_classifier.py --epochs 100 --gpu_number 1 --cell_mask non_nuc --fold_number $fold --random_seed 12 &
	python fine_tuning_classifier.py --epochs 100 --gpu_number 2 --fold_number $fold --random_seed 12 &
	wait
done
