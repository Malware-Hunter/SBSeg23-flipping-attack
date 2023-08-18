#!/bin/bash

for flip in 0.00 0.05 0.10 0.15 0.20 0.25 0.30
do
	output="../outputs/output_$flip"
	for classifier in svm rf dt
	do
		for file in ../datasets/originals/*.csv 
		do
			python3 run_ml.py -d "$file" -c "$classifier" -f "$flip" -n "$output"
		done
	done
done