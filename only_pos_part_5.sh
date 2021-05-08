#!/bin/bash
which python

for window_size in 3 4 5
do
   for filter_num in 20 30 40
   do
     echo "Output:"
      echo "part_5_task_pos_window_size_${window_size}_filter_num_${filter_num}"

       python main.py \
       --task pos \
       --part 5 \
       --optimizer AdamW \
       --batch_size 32 \
       --hidden_dim 200 \
       --l_r 0.001 \
       --window_size $window_size \
       --filter_num $filter_num
      done
 done