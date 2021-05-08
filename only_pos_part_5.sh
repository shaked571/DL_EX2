#!/bin/bash
which python

for hidden_dim in 200 300
do
  for l_r in 0.01 0.001
  do
   for batch_size in 16 32 128
   do
     echo "Output:"
      echo "part_5_task_pos_hiddendim${hidden_dim}_lr${l_r}_batch_size${batch_size}"

       python main.py \
       --task pos \
       --part 5 \
       --optimizer AdamW \
       --batch_size $batch_size \
       --hidden_dim $hidden_dim \
       --l_r $l_r \
       --window_size 3 \
       --filter_num 30
      done
   done
 done