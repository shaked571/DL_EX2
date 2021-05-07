#!/bin/bash
which python

for part in 4
do
 for task in pos
 do
    for hidden_dim in 200 300
    do
      for optimizer in AdamW
      do
      for l_r in 0.01 0.001
      do
       for batch_size in 16 32 128
       do
       echo "Output:"
        echo "part${part}_task${task}_hiddendim${hidden_dim}_optim_${optimizer}_lr${l_r}_batch_size${batch_size}"

         python main.py \
         --task ${task} \
         --part $part \
         --optimizer $optimizer \
         --batch_size $batch_size \
         --hidden_dim $hidden_dim \
         --l_r $l_r

      done
     done
    done
   done
  done
 done