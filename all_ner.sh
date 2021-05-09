for part in 1 3 4 5
do
     echo "Output:"
      echo "part_${part}_task_ner_window_size_5_filter_num_30"

       python main.py \
       --task ner \
       --part ${part} \
       --optimizer AdamW \
       --batch_size 32 \
       --hidden_dim 200 \
       --l_r 0.001 \
       --window_size 5 \
       --filter_num 30
done