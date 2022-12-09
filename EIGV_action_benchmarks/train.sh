## Actions with Grounded Features
# python train.py \
# --ans_num 5 \
# --gpu 0 \
# --logs 0 \
# --epoch 35 \
# --num_workers 8 \
# --bs 128 \
# --bert_type action \
# --video_type ground \
# --feat_path ./data \
# --sample_list_path data/CSVs 

## Actions with Next Features
# python train.py \
# --ans_num 5 \
# --num_frames 16 \
# --word_dim 768 \
# --module_dim 512 \
# --app_pool5_dim 2048 \
# --motion_dim 2048 \
# --gpu 1 \
# --logs 1 \
# --epoch 35 \
# --num_workers 8 \
# --bs 128 \
# --bert_type action \
# --video_type next \
# --feat_path ./data \
# --sample_list_path data/CSVs 

## With OG Next Features
# python train.py \
# --ans_num 5 \
# --num_frames 16 \
# --word_dim 768 \
# --module_dim 512 \
# --app_pool5_dim 2048 \
# --motion_dim 2048 \
# --gpu 2 \
# --logs 2 \
# --epoch 35 \
# --num_workers 8 \
# --bs 128 \
# --bert_type next \
# --video_type next \
# --feat_path ./data \
# --sample_list_path data/CSVs 

## With Next + Grounded Features + 32 Frames
# python train.py \
# --ans_num 5 \
# --num_frames 32 \
# --gpu 3 \
# --logs 3 \
# --epoch 35 \
# --num_workers 8 \
# --bs 128 \
# --bert_type next \
# --video_type ground \
# --feat_path ./data \
# --sample_list_path data/CSVs 


## With Next + Grounded Features + 16 Frames
# python train.py \
# --ans_num 5 \
# --num_frames 16 \
# --gpu 4 \
# --logs 4 \
# --epoch 35 \
# --num_workers 8 \
# --bs 128 \
# --bert_type next \
# --video_type ground \
# --feat_path ./data \
# --sample_list_path data/CSVs 