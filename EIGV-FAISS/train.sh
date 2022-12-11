#######################
## Using 3 Neighbors ##
#######################
## With Next + Grounded Features + 32 Frames
# python train.py \
# --ans_num 5 \
# --num_frames 32 \
# --gpu 0 \
# --logs 0 \
# --epoch 35 \
# --num_workers 8 \
# --bs 128 \
# --bert_type next \
# --video_type ground \
# --feat_path ./data \
# --num_workers 16 \
# --sample_list_path data/CSVs 

## With Next + Grounded Features + 16 Frames
# python train.py \
# --ans_num 5 \
# --num_frames 16 \
# --gpu 1 \
# --logs 1 \
# --epoch 35 \
# --num_workers 8 \
# --bs 128 \
# --bert_type next \
# --video_type ground \
# --feat_path ./data \
# --num_workers 16 \
# --sample_list_path data/CSVs 

#######################
## Using 5 Neighbors ##
#######################

# # With Next + Grounded Features + 32 Frames
# python train.py \
# --ans_num 5 \
# --num_frames 32 \
# --gpu 2 \
# --logs 2 \
# --epoch 35 \
# --num_neighbours 5 \
# --bs 128 \
# --bert_type next \
# --video_type ground \
# --feat_path ./data \
# --num_workers 16 \
# --sample_list_path data/CSVs 

## With Next + Grounded Features + 16 Frames
python train.py \
--ans_num 5 \
--num_frames 16 \
--gpu 3 \
--logs 3 \
--epoch 35 \
--num_neighbours 5 \
--bs 128 \
--bert_type next \
--video_type ground \
--feat_path ./data \
--num_workers 16 \
--sample_list_path data/CSVs 
