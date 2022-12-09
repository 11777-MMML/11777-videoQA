eval "$(conda shell.bash hook)"
python train.py \
--video_features_path /home/ubuntu/11777-videoQA/causal_reasoning_model/data \
--text_features_path ./data/qas_bert \
--csv_dataset_path /home/ubuntu/11777-videoQA/next-dataset \
--batch_size 256 \
--clip_frames 80 \
--lr 5e-5 \
--n_layers 2 \
--n_heads 2 \
--text_clip \
--n_cands 0 \
--exp_name exp_qva_frozen_80clips_shuffle_1 \
--seed 0 \
--use_text_cands \
--num_workers 4

# eval "$(conda shell.bash hook)"
# python train.py \
# --video_features_path /home/ubuntu/11777-videoQA/atp-video-language/data \
# --text_features_path ./ \
# --batch_size 128 \
# --csv_dataset_path /home/ubuntu/11777-videoQA//next-dataset \
# --clip_frames 16 \
# --text_clip \
# --use_text_cands \
# --use_text_query \
# --num_workers 4
