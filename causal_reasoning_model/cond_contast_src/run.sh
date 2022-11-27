eval "$(conda shell.bash hook)"
python train.py \
--video_features_path ./data/data \
--text_features_path ./data/qas_bert \
--csv_dataset_path ./data/next-dataset \
--batch_size 512 \
--clip_frames 16 \
--lr 1e-4 \
--text_clip \
--n_cands 0 \
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
