# eval "$(conda shell.bash hook)"
# conda activate atp
# python train.py \
# --video_features_path /home/tgkulkar/Courses/Multimodal/data/vid_feat \
# --text_features_path /home/tgkulkar/Courses/Multimodal/data/qas_bert \
# --csv_dataset_path /home/tgkulkar/Courses/Multimodal/11777-videoQA/next-dataset \
# --use_text_cands \
# --num_workers 4

eval "$(conda shell.bash hook)"
conda activate atp
python train.py \
--video_features_path /home/tgkulkar/Courses/Multimodal/data/ \
--text_features_path /home/tgkulkar/Courses/Multimodal/data/ \
--csv_dataset_path /home/tgkulkar/Courses/Multimodal/11777-videoQA/next-dataset \
--clip_frames 16 \
--use_text_cands \
--use_text_query \
--num_workers 4