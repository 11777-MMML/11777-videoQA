eval "$(conda shell.bash hook)"
conda activate atp
python run_eval.py \
--video_features_path /home/tgkulkar/Courses/Multimodal/data/vid_feat \
--text_features_path /home/tgkulkar/Courses/Multimodal/data/qas_bert \
--csv_dataset_path /home/tgkulkar/Courses/Multimodal/11777-videoQA/next-dataset \
--use_text_cands \
--checkpoint ./checkpoints/best_atp_364.pt \
--num_workers 4