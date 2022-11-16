GPU=$1
CUDA_VISIBLE_DEVICES=$GPU python main.py --checkpoint_dir=nextqa \
	--dataset=nextqa \
	--mc=5 \
	--bnum=5 \
	--test=1 \
	--qmax_words=0 \
	--amax_words=38 \
	--max_feats=32 \
	--batch_size=64 \
	--batch_size_val=64 \
	--num_thread_reader=0 \
	--mlm_prob=0 \
	--n_layers=1 \
	--embd_dim=512 \
	--ff_dim=1024 \
	--dropout=0.3 \
	--save_dir='../data/save_models/nextqa/VGT_B5/' \
	--pretrain_path=../data/save_models/nextqa/VGT_B5/best_model.pth \
	#--CM_PT=1
