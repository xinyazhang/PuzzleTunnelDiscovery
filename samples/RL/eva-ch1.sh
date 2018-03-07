# AVI

./pretrain-d.sh --ferev 5 --elu \
	--eval \
	--ckptdir ackpt/AVI-View-0-of-14-Action12-Rev-5-Feat-2048-524288 \
	--ckptprefix working- \
	--batch 2 --queuemax 64 --threads 1 \
	--iter 524288 \
	--view 0 \
	--avi \
	--featnum 2048 \
	--imhidden 2048 2048 \
	--samplein sample/batch2-view14-norgbd-T6-R6-2M/ \
	--sampletouse 524288 \
	--samplebatching 64 \
	--samplebase 1048576 > evaluation/AVI-View-0-of-14-Rev-5-Feat-2048-524288.out
exit

# 224 x 224

./pretrain-d.sh --ferev 10 --elu \
	--eval \
	--ckptdir ackpt/Res-224-View-0-of-14-Action12-Rev-10-Feat-2048-524288 \
       	--ckptprefix working- \
	--batch 2 --queuemax 64 --threads 1 \
	--iter 524288 \
	--view 0 \
	--res 224 \
	--featnum 2048 \
	--imhidden 2048 2048 \
	--samplein sample/batch2-view14-norgbd-T6-R6-2M/ \
	--sampletouse 524288 \
	--samplebatching 64 \
	--samplebase 1048576 >  evaluation/Res-224-View-0-of-14-Rev-10-Feat-2048-524288.out

