# Resnet-18 FV size 1024
# 
OUT=evaluation/Resnet-SMView-of-6-A12-Rev-11-MISP-1024/ 
mkdir -p $OUT

./pretrain-d.sh --ferev 11 --elu \
	--ckptdir ackpt/Resnet-View-0-of-14-Action12-Rev-11-Feat-1024-524288-fixcam/ --ckptprefix working-try11 \
	--batch 2 --queuemax 64 --threads 1 \
	--avi \
	--res 224 \
	--iter 524288 \
	--eval \
	--mispout $OUT \
	--sharedmultiview \
	--viewset cube \
	--featnum 1024 \
	--imhidden 1024 1024 \
	--samplein sample/batch2-view14-norgbd-T6-R6-2M/ \
	--sampletouse 524288 \
	--samplebatching 64 \
	--samplebase 1048576 > evaluation/Resnet-SMView-of-6-Rev-11-Feat-1024-524288-fixcam.out

exit

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

