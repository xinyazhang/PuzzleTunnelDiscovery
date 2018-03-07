./pretrain-d.sh --ferev 9 --elu \
	--ckptdir ackpt/View-0-of-14-Action4-Rev-9-524288/ --ckptprefix working-try11 \
	--batch 2 --queuemax 64 --threads 1 \
	--iter 262144 \
	--eval \
	--view 0 \
	--samplein sample/batch2-view14-norgbd-T2-R2-2M/ \
	--sampletouse 262144 \
	--samplebase 1048576 > evaluation/Action-4-Error-Base-0-Rev-9.out

