
./pretrain-d.sh --ferev 4 --elu \
	--ckptdir ckpt/Action4-Rev-4-262144/ --ckptprefix working-try11 \
	--batch 2 --queuemax 64 --threads 1 \
	--iter 262144 \
	--eval \
	--samplein sample/batch2-view1-T2-R2-2M/ \
	--sampletouse 262144 \
	--samplebatching 8 \
	--samplebase 0 > evaluation/Action-4-Error-Base-0-Rev-4.out

./pretrain-d.sh --ferev 4 --elu \
	--ckptdir ckpt/Action4-Rev-4-262144/ --ckptprefix working-try11 \
	--batch 2 --queuemax 64 --threads 1 \
	--iter 262144 \
	--eval \
	--samplein sample/batch2-view1-T2-R2-2M/ \
	--sampletouse 262144 \
	--samplebatching 8 \
	--samplebase 524288 > evaluation/Action-4-Error-Base-524288-Rev-4.out

./pretrain-d.sh --ferev 5 --elu \
	--ckptdir ckpt/Action4-Rev-5-262144/ --ckptprefix working-try11 \
	--batch 2 --queuemax 64 --threads 1 \
	--iter 262144 \
	--eval \
	--samplein sample/batch2-view1-T2-R2-2M/ \
	--sampletouse 262144 \
	--samplebatching 8 \
	--samplebase 0 > evaluation/Action-4-Error-Base-0-Rev-5.out

./pretrain-d.sh --ferev 5 --elu \
	--ckptdir ckpt/Action4-Rev-5-262144/ --ckptprefix working-try11 \
	--batch 2 --queuemax 64 --threads 1 \
	--iter 262144 \
	--eval \
	--samplein sample/batch2-view1-T2-R2-2M/ \
	--sampletouse 262144 \
	--samplebatching 8 \
	--samplebase 524288 > evaluation/Action-4-Error-Base-524288-Rev-5.out

./pretrain-d.sh --ferev 6 --elu \
	--ckptdir ckpt/Action4-Rev-6-262144/ --ckptprefix working-try11 \
	--batch 2 --queuemax 64 --threads 1 \
	--iter 262144 \
	--eval \
	--samplein sample/batch2-view1-T2-R2-2M/ \
	--sampletouse 262144 \
	--samplebatching 8 \
	--samplebase 0 > evaluation/Action-4-Error-Base-0-Rev-6.out

./pretrain-d.sh --ferev 6 --elu \
	--ckptdir ckpt/Action4-Rev-6-262144/ --ckptprefix working-try11 \
	--batch 2 --queuemax 64 --threads 1 \
	--iter 262144 \
	--eval \
	--samplein sample/batch2-view1-T2-R2-2M/ \
	--sampletouse 262144 \
	--samplebatching 8 \
	--samplebase 524288 > evaluation/Action-4-Error-Base-524288-Rev-6.out

./pretrain-d.sh --ferev 7 --elu \
	--ckptdir ckpt/Action4-Rev-7-262144/ --ckptprefix working-try11 \
	--batch 2 --queuemax 64 --threads 1 \
	--iter 262144 \
	--eval \
	--samplein sample/batch2-view1-T2-R2-2M/ \
	--sampletouse 262144 \
	--samplebatching 8 \
	--samplebase 0 > evaluation/Action-4-Error-Base-0-Rev-7.out

./pretrain-d.sh --ferev 7 --elu \
	--ckptdir ckpt/Action4-Rev-7-262144/ --ckptprefix working-try11 \
	--batch 2 --queuemax 64 --threads 1 \
	--iter 262144 \
	--eval \
	--samplein sample/batch2-view1-T2-R2-2M/ \
	--sampletouse 262144 \
	--samplebatching 8 \
	--samplebase 524288 > evaluation/Action-4-Error-Base-524288-Rev-7.out

./pretrain-d.sh --ferev 8 --elu \
	--ckptdir ckpt/Action4-Rev-8-262144/ --ckptprefix working-try11 \
	--batch 2 --queuemax 64 --threads 1 \
	--iter 262144 \
	--eval \
	--samplein sample/batch2-view1-T2-R2-2M/ \
	--sampletouse 262144 \
	--samplebatching 8 \
	--samplebase 0 > evaluation/Action-4-Error-Base-0-Rev-8.out

./pretrain-d.sh --ferev 8 --elu \
	--ckptdir ckpt/Action4-Rev-8-262144/ --ckptprefix working-try11 \
	--batch 2 --queuemax 64 --threads 1 \
	--iter 262144 \
	--eval \
	--samplein sample/batch2-view1-T2-R2-2M/ \
	--sampletouse 262144 \
	--samplebatching 8 \
	--samplebase 524288 > evaluation/Action-4-Error-Base-524288-Rev-8.out
