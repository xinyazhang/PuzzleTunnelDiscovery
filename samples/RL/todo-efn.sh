
	./pretrain-d.sh --ferev 7 --elu \
		--ckptdir ackpt/View-0-of-14-Action4-Rev-7-Feat-512-524288/ --ckptprefix working-try11 \
		--batch 2 --queuemax 64 --threads 1 \
		--iter 524288 \
		--eval \
		--view 0 \
		--featnum 512 \
		--imhidden 512 512 \
		--samplein sample/batch2-view14-norgbd-T2-R2-2M/ \
		--sampletouse 524288 \
		--samplebatching 8 \
		--samplebase 0 > evaluation/Action-4-Error-Base-0-Rev-7-Feat-512.out

	./pretrain-d.sh --ferev 7 --elu \
		--ckptdir ackpt/View-0-of-14-Action4-Rev-7-Feat-512-524288/ --ckptprefix working-try11 \
		--batch 2 --queuemax 64 --threads 1 \
		--iter 524288 \
		--eval \
		--view 0 \
		--featnum 512 \
		--imhidden 512 512 \
		--samplein sample/batch2-view14-norgbd-T2-R2-2M/ \
		--sampletouse 524288 \
		--samplebatching 8 \
		--samplebase 1048576 > evaluation/Action-4-Error-Base-1048576-Rev-7-Feat-512.out

	./pretrain-d.sh --ferev 7 --elu \
		--ckptdir ackpt/View-0-of-14-Action4-Rev-7-Feat-1024-524288/ --ckptprefix working-try11 \
		--batch 2 --queuemax 64 --threads 1 \
		--iter 524288 \
		--eval \
		--view 0 \
		--featnum 1024 \
		--imhidden 1024 1024 \
		--samplein sample/batch2-view14-norgbd-T2-R2-2M/ \
		--sampletouse 524288 \
		--samplebatching 8 \
		--samplebase 0 > evaluation/Action-4-Error-Base-0-Rev-7-Feat-1024.out

	./pretrain-d.sh --ferev 7 --elu \
		--ckptdir ackpt/View-0-of-14-Action4-Rev-7-Feat-1024-524288/ --ckptprefix working-try11 \
		--batch 2 --queuemax 64 --threads 1 \
		--iter 524288 \
		--eval \
		--view 0 \
		--featnum 1024 \
		--imhidden 1024 1024 \
		--samplein sample/batch2-view14-norgbd-T2-R2-2M/ \
		--sampletouse 524288 \
		--samplebatching 8 \
		--samplebase 1048576 > evaluation/Action-4-Error-Base-1048576-Rev-7-Feat-1024.out

	./pretrain-d.sh --ferev 7 --elu \
		--ckptdir ackpt/View-0-of-14-Action4-Rev-7-Feat-2048-524288/ --ckptprefix working-try11 \
		--batch 2 --queuemax 64 --threads 1 \
		--iter 524288 \
		--eval \
		--view 0 \
		--featnum 2048 \
		--imhidden 2048 2048 \
		--samplein sample/batch2-view14-norgbd-T2-R2-2M/ \
		--sampletouse 524288 \
		--samplebatching 8 \
		--samplebase 0 > evaluation/Action-4-Error-Base-0-Rev-7-Feat-2048.out

	./pretrain-d.sh --ferev 7 --elu \
		--ckptdir ackpt/View-0-of-14-Action4-Rev-7-Feat-2048-524288/ --ckptprefix working-try11 \
		--batch 2 --queuemax 64 --threads 1 \
		--iter 524288 \
		--eval \
		--view 0 \
		--featnum 2048 \
		--imhidden 2048 2048 \
		--samplein sample/batch2-view14-norgbd-T2-R2-2M/ \
		--sampletouse 524288 \
		--samplebatching 8 \
		--samplebase 1048576 > evaluation/Action-4-Error-Base-1048576-Rev-7-Feat-2048.out

	./pretrain-d.sh --ferev 8 --elu \
		--ckptdir ackpt/View-0-of-14-Action4-Rev-8-Feat-512-524288/ --ckptprefix working-try11 \
		--batch 2 --queuemax 64 --threads 1 \
		--iter 524288 \
		--eval \
		--view 0 \
		--featnum 512 \
		--imhidden 512 512 \
		--samplein sample/batch2-view14-norgbd-T2-R2-2M/ \
		--sampletouse 524288 \
		--samplebatching 8 \
		--samplebase 0 > evaluation/Action-4-Error-Base-0-Rev-8-Feat-512.out

	./pretrain-d.sh --ferev 8 --elu \
		--ckptdir ackpt/View-0-of-14-Action4-Rev-8-Feat-512-524288/ --ckptprefix working-try11 \
		--batch 2 --queuemax 64 --threads 1 \
		--iter 524288 \
		--eval \
		--view 0 \
		--featnum 512 \
		--imhidden 512 512 \
		--samplein sample/batch2-view14-norgbd-T2-R2-2M/ \
		--sampletouse 524288 \
		--samplebatching 8 \
		--samplebase 1048576 > evaluation/Action-4-Error-Base-1048576-Rev-8-Feat-512.out

	./pretrain-d.sh --ferev 8 --elu \
		--ckptdir ackpt/View-0-of-14-Action4-Rev-8-Feat-1024-524288/ --ckptprefix working-try11 \
		--batch 2 --queuemax 64 --threads 1 \
		--iter 524288 \
		--eval \
		--view 0 \
		--featnum 1024 \
		--imhidden 1024 1024 \
		--samplein sample/batch2-view14-norgbd-T2-R2-2M/ \
		--sampletouse 524288 \
		--samplebatching 8 \
		--samplebase 0 > evaluation/Action-4-Error-Base-0-Rev-8-Feat-1024.out

	./pretrain-d.sh --ferev 8 --elu \
		--ckptdir ackpt/View-0-of-14-Action4-Rev-8-Feat-1024-524288/ --ckptprefix working-try11 \
		--batch 2 --queuemax 64 --threads 1 \
		--iter 524288 \
		--eval \
		--view 0 \
		--featnum 1024 \
		--imhidden 1024 1024 \
		--samplein sample/batch2-view14-norgbd-T2-R2-2M/ \
		--sampletouse 524288 \
		--samplebatching 8 \
		--samplebase 1048576 > evaluation/Action-4-Error-Base-1048576-Rev-8-Feat-1024.out

	./pretrain-d.sh --ferev 8 --elu \
		--ckptdir ackpt/View-0-of-14-Action4-Rev-8-Feat-2048-524288/ --ckptprefix working-try11 \
		--batch 2 --queuemax 64 --threads 1 \
		--iter 524288 \
		--eval \
		--view 0 \
		--featnum 2048 \
		--imhidden 2048 2048 \
		--samplein sample/batch2-view14-norgbd-T2-R2-2M/ \
		--sampletouse 524288 \
		--samplebatching 8 \
		--samplebase 0 > evaluation/Action-4-Error-Base-0-Rev-8-Feat-2048.out

	./pretrain-d.sh --ferev 8 --elu \
		--ckptdir ackpt/View-0-of-14-Action4-Rev-8-Feat-2048-524288/ --ckptprefix working-try11 \
		--batch 2 --queuemax 64 --threads 1 \
		--iter 524288 \
		--eval \
		--view 0 \
		--featnum 2048 \
		--imhidden 2048 2048 \
		--samplein sample/batch2-view14-norgbd-T2-R2-2M/ \
		--sampletouse 524288 \
		--samplebatching 8 \
		--samplebase 1048576 > evaluation/Action-4-Error-Base-1048576-Rev-8-Feat-2048.out
