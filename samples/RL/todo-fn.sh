mkdir -p nsample-idmv-action4

mkdir -p ckpt/pretrain-d-elu-view-0-of-14-rev-5-action4-Feat-512-working

	./pretrain-d.sh --ferev 5 --elu \
		--ckptdir ckpt/pretrain-d-elu-view-0-of-14-rev-5-action4-Feat-512-working/ --ckptprefix working-try11 \
		--batch 2 --queuemax 64 --threads 1 \
		--iter 4194304 \
		--view 0 \
		--featnum 512 \
		--imhidden 512 512 \
		--samplein sample/batch2-view14-norgbd-T2-R2-2M/ \
		--sampletouse 524288 \
		--samplebatching 32 \
		--samplebase 0 > nsample-idmv-action4/View-0-of-14-Rev-5-Feat-512-524288.out
			
cp -a ckpt/pretrain-d-elu-view-0-of-14-rev-5-action4-Feat-512-working ackpt/View-0-of-14-Action4-Rev-5-Feat-512-524288


mkdir -p ckpt/pretrain-d-elu-view-0-of-14-rev-5-action4-Feat-1024-working

	./pretrain-d.sh --ferev 5 --elu \
		--ckptdir ckpt/pretrain-d-elu-view-0-of-14-rev-5-action4-Feat-1024-working/ --ckptprefix working-try11 \
		--batch 2 --queuemax 64 --threads 1 \
		--iter 4194304 \
		--view 0 \
		--featnum 1024 \
		--imhidden 1024 1024 \
		--samplein sample/batch2-view14-norgbd-T2-R2-2M/ \
		--sampletouse 524288 \
		--samplebatching 32 \
		--samplebase 0 > nsample-idmv-action4/View-0-of-14-Rev-5-Feat-1024-524288.out
			
cp -a ckpt/pretrain-d-elu-view-0-of-14-rev-5-action4-Feat-1024-working ackpt/View-0-of-14-Action4-Rev-5-Feat-1024-524288


mkdir -p ckpt/pretrain-d-elu-view-0-of-14-rev-5-action4-Feat-2048-working

	./pretrain-d.sh --ferev 5 --elu \
		--ckptdir ckpt/pretrain-d-elu-view-0-of-14-rev-5-action4-Feat-2048-working/ --ckptprefix working-try11 \
		--batch 2 --queuemax 64 --threads 1 \
		--iter 4194304 \
		--view 0 \
		--featnum 2048 \
		--imhidden 2048 2048 \
		--samplein sample/batch2-view14-norgbd-T2-R2-2M/ \
		--sampletouse 524288 \
		--samplebatching 32 \
		--samplebase 0 > nsample-idmv-action4/View-0-of-14-Rev-5-Feat-2048-524288.out
			
cp -a ckpt/pretrain-d-elu-view-0-of-14-rev-5-action4-Feat-2048-working ackpt/View-0-of-14-Action4-Rev-5-Feat-2048-524288


mkdir -p ckpt/pretrain-d-elu-view-0-of-14-rev-6-action4-Feat-512-working

	./pretrain-d.sh --ferev 6 --elu \
		--ckptdir ckpt/pretrain-d-elu-view-0-of-14-rev-6-action4-Feat-512-working/ --ckptprefix working-try11 \
		--batch 2 --queuemax 64 --threads 1 \
		--iter 4194304 \
		--view 0 \
		--featnum 512 \
		--imhidden 512 512 \
		--samplein sample/batch2-view14-norgbd-T2-R2-2M/ \
		--sampletouse 524288 \
		--samplebatching 32 \
		--samplebase 0 > nsample-idmv-action4/View-0-of-14-Rev-6-Feat-512-524288.out
			
cp -a ckpt/pretrain-d-elu-view-0-of-14-rev-6-action4-Feat-512-working ackpt/View-0-of-14-Action4-Rev-6-Feat-512-524288


mkdir -p ckpt/pretrain-d-elu-view-0-of-14-rev-6-action4-Feat-1024-working

	./pretrain-d.sh --ferev 6 --elu \
		--ckptdir ckpt/pretrain-d-elu-view-0-of-14-rev-6-action4-Feat-1024-working/ --ckptprefix working-try11 \
		--batch 2 --queuemax 64 --threads 1 \
		--iter 4194304 \
		--view 0 \
		--featnum 1024 \
		--imhidden 1024 1024 \
		--samplein sample/batch2-view14-norgbd-T2-R2-2M/ \
		--sampletouse 524288 \
		--samplebatching 32 \
		--samplebase 0 > nsample-idmv-action4/View-0-of-14-Rev-6-Feat-1024-524288.out
			
cp -a ckpt/pretrain-d-elu-view-0-of-14-rev-6-action4-Feat-1024-working ackpt/View-0-of-14-Action4-Rev-6-Feat-1024-524288


mkdir -p ckpt/pretrain-d-elu-view-0-of-14-rev-6-action4-Feat-2048-working

	./pretrain-d.sh --ferev 6 --elu \
		--ckptdir ckpt/pretrain-d-elu-view-0-of-14-rev-6-action4-Feat-2048-working/ --ckptprefix working-try11 \
		--batch 2 --queuemax 64 --threads 1 \
		--iter 4194304 \
		--view 0 \
		--featnum 2048 \
		--imhidden 2048 2048 \
		--samplein sample/batch2-view14-norgbd-T2-R2-2M/ \
		--sampletouse 524288 \
		--samplebatching 32 \
		--samplebase 0 > nsample-idmv-action4/View-0-of-14-Rev-6-Feat-2048-524288.out
			
cp -a ckpt/pretrain-d-elu-view-0-of-14-rev-6-action4-Feat-2048-working ackpt/View-0-of-14-Action4-Rev-6-Feat-2048-524288


mkdir -p ckpt/pretrain-d-elu-view-0-of-14-rev-7-action4-Feat-512-working

	./pretrain-d.sh --ferev 7 --elu \
		--ckptdir ckpt/pretrain-d-elu-view-0-of-14-rev-7-action4-Feat-512-working/ --ckptprefix working-try11 \
		--batch 2 --queuemax 64 --threads 1 \
		--iter 4194304 \
		--view 0 \
		--featnum 512 \
		--imhidden 512 512 \
		--samplein sample/batch2-view14-norgbd-T2-R2-2M/ \
		--sampletouse 524288 \
		--samplebatching 32 \
		--samplebase 0 > nsample-idmv-action4/View-0-of-14-Rev-7-Feat-512-524288.out
			
cp -a ckpt/pretrain-d-elu-view-0-of-14-rev-7-action4-Feat-512-working ackpt/View-0-of-14-Action4-Rev-7-Feat-512-524288


mkdir -p ckpt/pretrain-d-elu-view-0-of-14-rev-7-action4-Feat-1024-working

	./pretrain-d.sh --ferev 7 --elu \
		--ckptdir ckpt/pretrain-d-elu-view-0-of-14-rev-7-action4-Feat-1024-working/ --ckptprefix working-try11 \
		--batch 2 --queuemax 64 --threads 1 \
		--iter 4194304 \
		--view 0 \
		--featnum 1024 \
		--imhidden 1024 1024 \
		--samplein sample/batch2-view14-norgbd-T2-R2-2M/ \
		--sampletouse 524288 \
		--samplebatching 32 \
		--samplebase 0 > nsample-idmv-action4/View-0-of-14-Rev-7-Feat-1024-524288.out
			
cp -a ckpt/pretrain-d-elu-view-0-of-14-rev-7-action4-Feat-1024-working ackpt/View-0-of-14-Action4-Rev-7-Feat-1024-524288


mkdir -p ckpt/pretrain-d-elu-view-0-of-14-rev-7-action4-Feat-2048-working

	./pretrain-d.sh --ferev 7 --elu \
		--ckptdir ckpt/pretrain-d-elu-view-0-of-14-rev-7-action4-Feat-2048-working/ --ckptprefix working-try11 \
		--batch 2 --queuemax 64 --threads 1 \
		--iter 4194304 \
		--view 0 \
		--featnum 2048 \
		--imhidden 2048 2048 \
		--samplein sample/batch2-view14-norgbd-T2-R2-2M/ \
		--sampletouse 524288 \
		--samplebatching 32 \
		--samplebase 0 > nsample-idmv-action4/View-0-of-14-Rev-7-Feat-2048-524288.out
			
cp -a ckpt/pretrain-d-elu-view-0-of-14-rev-7-action4-Feat-2048-working ackpt/View-0-of-14-Action4-Rev-7-Feat-2048-524288


mkdir -p ckpt/pretrain-d-elu-view-0-of-14-rev-8-action4-Feat-512-working

	./pretrain-d.sh --ferev 8 --elu \
		--ckptdir ckpt/pretrain-d-elu-view-0-of-14-rev-8-action4-Feat-512-working/ --ckptprefix working-try11 \
		--batch 2 --queuemax 64 --threads 1 \
		--iter 4194304 \
		--view 0 \
		--featnum 512 \
		--imhidden 512 512 \
		--samplein sample/batch2-view14-norgbd-T2-R2-2M/ \
		--sampletouse 524288 \
		--samplebatching 32 \
		--samplebase 0 > nsample-idmv-action4/View-0-of-14-Rev-8-Feat-512-524288.out
			
cp -a ckpt/pretrain-d-elu-view-0-of-14-rev-8-action4-Feat-512-working ackpt/View-0-of-14-Action4-Rev-8-Feat-512-524288


mkdir -p ckpt/pretrain-d-elu-view-0-of-14-rev-8-action4-Feat-1024-working

	./pretrain-d.sh --ferev 8 --elu \
		--ckptdir ckpt/pretrain-d-elu-view-0-of-14-rev-8-action4-Feat-1024-working/ --ckptprefix working-try11 \
		--batch 2 --queuemax 64 --threads 1 \
		--iter 4194304 \
		--view 0 \
		--featnum 1024 \
		--imhidden 1024 1024 \
		--samplein sample/batch2-view14-norgbd-T2-R2-2M/ \
		--sampletouse 524288 \
		--samplebatching 32 \
		--samplebase 0 > nsample-idmv-action4/View-0-of-14-Rev-8-Feat-1024-524288.out
			
cp -a ckpt/pretrain-d-elu-view-0-of-14-rev-8-action4-Feat-1024-working ackpt/View-0-of-14-Action4-Rev-8-Feat-1024-524288


mkdir -p ckpt/pretrain-d-elu-view-0-of-14-rev-8-action4-Feat-2048-working

	./pretrain-d.sh --ferev 8 --elu \
		--ckptdir ckpt/pretrain-d-elu-view-0-of-14-rev-8-action4-Feat-2048-working/ --ckptprefix working-try11 \
		--batch 2 --queuemax 64 --threads 1 \
		--iter 4194304 \
		--view 0 \
		--featnum 2048 \
		--imhidden 2048 2048 \
		--samplein sample/batch2-view14-norgbd-T2-R2-2M/ \
		--sampletouse 524288 \
		--samplebatching 32 \
		--samplebase 0 > nsample-idmv-action4/View-0-of-14-Rev-8-Feat-2048-524288.out
			
cp -a ckpt/pretrain-d-elu-view-0-of-14-rev-8-action4-Feat-2048-working ackpt/View-0-of-14-Action4-Rev-8-Feat-2048-524288

