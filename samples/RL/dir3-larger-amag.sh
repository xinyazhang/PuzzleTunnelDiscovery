mkdir -p nsample-idmv-action12

mkdir -p ckpt/pretrain-d-elu-view-0-of-14-rev-5-action12-Feat-2048-8thAMAG-working

	./pretrain-d.sh --ferev 5 --elu \
		--ckptdir ckpt/pretrain-d-elu-view-0-of-14-rev-5-action12-Feat-2048-8thAMAG-working/ --ckptprefix working-try11 \
		--batch 2 --queuemax 64 --threads 1 \
		--iter 4194304 \
		--view 0 \
		--featnum 2048 \
		--imhidden 2048 2048 \
		--samplein sample/batch2-view14-norgbd-T6-R6-2M-8thAMAG/ \
		--sampletouse 524288 \
		--samplebatching 64 \
		--samplebase 0 > nsample-idmv-action12/View-0-of-14-Rev-5-Feat-2048-524288-8thAMAG.out
			
cp -a ckpt/pretrain-d-elu-view-0-of-14-rev-5-action12-Feat-2048-8thAMAG-working ackpt/View-0-of-14-Action12-Rev-5-Feat-2048-524288-8thAMAG

exit

mkdir -p ckpt/pretrain-d-elu-view-1-of-14-rev-5-action12-Feat-2048-8thAMAG-working

	./pretrain-d.sh --ferev 5 --elu \
		--ckptdir ckpt/pretrain-d-elu-view-1-of-14-rev-5-action12-Feat-2048-8thAMAG-working/ --ckptprefix working-try11 \
		--batch 2 --queuemax 64 --threads 1 \
		--iter 4194304 \
		--view 1 \
		--featnum 2048 \
		--imhidden 2048 2048 \
		--samplein sample/batch2-view14-norgbd-T6-R6-2M-8thAMAG/ \
		--sampletouse 524288 \
		--samplebatching 64 \
		--samplebase 0 > nsample-idmv-action12/View-1-of-14-Rev-5-Feat-2048-524288-8thAMAG.out
			
cp -a ckpt/pretrain-d-elu-view-1-of-14-rev-5-action12-Feat-2048-8thAMAG-working ackpt/View-1-of-14-Action12-Rev-5-Feat-2048-524288-8thAMAG


mkdir -p ckpt/pretrain-d-elu-view-2-of-14-rev-5-action12-Feat-2048-8thAMAG-working

	./pretrain-d.sh --ferev 5 --elu \
		--ckptdir ckpt/pretrain-d-elu-view-2-of-14-rev-5-action12-Feat-2048-8thAMAG-working/ --ckptprefix working-try11 \
		--batch 2 --queuemax 64 --threads 1 \
		--iter 4194304 \
		--view 2 \
		--featnum 2048 \
		--imhidden 2048 2048 \
		--samplein sample/batch2-view14-norgbd-T6-R6-2M-8thAMAG/ \
		--sampletouse 524288 \
		--samplebatching 64 \
		--samplebase 0 > nsample-idmv-action12/View-2-of-14-Rev-5-Feat-2048-524288-8thAMAG.out
			
cp -a ckpt/pretrain-d-elu-view-2-of-14-rev-5-action12-Feat-2048-8thAMAG-working ackpt/View-2-of-14-Action12-Rev-5-Feat-2048-524288-8thAMAG


mkdir -p ckpt/pretrain-d-elu-view-3-of-14-rev-5-action12-Feat-2048-8thAMAG-working

	./pretrain-d.sh --ferev 5 --elu \
		--ckptdir ckpt/pretrain-d-elu-view-3-of-14-rev-5-action12-Feat-2048-8thAMAG-working/ --ckptprefix working-try11 \
		--batch 2 --queuemax 64 --threads 1 \
		--iter 4194304 \
		--view 3 \
		--featnum 2048 \
		--imhidden 2048 2048 \
		--samplein sample/batch2-view14-norgbd-T6-R6-2M-8thAMAG/ \
		--sampletouse 524288 \
		--samplebatching 64 \
		--samplebase 0 > nsample-idmv-action12/View-3-of-14-Rev-5-Feat-2048-524288-8thAMAG.out
			
cp -a ckpt/pretrain-d-elu-view-3-of-14-rev-5-action12-Feat-2048-8thAMAG-working ackpt/View-3-of-14-Action12-Rev-5-Feat-2048-524288-8thAMAG


mkdir -p ckpt/pretrain-d-elu-view-4-of-14-rev-5-action12-Feat-2048-8thAMAG-working

	./pretrain-d.sh --ferev 5 --elu \
		--ckptdir ckpt/pretrain-d-elu-view-4-of-14-rev-5-action12-Feat-2048-8thAMAG-working/ --ckptprefix working-try11 \
		--batch 2 --queuemax 64 --threads 1 \
		--iter 4194304 \
		--view 4 \
		--featnum 2048 \
		--imhidden 2048 2048 \
		--samplein sample/batch2-view14-norgbd-T6-R6-2M-8thAMAG/ \
		--sampletouse 524288 \
		--samplebatching 64 \
		--samplebase 0 > nsample-idmv-action12/View-4-of-14-Rev-5-Feat-2048-524288-8thAMAG.out
			
cp -a ckpt/pretrain-d-elu-view-4-of-14-rev-5-action12-Feat-2048-8thAMAG-working ackpt/View-4-of-14-Action12-Rev-5-Feat-2048-524288-8thAMAG


mkdir -p ckpt/pretrain-d-elu-view-5-of-14-rev-5-action12-Feat-2048-8thAMAG-working

	./pretrain-d.sh --ferev 5 --elu \
		--ckptdir ckpt/pretrain-d-elu-view-5-of-14-rev-5-action12-Feat-2048-8thAMAG-working/ --ckptprefix working-try11 \
		--batch 2 --queuemax 64 --threads 1 \
		--iter 4194304 \
		--view 5 \
		--featnum 2048 \
		--imhidden 2048 2048 \
		--samplein sample/batch2-view14-norgbd-T6-R6-2M-8thAMAG/ \
		--sampletouse 524288 \
		--samplebatching 64 \
		--samplebase 0 > nsample-idmv-action12/View-5-of-14-Rev-5-Feat-2048-524288-8thAMAG.out
			
cp -a ckpt/pretrain-d-elu-view-5-of-14-rev-5-action12-Feat-2048-8thAMAG-working ackpt/View-5-of-14-Action12-Rev-5-Feat-2048-524288-8thAMAG


mkdir -p ckpt/pretrain-d-elu-view-6-of-14-rev-5-action12-Feat-2048-8thAMAG-working

	./pretrain-d.sh --ferev 5 --elu \
		--ckptdir ckpt/pretrain-d-elu-view-6-of-14-rev-5-action12-Feat-2048-8thAMAG-working/ --ckptprefix working-try11 \
		--batch 2 --queuemax 64 --threads 1 \
		--iter 4194304 \
		--view 6 \
		--featnum 2048 \
		--imhidden 2048 2048 \
		--samplein sample/batch2-view14-norgbd-T6-R6-2M-8thAMAG/ \
		--sampletouse 524288 \
		--samplebatching 64 \
		--samplebase 0 > nsample-idmv-action12/View-6-of-14-Rev-5-Feat-2048-524288-8thAMAG.out
			
cp -a ckpt/pretrain-d-elu-view-6-of-14-rev-5-action12-Feat-2048-8thAMAG-working ackpt/View-6-of-14-Action12-Rev-5-Feat-2048-524288-8thAMAG


mkdir -p ckpt/pretrain-d-elu-view-7-of-14-rev-5-action12-Feat-2048-8thAMAG-working

	./pretrain-d.sh --ferev 5 --elu \
		--ckptdir ckpt/pretrain-d-elu-view-7-of-14-rev-5-action12-Feat-2048-8thAMAG-working/ --ckptprefix working-try11 \
		--batch 2 --queuemax 64 --threads 1 \
		--iter 4194304 \
		--view 7 \
		--featnum 2048 \
		--imhidden 2048 2048 \
		--samplein sample/batch2-view14-norgbd-T6-R6-2M-8thAMAG/ \
		--sampletouse 524288 \
		--samplebatching 64 \
		--samplebase 0 > nsample-idmv-action12/View-7-of-14-Rev-5-Feat-2048-524288-8thAMAG.out
			
cp -a ckpt/pretrain-d-elu-view-7-of-14-rev-5-action12-Feat-2048-8thAMAG-working ackpt/View-7-of-14-Action12-Rev-5-Feat-2048-524288-8thAMAG


mkdir -p ckpt/pretrain-d-elu-view-8-of-14-rev-5-action12-Feat-2048-8thAMAG-working

	./pretrain-d.sh --ferev 5 --elu \
		--ckptdir ckpt/pretrain-d-elu-view-8-of-14-rev-5-action12-Feat-2048-8thAMAG-working/ --ckptprefix working-try11 \
		--batch 2 --queuemax 64 --threads 1 \
		--iter 4194304 \
		--view 8 \
		--featnum 2048 \
		--imhidden 2048 2048 \
		--samplein sample/batch2-view14-norgbd-T6-R6-2M-8thAMAG/ \
		--sampletouse 524288 \
		--samplebatching 64 \
		--samplebase 0 > nsample-idmv-action12/View-8-of-14-Rev-5-Feat-2048-524288-8thAMAG.out
			
cp -a ckpt/pretrain-d-elu-view-8-of-14-rev-5-action12-Feat-2048-8thAMAG-working ackpt/View-8-of-14-Action12-Rev-5-Feat-2048-524288-8thAMAG


mkdir -p ckpt/pretrain-d-elu-view-9-of-14-rev-5-action12-Feat-2048-8thAMAG-working

	./pretrain-d.sh --ferev 5 --elu \
		--ckptdir ckpt/pretrain-d-elu-view-9-of-14-rev-5-action12-Feat-2048-8thAMAG-working/ --ckptprefix working-try11 \
		--batch 2 --queuemax 64 --threads 1 \
		--iter 4194304 \
		--view 9 \
		--featnum 2048 \
		--imhidden 2048 2048 \
		--samplein sample/batch2-view14-norgbd-T6-R6-2M-8thAMAG/ \
		--sampletouse 524288 \
		--samplebatching 64 \
		--samplebase 0 > nsample-idmv-action12/View-9-of-14-Rev-5-Feat-2048-524288-8thAMAG.out
			
cp -a ckpt/pretrain-d-elu-view-9-of-14-rev-5-action12-Feat-2048-8thAMAG-working ackpt/View-9-of-14-Action12-Rev-5-Feat-2048-524288-8thAMAG


mkdir -p ckpt/pretrain-d-elu-view-10-of-14-rev-5-action12-Feat-2048-8thAMAG-working

	./pretrain-d.sh --ferev 5 --elu \
		--ckptdir ckpt/pretrain-d-elu-view-10-of-14-rev-5-action12-Feat-2048-8thAMAG-working/ --ckptprefix working-try11 \
		--batch 2 --queuemax 64 --threads 1 \
		--iter 4194304 \
		--view 10 \
		--featnum 2048 \
		--imhidden 2048 2048 \
		--samplein sample/batch2-view14-norgbd-T6-R6-2M-8thAMAG/ \
		--sampletouse 524288 \
		--samplebatching 64 \
		--samplebase 0 > nsample-idmv-action12/View-10-of-14-Rev-5-Feat-2048-524288-8thAMAG.out
			
cp -a ckpt/pretrain-d-elu-view-10-of-14-rev-5-action12-Feat-2048-8thAMAG-working ackpt/View-10-of-14-Action12-Rev-5-Feat-2048-524288-8thAMAG


mkdir -p ckpt/pretrain-d-elu-view-11-of-14-rev-5-action12-Feat-2048-8thAMAG-working

	./pretrain-d.sh --ferev 5 --elu \
		--ckptdir ckpt/pretrain-d-elu-view-11-of-14-rev-5-action12-Feat-2048-8thAMAG-working/ --ckptprefix working-try11 \
		--batch 2 --queuemax 64 --threads 1 \
		--iter 4194304 \
		--view 11 \
		--featnum 2048 \
		--imhidden 2048 2048 \
		--samplein sample/batch2-view14-norgbd-T6-R6-2M-8thAMAG/ \
		--sampletouse 524288 \
		--samplebatching 64 \
		--samplebase 0 > nsample-idmv-action12/View-11-of-14-Rev-5-Feat-2048-524288-8thAMAG.out
			
cp -a ckpt/pretrain-d-elu-view-11-of-14-rev-5-action12-Feat-2048-8thAMAG-working ackpt/View-11-of-14-Action12-Rev-5-Feat-2048-524288-8thAMAG


mkdir -p ckpt/pretrain-d-elu-view-12-of-14-rev-5-action12-Feat-2048-8thAMAG-working

	./pretrain-d.sh --ferev 5 --elu \
		--ckptdir ckpt/pretrain-d-elu-view-12-of-14-rev-5-action12-Feat-2048-8thAMAG-working/ --ckptprefix working-try11 \
		--batch 2 --queuemax 64 --threads 1 \
		--iter 4194304 \
		--view 12 \
		--featnum 2048 \
		--imhidden 2048 2048 \
		--samplein sample/batch2-view14-norgbd-T6-R6-2M-8thAMAG/ \
		--sampletouse 524288 \
		--samplebatching 64 \
		--samplebase 0 > nsample-idmv-action12/View-12-of-14-Rev-5-Feat-2048-524288-8thAMAG.out
			
cp -a ckpt/pretrain-d-elu-view-12-of-14-rev-5-action12-Feat-2048-8thAMAG-working ackpt/View-12-of-14-Action12-Rev-5-Feat-2048-524288-8thAMAG


mkdir -p ckpt/pretrain-d-elu-view-13-of-14-rev-5-action12-Feat-2048-8thAMAG-working

	./pretrain-d.sh --ferev 5 --elu \
		--ckptdir ckpt/pretrain-d-elu-view-13-of-14-rev-5-action12-Feat-2048-8thAMAG-working/ --ckptprefix working-try11 \
		--batch 2 --queuemax 64 --threads 1 \
		--iter 4194304 \
		--view 13 \
		--featnum 2048 \
		--imhidden 2048 2048 \
		--samplein sample/batch2-view14-norgbd-T6-R6-2M-8thAMAG/ \
		--sampletouse 524288 \
		--samplebatching 64 \
		--samplebase 0 > nsample-idmv-action12/View-13-of-14-Rev-5-Feat-2048-524288-8thAMAG.out
			
cp -a ckpt/pretrain-d-elu-view-13-of-14-rev-5-action12-Feat-2048-8thAMAG-working ackpt/View-13-of-14-Action12-Rev-5-Feat-2048-524288-8thAMAG

