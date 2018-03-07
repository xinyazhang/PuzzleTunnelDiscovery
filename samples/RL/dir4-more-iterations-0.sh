mkdir -p nsample-idmv-action12

mkdir -p ckpt/pretrain-d-elu-view-0-of-14-rev-5-action12-Feat-2048-working

	./pretrain-d.sh --ferev 5 --elu \
		--ckptdir ckpt/pretrain-d-elu-view-0-of-14-rev-5-action12-Feat-2048-working/ --ckptprefix working- \
		--batch 2 --queuemax 64 --threads 1 \
		--iter 12582912 \
		--view 0 \
		--featnum 2048 \
		--imhidden 2048 2048 \
		--samplein sample/batch2-view14-norgbd-T6-R6-2M/ \
		--sampletouse 524288 \
		--samplebatching 64 \
		--samplebase 0 > nsample-idmv-action12/View-0-of-14-Rev-5-Feat-2048-524288-Pass2.out
			
cp -a ckpt/pretrain-d-elu-view-0-of-14-rev-5-action12-Feat-2048-working ackpt/View-0-of-14-Action12-Rev-5-Feat-2048-524288-Pass2

exit # Pause at here

mkdir -p ckpt/pretrain-d-elu-view-1-of-14-rev-5-action12-Feat-2048-working

	./pretrain-d.sh --ferev 5 --elu \
		--ckptdir ckpt/pretrain-d-elu-view-1-of-14-rev-5-action12-Feat-2048-working/ --ckptprefix working- \
		--batch 2 --queuemax 64 --threads 1 \
		--iter 12582912 \
		--view 1 \
		--featnum 2048 \
		--imhidden 2048 2048 \
		--samplein sample/batch2-view14-norgbd-T6-R6-2M/ \
		--sampletouse 524288 \
		--samplebatching 64 \
		--samplebase 0 > nsample-idmv-action12/View-1-of-14-Rev-5-Feat-2048-524288-Pass2.out
			
cp -a ckpt/pretrain-d-elu-view-1-of-14-rev-5-action12-Feat-2048-working ackpt/View-1-of-14-Action12-Rev-5-Feat-2048-524288-Pass2


mkdir -p ckpt/pretrain-d-elu-view-2-of-14-rev-5-action12-Feat-2048-working

	./pretrain-d.sh --ferev 5 --elu \
		--ckptdir ckpt/pretrain-d-elu-view-2-of-14-rev-5-action12-Feat-2048-working/ --ckptprefix working- \
		--batch 2 --queuemax 64 --threads 1 \
		--iter 12582912 \
		--view 2 \
		--featnum 2048 \
		--imhidden 2048 2048 \
		--samplein sample/batch2-view14-norgbd-T6-R6-2M/ \
		--sampletouse 524288 \
		--samplebatching 64 \
		--samplebase 0 > nsample-idmv-action12/View-2-of-14-Rev-5-Feat-2048-524288-Pass2.out
			
cp -a ckpt/pretrain-d-elu-view-2-of-14-rev-5-action12-Feat-2048-working ackpt/View-2-of-14-Action12-Rev-5-Feat-2048-524288-Pass2


mkdir -p ckpt/pretrain-d-elu-view-3-of-14-rev-5-action12-Feat-2048-working

	./pretrain-d.sh --ferev 5 --elu \
		--ckptdir ckpt/pretrain-d-elu-view-3-of-14-rev-5-action12-Feat-2048-working/ --ckptprefix working- \
		--batch 2 --queuemax 64 --threads 1 \
		--iter 12582912 \
		--view 3 \
		--featnum 2048 \
		--imhidden 2048 2048 \
		--samplein sample/batch2-view14-norgbd-T6-R6-2M/ \
		--sampletouse 524288 \
		--samplebatching 64 \
		--samplebase 0 > nsample-idmv-action12/View-3-of-14-Rev-5-Feat-2048-524288-Pass2.out
			
cp -a ckpt/pretrain-d-elu-view-3-of-14-rev-5-action12-Feat-2048-working ackpt/View-3-of-14-Action12-Rev-5-Feat-2048-524288-Pass2


mkdir -p ckpt/pretrain-d-elu-view-4-of-14-rev-5-action12-Feat-2048-working

	./pretrain-d.sh --ferev 5 --elu \
		--ckptdir ckpt/pretrain-d-elu-view-4-of-14-rev-5-action12-Feat-2048-working/ --ckptprefix working- \
		--batch 2 --queuemax 64 --threads 1 \
		--iter 12582912 \
		--view 4 \
		--featnum 2048 \
		--imhidden 2048 2048 \
		--samplein sample/batch2-view14-norgbd-T6-R6-2M/ \
		--sampletouse 524288 \
		--samplebatching 64 \
		--samplebase 0 > nsample-idmv-action12/View-4-of-14-Rev-5-Feat-2048-524288-Pass2.out
			
cp -a ckpt/pretrain-d-elu-view-4-of-14-rev-5-action12-Feat-2048-working ackpt/View-4-of-14-Action12-Rev-5-Feat-2048-524288-Pass2


mkdir -p ckpt/pretrain-d-elu-view-5-of-14-rev-5-action12-Feat-2048-working

	./pretrain-d.sh --ferev 5 --elu \
		--ckptdir ckpt/pretrain-d-elu-view-5-of-14-rev-5-action12-Feat-2048-working/ --ckptprefix working- \
		--batch 2 --queuemax 64 --threads 1 \
		--iter 12582912 \
		--view 5 \
		--featnum 2048 \
		--imhidden 2048 2048 \
		--samplein sample/batch2-view14-norgbd-T6-R6-2M/ \
		--sampletouse 524288 \
		--samplebatching 64 \
		--samplebase 0 > nsample-idmv-action12/View-5-of-14-Rev-5-Feat-2048-524288-Pass2.out
			
cp -a ckpt/pretrain-d-elu-view-5-of-14-rev-5-action12-Feat-2048-working ackpt/View-5-of-14-Action12-Rev-5-Feat-2048-524288-Pass2


mkdir -p ckpt/pretrain-d-elu-view-6-of-14-rev-5-action12-Feat-2048-working

	./pretrain-d.sh --ferev 5 --elu \
		--ckptdir ckpt/pretrain-d-elu-view-6-of-14-rev-5-action12-Feat-2048-working/ --ckptprefix working- \
		--batch 2 --queuemax 64 --threads 1 \
		--iter 12582912 \
		--view 6 \
		--featnum 2048 \
		--imhidden 2048 2048 \
		--samplein sample/batch2-view14-norgbd-T6-R6-2M/ \
		--sampletouse 524288 \
		--samplebatching 64 \
		--samplebase 0 > nsample-idmv-action12/View-6-of-14-Rev-5-Feat-2048-524288-Pass2.out
			
cp -a ckpt/pretrain-d-elu-view-6-of-14-rev-5-action12-Feat-2048-working ackpt/View-6-of-14-Action12-Rev-5-Feat-2048-524288-Pass2

