mkdir -p nsample-idmv-action12

mkdir -p ckpt/pretrain-d-elu-res-224-view-0-of-14-rev-10-action12-Feat-2048-working

	./pretrain-d.sh --ferev 10 --elu \
		--ckptdir ckpt/pretrain-d-elu-res-224-view-0-of-14-rev-10-action12-Feat-2048-working/ --ckptprefix working- \
		--batch 2 --queuemax 64 --threads 1 \
		--iter 12582912 \
		--view 0 \
		--res 224 \
		--featnum 2048 \
		--imhidden 2048 2048 \
		--samplein sample/batch2-view14-norgbd-T6-R6-2M/ \
		--sampletouse 524288 \
		--samplebatching 64 \
		--samplebase 0 > nsample-idmv-action12/Res-224-View-0-of-14-Rev-10-Feat-2048-524288.out
			
cp -a ckpt/pretrain-d-elu-res-224-view-0-of-14-rev-10-action12-Feat-2048-working ackpt/Res-224-View-0-of-14-Action12-Rev-10-Feat-2048-524288


mkdir -p ckpt/pretrain-d-elu-res-224-view-1-of-14-rev-10-action12-Feat-2048-working

	./pretrain-d.sh --ferev 10 --elu \
		--ckptdir ckpt/pretrain-d-elu-res-224-view-1-of-14-rev-10-action12-Feat-2048-working/ --ckptprefix working- \
		--batch 2 --queuemax 64 --threads 1 \
		--iter 12582912 \
		--view 1 \
		--res 224 \
		--featnum 2048 \
		--imhidden 2048 2048 \
		--samplein sample/batch2-view14-norgbd-T6-R6-2M/ \
		--sampletouse 524288 \
		--samplebatching 64 \
		--samplebase 0 > nsample-idmv-action12/Res-224-View-1-of-14-Rev-10-Feat-2048-524288.out
			
cp -a ckpt/pretrain-d-elu-res-224-view-1-of-14-rev-10-action12-Feat-2048-working ackpt/Res-224-View-1-of-14-Action12-Rev-10-Feat-2048-524288


mkdir -p ckpt/pretrain-d-elu-res-224-view-2-of-14-rev-10-action12-Feat-2048-working

	./pretrain-d.sh --ferev 10 --elu \
		--ckptdir ckpt/pretrain-d-elu-res-224-view-2-of-14-rev-10-action12-Feat-2048-working/ --ckptprefix working- \
		--batch 2 --queuemax 64 --threads 1 \
		--iter 12582912 \
		--view 2 \
		--res 224 \
		--featnum 2048 \
		--imhidden 2048 2048 \
		--samplein sample/batch2-view14-norgbd-T6-R6-2M/ \
		--sampletouse 524288 \
		--samplebatching 64 \
		--samplebase 0 > nsample-idmv-action12/Res-224-View-2-of-14-Rev-10-Feat-2048-524288.out
			
cp -a ckpt/pretrain-d-elu-res-224-view-2-of-14-rev-10-action12-Feat-2048-working ackpt/Res-224-View-2-of-14-Action12-Rev-10-Feat-2048-524288


mkdir -p ckpt/pretrain-d-elu-res-224-view-3-of-14-rev-10-action12-Feat-2048-working

	./pretrain-d.sh --ferev 10 --elu \
		--ckptdir ckpt/pretrain-d-elu-res-224-view-3-of-14-rev-10-action12-Feat-2048-working/ --ckptprefix working- \
		--batch 2 --queuemax 64 --threads 1 \
		--iter 12582912 \
		--view 3 \
		--res 224 \
		--featnum 2048 \
		--imhidden 2048 2048 \
		--samplein sample/batch2-view14-norgbd-T6-R6-2M/ \
		--sampletouse 524288 \
		--samplebatching 64 \
		--samplebase 0 > nsample-idmv-action12/Res-224-View-3-of-14-Rev-10-Feat-2048-524288.out
			
cp -a ckpt/pretrain-d-elu-res-224-view-3-of-14-rev-10-action12-Feat-2048-working ackpt/Res-224-View-3-of-14-Action12-Rev-10-Feat-2048-524288


mkdir -p ckpt/pretrain-d-elu-res-224-view-4-of-14-rev-10-action12-Feat-2048-working

	./pretrain-d.sh --ferev 10 --elu \
		--ckptdir ckpt/pretrain-d-elu-res-224-view-4-of-14-rev-10-action12-Feat-2048-working/ --ckptprefix working- \
		--batch 2 --queuemax 64 --threads 1 \
		--iter 12582912 \
		--view 4 \
		--res 224 \
		--featnum 2048 \
		--imhidden 2048 2048 \
		--samplein sample/batch2-view14-norgbd-T6-R6-2M/ \
		--sampletouse 524288 \
		--samplebatching 64 \
		--samplebase 0 > nsample-idmv-action12/Res-224-View-4-of-14-Rev-10-Feat-2048-524288.out
			
cp -a ckpt/pretrain-d-elu-res-224-view-4-of-14-rev-10-action12-Feat-2048-working ackpt/Res-224-View-4-of-14-Action12-Rev-10-Feat-2048-524288


mkdir -p ckpt/pretrain-d-elu-res-224-view-5-of-14-rev-10-action12-Feat-2048-working

	./pretrain-d.sh --ferev 10 --elu \
		--ckptdir ckpt/pretrain-d-elu-res-224-view-5-of-14-rev-10-action12-Feat-2048-working/ --ckptprefix working- \
		--batch 2 --queuemax 64 --threads 1 \
		--iter 12582912 \
		--view 5 \
		--res 224 \
		--featnum 2048 \
		--imhidden 2048 2048 \
		--samplein sample/batch2-view14-norgbd-T6-R6-2M/ \
		--sampletouse 524288 \
		--samplebatching 64 \
		--samplebase 0 > nsample-idmv-action12/Res-224-View-5-of-14-Rev-10-Feat-2048-524288.out
			
cp -a ckpt/pretrain-d-elu-res-224-view-5-of-14-rev-10-action12-Feat-2048-working ackpt/Res-224-View-5-of-14-Action12-Rev-10-Feat-2048-524288


mkdir -p ckpt/pretrain-d-elu-res-224-view-6-of-14-rev-10-action12-Feat-2048-working

	./pretrain-d.sh --ferev 10 --elu \
		--ckptdir ckpt/pretrain-d-elu-res-224-view-6-of-14-rev-10-action12-Feat-2048-working/ --ckptprefix working- \
		--batch 2 --queuemax 64 --threads 1 \
		--iter 12582912 \
		--view 6 \
		--res 224 \
		--featnum 2048 \
		--imhidden 2048 2048 \
		--samplein sample/batch2-view14-norgbd-T6-R6-2M/ \
		--sampletouse 524288 \
		--samplebatching 64 \
		--samplebase 0 > nsample-idmv-action12/Res-224-View-6-of-14-Rev-10-Feat-2048-524288.out
			
cp -a ckpt/pretrain-d-elu-res-224-view-6-of-14-rev-10-action12-Feat-2048-working ackpt/Res-224-View-6-of-14-Action12-Rev-10-Feat-2048-524288

