mkdir -p nsample-idmv-action12

mkdir -p ckpt/pretrain-d-elu-view-Resnet-0-of-14-rev-11-action12-Feat-2048-working

	./pretrain-d.sh --ferev 11 --elu \
		--ckptdir ckpt/pretrain-d-elu-view-Resnet-0-of-14-rev-11-action12-Feat-2048-working/ --ckptprefix working-try11 \
		--batch 2 --queuemax 64 --threads 1 \
		--avi \
		--res 224 \
		--iter 16777216 \
		--view 0 \
		--featnum 2048 \
		--imhidden 2048 2048 \
		--samplein sample/batch2-view14-norgbd-T6-R6-2M/ \
		--sampletouse 524288 \
		--samplebatching 64 \
		--samplebase 0 > nsample-idmv-action12/Resnet-View-0-of-14-Rev-11-Feat-2048-524288.out
			
cp -a ckpt/pretrain-d-elu-view-Resnet-0-of-14-rev-11-action12-Feat-2048-working ackpt/Resnet-View-0-of-14-Action12-Rev-11-Feat-2048-524288

exit

mkdir -p ckpt/pretrain-d-elu-view-Resnet-1-of-14-rev-11-action12-Feat-2048-working

	./pretrain-d.sh --ferev 11 --elu \
		--ckptdir ckpt/pretrain-d-elu-view-Resnet-1-of-14-rev-11-action12-Feat-2048-working/ --ckptprefix working-try11 \
		--batch 2 --queuemax 64 --threads 1 \
		--avi \
		--res 224 \
		--iter 16777216 \
		--view 1 \
		--featnum 2048 \
		--imhidden 2048 2048 \
		--samplein sample/batch2-view14-norgbd-T6-R6-2M/ \
		--sampletouse 524288 \
		--samplebatching 64 \
		--samplebase 0 > nsample-idmv-action12/Resnet-View-1-of-14-Rev-11-Feat-2048-524288.out
			
cp -a ckpt/pretrain-d-elu-view-Resnet-1-of-14-rev-11-action12-Feat-2048-working ackpt/Resnet-View-1-of-14-Action12-Rev-11-Feat-2048-524288


mkdir -p ckpt/pretrain-d-elu-view-Resnet-2-of-14-rev-11-action12-Feat-2048-working

	./pretrain-d.sh --ferev 11 --elu \
		--ckptdir ckpt/pretrain-d-elu-view-Resnet-2-of-14-rev-11-action12-Feat-2048-working/ --ckptprefix working-try11 \
		--batch 2 --queuemax 64 --threads 1 \
		--avi \
		--res 224 \
		--iter 16777216 \
		--view 2 \
		--featnum 2048 \
		--imhidden 2048 2048 \
		--samplein sample/batch2-view14-norgbd-T6-R6-2M/ \
		--sampletouse 524288 \
		--samplebatching 64 \
		--samplebase 0 > nsample-idmv-action12/Resnet-View-2-of-14-Rev-11-Feat-2048-524288.out
			
cp -a ckpt/pretrain-d-elu-view-Resnet-2-of-14-rev-11-action12-Feat-2048-working ackpt/Resnet-View-2-of-14-Action12-Rev-11-Feat-2048-524288


mkdir -p ckpt/pretrain-d-elu-view-Resnet-3-of-14-rev-11-action12-Feat-2048-working

	./pretrain-d.sh --ferev 11 --elu \
		--ckptdir ckpt/pretrain-d-elu-view-Resnet-3-of-14-rev-11-action12-Feat-2048-working/ --ckptprefix working-try11 \
		--batch 2 --queuemax 64 --threads 1 \
		--avi \
		--res 224 \
		--iter 16777216 \
		--view 3 \
		--featnum 2048 \
		--imhidden 2048 2048 \
		--samplein sample/batch2-view14-norgbd-T6-R6-2M/ \
		--sampletouse 524288 \
		--samplebatching 64 \
		--samplebase 0 > nsample-idmv-action12/Resnet-View-3-of-14-Rev-11-Feat-2048-524288.out
			
cp -a ckpt/pretrain-d-elu-view-Resnet-3-of-14-rev-11-action12-Feat-2048-working ackpt/Resnet-View-3-of-14-Action12-Rev-11-Feat-2048-524288


mkdir -p ckpt/pretrain-d-elu-view-Resnet-4-of-14-rev-11-action12-Feat-2048-working

	./pretrain-d.sh --ferev 11 --elu \
		--ckptdir ckpt/pretrain-d-elu-view-Resnet-4-of-14-rev-11-action12-Feat-2048-working/ --ckptprefix working-try11 \
		--batch 2 --queuemax 64 --threads 1 \
		--avi \
		--res 224 \
		--iter 16777216 \
		--view 4 \
		--featnum 2048 \
		--imhidden 2048 2048 \
		--samplein sample/batch2-view14-norgbd-T6-R6-2M/ \
		--sampletouse 524288 \
		--samplebatching 64 \
		--samplebase 0 > nsample-idmv-action12/Resnet-View-4-of-14-Rev-11-Feat-2048-524288.out
			
cp -a ckpt/pretrain-d-elu-view-Resnet-4-of-14-rev-11-action12-Feat-2048-working ackpt/Resnet-View-4-of-14-Action12-Rev-11-Feat-2048-524288


mkdir -p ckpt/pretrain-d-elu-view-Resnet-5-of-14-rev-11-action12-Feat-2048-working

	./pretrain-d.sh --ferev 11 --elu \
		--ckptdir ckpt/pretrain-d-elu-view-Resnet-5-of-14-rev-11-action12-Feat-2048-working/ --ckptprefix working-try11 \
		--batch 2 --queuemax 64 --threads 1 \
		--avi \
		--res 224 \
		--iter 16777216 \
		--view 5 \
		--featnum 2048 \
		--imhidden 2048 2048 \
		--samplein sample/batch2-view14-norgbd-T6-R6-2M/ \
		--sampletouse 524288 \
		--samplebatching 64 \
		--samplebase 0 > nsample-idmv-action12/Resnet-View-5-of-14-Rev-11-Feat-2048-524288.out
			
cp -a ckpt/pretrain-d-elu-view-Resnet-5-of-14-rev-11-action12-Feat-2048-working ackpt/Resnet-View-5-of-14-Action12-Rev-11-Feat-2048-524288


mkdir -p ckpt/pretrain-d-elu-view-Resnet-6-of-14-rev-11-action12-Feat-2048-working

	./pretrain-d.sh --ferev 11 --elu \
		--ckptdir ckpt/pretrain-d-elu-view-Resnet-6-of-14-rev-11-action12-Feat-2048-working/ --ckptprefix working-try11 \
		--batch 2 --queuemax 64 --threads 1 \
		--avi \
		--res 224 \
		--iter 16777216 \
		--view 6 \
		--featnum 2048 \
		--imhidden 2048 2048 \
		--samplein sample/batch2-view14-norgbd-T6-R6-2M/ \
		--sampletouse 524288 \
		--samplebatching 64 \
		--samplebase 0 > nsample-idmv-action12/Resnet-View-6-of-14-Rev-11-Feat-2048-524288.out
			
cp -a ckpt/pretrain-d-elu-view-Resnet-6-of-14-rev-11-action12-Feat-2048-working ackpt/Resnet-View-6-of-14-Action12-Rev-11-Feat-2048-524288


mkdir -p ckpt/pretrain-d-elu-view-Resnet-7-of-14-rev-11-action12-Feat-2048-working

	./pretrain-d.sh --ferev 11 --elu \
		--ckptdir ckpt/pretrain-d-elu-view-Resnet-7-of-14-rev-11-action12-Feat-2048-working/ --ckptprefix working-try11 \
		--batch 2 --queuemax 64 --threads 1 \
		--avi \
		--res 224 \
		--iter 16777216 \
		--view 7 \
		--featnum 2048 \
		--imhidden 2048 2048 \
		--samplein sample/batch2-view14-norgbd-T6-R6-2M/ \
		--sampletouse 524288 \
		--samplebatching 64 \
		--samplebase 0 > nsample-idmv-action12/Resnet-View-7-of-14-Rev-11-Feat-2048-524288.out
			
cp -a ckpt/pretrain-d-elu-view-Resnet-7-of-14-rev-11-action12-Feat-2048-working ackpt/Resnet-View-7-of-14-Action12-Rev-11-Feat-2048-524288


mkdir -p ckpt/pretrain-d-elu-view-Resnet-8-of-14-rev-11-action12-Feat-2048-working

	./pretrain-d.sh --ferev 11 --elu \
		--ckptdir ckpt/pretrain-d-elu-view-Resnet-8-of-14-rev-11-action12-Feat-2048-working/ --ckptprefix working-try11 \
		--batch 2 --queuemax 64 --threads 1 \
		--avi \
		--res 224 \
		--iter 16777216 \
		--view 8 \
		--featnum 2048 \
		--imhidden 2048 2048 \
		--samplein sample/batch2-view14-norgbd-T6-R6-2M/ \
		--sampletouse 524288 \
		--samplebatching 64 \
		--samplebase 0 > nsample-idmv-action12/Resnet-View-8-of-14-Rev-11-Feat-2048-524288.out
			
cp -a ckpt/pretrain-d-elu-view-Resnet-8-of-14-rev-11-action12-Feat-2048-working ackpt/Resnet-View-8-of-14-Action12-Rev-11-Feat-2048-524288


mkdir -p ckpt/pretrain-d-elu-view-Resnet-9-of-14-rev-11-action12-Feat-2048-working

	./pretrain-d.sh --ferev 11 --elu \
		--ckptdir ckpt/pretrain-d-elu-view-Resnet-9-of-14-rev-11-action12-Feat-2048-working/ --ckptprefix working-try11 \
		--batch 2 --queuemax 64 --threads 1 \
		--avi \
		--res 224 \
		--iter 16777216 \
		--view 9 \
		--featnum 2048 \
		--imhidden 2048 2048 \
		--samplein sample/batch2-view14-norgbd-T6-R6-2M/ \
		--sampletouse 524288 \
		--samplebatching 64 \
		--samplebase 0 > nsample-idmv-action12/Resnet-View-9-of-14-Rev-11-Feat-2048-524288.out
			
cp -a ckpt/pretrain-d-elu-view-Resnet-9-of-14-rev-11-action12-Feat-2048-working ackpt/Resnet-View-9-of-14-Action12-Rev-11-Feat-2048-524288


mkdir -p ckpt/pretrain-d-elu-view-Resnet-10-of-14-rev-11-action12-Feat-2048-working

	./pretrain-d.sh --ferev 11 --elu \
		--ckptdir ckpt/pretrain-d-elu-view-Resnet-10-of-14-rev-11-action12-Feat-2048-working/ --ckptprefix working-try11 \
		--batch 2 --queuemax 64 --threads 1 \
		--avi \
		--res 224 \
		--iter 16777216 \
		--view 10 \
		--featnum 2048 \
		--imhidden 2048 2048 \
		--samplein sample/batch2-view14-norgbd-T6-R6-2M/ \
		--sampletouse 524288 \
		--samplebatching 64 \
		--samplebase 0 > nsample-idmv-action12/Resnet-View-10-of-14-Rev-11-Feat-2048-524288.out
			
cp -a ckpt/pretrain-d-elu-view-Resnet-10-of-14-rev-11-action12-Feat-2048-working ackpt/Resnet-View-10-of-14-Action12-Rev-11-Feat-2048-524288


mkdir -p ckpt/pretrain-d-elu-view-Resnet-11-of-14-rev-11-action12-Feat-2048-working

	./pretrain-d.sh --ferev 11 --elu \
		--ckptdir ckpt/pretrain-d-elu-view-Resnet-11-of-14-rev-11-action12-Feat-2048-working/ --ckptprefix working-try11 \
		--batch 2 --queuemax 64 --threads 1 \
		--avi \
		--res 224 \
		--iter 16777216 \
		--view 11 \
		--featnum 2048 \
		--imhidden 2048 2048 \
		--samplein sample/batch2-view14-norgbd-T6-R6-2M/ \
		--sampletouse 524288 \
		--samplebatching 64 \
		--samplebase 0 > nsample-idmv-action12/Resnet-View-11-of-14-Rev-11-Feat-2048-524288.out
			
cp -a ckpt/pretrain-d-elu-view-Resnet-11-of-14-rev-11-action12-Feat-2048-working ackpt/Resnet-View-11-of-14-Action12-Rev-11-Feat-2048-524288


mkdir -p ckpt/pretrain-d-elu-view-Resnet-12-of-14-rev-11-action12-Feat-2048-working

	./pretrain-d.sh --ferev 11 --elu \
		--ckptdir ckpt/pretrain-d-elu-view-Resnet-12-of-14-rev-11-action12-Feat-2048-working/ --ckptprefix working-try11 \
		--batch 2 --queuemax 64 --threads 1 \
		--avi \
		--res 224 \
		--iter 16777216 \
		--view 12 \
		--featnum 2048 \
		--imhidden 2048 2048 \
		--samplein sample/batch2-view14-norgbd-T6-R6-2M/ \
		--sampletouse 524288 \
		--samplebatching 64 \
		--samplebase 0 > nsample-idmv-action12/Resnet-View-12-of-14-Rev-11-Feat-2048-524288.out
			
cp -a ckpt/pretrain-d-elu-view-Resnet-12-of-14-rev-11-action12-Feat-2048-working ackpt/Resnet-View-12-of-14-Action12-Rev-11-Feat-2048-524288


mkdir -p ckpt/pretrain-d-elu-view-Resnet-13-of-14-rev-11-action12-Feat-2048-working

	./pretrain-d.sh --ferev 11 --elu \
		--ckptdir ckpt/pretrain-d-elu-view-Resnet-13-of-14-rev-11-action12-Feat-2048-working/ --ckptprefix working-try11 \
		--batch 2 --queuemax 64 --threads 1 \
		--avi \
		--res 224 \
		--iter 16777216 \
		--view 13 \
		--featnum 2048 \
		--imhidden 2048 2048 \
		--samplein sample/batch2-view14-norgbd-T6-R6-2M/ \
		--sampletouse 524288 \
		--samplebatching 64 \
		--samplebase 0 > nsample-idmv-action12/Resnet-View-13-of-14-Rev-11-Feat-2048-524288.out
			
cp -a ckpt/pretrain-d-elu-view-Resnet-13-of-14-rev-11-action12-Feat-2048-working ackpt/Resnet-View-13-of-14-Action12-Rev-11-Feat-2048-524288

