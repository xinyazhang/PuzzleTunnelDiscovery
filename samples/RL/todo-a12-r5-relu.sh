mkdir -p nsample-idmv-action12

mkdir -p ckpt/pretrain-d-relu-view-0-of-14-rev-5-action12-Feat-2048-working

	./pretrain-d.sh --ferev 5 \
		--ckptdir ckpt/pretrain-d-relu-view-0-of-14-rev-5-action12-Feat-2048-working/ --ckptprefix working-try11 \
		--batch 2 --queuemax 64 --threads 1 \
		--iter 4194304 \
		--view 0 \
		--featnum 2048 \
		--imhidden 2048 2048 \
		--samplein sample/batch2-view14-norgbd-T6-R6-2M/ \
		--sampletouse 524288 \
		--samplebatching 64 \
		--samplebase 0 > nsample-idmv-action12/ReLU-View-0-of-14-Rev-5-Feat-2048-524288.out
			
cp -a ckpt/pretrain-d-relu-view-0-of-14-rev-5-action12-Feat-2048-working ackpt/ReLU-View-0-of-14-Action12-Rev-5-Feat-2048-524288


mkdir -p ckpt/pretrain-d-relu-view-1-of-14-rev-5-action12-Feat-2048-working

	./pretrain-d.sh --ferev 5 \
		--ckptdir ckpt/pretrain-d-relu-view-1-of-14-rev-5-action12-Feat-2048-working/ --ckptprefix working-try11 \
		--batch 2 --queuemax 64 --threads 1 \
		--iter 4194304 \
		--view 1 \
		--featnum 2048 \
		--imhidden 2048 2048 \
		--samplein sample/batch2-view14-norgbd-T6-R6-2M/ \
		--sampletouse 524288 \
		--samplebatching 64 \
		--samplebase 0 > nsample-idmv-action12/ReLU-View-1-of-14-Rev-5-Feat-2048-524288.out
			
cp -a ckpt/pretrain-d-relu-view-1-of-14-rev-5-action12-Feat-2048-working ackpt/ReLU-View-1-of-14-Action12-Rev-5-Feat-2048-524288


mkdir -p ckpt/pretrain-d-relu-view-2-of-14-rev-5-action12-Feat-2048-working

	./pretrain-d.sh --ferev 5 \
		--ckptdir ckpt/pretrain-d-relu-view-2-of-14-rev-5-action12-Feat-2048-working/ --ckptprefix working-try11 \
		--batch 2 --queuemax 64 --threads 1 \
		--iter 4194304 \
		--view 2 \
		--featnum 2048 \
		--imhidden 2048 2048 \
		--samplein sample/batch2-view14-norgbd-T6-R6-2M/ \
		--sampletouse 524288 \
		--samplebatching 64 \
		--samplebase 0 > nsample-idmv-action12/ReLU-View-2-of-14-Rev-5-Feat-2048-524288.out
			
cp -a ckpt/pretrain-d-relu-view-2-of-14-rev-5-action12-Feat-2048-working ackpt/ReLU-View-2-of-14-Action12-Rev-5-Feat-2048-524288


mkdir -p ckpt/pretrain-d-relu-view-3-of-14-rev-5-action12-Feat-2048-working

	./pretrain-d.sh --ferev 5 \
		--ckptdir ckpt/pretrain-d-relu-view-3-of-14-rev-5-action12-Feat-2048-working/ --ckptprefix working-try11 \
		--batch 2 --queuemax 64 --threads 1 \
		--iter 4194304 \
		--view 3 \
		--featnum 2048 \
		--imhidden 2048 2048 \
		--samplein sample/batch2-view14-norgbd-T6-R6-2M/ \
		--sampletouse 524288 \
		--samplebatching 64 \
		--samplebase 0 > nsample-idmv-action12/ReLU-View-3-of-14-Rev-5-Feat-2048-524288.out
			
cp -a ckpt/pretrain-d-relu-view-3-of-14-rev-5-action12-Feat-2048-working ackpt/ReLU-View-3-of-14-Action12-Rev-5-Feat-2048-524288


mkdir -p ckpt/pretrain-d-relu-view-4-of-14-rev-5-action12-Feat-2048-working

	./pretrain-d.sh --ferev 5 \
		--ckptdir ckpt/pretrain-d-relu-view-4-of-14-rev-5-action12-Feat-2048-working/ --ckptprefix working-try11 \
		--batch 2 --queuemax 64 --threads 1 \
		--iter 4194304 \
		--view 4 \
		--featnum 2048 \
		--imhidden 2048 2048 \
		--samplein sample/batch2-view14-norgbd-T6-R6-2M/ \
		--sampletouse 524288 \
		--samplebatching 64 \
		--samplebase 0 > nsample-idmv-action12/ReLU-View-4-of-14-Rev-5-Feat-2048-524288.out
			
cp -a ckpt/pretrain-d-relu-view-4-of-14-rev-5-action12-Feat-2048-working ackpt/ReLU-View-4-of-14-Action12-Rev-5-Feat-2048-524288


mkdir -p ckpt/pretrain-d-relu-view-5-of-14-rev-5-action12-Feat-2048-working

	./pretrain-d.sh --ferev 5 \
		--ckptdir ckpt/pretrain-d-relu-view-5-of-14-rev-5-action12-Feat-2048-working/ --ckptprefix working-try11 \
		--batch 2 --queuemax 64 --threads 1 \
		--iter 4194304 \
		--view 5 \
		--featnum 2048 \
		--imhidden 2048 2048 \
		--samplein sample/batch2-view14-norgbd-T6-R6-2M/ \
		--sampletouse 524288 \
		--samplebatching 64 \
		--samplebase 0 > nsample-idmv-action12/ReLU-View-5-of-14-Rev-5-Feat-2048-524288.out
			
cp -a ckpt/pretrain-d-relu-view-5-of-14-rev-5-action12-Feat-2048-working ackpt/ReLU-View-5-of-14-Action12-Rev-5-Feat-2048-524288


mkdir -p ckpt/pretrain-d-relu-view-6-of-14-rev-5-action12-Feat-2048-working

	./pretrain-d.sh --ferev 5 \
		--ckptdir ckpt/pretrain-d-relu-view-6-of-14-rev-5-action12-Feat-2048-working/ --ckptprefix working-try11 \
		--batch 2 --queuemax 64 --threads 1 \
		--iter 4194304 \
		--view 6 \
		--featnum 2048 \
		--imhidden 2048 2048 \
		--samplein sample/batch2-view14-norgbd-T6-R6-2M/ \
		--sampletouse 524288 \
		--samplebatching 64 \
		--samplebase 0 > nsample-idmv-action12/ReLU-View-6-of-14-Rev-5-Feat-2048-524288.out
			
cp -a ckpt/pretrain-d-relu-view-6-of-14-rev-5-action12-Feat-2048-working ackpt/ReLU-View-6-of-14-Action12-Rev-5-Feat-2048-524288


mkdir -p ckpt/pretrain-d-relu-view-7-of-14-rev-5-action12-Feat-2048-working

	./pretrain-d.sh --ferev 5 \
		--ckptdir ckpt/pretrain-d-relu-view-7-of-14-rev-5-action12-Feat-2048-working/ --ckptprefix working-try11 \
		--batch 2 --queuemax 64 --threads 1 \
		--iter 4194304 \
		--view 7 \
		--featnum 2048 \
		--imhidden 2048 2048 \
		--samplein sample/batch2-view14-norgbd-T6-R6-2M/ \
		--sampletouse 524288 \
		--samplebatching 64 \
		--samplebase 0 > nsample-idmv-action12/ReLU-View-7-of-14-Rev-5-Feat-2048-524288.out
			
cp -a ckpt/pretrain-d-relu-view-7-of-14-rev-5-action12-Feat-2048-working ackpt/ReLU-View-7-of-14-Action12-Rev-5-Feat-2048-524288


mkdir -p ckpt/pretrain-d-relu-view-8-of-14-rev-5-action12-Feat-2048-working

	./pretrain-d.sh --ferev 5 \
		--ckptdir ckpt/pretrain-d-relu-view-8-of-14-rev-5-action12-Feat-2048-working/ --ckptprefix working-try11 \
		--batch 2 --queuemax 64 --threads 1 \
		--iter 4194304 \
		--view 8 \
		--featnum 2048 \
		--imhidden 2048 2048 \
		--samplein sample/batch2-view14-norgbd-T6-R6-2M/ \
		--sampletouse 524288 \
		--samplebatching 64 \
		--samplebase 0 > nsample-idmv-action12/ReLU-View-8-of-14-Rev-5-Feat-2048-524288.out
			
cp -a ckpt/pretrain-d-relu-view-8-of-14-rev-5-action12-Feat-2048-working ackpt/ReLU-View-8-of-14-Action12-Rev-5-Feat-2048-524288


mkdir -p ckpt/pretrain-d-relu-view-9-of-14-rev-5-action12-Feat-2048-working

	./pretrain-d.sh --ferev 5 \
		--ckptdir ckpt/pretrain-d-relu-view-9-of-14-rev-5-action12-Feat-2048-working/ --ckptprefix working-try11 \
		--batch 2 --queuemax 64 --threads 1 \
		--iter 4194304 \
		--view 9 \
		--featnum 2048 \
		--imhidden 2048 2048 \
		--samplein sample/batch2-view14-norgbd-T6-R6-2M/ \
		--sampletouse 524288 \
		--samplebatching 64 \
		--samplebase 0 > nsample-idmv-action12/ReLU-View-9-of-14-Rev-5-Feat-2048-524288.out
			
cp -a ckpt/pretrain-d-relu-view-9-of-14-rev-5-action12-Feat-2048-working ackpt/ReLU-View-9-of-14-Action12-Rev-5-Feat-2048-524288


mkdir -p ckpt/pretrain-d-relu-view-10-of-14-rev-5-action12-Feat-2048-working

	./pretrain-d.sh --ferev 5 \
		--ckptdir ckpt/pretrain-d-relu-view-10-of-14-rev-5-action12-Feat-2048-working/ --ckptprefix working-try11 \
		--batch 2 --queuemax 64 --threads 1 \
		--iter 4194304 \
		--view 10 \
		--featnum 2048 \
		--imhidden 2048 2048 \
		--samplein sample/batch2-view14-norgbd-T6-R6-2M/ \
		--sampletouse 524288 \
		--samplebatching 64 \
		--samplebase 0 > nsample-idmv-action12/ReLU-View-10-of-14-Rev-5-Feat-2048-524288.out
			
cp -a ckpt/pretrain-d-relu-view-10-of-14-rev-5-action12-Feat-2048-working ackpt/ReLU-View-10-of-14-Action12-Rev-5-Feat-2048-524288


mkdir -p ckpt/pretrain-d-relu-view-11-of-14-rev-5-action12-Feat-2048-working

	./pretrain-d.sh --ferev 5 \
		--ckptdir ckpt/pretrain-d-relu-view-11-of-14-rev-5-action12-Feat-2048-working/ --ckptprefix working-try11 \
		--batch 2 --queuemax 64 --threads 1 \
		--iter 4194304 \
		--view 11 \
		--featnum 2048 \
		--imhidden 2048 2048 \
		--samplein sample/batch2-view14-norgbd-T6-R6-2M/ \
		--sampletouse 524288 \
		--samplebatching 64 \
		--samplebase 0 > nsample-idmv-action12/ReLU-View-11-of-14-Rev-5-Feat-2048-524288.out
			
cp -a ckpt/pretrain-d-relu-view-11-of-14-rev-5-action12-Feat-2048-working ackpt/ReLU-View-11-of-14-Action12-Rev-5-Feat-2048-524288


mkdir -p ckpt/pretrain-d-relu-view-12-of-14-rev-5-action12-Feat-2048-working

	./pretrain-d.sh --ferev 5 \
		--ckptdir ckpt/pretrain-d-relu-view-12-of-14-rev-5-action12-Feat-2048-working/ --ckptprefix working-try11 \
		--batch 2 --queuemax 64 --threads 1 \
		--iter 4194304 \
		--view 12 \
		--featnum 2048 \
		--imhidden 2048 2048 \
		--samplein sample/batch2-view14-norgbd-T6-R6-2M/ \
		--sampletouse 524288 \
		--samplebatching 64 \
		--samplebase 0 > nsample-idmv-action12/ReLU-View-12-of-14-Rev-5-Feat-2048-524288.out
			
cp -a ckpt/pretrain-d-relu-view-12-of-14-rev-5-action12-Feat-2048-working ackpt/ReLU-View-12-of-14-Action12-Rev-5-Feat-2048-524288


mkdir -p ckpt/pretrain-d-relu-view-13-of-14-rev-5-action12-Feat-2048-working

	./pretrain-d.sh --ferev 5 \
		--ckptdir ckpt/pretrain-d-relu-view-13-of-14-rev-5-action12-Feat-2048-working/ --ckptprefix working-try11 \
		--batch 2 --queuemax 64 --threads 1 \
		--iter 4194304 \
		--view 13 \
		--featnum 2048 \
		--imhidden 2048 2048 \
		--samplein sample/batch2-view14-norgbd-T6-R6-2M/ \
		--sampletouse 524288 \
		--samplebatching 64 \
		--samplebase 0 > nsample-idmv-action12/ReLU-View-13-of-14-Rev-5-Feat-2048-524288.out
			
cp -a ckpt/pretrain-d-relu-view-13-of-14-rev-5-action12-Feat-2048-working ackpt/ReLU-View-13-of-14-Action12-Rev-5-Feat-2048-524288

