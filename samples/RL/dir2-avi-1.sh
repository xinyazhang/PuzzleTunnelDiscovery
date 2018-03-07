ssh linux.cs.utexas.edu 'mail -s "Update your TF to 1.5 (1)" < /devl/null'
exit # Pause at here

mkdir -p ckpt/pretrain-d-elu-avi-view-7-of-14-rev-5-action12-Feat-2048-working

	./pretrain-d.sh --ferev 5 --elu \
		--ckptdir ckpt/pretrain-d-elu-avi-view-7-of-14-rev-5-action12-Feat-2048-working/ --ckptprefix working- \
		--batch 2 --queuemax 64 --threads 1 \
		--iter 12582912 \
		--view 7 \
		--avi \
		--featnum 2048 \
		--imhidden 2048 2048 \
		--samplein sample/batch2-view14-norgbd-T6-R6-2M/ \
		--sampletouse 524288 \
		--samplebatching 64 \
		--samplebase 0 > nsample-idmv-action12/AVI-View-7-of-14-Rev-5-Feat-2048-524288.out
			
cp -a ckpt/pretrain-d-elu-avi-view-7-of-14-rev-5-action12-Feat-2048-working ackpt/AVI-View-7-of-14-Action12-Rev-5-Feat-2048-524288


mkdir -p ckpt/pretrain-d-elu-avi-view-8-of-14-rev-5-action12-Feat-2048-working

	./pretrain-d.sh --ferev 5 --elu \
		--ckptdir ckpt/pretrain-d-elu-avi-view-8-of-14-rev-5-action12-Feat-2048-working/ --ckptprefix working- \
		--batch 2 --queuemax 64 --threads 1 \
		--iter 12582912 \
		--view 8 \
		--avi \
		--featnum 2048 \
		--imhidden 2048 2048 \
		--samplein sample/batch2-view14-norgbd-T6-R6-2M/ \
		--sampletouse 524288 \
		--samplebatching 64 \
		--samplebase 0 > nsample-idmv-action12/AVI-View-8-of-14-Rev-5-Feat-2048-524288.out
			
cp -a ckpt/pretrain-d-elu-avi-view-8-of-14-rev-5-action12-Feat-2048-working ackpt/AVI-View-8-of-14-Action12-Rev-5-Feat-2048-524288


mkdir -p ckpt/pretrain-d-elu-avi-view-9-of-14-rev-5-action12-Feat-2048-working

	./pretrain-d.sh --ferev 5 --elu \
		--ckptdir ckpt/pretrain-d-elu-avi-view-9-of-14-rev-5-action12-Feat-2048-working/ --ckptprefix working- \
		--batch 2 --queuemax 64 --threads 1 \
		--iter 12582912 \
		--view 9 \
		--avi \
		--featnum 2048 \
		--imhidden 2048 2048 \
		--samplein sample/batch2-view14-norgbd-T6-R6-2M/ \
		--sampletouse 524288 \
		--samplebatching 64 \
		--samplebase 0 > nsample-idmv-action12/AVI-View-9-of-14-Rev-5-Feat-2048-524288.out
			
cp -a ckpt/pretrain-d-elu-avi-view-9-of-14-rev-5-action12-Feat-2048-working ackpt/AVI-View-9-of-14-Action12-Rev-5-Feat-2048-524288


mkdir -p ckpt/pretrain-d-elu-avi-view-10-of-14-rev-5-action12-Feat-2048-working

	./pretrain-d.sh --ferev 5 --elu \
		--ckptdir ckpt/pretrain-d-elu-avi-view-10-of-14-rev-5-action12-Feat-2048-working/ --ckptprefix working- \
		--batch 2 --queuemax 64 --threads 1 \
		--iter 12582912 \
		--view 10 \
		--avi \
		--featnum 2048 \
		--imhidden 2048 2048 \
		--samplein sample/batch2-view14-norgbd-T6-R6-2M/ \
		--sampletouse 524288 \
		--samplebatching 64 \
		--samplebase 0 > nsample-idmv-action12/AVI-View-10-of-14-Rev-5-Feat-2048-524288.out
			
cp -a ckpt/pretrain-d-elu-avi-view-10-of-14-rev-5-action12-Feat-2048-working ackpt/AVI-View-10-of-14-Action12-Rev-5-Feat-2048-524288


mkdir -p ckpt/pretrain-d-elu-avi-view-11-of-14-rev-5-action12-Feat-2048-working

	./pretrain-d.sh --ferev 5 --elu \
		--ckptdir ckpt/pretrain-d-elu-avi-view-11-of-14-rev-5-action12-Feat-2048-working/ --ckptprefix working- \
		--batch 2 --queuemax 64 --threads 1 \
		--iter 12582912 \
		--view 11 \
		--avi \
		--featnum 2048 \
		--imhidden 2048 2048 \
		--samplein sample/batch2-view14-norgbd-T6-R6-2M/ \
		--sampletouse 524288 \
		--samplebatching 64 \
		--samplebase 0 > nsample-idmv-action12/AVI-View-11-of-14-Rev-5-Feat-2048-524288.out
			
cp -a ckpt/pretrain-d-elu-avi-view-11-of-14-rev-5-action12-Feat-2048-working ackpt/AVI-View-11-of-14-Action12-Rev-5-Feat-2048-524288


mkdir -p ckpt/pretrain-d-elu-avi-view-12-of-14-rev-5-action12-Feat-2048-working

	./pretrain-d.sh --ferev 5 --elu \
		--ckptdir ckpt/pretrain-d-elu-avi-view-12-of-14-rev-5-action12-Feat-2048-working/ --ckptprefix working- \
		--batch 2 --queuemax 64 --threads 1 \
		--iter 12582912 \
		--view 12 \
		--avi \
		--featnum 2048 \
		--imhidden 2048 2048 \
		--samplein sample/batch2-view14-norgbd-T6-R6-2M/ \
		--sampletouse 524288 \
		--samplebatching 64 \
		--samplebase 0 > nsample-idmv-action12/AVI-View-12-of-14-Rev-5-Feat-2048-524288.out
			
cp -a ckpt/pretrain-d-elu-avi-view-12-of-14-rev-5-action12-Feat-2048-working ackpt/AVI-View-12-of-14-Action12-Rev-5-Feat-2048-524288


mkdir -p ckpt/pretrain-d-elu-avi-view-13-of-14-rev-5-action12-Feat-2048-working

	./pretrain-d.sh --ferev 5 --elu \
		--ckptdir ckpt/pretrain-d-elu-avi-view-13-of-14-rev-5-action12-Feat-2048-working/ --ckptprefix working- \
		--batch 2 --queuemax 64 --threads 1 \
		--iter 12582912 \
		--view 13 \
		--avi \
		--featnum 2048 \
		--imhidden 2048 2048 \
		--samplein sample/batch2-view14-norgbd-T6-R6-2M/ \
		--sampletouse 524288 \
		--samplebatching 64 \
		--samplebase 0 > nsample-idmv-action12/AVI-View-13-of-14-Rev-5-Feat-2048-524288.out
			
cp -a ckpt/pretrain-d-elu-avi-view-13-of-14-rev-5-action12-Feat-2048-working ackpt/AVI-View-13-of-14-Action12-Rev-5-Feat-2048-524288

