mkdir -p nsample-idmv-action12

mkdir -p ckpt/pretrain-d-elu-avi-view-0-of-14-rev-5-action12-Feat-2048-working

	./pretrain-d.sh --ferev 5 --elu \
		--ckptdir ckpt/pretrain-d-elu-avi-view-0-of-14-rev-5-action12-Feat-2048-working/ --ckptprefix working- \
		--batch 2 --queuemax 64 --threads 1 \
		--iter 12582912 \
		--view 0 \
		--avi \
		--featnum 2048 \
		--imhidden 2048 2048 \
		--samplein sample/batch2-view14-norgbd-T6-R6-2M/ \
		--sampletouse 524288 \
		--samplebatching 64 \
		--samplebase 0 > nsample-idmv-action12/AVI-View-0-of-14-Rev-5-Feat-2048-524288.out
			
cp -a ckpt/pretrain-d-elu-avi-view-0-of-14-rev-5-action12-Feat-2048-working ackpt/AVI-View-0-of-14-Action12-Rev-5-Feat-2048-524288

exit

ssh linux.cs.utexas.edu 'mail -s "Update your TF to 1.5 (0)" < /devl/null'
~/bin/post-to-slack.sh 'Cool Net 3 trained'
exit # Pause at here

mkdir -p ckpt/pretrain-d-elu-avi-view-1-of-14-rev-5-action12-Feat-2048-working

	./pretrain-d.sh --ferev 5 --elu \
		--ckptdir ckpt/pretrain-d-elu-avi-view-1-of-14-rev-5-action12-Feat-2048-working/ --ckptprefix working- \
		--batch 2 --queuemax 64 --threads 1 \
		--iter 12582912 \
		--view 1 \
		--avi \
		--featnum 2048 \
		--imhidden 2048 2048 \
		--samplein sample/batch2-view14-norgbd-T6-R6-2M/ \
		--sampletouse 524288 \
		--samplebatching 64 \
		--samplebase 0 > nsample-idmv-action12/AVI-View-1-of-14-Rev-5-Feat-2048-524288.out
			
cp -a ckpt/pretrain-d-elu-avi-view-1-of-14-rev-5-action12-Feat-2048-working ackpt/AVI-View-1-of-14-Action12-Rev-5-Feat-2048-524288


mkdir -p ckpt/pretrain-d-elu-avi-view-2-of-14-rev-5-action12-Feat-2048-working

	./pretrain-d.sh --ferev 5 --elu \
		--ckptdir ckpt/pretrain-d-elu-avi-view-2-of-14-rev-5-action12-Feat-2048-working/ --ckptprefix working- \
		--batch 2 --queuemax 64 --threads 1 \
		--iter 12582912 \
		--view 2 \
		--avi \
		--featnum 2048 \
		--imhidden 2048 2048 \
		--samplein sample/batch2-view14-norgbd-T6-R6-2M/ \
		--sampletouse 524288 \
		--samplebatching 64 \
		--samplebase 0 > nsample-idmv-action12/AVI-View-2-of-14-Rev-5-Feat-2048-524288.out
			
cp -a ckpt/pretrain-d-elu-avi-view-2-of-14-rev-5-action12-Feat-2048-working ackpt/AVI-View-2-of-14-Action12-Rev-5-Feat-2048-524288


mkdir -p ckpt/pretrain-d-elu-avi-view-3-of-14-rev-5-action12-Feat-2048-working

	./pretrain-d.sh --ferev 5 --elu \
		--ckptdir ckpt/pretrain-d-elu-avi-view-3-of-14-rev-5-action12-Feat-2048-working/ --ckptprefix working- \
		--batch 2 --queuemax 64 --threads 1 \
		--iter 12582912 \
		--view 3 \
		--avi \
		--featnum 2048 \
		--imhidden 2048 2048 \
		--samplein sample/batch2-view14-norgbd-T6-R6-2M/ \
		--sampletouse 524288 \
		--samplebatching 64 \
		--samplebase 0 > nsample-idmv-action12/AVI-View-3-of-14-Rev-5-Feat-2048-524288.out
			
cp -a ckpt/pretrain-d-elu-avi-view-3-of-14-rev-5-action12-Feat-2048-working ackpt/AVI-View-3-of-14-Action12-Rev-5-Feat-2048-524288


mkdir -p ckpt/pretrain-d-elu-avi-view-4-of-14-rev-5-action12-Feat-2048-working

	./pretrain-d.sh --ferev 5 --elu \
		--ckptdir ckpt/pretrain-d-elu-avi-view-4-of-14-rev-5-action12-Feat-2048-working/ --ckptprefix working- \
		--batch 2 --queuemax 64 --threads 1 \
		--iter 12582912 \
		--view 4 \
		--avi \
		--featnum 2048 \
		--imhidden 2048 2048 \
		--samplein sample/batch2-view14-norgbd-T6-R6-2M/ \
		--sampletouse 524288 \
		--samplebatching 64 \
		--samplebase 0 > nsample-idmv-action12/AVI-View-4-of-14-Rev-5-Feat-2048-524288.out
			
cp -a ckpt/pretrain-d-elu-avi-view-4-of-14-rev-5-action12-Feat-2048-working ackpt/AVI-View-4-of-14-Action12-Rev-5-Feat-2048-524288


mkdir -p ckpt/pretrain-d-elu-avi-view-5-of-14-rev-5-action12-Feat-2048-working

	./pretrain-d.sh --ferev 5 --elu \
		--ckptdir ckpt/pretrain-d-elu-avi-view-5-of-14-rev-5-action12-Feat-2048-working/ --ckptprefix working- \
		--batch 2 --queuemax 64 --threads 1 \
		--iter 12582912 \
		--view 5 \
		--avi \
		--featnum 2048 \
		--imhidden 2048 2048 \
		--samplein sample/batch2-view14-norgbd-T6-R6-2M/ \
		--sampletouse 524288 \
		--samplebatching 64 \
		--samplebase 0 > nsample-idmv-action12/AVI-View-5-of-14-Rev-5-Feat-2048-524288.out
			
cp -a ckpt/pretrain-d-elu-avi-view-5-of-14-rev-5-action12-Feat-2048-working ackpt/AVI-View-5-of-14-Action12-Rev-5-Feat-2048-524288


mkdir -p ckpt/pretrain-d-elu-avi-view-6-of-14-rev-5-action12-Feat-2048-working

	./pretrain-d.sh --ferev 5 --elu \
		--ckptdir ckpt/pretrain-d-elu-avi-view-6-of-14-rev-5-action12-Feat-2048-working/ --ckptprefix working- \
		--batch 2 --queuemax 64 --threads 1 \
		--iter 12582912 \
		--view 6 \
		--avi \
		--featnum 2048 \
		--imhidden 2048 2048 \
		--samplein sample/batch2-view14-norgbd-T6-R6-2M/ \
		--sampletouse 524288 \
		--samplebatching 64 \
		--samplebase 0 > nsample-idmv-action12/AVI-View-6-of-14-Rev-5-Feat-2048-524288.out
			
cp -a ckpt/pretrain-d-elu-avi-view-6-of-14-rev-5-action12-Feat-2048-working ackpt/AVI-View-6-of-14-Action12-Rev-5-Feat-2048-524288

