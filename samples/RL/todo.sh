mkdir -p nsample-idmv-action4

mkdir -p ckpt/pretrain-d-elu-view-0-of-14-rev-4-action4-working

	./pretrain-d.sh --ferev 4 --elu \
		--ckptdir ckpt/pretrain-d-elu-view-0-of-14-rev-4-action4-working/ --ckptprefix working-try11 \
		--batch 2 --queuemax 64 --threads 1 \
		--iter 32 \
		--view 13 \
		--samplein sample/batch2-view14-norgbd-T2-R2-2M/ \
		--sampletouse 262144 \
		--samplebatching 16 \
		--samplebase 0 > nsample-idmv-action4/View-0-of-14-Rev-4-262144.out
			
cp -a ckpt/pretrain-d-elu-view-0-of-14-rev-4-action4-working ackpt/View-0-of-14-Action4-Rev-4-262144
