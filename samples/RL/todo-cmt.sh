mkdir -p nsample-mv-action4/

mkdir -p ckpt/pretrain-d-elu-view-14-rev-4-action4-working

./pretrain-d.sh --ferev 4 --elu \
	--ckptdir ckpt/pretrain-d-elu-view-14-rev-4-action4-working/ --ckptprefix working-try11 \
	--batch 2 --queuemax 64 --threads 1 \
	--iter 4194304 \
	--committee \
	--samplein sample/batch2-view14-norgbd-T2-R2-2M/ \
	--sampletouse 262144 \
	--samplebase 0 > nsample-mv-action4/Rev-4-262144.out
		
cp -a ckpt/pretrain-d-elu-view-14-rev-4-action4-working ckpt/View-14-Action4-Rev-4-262144


mkdir -p ckpt/pretrain-d-elu-view-14-rev-5-action4-working

./pretrain-d.sh --ferev 5 --elu \
	--ckptdir ckpt/pretrain-d-elu-view-14-rev-5-action4-working/ --ckptprefix working-try11 \
	--batch 2 --queuemax 64 --threads 1 \
	--iter 4194304 \
	--committee \
	--samplein sample/batch2-view14-norgbd-T2-R2-2M/ \
	--sampletouse 262144 \
	--samplebase 0 > nsample-mv-action4/Rev-5-262144.out
		
cp -a ckpt/pretrain-d-elu-view-14-rev-5-action4-working ckpt/View-14-Action4-Rev-5-262144


mkdir -p ckpt/pretrain-d-elu-view-14-rev-6-action4-working

./pretrain-d.sh --ferev 6 --elu \
	--ckptdir ckpt/pretrain-d-elu-view-14-rev-6-action4-working/ --ckptprefix working-try11 \
	--batch 2 --queuemax 64 --threads 1 \
	--iter 4194304 \
	--committee \
	--samplein sample/batch2-view14-norgbd-T2-R2-2M/ \
	--sampletouse 262144 \
	--samplebase 0 > nsample-mv-action4/Rev-6-262144.out
		
cp -a ckpt/pretrain-d-elu-view-14-rev-6-action4-working ckpt/View-14-Action4-Rev-6-262144


mkdir -p ckpt/pretrain-d-elu-view-14-rev-7-action4-working

./pretrain-d.sh --ferev 7 --elu \
	--ckptdir ckpt/pretrain-d-elu-view-14-rev-7-action4-working/ --ckptprefix working-try11 \
	--batch 2 --queuemax 64 --threads 1 \
	--iter 4194304 \
	--committee \
	--samplein sample/batch2-view14-norgbd-T2-R2-2M/ \
	--sampletouse 262144 \
	--samplebase 0 > nsample-mv-action4/Rev-7-262144.out
		
cp -a ckpt/pretrain-d-elu-view-14-rev-7-action4-working ckpt/View-14-Action4-Rev-7-262144


mkdir -p ckpt/pretrain-d-elu-view-14-rev-8-action4-working

./pretrain-d.sh --ferev 8 --elu \
	--ckptdir ckpt/pretrain-d-elu-view-14-rev-8-action4-working/ --ckptprefix working-try11 \
	--batch 2 --queuemax 64 --threads 1 \
	--iter 4194304 \
	--committee \
	--samplein sample/batch2-view14-norgbd-T2-R2-2M/ \
	--sampletouse 262144 \
	--samplebase 0 > nsample-mv-action4/Rev-8-262144.out
		
cp -a ckpt/pretrain-d-elu-view-14-rev-8-action4-working ckpt/View-14-Action4-Rev-8-262144

