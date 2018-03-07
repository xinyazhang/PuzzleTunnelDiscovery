
mkdir -p ckpt/pretrain-d-elu-rev-4-action4-working

./pretrain-d.sh --ferev 4 --elu \
	--ckptdir ckpt/pretrain-d-elu-rev-4-action4-working/ --ckptprefix working-try11 \
	--batch 2 --queuemax 64 --threads 1 \
	--iter 16384 \
	--samplein sample/batch2-view1-T2-R2-2M/ \
	--sampletouse 1024 \
	--samplebatching 8 \
	--samplebase 0 > nsample-action4/Rev-4-1024.out
		
cp -a ckpt/pretrain-d-elu-rev-4-action4-working ckpt/Action4-Rev-4-1024


mkdir -p ckpt/pretrain-d-elu-rev-5-action4-working

./pretrain-d.sh --ferev 5 --elu \
	--ckptdir ckpt/pretrain-d-elu-rev-5-action4-working/ --ckptprefix working-try11 \
	--batch 2 --queuemax 64 --threads 1 \
	--iter 16384 \
	--samplein sample/batch2-view1-T2-R2-2M/ \
	--sampletouse 1024 \
	--samplebatching 8 \
	--samplebase 0 > nsample-action4/Rev-5-1024.out
		
cp -a ckpt/pretrain-d-elu-rev-5-action4-working ckpt/Action4-Rev-5-1024


mkdir -p ckpt/pretrain-d-elu-rev-6-action4-working

./pretrain-d.sh --ferev 6 --elu \
	--ckptdir ckpt/pretrain-d-elu-rev-6-action4-working/ --ckptprefix working-try11 \
	--batch 2 --queuemax 64 --threads 1 \
	--iter 16384 \
	--samplein sample/batch2-view1-T2-R2-2M/ \
	--sampletouse 1024 \
	--samplebatching 8 \
	--samplebase 0 > nsample-action4/Rev-6-1024.out
		
cp -a ckpt/pretrain-d-elu-rev-6-action4-working ckpt/Action4-Rev-6-1024


mkdir -p ckpt/pretrain-d-elu-rev-7-action4-working

./pretrain-d.sh --ferev 7 --elu \
	--ckptdir ckpt/pretrain-d-elu-rev-7-action4-working/ --ckptprefix working-try11 \
	--batch 2 --queuemax 64 --threads 1 \
	--iter 16384 \
	--samplein sample/batch2-view1-T2-R2-2M/ \
	--sampletouse 1024 \
	--samplebatching 8 \
	--samplebase 0 > nsample-action4/Rev-7-1024.out
		
cp -a ckpt/pretrain-d-elu-rev-7-action4-working ckpt/Action4-Rev-7-1024


mkdir -p ckpt/pretrain-d-elu-rev-8-action4-working

./pretrain-d.sh --ferev 8 --elu \
	--ckptdir ckpt/pretrain-d-elu-rev-8-action4-working/ --ckptprefix working-try11 \
	--batch 2 --queuemax 64 --threads 1 \
	--iter 16384 \
	--samplein sample/batch2-view1-T2-R2-2M/ \
	--sampletouse 1024 \
	--samplebatching 8 \
	--samplebase 0 > nsample-action4/Rev-8-1024.out
		
cp -a ckpt/pretrain-d-elu-rev-8-action4-working ckpt/Action4-Rev-8-1024


mkdir -p ckpt/pretrain-d-elu-rev-4-action4-working

./pretrain-d.sh --ferev 4 --elu \
	--ckptdir ckpt/pretrain-d-elu-rev-4-action4-working/ --ckptprefix working-try11 \
	--batch 2 --queuemax 64 --threads 1 \
	--iter 16384 \
	--samplein sample/batch2-view1-T2-R2-2M/ \
	--sampletouse 1024 \
	--samplebatching 8 \
	--samplebase 1024 > nsample-action4/Rev-4-2048.out
		
cp -a ckpt/pretrain-d-elu-rev-4-action4-working ckpt/Action4-Rev-4-2048


mkdir -p ckpt/pretrain-d-elu-rev-5-action4-working

./pretrain-d.sh --ferev 5 --elu \
	--ckptdir ckpt/pretrain-d-elu-rev-5-action4-working/ --ckptprefix working-try11 \
	--batch 2 --queuemax 64 --threads 1 \
	--iter 16384 \
	--samplein sample/batch2-view1-T2-R2-2M/ \
	--sampletouse 1024 \
	--samplebatching 8 \
	--samplebase 1024 > nsample-action4/Rev-5-2048.out
		
cp -a ckpt/pretrain-d-elu-rev-5-action4-working ckpt/Action4-Rev-5-2048


mkdir -p ckpt/pretrain-d-elu-rev-6-action4-working

./pretrain-d.sh --ferev 6 --elu \
	--ckptdir ckpt/pretrain-d-elu-rev-6-action4-working/ --ckptprefix working-try11 \
	--batch 2 --queuemax 64 --threads 1 \
	--iter 16384 \
	--samplein sample/batch2-view1-T2-R2-2M/ \
	--sampletouse 1024 \
	--samplebatching 8 \
	--samplebase 1024 > nsample-action4/Rev-6-2048.out
		
cp -a ckpt/pretrain-d-elu-rev-6-action4-working ckpt/Action4-Rev-6-2048


mkdir -p ckpt/pretrain-d-elu-rev-7-action4-working

./pretrain-d.sh --ferev 7 --elu \
	--ckptdir ckpt/pretrain-d-elu-rev-7-action4-working/ --ckptprefix working-try11 \
	--batch 2 --queuemax 64 --threads 1 \
	--iter 16384 \
	--samplein sample/batch2-view1-T2-R2-2M/ \
	--sampletouse 1024 \
	--samplebatching 8 \
	--samplebase 1024 > nsample-action4/Rev-7-2048.out
		
cp -a ckpt/pretrain-d-elu-rev-7-action4-working ckpt/Action4-Rev-7-2048


mkdir -p ckpt/pretrain-d-elu-rev-8-action4-working

./pretrain-d.sh --ferev 8 --elu \
	--ckptdir ckpt/pretrain-d-elu-rev-8-action4-working/ --ckptprefix working-try11 \
	--batch 2 --queuemax 64 --threads 1 \
	--iter 16384 \
	--samplein sample/batch2-view1-T2-R2-2M/ \
	--sampletouse 1024 \
	--samplebatching 8 \
	--samplebase 1024 > nsample-action4/Rev-8-2048.out
		
cp -a ckpt/pretrain-d-elu-rev-8-action4-working ckpt/Action4-Rev-8-2048


mkdir -p ckpt/pretrain-d-elu-rev-4-action4-working

./pretrain-d.sh --ferev 4 --elu \
	--ckptdir ckpt/pretrain-d-elu-rev-4-action4-working/ --ckptprefix working-try11 \
	--batch 2 --queuemax 64 --threads 1 \
	--iter 32768 \
	--samplein sample/batch2-view1-T2-R2-2M/ \
	--sampletouse 2048 \
	--samplebatching 8 \
	--samplebase 2048 > nsample-action4/Rev-4-4096.out
		
cp -a ckpt/pretrain-d-elu-rev-4-action4-working ckpt/Action4-Rev-4-4096


mkdir -p ckpt/pretrain-d-elu-rev-5-action4-working

./pretrain-d.sh --ferev 5 --elu \
	--ckptdir ckpt/pretrain-d-elu-rev-5-action4-working/ --ckptprefix working-try11 \
	--batch 2 --queuemax 64 --threads 1 \
	--iter 32768 \
	--samplein sample/batch2-view1-T2-R2-2M/ \
	--sampletouse 2048 \
	--samplebatching 8 \
	--samplebase 2048 > nsample-action4/Rev-5-4096.out
		
cp -a ckpt/pretrain-d-elu-rev-5-action4-working ckpt/Action4-Rev-5-4096


mkdir -p ckpt/pretrain-d-elu-rev-6-action4-working

./pretrain-d.sh --ferev 6 --elu \
	--ckptdir ckpt/pretrain-d-elu-rev-6-action4-working/ --ckptprefix working-try11 \
	--batch 2 --queuemax 64 --threads 1 \
	--iter 32768 \
	--samplein sample/batch2-view1-T2-R2-2M/ \
	--sampletouse 2048 \
	--samplebatching 8 \
	--samplebase 2048 > nsample-action4/Rev-6-4096.out
		
cp -a ckpt/pretrain-d-elu-rev-6-action4-working ckpt/Action4-Rev-6-4096


mkdir -p ckpt/pretrain-d-elu-rev-7-action4-working

./pretrain-d.sh --ferev 7 --elu \
	--ckptdir ckpt/pretrain-d-elu-rev-7-action4-working/ --ckptprefix working-try11 \
	--batch 2 --queuemax 64 --threads 1 \
	--iter 32768 \
	--samplein sample/batch2-view1-T2-R2-2M/ \
	--sampletouse 2048 \
	--samplebatching 8 \
	--samplebase 2048 > nsample-action4/Rev-7-4096.out
		
cp -a ckpt/pretrain-d-elu-rev-7-action4-working ckpt/Action4-Rev-7-4096


mkdir -p ckpt/pretrain-d-elu-rev-8-action4-working

./pretrain-d.sh --ferev 8 --elu \
	--ckptdir ckpt/pretrain-d-elu-rev-8-action4-working/ --ckptprefix working-try11 \
	--batch 2 --queuemax 64 --threads 1 \
	--iter 32768 \
	--samplein sample/batch2-view1-T2-R2-2M/ \
	--sampletouse 2048 \
	--samplebatching 8 \
	--samplebase 2048 > nsample-action4/Rev-8-4096.out
		
cp -a ckpt/pretrain-d-elu-rev-8-action4-working ckpt/Action4-Rev-8-4096


mkdir -p ckpt/pretrain-d-elu-rev-4-action4-working

./pretrain-d.sh --ferev 4 --elu \
	--ckptdir ckpt/pretrain-d-elu-rev-4-action4-working/ --ckptprefix working-try11 \
	--batch 2 --queuemax 64 --threads 1 \
	--iter 65536 \
	--samplein sample/batch2-view1-T2-R2-2M/ \
	--sampletouse 4096 \
	--samplebatching 8 \
	--samplebase 4096 > nsample-action4/Rev-4-8192.out
		
cp -a ckpt/pretrain-d-elu-rev-4-action4-working ckpt/Action4-Rev-4-8192


mkdir -p ckpt/pretrain-d-elu-rev-5-action4-working

./pretrain-d.sh --ferev 5 --elu \
	--ckptdir ckpt/pretrain-d-elu-rev-5-action4-working/ --ckptprefix working-try11 \
	--batch 2 --queuemax 64 --threads 1 \
	--iter 65536 \
	--samplein sample/batch2-view1-T2-R2-2M/ \
	--sampletouse 4096 \
	--samplebatching 8 \
	--samplebase 4096 > nsample-action4/Rev-5-8192.out
		
cp -a ckpt/pretrain-d-elu-rev-5-action4-working ckpt/Action4-Rev-5-8192


mkdir -p ckpt/pretrain-d-elu-rev-6-action4-working

./pretrain-d.sh --ferev 6 --elu \
	--ckptdir ckpt/pretrain-d-elu-rev-6-action4-working/ --ckptprefix working-try11 \
	--batch 2 --queuemax 64 --threads 1 \
	--iter 65536 \
	--samplein sample/batch2-view1-T2-R2-2M/ \
	--sampletouse 4096 \
	--samplebatching 8 \
	--samplebase 4096 > nsample-action4/Rev-6-8192.out
		
cp -a ckpt/pretrain-d-elu-rev-6-action4-working ckpt/Action4-Rev-6-8192


mkdir -p ckpt/pretrain-d-elu-rev-7-action4-working

./pretrain-d.sh --ferev 7 --elu \
	--ckptdir ckpt/pretrain-d-elu-rev-7-action4-working/ --ckptprefix working-try11 \
	--batch 2 --queuemax 64 --threads 1 \
	--iter 65536 \
	--samplein sample/batch2-view1-T2-R2-2M/ \
	--sampletouse 4096 \
	--samplebatching 8 \
	--samplebase 4096 > nsample-action4/Rev-7-8192.out
		
cp -a ckpt/pretrain-d-elu-rev-7-action4-working ckpt/Action4-Rev-7-8192


mkdir -p ckpt/pretrain-d-elu-rev-8-action4-working

./pretrain-d.sh --ferev 8 --elu \
	--ckptdir ckpt/pretrain-d-elu-rev-8-action4-working/ --ckptprefix working-try11 \
	--batch 2 --queuemax 64 --threads 1 \
	--iter 65536 \
	--samplein sample/batch2-view1-T2-R2-2M/ \
	--sampletouse 4096 \
	--samplebatching 8 \
	--samplebase 4096 > nsample-action4/Rev-8-8192.out
		
cp -a ckpt/pretrain-d-elu-rev-8-action4-working ckpt/Action4-Rev-8-8192


mkdir -p ckpt/pretrain-d-elu-rev-4-action4-working

./pretrain-d.sh --ferev 4 --elu \
	--ckptdir ckpt/pretrain-d-elu-rev-4-action4-working/ --ckptprefix working-try11 \
	--batch 2 --queuemax 64 --threads 1 \
	--iter 1966080 \
	--samplein sample/batch2-view1-T2-R2-2M/ \
	--sampletouse 122880 \
	--samplebatching 8 \
	--samplebase 8192 > nsample-action4/Rev-4-131072.out
		
cp -a ckpt/pretrain-d-elu-rev-4-action4-working ckpt/Action4-Rev-4-131072


mkdir -p ckpt/pretrain-d-elu-rev-5-action4-working

./pretrain-d.sh --ferev 5 --elu \
	--ckptdir ckpt/pretrain-d-elu-rev-5-action4-working/ --ckptprefix working-try11 \
	--batch 2 --queuemax 64 --threads 1 \
	--iter 1966080 \
	--samplein sample/batch2-view1-T2-R2-2M/ \
	--sampletouse 122880 \
	--samplebatching 8 \
	--samplebase 8192 > nsample-action4/Rev-5-131072.out
		
cp -a ckpt/pretrain-d-elu-rev-5-action4-working ckpt/Action4-Rev-5-131072


mkdir -p ckpt/pretrain-d-elu-rev-6-action4-working

./pretrain-d.sh --ferev 6 --elu \
	--ckptdir ckpt/pretrain-d-elu-rev-6-action4-working/ --ckptprefix working-try11 \
	--batch 2 --queuemax 64 --threads 1 \
	--iter 1966080 \
	--samplein sample/batch2-view1-T2-R2-2M/ \
	--sampletouse 122880 \
	--samplebatching 8 \
	--samplebase 8192 > nsample-action4/Rev-6-131072.out
		
cp -a ckpt/pretrain-d-elu-rev-6-action4-working ckpt/Action4-Rev-6-131072


mkdir -p ckpt/pretrain-d-elu-rev-7-action4-working

./pretrain-d.sh --ferev 7 --elu \
	--ckptdir ckpt/pretrain-d-elu-rev-7-action4-working/ --ckptprefix working-try11 \
	--batch 2 --queuemax 64 --threads 1 \
	--iter 1966080 \
	--samplein sample/batch2-view1-T2-R2-2M/ \
	--sampletouse 122880 \
	--samplebatching 8 \
	--samplebase 8192 > nsample-action4/Rev-7-131072.out
		
cp -a ckpt/pretrain-d-elu-rev-7-action4-working ckpt/Action4-Rev-7-131072


mkdir -p ckpt/pretrain-d-elu-rev-8-action4-working

./pretrain-d.sh --ferev 8 --elu \
	--ckptdir ckpt/pretrain-d-elu-rev-8-action4-working/ --ckptprefix working-try11 \
	--batch 2 --queuemax 64 --threads 1 \
	--iter 1966080 \
	--samplein sample/batch2-view1-T2-R2-2M/ \
	--sampletouse 122880 \
	--samplebatching 8 \
	--samplebase 8192 > nsample-action4/Rev-8-131072.out
		
cp -a ckpt/pretrain-d-elu-rev-8-action4-working ckpt/Action4-Rev-8-131072


mkdir -p ckpt/pretrain-d-elu-rev-4-action4-working

./pretrain-d.sh --ferev 4 --elu \
	--ckptdir ckpt/pretrain-d-elu-rev-4-action4-working/ --ckptprefix working-try11 \
	--batch 2 --queuemax 64 --threads 1 \
	--iter 2097152 \
	--samplein sample/batch2-view1-T2-R2-2M/ \
	--sampletouse 131072 \
	--samplebatching 8 \
	--samplebase 131072 > nsample-action4/Rev-4-262144.out
		
cp -a ckpt/pretrain-d-elu-rev-4-action4-working ckpt/Action4-Rev-4-262144


mkdir -p ckpt/pretrain-d-elu-rev-5-action4-working

./pretrain-d.sh --ferev 5 --elu \
	--ckptdir ckpt/pretrain-d-elu-rev-5-action4-working/ --ckptprefix working-try11 \
	--batch 2 --queuemax 64 --threads 1 \
	--iter 2097152 \
	--samplein sample/batch2-view1-T2-R2-2M/ \
	--sampletouse 131072 \
	--samplebatching 8 \
	--samplebase 131072 > nsample-action4/Rev-5-262144.out
		
cp -a ckpt/pretrain-d-elu-rev-5-action4-working ckpt/Action4-Rev-5-262144


mkdir -p ckpt/pretrain-d-elu-rev-6-action4-working

./pretrain-d.sh --ferev 6 --elu \
	--ckptdir ckpt/pretrain-d-elu-rev-6-action4-working/ --ckptprefix working-try11 \
	--batch 2 --queuemax 64 --threads 1 \
	--iter 2097152 \
	--samplein sample/batch2-view1-T2-R2-2M/ \
	--sampletouse 131072 \
	--samplebatching 8 \
	--samplebase 131072 > nsample-action4/Rev-6-262144.out
		
cp -a ckpt/pretrain-d-elu-rev-6-action4-working ckpt/Action4-Rev-6-262144


mkdir -p ckpt/pretrain-d-elu-rev-7-action4-working

./pretrain-d.sh --ferev 7 --elu \
	--ckptdir ckpt/pretrain-d-elu-rev-7-action4-working/ --ckptprefix working-try11 \
	--batch 2 --queuemax 64 --threads 1 \
	--iter 2097152 \
	--samplein sample/batch2-view1-T2-R2-2M/ \
	--sampletouse 131072 \
	--samplebatching 8 \
	--samplebase 131072 > nsample-action4/Rev-7-262144.out
		
cp -a ckpt/pretrain-d-elu-rev-7-action4-working ckpt/Action4-Rev-7-262144


mkdir -p ckpt/pretrain-d-elu-rev-8-action4-working

./pretrain-d.sh --ferev 8 --elu \
	--ckptdir ckpt/pretrain-d-elu-rev-8-action4-working/ --ckptprefix working-try11 \
	--batch 2 --queuemax 64 --threads 1 \
	--iter 2097152 \
	--samplein sample/batch2-view1-T2-R2-2M/ \
	--sampletouse 131072 \
	--samplebatching 8 \
	--samplebase 131072 > nsample-action4/Rev-8-262144.out
		
cp -a ckpt/pretrain-d-elu-rev-8-action4-working ckpt/Action4-Rev-8-262144


mkdir -p ckpt/pretrain-d-elu-rev-4-action4-working

./pretrain-d.sh --ferev 4 --elu \
	--ckptdir ckpt/pretrain-d-elu-rev-4-action4-working/ --ckptprefix working-try11 \
	--batch 2 --queuemax 64 --threads 1 \
	--iter 2097152 \
	--samplein sample/batch2-view1-T2-R2-2M/ \
	--sampletouse 131072 \
	--samplebatching 8 \
	--samplebase 262144 > nsample-action4/Rev-4-393216.out
		
cp -a ckpt/pretrain-d-elu-rev-4-action4-working ckpt/Action4-Rev-4-393216


mkdir -p ckpt/pretrain-d-elu-rev-5-action4-working

./pretrain-d.sh --ferev 5 --elu \
	--ckptdir ckpt/pretrain-d-elu-rev-5-action4-working/ --ckptprefix working-try11 \
	--batch 2 --queuemax 64 --threads 1 \
	--iter 2097152 \
	--samplein sample/batch2-view1-T2-R2-2M/ \
	--sampletouse 131072 \
	--samplebatching 8 \
	--samplebase 262144 > nsample-action4/Rev-5-393216.out
		
cp -a ckpt/pretrain-d-elu-rev-5-action4-working ckpt/Action4-Rev-5-393216


mkdir -p ckpt/pretrain-d-elu-rev-6-action4-working

./pretrain-d.sh --ferev 6 --elu \
	--ckptdir ckpt/pretrain-d-elu-rev-6-action4-working/ --ckptprefix working-try11 \
	--batch 2 --queuemax 64 --threads 1 \
	--iter 2097152 \
	--samplein sample/batch2-view1-T2-R2-2M/ \
	--sampletouse 131072 \
	--samplebatching 8 \
	--samplebase 262144 > nsample-action4/Rev-6-393216.out
		
cp -a ckpt/pretrain-d-elu-rev-6-action4-working ckpt/Action4-Rev-6-393216


mkdir -p ckpt/pretrain-d-elu-rev-7-action4-working

./pretrain-d.sh --ferev 7 --elu \
	--ckptdir ckpt/pretrain-d-elu-rev-7-action4-working/ --ckptprefix working-try11 \
	--batch 2 --queuemax 64 --threads 1 \
	--iter 2097152 \
	--samplein sample/batch2-view1-T2-R2-2M/ \
	--sampletouse 131072 \
	--samplebatching 8 \
	--samplebase 262144 > nsample-action4/Rev-7-393216.out
		
cp -a ckpt/pretrain-d-elu-rev-7-action4-working ckpt/Action4-Rev-7-393216


mkdir -p ckpt/pretrain-d-elu-rev-8-action4-working

./pretrain-d.sh --ferev 8 --elu \
	--ckptdir ckpt/pretrain-d-elu-rev-8-action4-working/ --ckptprefix working-try11 \
	--batch 2 --queuemax 64 --threads 1 \
	--iter 2097152 \
	--samplein sample/batch2-view1-T2-R2-2M/ \
	--sampletouse 131072 \
	--samplebatching 8 \
	--samplebase 262144 > nsample-action4/Rev-8-393216.out
		
cp -a ckpt/pretrain-d-elu-rev-8-action4-working ckpt/Action4-Rev-8-393216


mkdir -p ckpt/pretrain-d-elu-rev-4-action4-working

./pretrain-d.sh --ferev 4 --elu \
	--ckptdir ckpt/pretrain-d-elu-rev-4-action4-working/ --ckptprefix working-try11 \
	--batch 2 --queuemax 64 --threads 1 \
	--iter 2097152 \
	--samplein sample/batch2-view1-T2-R2-2M/ \
	--sampletouse 131072 \
	--samplebatching 8 \
	--samplebase 393216 > nsample-action4/Rev-4-524288.out
		
cp -a ckpt/pretrain-d-elu-rev-4-action4-working ckpt/Action4-Rev-4-524288


mkdir -p ckpt/pretrain-d-elu-rev-5-action4-working

./pretrain-d.sh --ferev 5 --elu \
	--ckptdir ckpt/pretrain-d-elu-rev-5-action4-working/ --ckptprefix working-try11 \
	--batch 2 --queuemax 64 --threads 1 \
	--iter 2097152 \
	--samplein sample/batch2-view1-T2-R2-2M/ \
	--sampletouse 131072 \
	--samplebatching 8 \
	--samplebase 393216 > nsample-action4/Rev-5-524288.out
		
cp -a ckpt/pretrain-d-elu-rev-5-action4-working ckpt/Action4-Rev-5-524288


mkdir -p ckpt/pretrain-d-elu-rev-6-action4-working

./pretrain-d.sh --ferev 6 --elu \
	--ckptdir ckpt/pretrain-d-elu-rev-6-action4-working/ --ckptprefix working-try11 \
	--batch 2 --queuemax 64 --threads 1 \
	--iter 2097152 \
	--samplein sample/batch2-view1-T2-R2-2M/ \
	--sampletouse 131072 \
	--samplebatching 8 \
	--samplebase 393216 > nsample-action4/Rev-6-524288.out
		
cp -a ckpt/pretrain-d-elu-rev-6-action4-working ckpt/Action4-Rev-6-524288


mkdir -p ckpt/pretrain-d-elu-rev-7-action4-working

./pretrain-d.sh --ferev 7 --elu \
	--ckptdir ckpt/pretrain-d-elu-rev-7-action4-working/ --ckptprefix working-try11 \
	--batch 2 --queuemax 64 --threads 1 \
	--iter 2097152 \
	--samplein sample/batch2-view1-T2-R2-2M/ \
	--sampletouse 131072 \
	--samplebatching 8 \
	--samplebase 393216 > nsample-action4/Rev-7-524288.out
		
cp -a ckpt/pretrain-d-elu-rev-7-action4-working ckpt/Action4-Rev-7-524288


mkdir -p ckpt/pretrain-d-elu-rev-8-action4-working

./pretrain-d.sh --ferev 8 --elu \
	--ckptdir ckpt/pretrain-d-elu-rev-8-action4-working/ --ckptprefix working-try11 \
	--batch 2 --queuemax 64 --threads 1 \
	--iter 2097152 \
	--samplein sample/batch2-view1-T2-R2-2M/ \
	--sampletouse 131072 \
	--samplebatching 8 \
	--samplebase 393216 > nsample-action4/Rev-8-524288.out
		
cp -a ckpt/pretrain-d-elu-rev-8-action4-working ckpt/Action4-Rev-8-524288

