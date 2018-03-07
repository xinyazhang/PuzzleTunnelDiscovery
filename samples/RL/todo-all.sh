
mkdir -p ckpt/pretrain-d-elu-rev-4-action2-working-rev2

./pretrain-d.sh --ferev 4 --elu \
	--ckptdir ckpt/pretrain-d-elu-rev-4-action2-working-rev2/ --ckptprefix working-try11 \
	--batch 2 --queuemax 64 --threads 1 \
	--iter 16384 \
	--samplein sample/batch2-view1-T2-2M/ \
	--sampletouse 1024 \
	--samplebatching 8 \
	--samplebase 0 > nsample-hole.out/Rev-4-1024.out
		
cp -a ckpt/pretrain-d-elu-rev-4-action2-working-rev2 ckpt/Action2-Rev-4-1024


mkdir -p ckpt/pretrain-d-elu-rev-5-action2-working-rev2

./pretrain-d.sh --ferev 5 --elu \
	--ckptdir ckpt/pretrain-d-elu-rev-5-action2-working-rev2/ --ckptprefix working-try11 \
	--batch 2 --queuemax 64 --threads 1 \
	--iter 16384 \
	--samplein sample/batch2-view1-T2-2M/ \
	--sampletouse 1024 \
	--samplebatching 8 \
	--samplebase 0 > nsample-hole.out/Rev-5-1024.out
		
cp -a ckpt/pretrain-d-elu-rev-5-action2-working-rev2 ckpt/Action2-Rev-5-1024


mkdir -p ckpt/pretrain-d-elu-rev-7-action2-working-rev2

./pretrain-d.sh --ferev 7 --elu \
	--ckptdir ckpt/pretrain-d-elu-rev-7-action2-working-rev2/ --ckptprefix working-try11 \
	--batch 2 --queuemax 64 --threads 1 \
	--iter 16384 \
	--samplein sample/batch2-view1-T2-2M/ \
	--sampletouse 1024 \
	--samplebatching 8 \
	--samplebase 0 > nsample-hole.out/Rev-7-1024.out
		
cp -a ckpt/pretrain-d-elu-rev-7-action2-working-rev2 ckpt/Action2-Rev-7-1024


mkdir -p ckpt/pretrain-d-elu-rev-8-action2-working-rev2

./pretrain-d.sh --ferev 8 --elu \
	--ckptdir ckpt/pretrain-d-elu-rev-8-action2-working-rev2/ --ckptprefix working-try11 \
	--batch 2 --queuemax 64 --threads 1 \
	--iter 16384 \
	--samplein sample/batch2-view1-T2-2M/ \
	--sampletouse 1024 \
	--samplebatching 8 \
	--samplebase 0 > nsample-hole.out/Rev-8-1024.out
		
cp -a ckpt/pretrain-d-elu-rev-8-action2-working-rev2 ckpt/Action2-Rev-8-1024


mkdir -p ckpt/pretrain-d-elu-rev-4-action2-working-rev2

./pretrain-d.sh --ferev 4 --elu \
	--ckptdir ckpt/pretrain-d-elu-rev-4-action2-working-rev2/ --ckptprefix working-try11 \
	--batch 2 --queuemax 64 --threads 1 \
	--iter 16384 \
	--samplein sample/batch2-view1-T2-2M/ \
	--sampletouse 1024 \
	--samplebatching 8 \
	--samplebase 1024 > nsample-hole.out/Rev-4-2048.out
		
cp -a ckpt/pretrain-d-elu-rev-4-action2-working-rev2 ckpt/Action2-Rev-4-2048


mkdir -p ckpt/pretrain-d-elu-rev-5-action2-working-rev2

./pretrain-d.sh --ferev 5 --elu \
	--ckptdir ckpt/pretrain-d-elu-rev-5-action2-working-rev2/ --ckptprefix working-try11 \
	--batch 2 --queuemax 64 --threads 1 \
	--iter 16384 \
	--samplein sample/batch2-view1-T2-2M/ \
	--sampletouse 1024 \
	--samplebatching 8 \
	--samplebase 1024 > nsample-hole.out/Rev-5-2048.out
		
cp -a ckpt/pretrain-d-elu-rev-5-action2-working-rev2 ckpt/Action2-Rev-5-2048


mkdir -p ckpt/pretrain-d-elu-rev-7-action2-working-rev2

./pretrain-d.sh --ferev 7 --elu \
	--ckptdir ckpt/pretrain-d-elu-rev-7-action2-working-rev2/ --ckptprefix working-try11 \
	--batch 2 --queuemax 64 --threads 1 \
	--iter 16384 \
	--samplein sample/batch2-view1-T2-2M/ \
	--sampletouse 1024 \
	--samplebatching 8 \
	--samplebase 1024 > nsample-hole.out/Rev-7-2048.out
		
cp -a ckpt/pretrain-d-elu-rev-7-action2-working-rev2 ckpt/Action2-Rev-7-2048


mkdir -p ckpt/pretrain-d-elu-rev-8-action2-working-rev2

./pretrain-d.sh --ferev 8 --elu \
	--ckptdir ckpt/pretrain-d-elu-rev-8-action2-working-rev2/ --ckptprefix working-try11 \
	--batch 2 --queuemax 64 --threads 1 \
	--iter 16384 \
	--samplein sample/batch2-view1-T2-2M/ \
	--sampletouse 1024 \
	--samplebatching 8 \
	--samplebase 1024 > nsample-hole.out/Rev-8-2048.out
		
cp -a ckpt/pretrain-d-elu-rev-8-action2-working-rev2 ckpt/Action2-Rev-8-2048


mkdir -p ckpt/pretrain-d-elu-rev-4-action2-working-rev2

./pretrain-d.sh --ferev 4 --elu \
	--ckptdir ckpt/pretrain-d-elu-rev-4-action2-working-rev2/ --ckptprefix working-try11 \
	--batch 2 --queuemax 64 --threads 1 \
	--iter 32768 \
	--samplein sample/batch2-view1-T2-2M/ \
	--sampletouse 2048 \
	--samplebatching 8 \
	--samplebase 2048 > nsample-hole.out/Rev-4-4096.out
		
cp -a ckpt/pretrain-d-elu-rev-4-action2-working-rev2 ckpt/Action2-Rev-4-4096


mkdir -p ckpt/pretrain-d-elu-rev-5-action2-working-rev2

./pretrain-d.sh --ferev 5 --elu \
	--ckptdir ckpt/pretrain-d-elu-rev-5-action2-working-rev2/ --ckptprefix working-try11 \
	--batch 2 --queuemax 64 --threads 1 \
	--iter 32768 \
	--samplein sample/batch2-view1-T2-2M/ \
	--sampletouse 2048 \
	--samplebatching 8 \
	--samplebase 2048 > nsample-hole.out/Rev-5-4096.out
		
cp -a ckpt/pretrain-d-elu-rev-5-action2-working-rev2 ckpt/Action2-Rev-5-4096


mkdir -p ckpt/pretrain-d-elu-rev-7-action2-working-rev2

./pretrain-d.sh --ferev 7 --elu \
	--ckptdir ckpt/pretrain-d-elu-rev-7-action2-working-rev2/ --ckptprefix working-try11 \
	--batch 2 --queuemax 64 --threads 1 \
	--iter 32768 \
	--samplein sample/batch2-view1-T2-2M/ \
	--sampletouse 2048 \
	--samplebatching 8 \
	--samplebase 2048 > nsample-hole.out/Rev-7-4096.out
		
cp -a ckpt/pretrain-d-elu-rev-7-action2-working-rev2 ckpt/Action2-Rev-7-4096


mkdir -p ckpt/pretrain-d-elu-rev-8-action2-working-rev2

./pretrain-d.sh --ferev 8 --elu \
	--ckptdir ckpt/pretrain-d-elu-rev-8-action2-working-rev2/ --ckptprefix working-try11 \
	--batch 2 --queuemax 64 --threads 1 \
	--iter 32768 \
	--samplein sample/batch2-view1-T2-2M/ \
	--sampletouse 2048 \
	--samplebatching 8 \
	--samplebase 2048 > nsample-hole.out/Rev-8-4096.out
		
cp -a ckpt/pretrain-d-elu-rev-8-action2-working-rev2 ckpt/Action2-Rev-8-4096


mkdir -p ckpt/pretrain-d-elu-rev-4-action2-working-rev2

./pretrain-d.sh --ferev 4 --elu \
	--ckptdir ckpt/pretrain-d-elu-rev-4-action2-working-rev2/ --ckptprefix working-try11 \
	--batch 2 --queuemax 64 --threads 1 \
	--iter 65536 \
	--samplein sample/batch2-view1-T2-2M/ \
	--sampletouse 4096 \
	--samplebatching 8 \
	--samplebase 4096 > nsample-hole.out/Rev-4-8192.out
		
cp -a ckpt/pretrain-d-elu-rev-4-action2-working-rev2 ckpt/Action2-Rev-4-8192


mkdir -p ckpt/pretrain-d-elu-rev-5-action2-working-rev2

./pretrain-d.sh --ferev 5 --elu \
	--ckptdir ckpt/pretrain-d-elu-rev-5-action2-working-rev2/ --ckptprefix working-try11 \
	--batch 2 --queuemax 64 --threads 1 \
	--iter 65536 \
	--samplein sample/batch2-view1-T2-2M/ \
	--sampletouse 4096 \
	--samplebatching 8 \
	--samplebase 4096 > nsample-hole.out/Rev-5-8192.out
		
cp -a ckpt/pretrain-d-elu-rev-5-action2-working-rev2 ckpt/Action2-Rev-5-8192


mkdir -p ckpt/pretrain-d-elu-rev-7-action2-working-rev2

./pretrain-d.sh --ferev 7 --elu \
	--ckptdir ckpt/pretrain-d-elu-rev-7-action2-working-rev2/ --ckptprefix working-try11 \
	--batch 2 --queuemax 64 --threads 1 \
	--iter 65536 \
	--samplein sample/batch2-view1-T2-2M/ \
	--sampletouse 4096 \
	--samplebatching 8 \
	--samplebase 4096 > nsample-hole.out/Rev-7-8192.out
		
cp -a ckpt/pretrain-d-elu-rev-7-action2-working-rev2 ckpt/Action2-Rev-7-8192


mkdir -p ckpt/pretrain-d-elu-rev-8-action2-working-rev2

./pretrain-d.sh --ferev 8 --elu \
	--ckptdir ckpt/pretrain-d-elu-rev-8-action2-working-rev2/ --ckptprefix working-try11 \
	--batch 2 --queuemax 64 --threads 1 \
	--iter 65536 \
	--samplein sample/batch2-view1-T2-2M/ \
	--sampletouse 4096 \
	--samplebatching 8 \
	--samplebase 4096 > nsample-hole.out/Rev-8-8192.out
		
cp -a ckpt/pretrain-d-elu-rev-8-action2-working-rev2 ckpt/Action2-Rev-8-8192


mkdir -p ckpt/pretrain-d-elu-rev-4-action2-working-rev2

./pretrain-d.sh --ferev 4 --elu \
	--ckptdir ckpt/pretrain-d-elu-rev-4-action2-working-rev2/ --ckptprefix working-try11 \
	--batch 2 --queuemax 64 --threads 1 \
	--iter 1966080 \
	--samplein sample/batch2-view1-T2-2M/ \
	--sampletouse 122880 \
	--samplebatching 8 \
	--samplebase 8192 > nsample-hole.out/Rev-4-131072.out
		
cp -a ckpt/pretrain-d-elu-rev-4-action2-working-rev2 ckpt/Action2-Rev-4-131072


mkdir -p ckpt/pretrain-d-elu-rev-5-action2-working-rev2

./pretrain-d.sh --ferev 5 --elu \
	--ckptdir ckpt/pretrain-d-elu-rev-5-action2-working-rev2/ --ckptprefix working-try11 \
	--batch 2 --queuemax 64 --threads 1 \
	--iter 1966080 \
	--samplein sample/batch2-view1-T2-2M/ \
	--sampletouse 122880 \
	--samplebatching 8 \
	--samplebase 8192 > nsample-hole.out/Rev-5-131072.out
		
cp -a ckpt/pretrain-d-elu-rev-5-action2-working-rev2 ckpt/Action2-Rev-5-131072


mkdir -p ckpt/pretrain-d-elu-rev-7-action2-working-rev2

./pretrain-d.sh --ferev 7 --elu \
	--ckptdir ckpt/pretrain-d-elu-rev-7-action2-working-rev2/ --ckptprefix working-try11 \
	--batch 2 --queuemax 64 --threads 1 \
	--iter 1966080 \
	--samplein sample/batch2-view1-T2-2M/ \
	--sampletouse 122880 \
	--samplebatching 8 \
	--samplebase 8192 > nsample-hole.out/Rev-7-131072.out
		
cp -a ckpt/pretrain-d-elu-rev-7-action2-working-rev2 ckpt/Action2-Rev-7-131072


mkdir -p ckpt/pretrain-d-elu-rev-8-action2-working-rev2

./pretrain-d.sh --ferev 8 --elu \
	--ckptdir ckpt/pretrain-d-elu-rev-8-action2-working-rev2/ --ckptprefix working-try11 \
	--batch 2 --queuemax 64 --threads 1 \
	--iter 1966080 \
	--samplein sample/batch2-view1-T2-2M/ \
	--sampletouse 122880 \
	--samplebatching 8 \
	--samplebase 8192 > nsample-hole.out/Rev-8-131072.out
		
cp -a ckpt/pretrain-d-elu-rev-8-action2-working-rev2 ckpt/Action2-Rev-8-131072


mkdir -p ckpt/pretrain-d-elu-rev-4-action2-working-rev2

./pretrain-d.sh --ferev 4 --elu \
	--ckptdir ckpt/pretrain-d-elu-rev-4-action2-working-rev2/ --ckptprefix working-try11 \
	--batch 2 --queuemax 64 --threads 1 \
	--iter 2097152 \
	--samplein sample/batch2-view1-T2-2M/ \
	--sampletouse 131072 \
	--samplebatching 8 \
	--samplebase 131072 > nsample-hole.out/Rev-4-262144.out
		
cp -a ckpt/pretrain-d-elu-rev-4-action2-working-rev2 ckpt/Action2-Rev-4-262144


mkdir -p ckpt/pretrain-d-elu-rev-5-action2-working-rev2

./pretrain-d.sh --ferev 5 --elu \
	--ckptdir ckpt/pretrain-d-elu-rev-5-action2-working-rev2/ --ckptprefix working-try11 \
	--batch 2 --queuemax 64 --threads 1 \
	--iter 2097152 \
	--samplein sample/batch2-view1-T2-2M/ \
	--sampletouse 131072 \
	--samplebatching 8 \
	--samplebase 131072 > nsample-hole.out/Rev-5-262144.out
		
cp -a ckpt/pretrain-d-elu-rev-5-action2-working-rev2 ckpt/Action2-Rev-5-262144


mkdir -p ckpt/pretrain-d-elu-rev-7-action2-working-rev2

./pretrain-d.sh --ferev 7 --elu \
	--ckptdir ckpt/pretrain-d-elu-rev-7-action2-working-rev2/ --ckptprefix working-try11 \
	--batch 2 --queuemax 64 --threads 1 \
	--iter 2097152 \
	--samplein sample/batch2-view1-T2-2M/ \
	--sampletouse 131072 \
	--samplebatching 8 \
	--samplebase 131072 > nsample-hole.out/Rev-7-262144.out
		
cp -a ckpt/pretrain-d-elu-rev-7-action2-working-rev2 ckpt/Action2-Rev-7-262144


mkdir -p ckpt/pretrain-d-elu-rev-8-action2-working-rev2

./pretrain-d.sh --ferev 8 --elu \
	--ckptdir ckpt/pretrain-d-elu-rev-8-action2-working-rev2/ --ckptprefix working-try11 \
	--batch 2 --queuemax 64 --threads 1 \
	--iter 2097152 \
	--samplein sample/batch2-view1-T2-2M/ \
	--sampletouse 131072 \
	--samplebatching 8 \
	--samplebase 131072 > nsample-hole.out/Rev-8-262144.out
		
cp -a ckpt/pretrain-d-elu-rev-8-action2-working-rev2 ckpt/Action2-Rev-8-262144


mkdir -p ckpt/pretrain-d-elu-rev-4-action2-working-rev2

./pretrain-d.sh --ferev 4 --elu \
	--ckptdir ckpt/pretrain-d-elu-rev-4-action2-working-rev2/ --ckptprefix working-try11 \
	--batch 2 --queuemax 64 --threads 1 \
	--iter 2097152 \
	--samplein sample/batch2-view1-T2-2M/ \
	--sampletouse 131072 \
	--samplebatching 8 \
	--samplebase 262144 > nsample-hole.out/Rev-4-393216.out
		
cp -a ckpt/pretrain-d-elu-rev-4-action2-working-rev2 ckpt/Action2-Rev-4-393216


mkdir -p ckpt/pretrain-d-elu-rev-5-action2-working-rev2

./pretrain-d.sh --ferev 5 --elu \
	--ckptdir ckpt/pretrain-d-elu-rev-5-action2-working-rev2/ --ckptprefix working-try11 \
	--batch 2 --queuemax 64 --threads 1 \
	--iter 2097152 \
	--samplein sample/batch2-view1-T2-2M/ \
	--sampletouse 131072 \
	--samplebatching 8 \
	--samplebase 262144 > nsample-hole.out/Rev-5-393216.out
		
cp -a ckpt/pretrain-d-elu-rev-5-action2-working-rev2 ckpt/Action2-Rev-5-393216


mkdir -p ckpt/pretrain-d-elu-rev-7-action2-working-rev2

./pretrain-d.sh --ferev 7 --elu \
	--ckptdir ckpt/pretrain-d-elu-rev-7-action2-working-rev2/ --ckptprefix working-try11 \
	--batch 2 --queuemax 64 --threads 1 \
	--iter 2097152 \
	--samplein sample/batch2-view1-T2-2M/ \
	--sampletouse 131072 \
	--samplebatching 8 \
	--samplebase 262144 > nsample-hole.out/Rev-7-393216.out
		
cp -a ckpt/pretrain-d-elu-rev-7-action2-working-rev2 ckpt/Action2-Rev-7-393216


mkdir -p ckpt/pretrain-d-elu-rev-8-action2-working-rev2

./pretrain-d.sh --ferev 8 --elu \
	--ckptdir ckpt/pretrain-d-elu-rev-8-action2-working-rev2/ --ckptprefix working-try11 \
	--batch 2 --queuemax 64 --threads 1 \
	--iter 2097152 \
	--samplein sample/batch2-view1-T2-2M/ \
	--sampletouse 131072 \
	--samplebatching 8 \
	--samplebase 262144 > nsample-hole.out/Rev-8-393216.out
		
cp -a ckpt/pretrain-d-elu-rev-8-action2-working-rev2 ckpt/Action2-Rev-8-393216


mkdir -p ckpt/pretrain-d-elu-rev-4-action2-working-rev2

./pretrain-d.sh --ferev 4 --elu \
	--ckptdir ckpt/pretrain-d-elu-rev-4-action2-working-rev2/ --ckptprefix working-try11 \
	--batch 2 --queuemax 64 --threads 1 \
	--iter 2097152 \
	--samplein sample/batch2-view1-T2-2M/ \
	--sampletouse 131072 \
	--samplebatching 8 \
	--samplebase 393216 > nsample-hole.out/Rev-4-524288.out
		
cp -a ckpt/pretrain-d-elu-rev-4-action2-working-rev2 ckpt/Action2-Rev-4-524288


mkdir -p ckpt/pretrain-d-elu-rev-5-action2-working-rev2

./pretrain-d.sh --ferev 5 --elu \
	--ckptdir ckpt/pretrain-d-elu-rev-5-action2-working-rev2/ --ckptprefix working-try11 \
	--batch 2 --queuemax 64 --threads 1 \
	--iter 2097152 \
	--samplein sample/batch2-view1-T2-2M/ \
	--sampletouse 131072 \
	--samplebatching 8 \
	--samplebase 393216 > nsample-hole.out/Rev-5-524288.out
		
cp -a ckpt/pretrain-d-elu-rev-5-action2-working-rev2 ckpt/Action2-Rev-5-524288


mkdir -p ckpt/pretrain-d-elu-rev-7-action2-working-rev2

./pretrain-d.sh --ferev 7 --elu \
	--ckptdir ckpt/pretrain-d-elu-rev-7-action2-working-rev2/ --ckptprefix working-try11 \
	--batch 2 --queuemax 64 --threads 1 \
	--iter 2097152 \
	--samplein sample/batch2-view1-T2-2M/ \
	--sampletouse 131072 \
	--samplebatching 8 \
	--samplebase 393216 > nsample-hole.out/Rev-7-524288.out
		
cp -a ckpt/pretrain-d-elu-rev-7-action2-working-rev2 ckpt/Action2-Rev-7-524288


mkdir -p ckpt/pretrain-d-elu-rev-8-action2-working-rev2

./pretrain-d.sh --ferev 8 --elu \
	--ckptdir ckpt/pretrain-d-elu-rev-8-action2-working-rev2/ --ckptprefix working-try11 \
	--batch 2 --queuemax 64 --threads 1 \
	--iter 2097152 \
	--samplein sample/batch2-view1-T2-2M/ \
	--sampletouse 131072 \
	--samplebatching 8 \
	--samplebase 393216 > nsample-hole.out/Rev-8-524288.out
		
cp -a ckpt/pretrain-d-elu-rev-8-action2-working-rev2 ckpt/Action2-Rev-8-524288


mkdir -p ckpt/pretrain-d-elu-rev-4-action2-working-rev2

./pretrain-d.sh --ferev 4 --elu \
	--ckptdir ckpt/pretrain-d-elu-rev-4-action2-working-rev2/ --ckptprefix working-try11 \
	--batch 2 --queuemax 64 --threads 1 \
	--iter 2097152 \
	--samplein sample/batch2-view1-T2-2M/ \
	--sampletouse 131072 \
	--samplebatching 8 \
	--samplebase 524288 > nsample-hole.out/Rev-4-655360.out
		
cp -a ckpt/pretrain-d-elu-rev-4-action2-working-rev2 ckpt/Action2-Rev-4-655360


mkdir -p ckpt/pretrain-d-elu-rev-5-action2-working-rev2

./pretrain-d.sh --ferev 5 --elu \
	--ckptdir ckpt/pretrain-d-elu-rev-5-action2-working-rev2/ --ckptprefix working-try11 \
	--batch 2 --queuemax 64 --threads 1 \
	--iter 2097152 \
	--samplein sample/batch2-view1-T2-2M/ \
	--sampletouse 131072 \
	--samplebatching 8 \
	--samplebase 524288 > nsample-hole.out/Rev-5-655360.out
		
cp -a ckpt/pretrain-d-elu-rev-5-action2-working-rev2 ckpt/Action2-Rev-5-655360


mkdir -p ckpt/pretrain-d-elu-rev-7-action2-working-rev2

./pretrain-d.sh --ferev 7 --elu \
	--ckptdir ckpt/pretrain-d-elu-rev-7-action2-working-rev2/ --ckptprefix working-try11 \
	--batch 2 --queuemax 64 --threads 1 \
	--iter 2097152 \
	--samplein sample/batch2-view1-T2-2M/ \
	--sampletouse 131072 \
	--samplebatching 8 \
	--samplebase 524288 > nsample-hole.out/Rev-7-655360.out
		
cp -a ckpt/pretrain-d-elu-rev-7-action2-working-rev2 ckpt/Action2-Rev-7-655360


mkdir -p ckpt/pretrain-d-elu-rev-8-action2-working-rev2

./pretrain-d.sh --ferev 8 --elu \
	--ckptdir ckpt/pretrain-d-elu-rev-8-action2-working-rev2/ --ckptprefix working-try11 \
	--batch 2 --queuemax 64 --threads 1 \
	--iter 2097152 \
	--samplein sample/batch2-view1-T2-2M/ \
	--sampletouse 131072 \
	--samplebatching 8 \
	--samplebase 524288 > nsample-hole.out/Rev-8-655360.out
		
cp -a ckpt/pretrain-d-elu-rev-8-action2-working-rev2 ckpt/Action2-Rev-8-655360


mkdir -p ckpt/pretrain-d-elu-rev-4-action2-working-rev2

./pretrain-d.sh --ferev 4 --elu \
	--ckptdir ckpt/pretrain-d-elu-rev-4-action2-working-rev2/ --ckptprefix working-try11 \
	--batch 2 --queuemax 64 --threads 1 \
	--iter 2097152 \
	--samplein sample/batch2-view1-T2-2M/ \
	--sampletouse 131072 \
	--samplebatching 8 \
	--samplebase 655360 > nsample-hole.out/Rev-4-786432.out
		
cp -a ckpt/pretrain-d-elu-rev-4-action2-working-rev2 ckpt/Action2-Rev-4-786432


mkdir -p ckpt/pretrain-d-elu-rev-5-action2-working-rev2

./pretrain-d.sh --ferev 5 --elu \
	--ckptdir ckpt/pretrain-d-elu-rev-5-action2-working-rev2/ --ckptprefix working-try11 \
	--batch 2 --queuemax 64 --threads 1 \
	--iter 2097152 \
	--samplein sample/batch2-view1-T2-2M/ \
	--sampletouse 131072 \
	--samplebatching 8 \
	--samplebase 655360 > nsample-hole.out/Rev-5-786432.out
		
cp -a ckpt/pretrain-d-elu-rev-5-action2-working-rev2 ckpt/Action2-Rev-5-786432


mkdir -p ckpt/pretrain-d-elu-rev-7-action2-working-rev2

./pretrain-d.sh --ferev 7 --elu \
	--ckptdir ckpt/pretrain-d-elu-rev-7-action2-working-rev2/ --ckptprefix working-try11 \
	--batch 2 --queuemax 64 --threads 1 \
	--iter 2097152 \
	--samplein sample/batch2-view1-T2-2M/ \
	--sampletouse 131072 \
	--samplebatching 8 \
	--samplebase 655360 > nsample-hole.out/Rev-7-786432.out
		
cp -a ckpt/pretrain-d-elu-rev-7-action2-working-rev2 ckpt/Action2-Rev-7-786432


mkdir -p ckpt/pretrain-d-elu-rev-8-action2-working-rev2

./pretrain-d.sh --ferev 8 --elu \
	--ckptdir ckpt/pretrain-d-elu-rev-8-action2-working-rev2/ --ckptprefix working-try11 \
	--batch 2 --queuemax 64 --threads 1 \
	--iter 2097152 \
	--samplein sample/batch2-view1-T2-2M/ \
	--sampletouse 131072 \
	--samplebatching 8 \
	--samplebase 655360 > nsample-hole.out/Rev-8-786432.out
		
cp -a ckpt/pretrain-d-elu-rev-8-action2-working-rev2 ckpt/Action2-Rev-8-786432


mkdir -p ckpt/pretrain-d-elu-rev-4-action2-working-rev2

./pretrain-d.sh --ferev 4 --elu \
	--ckptdir ckpt/pretrain-d-elu-rev-4-action2-working-rev2/ --ckptprefix working-try11 \
	--batch 2 --queuemax 64 --threads 1 \
	--iter 2097152 \
	--samplein sample/batch2-view1-T2-2M/ \
	--sampletouse 131072 \
	--samplebatching 8 \
	--samplebase 786432 > nsample-hole.out/Rev-4-917504.out
		
cp -a ckpt/pretrain-d-elu-rev-4-action2-working-rev2 ckpt/Action2-Rev-4-917504


mkdir -p ckpt/pretrain-d-elu-rev-5-action2-working-rev2

./pretrain-d.sh --ferev 5 --elu \
	--ckptdir ckpt/pretrain-d-elu-rev-5-action2-working-rev2/ --ckptprefix working-try11 \
	--batch 2 --queuemax 64 --threads 1 \
	--iter 2097152 \
	--samplein sample/batch2-view1-T2-2M/ \
	--sampletouse 131072 \
	--samplebatching 8 \
	--samplebase 786432 > nsample-hole.out/Rev-5-917504.out
		
cp -a ckpt/pretrain-d-elu-rev-5-action2-working-rev2 ckpt/Action2-Rev-5-917504


mkdir -p ckpt/pretrain-d-elu-rev-7-action2-working-rev2

./pretrain-d.sh --ferev 7 --elu \
	--ckptdir ckpt/pretrain-d-elu-rev-7-action2-working-rev2/ --ckptprefix working-try11 \
	--batch 2 --queuemax 64 --threads 1 \
	--iter 2097152 \
	--samplein sample/batch2-view1-T2-2M/ \
	--sampletouse 131072 \
	--samplebatching 8 \
	--samplebase 786432 > nsample-hole.out/Rev-7-917504.out
		
cp -a ckpt/pretrain-d-elu-rev-7-action2-working-rev2 ckpt/Action2-Rev-7-917504


mkdir -p ckpt/pretrain-d-elu-rev-8-action2-working-rev2

./pretrain-d.sh --ferev 8 --elu \
	--ckptdir ckpt/pretrain-d-elu-rev-8-action2-working-rev2/ --ckptprefix working-try11 \
	--batch 2 --queuemax 64 --threads 1 \
	--iter 2097152 \
	--samplein sample/batch2-view1-T2-2M/ \
	--sampletouse 131072 \
	--samplebatching 8 \
	--samplebase 786432 > nsample-hole.out/Rev-8-917504.out
		
cp -a ckpt/pretrain-d-elu-rev-8-action2-working-rev2 ckpt/Action2-Rev-8-917504


mkdir -p ckpt/pretrain-d-elu-rev-4-action2-working-rev2

./pretrain-d.sh --ferev 4 --elu \
	--ckptdir ckpt/pretrain-d-elu-rev-4-action2-working-rev2/ --ckptprefix working-try11 \
	--batch 2 --queuemax 64 --threads 1 \
	--iter 2097152 \
	--samplein sample/batch2-view1-T2-2M/ \
	--sampletouse 131072 \
	--samplebatching 8 \
	--samplebase 917504 > nsample-hole.out/Rev-4-1048576.out
		
cp -a ckpt/pretrain-d-elu-rev-4-action2-working-rev2 ckpt/Action2-Rev-4-1048576


mkdir -p ckpt/pretrain-d-elu-rev-5-action2-working-rev2

./pretrain-d.sh --ferev 5 --elu \
	--ckptdir ckpt/pretrain-d-elu-rev-5-action2-working-rev2/ --ckptprefix working-try11 \
	--batch 2 --queuemax 64 --threads 1 \
	--iter 2097152 \
	--samplein sample/batch2-view1-T2-2M/ \
	--sampletouse 131072 \
	--samplebatching 8 \
	--samplebase 917504 > nsample-hole.out/Rev-5-1048576.out
		
cp -a ckpt/pretrain-d-elu-rev-5-action2-working-rev2 ckpt/Action2-Rev-5-1048576


mkdir -p ckpt/pretrain-d-elu-rev-7-action2-working-rev2

./pretrain-d.sh --ferev 7 --elu \
	--ckptdir ckpt/pretrain-d-elu-rev-7-action2-working-rev2/ --ckptprefix working-try11 \
	--batch 2 --queuemax 64 --threads 1 \
	--iter 2097152 \
	--samplein sample/batch2-view1-T2-2M/ \
	--sampletouse 131072 \
	--samplebatching 8 \
	--samplebase 917504 > nsample-hole.out/Rev-7-1048576.out
		
cp -a ckpt/pretrain-d-elu-rev-7-action2-working-rev2 ckpt/Action2-Rev-7-1048576


mkdir -p ckpt/pretrain-d-elu-rev-8-action2-working-rev2

./pretrain-d.sh --ferev 8 --elu \
	--ckptdir ckpt/pretrain-d-elu-rev-8-action2-working-rev2/ --ckptprefix working-try11 \
	--batch 2 --queuemax 64 --threads 1 \
	--iter 2097152 \
	--samplein sample/batch2-view1-T2-2M/ \
	--sampletouse 131072 \
	--samplebatching 8 \
	--samplebase 917504 > nsample-hole.out/Rev-8-1048576.out
		
cp -a ckpt/pretrain-d-elu-rev-8-action2-working-rev2 ckpt/Action2-Rev-8-1048576

