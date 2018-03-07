# Only translations

TRY=try11
REV=5
ITER=1024
PREFIX=XOnly-$TRY

mkdir -p sample/batch2-view14-norgbd-T6-R0-2M/

./pretrain-d.sh --ferev $REV --elu \
	--dryrun3 \
	--ckptdir ckpt/pretrain-d-elu-rev$REV-$TRY/ --ckptprefix $PREFIX \
	--batch 2 --queuemax 64 --threads 8 \
	--iter 262144 \
	--norgbd \
	--sampleout sample/batch2-view14-norgbd-T6-R0-2M/ \
	--actionset 0 1 2 3 4 5

exit

# Larger Action Magnitude
TRY=try11
REV=5
ITER=1024
PREFIX=XOnly-$TRY

mkdir -p sample/batch2-view14-norgbd-T6-R6-2M-8thAMAG/

./pretrain-d.sh --ferev $REV --elu \
	--dryrun3 \
	--amag 0.125 \
	--ckptdir ckpt/pretrain-d-elu-rev$REV-$TRY/ --ckptprefix $PREFIX \
	--batch 2 --queuemax 64 --threads 4 \
	--iter 524288 \
	--norgbd \
	--sampleout sample/batch2-view14-norgbd-T6-R6-2M-8thAMAG/ \
	--actionset -1

exit

# All 12 actions

TRY=try11
REV=5
ITER=1024
PREFIX=XOnly-$TRY

mkdir -p sample/batch2-view14-norgbd-T6-R6-2M/

./pretrain-d.sh --ferev $REV --elu \
	--dryrun3 \
	--ckptdir ckpt/pretrain-d-elu-rev$REV-$TRY/ --ckptprefix $PREFIX \
	--batch 2 --queuemax 64 --threads 4 \
	--iter 524288 \
	--norgbd \
	--sampleout sample/batch2-view14-norgbd-T6-R6-2M/ \
	--actionset -1

exit

mkdir -p sample/batch2-view14-norgbd-T2-R2-2M/

./pretrain-d.sh --ferev $REV --elu \
	--dryrun3 \
	--ckptdir ckpt/pretrain-d-elu-rev$REV-$TRY/ --ckptprefix $PREFIX \
	--batch 2 --queuemax 64 --threads 8 \
	--iter 262144 \
	--norgbd \
	--sampleout sample/batch2-view14-norgbd-T2-R2-2M/ \
	--actionset 0 1 10 11

exit

mkdir -p sample/batch2-view1-T2-R2-2M/ 

./pretrain-d.sh --ferev $REV --elu \
	--dryrun3 \
	--ckptdir ckpt/pretrain-d-elu-rev$REV-$TRY/ --ckptprefix $PREFIX \
	--batch 2 --queuemax 64 --threads 8 \
	--iter 262144 \
	--sampleout sample/batch2-view1-T2-R2-2M/ \
	--actionset 0 1 10 11

exit

./pretrain-d.sh --ferev $REV --elu \
	--dryrun3 \
	--ckptdir ckpt/pretrain-d-elu-rev$REV-$TRY/ --ckptprefix $PREFIX \
	--batch 2 --queuemax 64 --threads 4 \
	--iter 524288 \
	--sampleout sample/batch2-view1-T2-2M/ \
	--actionset 0 1

exit 
./pretrain-d.sh --ferev $REV --elu \
	--dryrun3 \
	--ckptdir ckpt/pretrain-d-elu-rev$REV-$TRY/ --ckptprefix $PREFIX \
	--batch 2 --queuemax 64 --threads 2 \
	--iter 524288 \
	--sampleout sample/batch2-view1-T4/ \
	--actionset 0 1 2 3

./pretrain-d.sh --ferev $REV --elu \
	--dryrun3 \
	--ckptdir ckpt/pretrain-d-elu-rev$REV-$TRY/ --ckptprefix $PREFIX \
	--batch 2 --queuemax 64 --threads 2 \
	--iter 524288 \
	--sampleout sample/batch2-view1-T6/ \
	--actionset 0 1 2 3 4 5

./pretrain-d.sh --ferev $REV --elu \
	--dryrun3 \
	--ckptdir ckpt/pretrain-d-elu-rev$REV-$TRY/ --ckptprefix $PREFIX \
	--batch 2 --queuemax 64 --threads 2 \
	--iter 524288 \
	--sampleout sample/batch2-view1-TXRY/ \
	--actionset 0 1 8 9

exit

for base in `seq 2049 2 32768`
do
	./pretrain-d.sh --ferev $REV --elu \
		--dryrun3 \
		--ckptdir ckpt/pretrain-d-elu-rev$REV-$TRY/ --ckptprefix $PREFIX \
		--batch 2 --queuemax 64 --threads 1 \
		--iter 1 \
		--sampleout sample/batch2-view1-XTrans/ \
		--samplebase $base \
		--uniqueaction 1
done

for base in `seq 2048 2 32768`
do
	./pretrain-d.sh --ferev $REV --elu \
		--dryrun3 \
		--ckptdir ckpt/pretrain-d-elu-rev$REV-$TRY/ --ckptprefix $PREFIX \
		--batch 2 --queuemax 64 --threads 1 \
		--iter 1 \
		--sampleout sample/batch2-view1-XTrans/ \
		--samplebase $base \
		--uniqueaction 0
done

#./pretrain-d.sh --ferev $REV --ckptdir ckpt/pretrain-d-rev$REV-$TRY/ --ckptprefix $PREFIX --batch 2 --queuemax 64 --threads 1 --iter $ITER
exit
./pretrain-d.sh --ferev $REV --elu --ckptdir ckpt/pretrain-d-elu-sb-rev$REV-$TRY/ --ckptprefix $PREFIX --batch 5 --queuemax 64 --threads 1 --iter 16384
./pretrain-d.sh --ferev $REV --elu --ckptdir ckpt/pretrain-d-elu-nb-rev$REV-$TRY/ --ckptprefix $PREFIX --batch 2 --queuemax 64 --threads 1 --iter 65536
./pretrain-d.sh --ferev 2 --elu --ckptdir ckpt/pretrain-d-elu-rev2-$TRY/ --ckptprefix $PREFIX --batch 16 --queuemax 64 --threads 1 --iter 2048
./pretrain-d.sh --ferev 3 --elu --ckptdir ckpt/pretrain-d-elu-rev3-$TRY/ --ckptprefix $PREFIX --batch 16 --queuemax 64 --threads 1 --iter 2048
./pretrain-d.sh --ferev 2 --elu --ckptdir ckpt/pretrain-d-elu-sb-rev2-$TRY/ --ckptprefix $PREFIX --batch 5 --queuemax 64 --threads 1 --iter 8192
./pretrain-d.sh --ferev 3 --elu --ckptdir ckpt/pretrain-d-elu-sb-rev3-$TRY/ --ckptprefix $PREFIX --batch 5 --queuemax 64 --threads 1 --iter 8192
./pretrain-d.sh --ferev 2 --elu --ckptdir ckpt/pretrain-d-elu-nb-rev2-$TRY/ --ckptprefix $PREFIX --batch 2 --queuemax 64 --threads 1 --iter 16384
./pretrain-d.sh --ferev 3 --elu --ckptdir ckpt/pretrain-d-elu-nb-rev3-$TRY/ --ckptprefix $PREFIX --batch 2 --queuemax 64 --threads 1 --iter 16384
