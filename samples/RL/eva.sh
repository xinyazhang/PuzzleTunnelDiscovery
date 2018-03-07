TRY=try11
NSAMPLE=524288

for REV in 7 8
do
	for FEAT in 512 1024 2048
	do
		CKPT_DIR=View-0-of-14-Action4-Rev-$REV-Feat-$FEAT-524288
		PREFIX=working-$TRY
		ITER=$((NSAMPLE))

		for BASE in 0 1048576
		do
			echo "
	./pretrain-d.sh --ferev $REV --elu \\
		--ckptdir ackpt/$CKPT_DIR/ --ckptprefix $PREFIX \\
		--batch 2 --queuemax 64 --threads 1 \\
		--iter $ITER \\
		--eval \\
		--view 0 \\
		--featnum $FEAT \\
		--imhidden $FEAT $FEAT \\
		--samplein sample/batch2-view14-norgbd-T2-R2-2M/ \\
		--sampletouse $NSAMPLE \\
		--samplebatching 8 \\
		--samplebase $BASE > evaluation/Action-4-Error-Base-$BASE-Rev-$REV-Feat-$FEAT.out"
		done
	done
done

exit

TRY=try11
NSAMPLE=524288

for REV in 7 8
do
	for FEAT in 512 1024 2048
	do
		CKPT_DIR=View-0-of-14-Action4-Rev-$REV-Feat-$FEAT-524288
		PREFIX=working-$TRY
		ITER=$((NSAMPLE))

		for BASE in 0 1048576
		do
			echo "
	./pretrain-d.sh --ferev $REV --elu \\
		--ckptdir ackpt/$CKPT_DIR/ --ckptprefix $PREFIX \\
		--batch 2 --queuemax 64 --threads 1 \\
		--iter $ITER \\
		--eval \\
		--view 0 \\
		--featnum $FEAT \\
		--imhidden $FEAT $FEAT \\
		--samplein sample/batch2-view14-norgbd-T2-R2-2M/ \\
		--sampletouse $NSAMPLE \\
		--samplebatching 8 \\
		--samplebase $BASE > evaluation/Action-4-Error-Base-$BASE-Rev-$REV-Feat-$FEAT.out"
		done
	done
done

exit

TRY=try11
REV=5

NSAMPLE=262144

for REV in 4 5 6 7 8
do
	CKPT_DIR=Action4-Rev-$REV-262144
	PREFIX=working-$TRY
	ITER=$((NSAMPLE))

	for BASE in 0 524288
	do
		echo "
./pretrain-d.sh --ferev $REV --elu \\
	--ckptdir ckpt/$CKPT_DIR/ --ckptprefix $PREFIX \\
	--batch 2 --queuemax 64 --threads 1 \\
	--iter $ITER \\
	--eval \\
	--samplein sample/batch2-view1-T2-R2-2M/ \\
	--sampletouse $NSAMPLE \\
	--samplebatching 8 \\
	--samplebase $BASE > evaluation/Action-4-Error-Base-$BASE-Rev-$REV.out"
	done
done

exit

ITER=1024
# PREFIX=singlesample-$TRY
for NSAMPLE in `seq 0 2 99`
do
	PREFIX=sample$NSAMPLE-$TRY
	CKPT_DIR=pretrain-d-elu-action2-sample$NSAMPLE-$TRY

	# ./pretrain-d.sh --ferev $REV --elu --sampleout sample/batch2-view1/ --ckptdir ckpt/pretrain-d-elu-rev$REV-$TRY/ --ckptprefix $PREFIX --batch 2 --queuemax 64 --threads 1 --iter $ITER
	./pretrain-d.sh --ferev $REV --elu \
		--eval \
		--ckptdir ckpt/$CKPT_DIR/ --ckptprefix $PREFIX \
		--batch 2 --queuemax 64 --threads 1 \
		--iter 100 \
		--samplein sample/batch2-view1-XTrans/ \
		--sampletouse $NSAMPLE \
		--samplebase 0
done
	
exit
for base in `seq 0 2 99`
do
	./pretrain-d.sh --ferev $REV --elu \
		--ckptdir ckpt/pretrain-d-elu-rev$REV-$TRY/ --ckptprefix $PREFIX \
		--batch 2 --queuemax 64 --threads 1 \
		--iter 1 \
		--sampleout sample/batch2-view1-XTrans/ \
		--samplebase $base \
		--uniqueaction 0
done
for base in `seq 1 2 99`
do
	./pretrain-d.sh --ferev $REV --elu \
		--ckptdir ckpt/pretrain-d-elu-rev$REV-$TRY/ --ckptprefix $PREFIX \
		--batch 2 --queuemax 64 --threads 1 \
		--iter 1 \
		--sampleout sample/batch2-view1-XTrans/ \
		--samplebase $base \
		--uniqueaction 1
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
