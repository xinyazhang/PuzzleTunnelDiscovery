TRY=try11

# 14 Views on 12 Actions with 2048 FV
# 512K training samples, *32 iterations
# AdVanced Illumination
# Res: 224 * 224
# Rev: 12 (ResNet, with gradb)
NACTION=12
SAMPLEDIR=sample/batch2-view14-norgbd-T6-R6-2M/
OUTDIR=nsample-idmv-action$NACTION
TOTALVIEW=14
echo "mkdir -p $OUTDIR"

BASE=0
for END in 524288
do
	NSAMPLE=$((END - BASE))
	for REV in 12
	do
		for FEATNUM in 2048
		do
		for VIEW in `seq 0 1 $((TOTALVIEW -1))`
		do
			CKPT_DIR=pretrain-d-elu-view-Resnet-$VIEW-of-$TOTALVIEW-rev-$REV-action$NACTION-Feat-$FEATNUM-working
			echo -e "\nmkdir -p ckpt/$CKPT_DIR"
			PREFIX=working-$TRY
			ITER=$((NSAMPLE * 32))

			echo "
	./pretrain-d.sh --ferev $REV --elu \\
		--ckptdir ckpt/$CKPT_DIR/ --ckptprefix $PREFIX \\
		--batch 2 --queuemax 64 --threads 1 \\
		--avi \\
		--res 224 \\
		--iter $ITER \\
		--view $VIEW \\
		--featnum $FEATNUM \\
		--imhidden $FEATNUM $FEATNUM \\
		--samplein $SAMPLEDIR \\
		--sampletouse $NSAMPLE \\
		--samplebatching 64 \\
		--samplebase $BASE > nsample-idmv-action$NACTION/Resnet-View-$VIEW-of-$TOTALVIEW-Rev-$REV-Feat-$FEATNUM-$END.out
			"
			ARCHIVED_CKPT=Resnet-View-$VIEW-of-$TOTALVIEW-Action$NACTION-Rev-$REV-Feat-$FEATNUM-$END
			echo -e "cp -a ckpt/$CKPT_DIR ackpt/$ARCHIVED_CKPT\n"
		done
		done
	done
	BASE=$END
done

exit

# 14 Views on 12 Actions with 2048 FV
# 512K training samples, *32 iterations
# AdVanced Illumination
# Res: 224 * 224
# Rev: 11 (ResNet)
NACTION=12
SAMPLEDIR=sample/batch2-view14-norgbd-T6-R6-2M/
OUTDIR=nsample-idmv-action$NACTION
TOTALVIEW=14
echo "mkdir -p $OUTDIR"

BASE=0
for END in 524288
do
	NSAMPLE=$((END - BASE))
	for REV in 11
	do
		for FEATNUM in 2048
		do
		for VIEW in `seq 0 1 $((TOTALVIEW -1))`
		do
			CKPT_DIR=pretrain-d-elu-view-Resnet-$VIEW-of-$TOTALVIEW-rev-$REV-action$NACTION-Feat-$FEATNUM-working
			echo -e "\nmkdir -p ckpt/$CKPT_DIR"
			PREFIX=working-$TRY
			ITER=$((NSAMPLE * 32))

			echo "
	./pretrain-d.sh --ferev $REV --elu \\
		--ckptdir ckpt/$CKPT_DIR/ --ckptprefix $PREFIX \\
		--batch 2 --queuemax 64 --threads 1 \\
		--avi \\
		--res 224 \\
		--iter $ITER \\
		--view $VIEW \\
		--featnum $FEATNUM \\
		--imhidden $FEATNUM $FEATNUM \\
		--samplein $SAMPLEDIR \\
		--sampletouse $NSAMPLE \\
		--samplebatching 64 \\
		--samplebase $BASE > nsample-idmv-action$NACTION/Resnet-View-$VIEW-of-$TOTALVIEW-Rev-$REV-Feat-$FEATNUM-$END.out
			"
			ARCHIVED_CKPT=Resnet-View-$VIEW-of-$TOTALVIEW-Action$NACTION-Rev-$REV-Feat-$FEATNUM-$END
			echo -e "cp -a ckpt/$CKPT_DIR ackpt/$ARCHIVED_CKPT\n"
		done
		done
	done
	BASE=$END
done

exit

TRY=try11

# 14 Views on 12 Actions with 2048 FV
# 512K training samples, *8 iterations
# Larger Magnitude
# Rev: 5
NACTION=12
SAMPLEDIR=sample/batch2-view14-norgbd-T6-R6-2M-8thAMAG/
OUTDIR=nsample-idmv-action$NACTION
TOTALVIEW=14
echo "mkdir -p $OUTDIR"

BASE=0
for END in 524288
do
	NSAMPLE=$((END - BASE))
	for REV in 5
	do
		for FEATNUM in 2048
		do
		for VIEW in `seq 0 1 $((TOTALVIEW -1))`
		do
			CKPT_DIR=pretrain-d-elu-view-$VIEW-of-$TOTALVIEW-rev-$REV-action$NACTION-Feat-$FEATNUM-8thAMAG-working
			echo -e "\nmkdir -p ckpt/$CKPT_DIR"
			PREFIX=working-$TRY
			ITER=$((NSAMPLE * 8))

			echo "
	./pretrain-d.sh --ferev $REV --elu \\
		--ckptdir ckpt/$CKPT_DIR/ --ckptprefix $PREFIX \\
		--batch 2 --queuemax 64 --threads 1 \\
		--iter $ITER \\
		--view $VIEW \\
		--featnum $FEATNUM \\
		--imhidden $FEATNUM $FEATNUM \\
		--samplein $SAMPLEDIR \\
		--sampletouse $NSAMPLE \\
		--samplebatching 64 \\
		--samplebase $BASE > nsample-idmv-action$NACTION/View-$VIEW-of-$TOTALVIEW-Rev-$REV-Feat-$FEATNUM-$END-8thAMAG.out
			"
			ARCHIVED_CKPT=View-$VIEW-of-$TOTALVIEW-Action$NACTION-Rev-$REV-Feat-$FEATNUM-$END-8thAMAG
			echo -e "cp -a ckpt/$CKPT_DIR ackpt/$ARCHIVED_CKPT\n"
		done
		done
	done
	BASE=$END
done

exit
# 14 Views on 12 Actions with 2048 FV
# 512K training samples, *24 iterations
# Res: 224 * 224
# Rev: 10 (Rev 5 with one additional layer)

NACTION=12
SAMPLEDIR=sample/batch2-view14-norgbd-T6-R6-2M/
OUTDIR=nsample-idmv-action$NACTION
TOTALVIEW=14
RES=224
echo "mkdir -p $OUTDIR"

BASE=0
for END in 524288
do
	NSAMPLE=$((END - BASE))
	for REV in 10
	do
		for FEATNUM in 2048
		do
		for VIEW in `seq 0 1 $((TOTALVIEW -1))`
		do
			CKPT_DIR=pretrain-d-elu-res-$RES-view-$VIEW-of-$TOTALVIEW-rev-$REV-action$NACTION-Feat-$FEATNUM-working
			echo "\nmkdir -p ckpt/$CKPT_DIR"
			PREFIX=working-$TRY
			ITER=$((NSAMPLE * 24))

			echo "
	./pretrain-d.sh --ferev $REV --elu \\
		--ckptdir ckpt/$CKPT_DIR/ --ckptprefix $PREFIX \\
		--batch 2 --queuemax 64 --threads 1 \\
		--iter $ITER \\
		--view $VIEW \\
		--res $RES \\
		--featnum $FEATNUM \\
		--imhidden $FEATNUM $FEATNUM \\
		--samplein $SAMPLEDIR \\
		--sampletouse $NSAMPLE \\
		--samplebatching 64 \\
		--samplebase $BASE > nsample-idmv-action$NACTION/Res-$RES-View-$VIEW-of-$TOTALVIEW-Rev-$REV-Feat-$FEATNUM-$END.out
			"
			ARCHIVED_CKPT=Res-$RES-View-$VIEW-of-$TOTALVIEW-Action$NACTION-Rev-$REV-Feat-$FEATNUM-$END
			echo "cp -a ckpt/$CKPT_DIR ackpt/$ARCHIVED_CKPT\n"
		done
		done
	done
	BASE=$END
done

exit
# 14 Views on 12 Actions with 2048 FV
# 512K training samples
# Second pass, now with *24 iterations
# Note: Rev 5 is a smaller model so we shall train multiple ones at the same time.

NACTION=12
SAMPLEDIR=sample/batch2-view14-norgbd-T6-R6-2M/
OUTDIR=nsample-idmv-action$NACTION
TOTALVIEW=14
echo "mkdir -p $OUTDIR"

BASE=0
for END in 524288
do
	NSAMPLE=$((END - BASE))
	for REV in 5
	do
		for FEATNUM in 2048
		do
		for VIEW in `seq 0 1 $((TOTALVIEW -1))`
		do
			CKPT_DIR=pretrain-d-elu-view-$VIEW-of-$TOTALVIEW-rev-$REV-action$NACTION-Feat-$FEATNUM-working
			echo "\nmkdir -p ckpt/$CKPT_DIR"
			PREFIX=working-$TRY
			ITER=$((NSAMPLE * 24))

			echo "
	./pretrain-d.sh --ferev $REV --elu \\
		--ckptdir ckpt/$CKPT_DIR/ --ckptprefix $PREFIX \\
		--batch 2 --queuemax 64 --threads 1 \\
		--iter $ITER \\
		--view $VIEW \\
		--featnum $FEATNUM \\
		--imhidden $FEATNUM $FEATNUM \\
		--samplein $SAMPLEDIR \\
		--sampletouse $NSAMPLE \\
		--samplebatching 64 \\
		--samplebase $BASE > nsample-idmv-action$NACTION/View-$VIEW-of-$TOTALVIEW-Rev-$REV-Feat-$FEATNUM-$END-Pass2.out
			"
			ARCHIVED_CKPT=View-$VIEW-of-$TOTALVIEW-Action$NACTION-Rev-$REV-Feat-$FEATNUM-$END-Pass2
			echo "cp -a ckpt/$CKPT_DIR ackpt/$ARCHIVED_CKPT\n"
		done
		done
	done
	BASE=$END
done

exit

TRY=try11
REV=5
# PREFIX=singlesample-$TRY

# ReLU, not ELU
NACTION=12
SAMPLEDIR=sample/batch2-view14-norgbd-T6-R6-2M/
OUTDIR=nsample-idmv-action$NACTION
TOTALVIEW=14
echo "mkdir -p $OUTDIR"

BASE=0
for END in 524288
do
	NSAMPLE=$((END - BASE))
	for REV in 5
	do
		for FEATNUM in 2048
		do
		for VIEW in `seq 0 1 $((TOTALVIEW -1))`
		do
			CKPT_DIR=pretrain-d-relu-view-$VIEW-of-$TOTALVIEW-rev-$REV-action$NACTION-Feat-$FEATNUM-working
			echo "\nmkdir -p ckpt/$CKPT_DIR"
			PREFIX=working-$TRY
			ITER=$((NSAMPLE * 8))

			echo "
	./pretrain-d.sh --ferev $REV \\
		--ckptdir ckpt/$CKPT_DIR/ --ckptprefix $PREFIX \\
		--batch 2 --queuemax 64 --threads 1 \\
		--iter $ITER \\
		--view $VIEW \\
		--featnum $FEATNUM \\
		--imhidden $FEATNUM $FEATNUM \\
		--samplein $SAMPLEDIR \\
		--sampletouse $NSAMPLE \\
		--samplebatching 64 \\
		--samplebase $BASE > nsample-idmv-action$NACTION/ReLU-View-$VIEW-of-$TOTALVIEW-Rev-$REV-Feat-$FEATNUM-$END.out
			"
			ARCHIVED_CKPT=ReLU-View-$VIEW-of-$TOTALVIEW-Action$NACTION-Rev-$REV-Feat-$FEATNUM-$END
			echo "cp -a ckpt/$CKPT_DIR ackpt/$ARCHIVED_CKPT\n"
		done
		done
	done
	BASE=$END
done

exit
 
# 14 Views on 12 Actions with 2048 FV
# 512K training samples, *8 iterations
NACTION=12
SAMPLEDIR=sample/batch2-view14-norgbd-T6-R6-2M/
OUTDIR=nsample-idmv-action$NACTION
TOTALVIEW=14
echo "mkdir -p $OUTDIR"

BASE=0
for END in 524288
do
	NSAMPLE=$((END - BASE))
	for REV in 5
	do
		for FEATNUM in 2048
		do
		for VIEW in `seq 0 1 $((TOTALVIEW -1))`
		do
			CKPT_DIR=pretrain-d-elu-view-$VIEW-of-$TOTALVIEW-rev-$REV-action$NACTION-Feat-$FEATNUM-working
			echo "\nmkdir -p ckpt/$CKPT_DIR"
			PREFIX=working-$TRY
			ITER=$((NSAMPLE * 8))

			echo "
	./pretrain-d.sh --ferev $REV --elu \\
		--ckptdir ckpt/$CKPT_DIR/ --ckptprefix $PREFIX \\
		--batch 2 --queuemax 64 --threads 1 \\
		--iter $ITER \\
		--view $VIEW \\
		--featnum $FEATNUM \\
		--imhidden $FEATNUM $FEATNUM \\
		--samplein $SAMPLEDIR \\
		--sampletouse $NSAMPLE \\
		--samplebatching 64 \\
		--samplebase $BASE > nsample-idmv-action$NACTION/View-$VIEW-of-$TOTALVIEW-Rev-$REV-Feat-$FEATNUM-$END.out
			"
			ARCHIVED_CKPT=View-$VIEW-of-$TOTALVIEW-Action$NACTION-Rev-$REV-Feat-$FEATNUM-$END
			echo "cp -a ckpt/$CKPT_DIR ackpt/$ARCHIVED_CKPT\n"
		done
		done
	done
	BASE=$END
done

exit

# Larger FC and deeper Inverse Model
OUTDIR=nsample-idmv-action4
TOTALVIEW=14
echo "mkdir -p $OUTDIR"

BASE=0
for END in 524288
do
	NSAMPLE=$((END - BASE))
	for REV in 5 6 7 8
	do
		for FEATNUM in 512 1024 2048
		do
		for VIEW in 0
		do
			CKPT_DIR=pretrain-d-elu-view-$VIEW-of-$TOTALVIEW-rev-$REV-action4-Feat-$FEATNUM-working
			echo "\nmkdir -p ckpt/$CKPT_DIR"
			PREFIX=working-$TRY
			ITER=$((NSAMPLE * 8))

			echo "
	./pretrain-d.sh --ferev $REV --elu \\
		--ckptdir ckpt/$CKPT_DIR/ --ckptprefix $PREFIX \\
		--batch 2 --queuemax 64 --threads 1 \\
		--iter $ITER \\
		--view $VIEW \\
		--featnum $FEATNUM \\
		--imhidden $FEATNUM $FEATNUM \\
		--samplein sample/batch2-view14-norgbd-T2-R2-2M/ \\
		--sampletouse $NSAMPLE \\
		--samplebatching 32 \\
		--samplebase $BASE > nsample-idmv-action4/View-$VIEW-of-$TOTALVIEW-Rev-$REV-Feat-$FEATNUM-$END.out
			"
			ARCHIVED_CKPT=View-$VIEW-of-$TOTALVIEW-Action4-Rev-$REV-Feat-$FEATNUM-$END
			echo "cp -a ckpt/$CKPT_DIR ackpt/$ARCHIVED_CKPT\n"
		done
		done
	done
	BASE=$END
done

exit


SAMPLED=131072
# echo "1024 2048 4096 8192 `seq $NSAMPLE $NSAMPLE 1048575`"

# VGG for 4 Actions
OUTDIR=nsample-idmv-action4
TOTALVIEW=14
echo "mkdir -p $OUTDIR"

BASE=0
for END in 524288
do
	NSAMPLE=$((END - BASE))
	for REV in 9
	do
		for VIEW in `seq 0 1 $((TOTALVIEW -1))`
		do
			CKPT_DIR=pretrain-d-elu-view-$VIEW-of-$TOTALVIEW-rev-$REV-action4-working
			echo "\nmkdir -p ckpt/$CKPT_DIR"
			PREFIX=working-$TRY
			ITER=$((NSAMPLE * 8))

			echo "
	./pretrain-d.sh --ferev $REV --elu \\
		--ckptdir ckpt/$CKPT_DIR/ --ckptprefix $PREFIX \\
		--batch 2 --queuemax 64 --threads 1 \\
		--iter $ITER \\
		--view $VIEW \\
		--samplein sample/batch2-view14-norgbd-T2-R2-2M/ \\
		--sampletouse $NSAMPLE \\
		--samplebatching 16 \\
		--samplebase $BASE > nsample-idmv-action4/View-$VIEW-of-$TOTALVIEW-Rev-$REV-$END.out
			"
			ARCHIVED_CKPT=View-$VIEW-of-$TOTALVIEW-Action4-Rev-$REV-$END
			echo "cp -a ckpt/$CKPT_DIR ackpt/$ARCHIVED_CKPT\n"
		done
	done
	BASE=$END
done

exit

# IDMV for InDependent MultiView
OUTDIR=nsample-idmv-action4
TOTALVIEW=14
echo "mkdir -p $OUTDIR"

BASE=0
for END in 262144
do
	NSAMPLE=$((END - BASE))
	for REV in 4 5 6 7 8
	do
		for VIEW in `seq 0 1 $((TOTALVIEW -1))`
		do
			CKPT_DIR=pretrain-d-elu-view-$VIEW-of-$TOTALVIEW-rev-$REV-action4-working
			echo "\nmkdir -p ckpt/$CKPT_DIR"
			PREFIX=working-$TRY
			ITER=$((NSAMPLE * 8))

			echo "
	./pretrain-d.sh --ferev $REV --elu \\
		--ckptdir ckpt/$CKPT_DIR/ --ckptprefix $PREFIX \\
		--batch 2 --queuemax 64 --threads 1 \\
		--iter $ITER \\
		--view $VIEW \\
		--samplein sample/batch2-view14-norgbd-T2-R2-2M/ \\
		--sampletouse $NSAMPLE \\
		--samplebatching 16 \\
		--samplebase $BASE > nsample-idmv-action4/View-$VIEW-of-$TOTALVIEW-Rev-$REV-$END.out
			"
			ARCHIVED_CKPT=View-$VIEW-of-$TOTALVIEW-Action4-Rev-$REV-$END
			echo "cp -a ckpt/$CKPT_DIR ackpt/$ARCHIVED_CKPT\n"
		done
	done
	BASE=$END
done

exit

echo 'mkdir -p nsample-mv-action4/'

BASE=0
for END in 262144
do
	NSAMPLE=$((END - BASE))
	for REV in 4 5 6 7 8
	do
		CKPT_DIR=pretrain-d-elu-view-14-rev-$REV-action4-working
		echo "\nmkdir -p ckpt/$CKPT_DIR"
		PREFIX=working-$TRY
		ITER=$((NSAMPLE * 16))

		echo "
./pretrain-d.sh --ferev $REV --elu \\
	--ckptdir ckpt/$CKPT_DIR/ --ckptprefix $PREFIX \\
	--batch 2 --queuemax 64 --threads 1 \\
	--iter $ITER \\
	--committee \\
	--samplein sample/batch2-view14-norgbd-T2-R2-2M/ \\
	--sampletouse $NSAMPLE \\
	--samplebase $BASE > nsample-mv-action4/Rev-$REV-$END.out
		"
		PARTIAL_CKPT=View-14-Action4-Rev-$REV-$END
		echo "cp -a ckpt/$CKPT_DIR ckpt/$PARTIAL_CKPT\n"
	done
	BASE=$END
done

exit

BASE=0
for END in 1024 2048 4096 8192 `seq $SAMPLED $SAMPLED 524288`
do
	# echo "$END\n"
	# echo $((END - BASE))
	NSAMPLE=$((END - BASE))
	for REV in 4 5 6 7 8
	do
		CKPT_DIR=pretrain-d-elu-rev-$REV-action4-working
		echo "\nmkdir -p ckpt/$CKPT_DIR"
		PREFIX=working-$TRY
		ITER=$((NSAMPLE * 16))

		echo "
./pretrain-d.sh --ferev $REV --elu \\
	--ckptdir ckpt/$CKPT_DIR/ --ckptprefix $PREFIX \\
	--batch 2 --queuemax 64 --threads 1 \\
	--iter $ITER \\
	--samplein sample/batch2-view1-T2-R2-2M/ \\
	--sampletouse $NSAMPLE \\
	--samplebatching 8 \\
	--samplebase $BASE > nsample-action4/Rev-$REV-$END.out
		"
		PARTIAL_CKPT=Action4-Rev-$REV-$END
		echo "cp -a ckpt/$CKPT_DIR ckpt/$PARTIAL_CKPT\n"
	done
	BASE=$END
done

exit 

BASE=0
for END in 1024 2048 4096 8192 `seq $SAMPLED $SAMPLED 1048576`
do
	# echo "$END\n"
	# echo $((END - BASE))
	NSAMPLE=$((END - BASE))
	for REV in 4 5 7 8
	do
		CKPT_DIR=pretrain-d-elu-rev-$REV-action2-working-rev2
		echo "\nmkdir -p ckpt/$CKPT_DIR"
		PREFIX=working-$TRY
		ITER=$((NSAMPLE * 16))

		echo "
./pretrain-d.sh --ferev $REV --elu \\
	--ckptdir ckpt/$CKPT_DIR/ --ckptprefix $PREFIX \\
	--batch 2 --queuemax 64 --threads 1 \\
	--iter $ITER \\
	--samplein sample/batch2-view1-T2-2M/ \\
	--sampletouse $NSAMPLE \\
	--samplebatching 8 \\
	--samplebase $BASE > nsample-hole.out/Rev-$REV-$END.out
		"
		PARTIAL_CKPT=Action2-Rev-$REV-$END
		echo "cp -a ckpt/$CKPT_DIR ckpt/$PARTIAL_CKPT\n"
	done
	BASE=$END
done

exit 

for NSAMPLE in `seq 1024 1024 32768`
do
	PREFIX=sample$NSAMPLE-$TRY
	CKPT_DIR=pretrain-d-elu-action2-sample$NSAMPLE-$TRY
	ITER=$((NSAMPLE * 12))

	# ./pretrain-d.sh --ferev $REV --elu --sampleout sample/batch2-view1/ --ckptdir ckpt/pretrain-d-elu-rev$REV-$TRY/ --ckptprefix $PREFIX --batch 2 --queuemax 64 --threads 1 --iter $ITER
	./pretrain-d.sh --ferev $REV --elu \
		--ckptdir ckpt/$CKPT_DIR/ --ckptprefix $PREFIX \
		--batch 2 --queuemax 64 --threads 1 \
		--iter $ITER \
		--samplein sample/batch2-view1-XTrans/ \
		--sampletouse $NSAMPLE \
		--samplebase 0 > nsample.out/$NSAMPLE.out
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
