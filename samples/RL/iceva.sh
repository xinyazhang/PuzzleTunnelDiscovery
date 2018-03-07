# Single Softmax with MISPOUT
REV=5

./pretrain-d.sh --ferev $REV --elu \
	--ckptdir ckpt/IC-Action12-Rev-$REV-262144/ --ckptprefix working-try11 \
	--batch 2 --queuemax 64 --threads 1 \
	--iter 524288 \
	--eval \
	--featnum 2048 \
	--imhidden 2048 2048 \
	--singlesoftmax \
	--samplein sample/batch2-view14-norgbd-T6-R6-2M/ \
	--mispout evaluation/IC-A12-MISP-1SF/ \
	--norgbd \
	--viewinitckpt \
ackpt/View-0-of-14-Action12-Rev-$REV-Feat-2048-524288 \
ackpt/View-1-of-14-Action12-Rev-$REV-Feat-2048-524288 \
ackpt/View-2-of-14-Action12-Rev-$REV-Feat-2048-524288 \
ackpt/View-3-of-14-Action12-Rev-$REV-Feat-2048-524288 \
ackpt/View-4-of-14-Action12-Rev-$REV-Feat-2048-524288 \
ackpt/View-5-of-14-Action12-Rev-$REV-Feat-2048-524288 \
ackpt/View-6-of-14-Action12-Rev-$REV-Feat-2048-524288 \
ackpt/View-7-of-14-Action12-Rev-$REV-Feat-2048-524288 \
ackpt/View-8-of-14-Action12-Rev-$REV-Feat-2048-524288 \
ackpt/View-9-of-14-Action12-Rev-$REV-Feat-2048-524288 \
ackpt/View-10-of-14-Action12-Rev-$REV-Feat-2048-524288 \
ackpt/View-11-of-14-Action12-Rev-$REV-Feat-2048-524288 \
ackpt/View-12-of-14-Action12-Rev-$REV-Feat-2048-524288 \
ackpt/View-13-of-14-Action12-Rev-$REV-Feat-2048-524288 \
	--sampletouse 524288 \
	--samplebase 1048576

exit

# NO EXIT
# exit
# RE-EVALUATE with PRED saved.

# MISPOUT
REV=5

./pretrain-d.sh --ferev $REV --elu \
	--ckptdir ckpt/IC-Action12-Rev-$REV-262144/ --ckptprefix working-try11 \
	--batch 2 --queuemax 64 --threads 1 \
	--iter 524288 \
	--eval \
	--featnum 2048 \
	--imhidden 2048 2048 \
	--samplein sample/batch2-view14-norgbd-T6-R6-2M/ \
	--mispout evaluation/IC-A12-MISP/ \
	--viewinitckpt \
ackpt/View-0-of-14-Action12-Rev-$REV-Feat-2048-524288 \
ackpt/View-1-of-14-Action12-Rev-$REV-Feat-2048-524288 \
ackpt/View-2-of-14-Action12-Rev-$REV-Feat-2048-524288 \
ackpt/View-3-of-14-Action12-Rev-$REV-Feat-2048-524288 \
ackpt/View-4-of-14-Action12-Rev-$REV-Feat-2048-524288 \
ackpt/View-5-of-14-Action12-Rev-$REV-Feat-2048-524288 \
ackpt/View-6-of-14-Action12-Rev-$REV-Feat-2048-524288 \
ackpt/View-7-of-14-Action12-Rev-$REV-Feat-2048-524288 \
ackpt/View-8-of-14-Action12-Rev-$REV-Feat-2048-524288 \
ackpt/View-9-of-14-Action12-Rev-$REV-Feat-2048-524288 \
ackpt/View-10-of-14-Action12-Rev-$REV-Feat-2048-524288 \
ackpt/View-11-of-14-Action12-Rev-$REV-Feat-2048-524288 \
ackpt/View-12-of-14-Action12-Rev-$REV-Feat-2048-524288 \
ackpt/View-13-of-14-Action12-Rev-$REV-Feat-2048-524288 \
	--sampletouse 1024 \
	--samplebase 1048576

# NO EXIT
# exit
# RE-EVALUATE TOTAL ACCURACY

# Single Softmax
REV=5

./pretrain-d.sh --ferev $REV --elu \
	--ckptdir ckpt/IC-Action12-Rev-$REV-262144/ --ckptprefix working-try11 \
	--batch 2 --queuemax 64 --threads 1 \
	--iter 524288 \
	--eval \
	--featnum 2048 \
	--imhidden 2048 2048 \
	--singlesoftmax \
	--samplein sample/batch2-view14-norgbd-T6-R6-2M/ \
	--viewinitckpt \
ackpt/View-0-of-14-Action12-Rev-$REV-Feat-2048-524288 \
ackpt/View-1-of-14-Action12-Rev-$REV-Feat-2048-524288 \
ackpt/View-2-of-14-Action12-Rev-$REV-Feat-2048-524288 \
ackpt/View-3-of-14-Action12-Rev-$REV-Feat-2048-524288 \
ackpt/View-4-of-14-Action12-Rev-$REV-Feat-2048-524288 \
ackpt/View-5-of-14-Action12-Rev-$REV-Feat-2048-524288 \
ackpt/View-6-of-14-Action12-Rev-$REV-Feat-2048-524288 \
ackpt/View-7-of-14-Action12-Rev-$REV-Feat-2048-524288 \
ackpt/View-8-of-14-Action12-Rev-$REV-Feat-2048-524288 \
ackpt/View-9-of-14-Action12-Rev-$REV-Feat-2048-524288 \
ackpt/View-10-of-14-Action12-Rev-$REV-Feat-2048-524288 \
ackpt/View-11-of-14-Action12-Rev-$REV-Feat-2048-524288 \
ackpt/View-12-of-14-Action12-Rev-$REV-Feat-2048-524288 \
ackpt/View-13-of-14-Action12-Rev-$REV-Feat-2048-524288 \
	--sampletouse 524288 \
	--samplebase 1048576 > evaluation/IC-A12-Base-1048576-Rev-$REV-SingleSoftmax.out

exit

# Eval 2048-width hidden layers
REV=5

./pretrain-d.sh --ferev $REV --elu \
	--ckptdir ckpt/IC-Action12-Rev-$REV-262144/ --ckptprefix working-try11 \
	--batch 2 --queuemax 64 --threads 1 \
	--iter 524288 \
	--eval \
	--featnum 2048 \
	--imhidden 2048 2048 \
	--samplein sample/batch2-view14-norgbd-T6-R6-2M/ \
	--viewinitckpt \
ackpt/View-0-of-14-Action12-Rev-$REV-Feat-2048-524288 \
ackpt/View-1-of-14-Action12-Rev-$REV-Feat-2048-524288 \
ackpt/View-2-of-14-Action12-Rev-$REV-Feat-2048-524288 \
ackpt/View-3-of-14-Action12-Rev-$REV-Feat-2048-524288 \
ackpt/View-4-of-14-Action12-Rev-$REV-Feat-2048-524288 \
ackpt/View-5-of-14-Action12-Rev-$REV-Feat-2048-524288 \
ackpt/View-6-of-14-Action12-Rev-$REV-Feat-2048-524288 \
ackpt/View-7-of-14-Action12-Rev-$REV-Feat-2048-524288 \
ackpt/View-8-of-14-Action12-Rev-$REV-Feat-2048-524288 \
ackpt/View-9-of-14-Action12-Rev-$REV-Feat-2048-524288 \
ackpt/View-10-of-14-Action12-Rev-$REV-Feat-2048-524288 \
ackpt/View-11-of-14-Action12-Rev-$REV-Feat-2048-524288 \
ackpt/View-12-of-14-Action12-Rev-$REV-Feat-2048-524288 \
ackpt/View-13-of-14-Action12-Rev-$REV-Feat-2048-524288 \
	--sampletouse 524288 \
	--samplebase 1048576 > evaluation/IC-A12-Base-1048576-Rev-$REV.out

exit

REV=5
# ReLU

./pretrain-d.sh --ferev $REV \
	--ckptdir ckpt/ReLU-IC-Action12-Rev-$REV-262144/ --ckptprefix working-try11 \
	--batch 2 --queuemax 64 --threads 1 \
	--iter 524288 \
	--eval \
	--featnum 2048 \
	--imhidden 2048 2048 \
	--samplein sample/batch2-view14-norgbd-T6-R6-2M/ \
	--viewinitckpt \
ackpt/ReLU-View-0-of-14-Action12-Rev-$REV-Feat-2048-524288 \
ackpt/ReLU-View-1-of-14-Action12-Rev-$REV-Feat-2048-524288 \
ackpt/ReLU-View-2-of-14-Action12-Rev-$REV-Feat-2048-524288 \
ackpt/ReLU-View-3-of-14-Action12-Rev-$REV-Feat-2048-524288 \
ackpt/ReLU-View-4-of-14-Action12-Rev-$REV-Feat-2048-524288 \
ackpt/ReLU-View-5-of-14-Action12-Rev-$REV-Feat-2048-524288 \
ackpt/ReLU-View-6-of-14-Action12-Rev-$REV-Feat-2048-524288 \
ackpt/ReLU-View-7-of-14-Action12-Rev-$REV-Feat-2048-524288 \
ackpt/ReLU-View-8-of-14-Action12-Rev-$REV-Feat-2048-524288 \
ackpt/ReLU-View-9-of-14-Action12-Rev-$REV-Feat-2048-524288 \
ackpt/ReLU-View-10-of-14-Action12-Rev-$REV-Feat-2048-524288 \
ackpt/ReLU-View-11-of-14-Action12-Rev-$REV-Feat-2048-524288 \
ackpt/ReLU-View-12-of-14-Action12-Rev-$REV-Feat-2048-524288 \
ackpt/ReLU-View-13-of-14-Action12-Rev-$REV-Feat-2048-524288 \
	--sampletouse 524288 \
	--samplebase 1048576 > evaluation/ReLU-IC-A12-Base-1048576-Rev-$REV.out

exit

REV=6

./pretrain-d.sh --ferev $REV --elu \
	--ckptdir ckpt/IC-Action4-Rev-$REV-262144/ --ckptprefix working-try11 \
	--batch 2 --queuemax 64 --threads 1 \
	--iter 262144 \
	--eval \
	--samplein sample/batch2-view14-norgbd-T2-R2-2M/ \
	--viewinitckpt \
ackpt/View-0-of-14-Action4-Rev-$REV-262144 \
ackpt/View-1-of-14-Action4-Rev-$REV-262144 \
ackpt/View-2-of-14-Action4-Rev-$REV-262144 \
ackpt/View-3-of-14-Action4-Rev-$REV-262144 \
ackpt/View-4-of-14-Action4-Rev-$REV-262144 \
ackpt/View-5-of-14-Action4-Rev-$REV-262144 \
ackpt/View-6-of-14-Action4-Rev-$REV-262144 \
ackpt/View-7-of-14-Action4-Rev-$REV-262144 \
ackpt/View-8-of-14-Action4-Rev-$REV-262144 \
ackpt/View-9-of-14-Action4-Rev-$REV-262144 \
ackpt/View-10-of-14-Action4-Rev-$REV-262144 \
ackpt/View-11-of-14-Action4-Rev-$REV-262144 \
ackpt/View-12-of-14-Action4-Rev-$REV-262144 \
ackpt/View-13-of-14-Action4-Rev-$REV-262144 \
	--sampletouse 262144 \
	--samplebase 524288 > evaluation/IC-A4-Base-524288-Rev-$REV.out

REV=7

./pretrain-d.sh --ferev $REV --elu \
	--ckptdir ckpt/IC-Action4-Rev-$REV-262144/ --ckptprefix working-try11 \
	--batch 2 --queuemax 64 --threads 1 \
	--iter 262144 \
	--eval \
	--samplein sample/batch2-view14-norgbd-T2-R2-2M/ \
	--viewinitckpt \
ackpt/View-0-of-14-Action4-Rev-$REV-262144 \
ackpt/View-1-of-14-Action4-Rev-$REV-262144 \
ackpt/View-2-of-14-Action4-Rev-$REV-262144 \
ackpt/View-3-of-14-Action4-Rev-$REV-262144 \
ackpt/View-4-of-14-Action4-Rev-$REV-262144 \
ackpt/View-5-of-14-Action4-Rev-$REV-262144 \
ackpt/View-6-of-14-Action4-Rev-$REV-262144 \
ackpt/View-7-of-14-Action4-Rev-$REV-262144 \
ackpt/View-8-of-14-Action4-Rev-$REV-262144 \
ackpt/View-9-of-14-Action4-Rev-$REV-262144 \
ackpt/View-10-of-14-Action4-Rev-$REV-262144 \
ackpt/View-11-of-14-Action4-Rev-$REV-262144 \
ackpt/View-12-of-14-Action4-Rev-$REV-262144 \
ackpt/View-13-of-14-Action4-Rev-$REV-262144 \
	--sampletouse 262144 \
	--samplebase 524288 > evaluation/IC-A4-Base-524288-Rev-$REV.out

exit


REV=5

./pretrain-d.sh --ferev $REV --elu \
	--ckptdir ckpt/IC-Action4-Rev-$REV-262144/ --ckptprefix working-try11 \
	--batch 2 --queuemax 64 --threads 1 \
	--iter 262144 \
	--eval \
	--samplein sample/batch2-view14-norgbd-T2-R2-2M/ \
	--viewinitckpt \
ackpt/View-0-of-14-Action4-Rev-$REV-262144 \
ackpt/View-1-of-14-Action4-Rev-$REV-262144 \
ackpt/View-2-of-14-Action4-Rev-$REV-262144 \
ackpt/View-3-of-14-Action4-Rev-$REV-262144 \
ackpt/View-4-of-14-Action4-Rev-$REV-262144 \
ackpt/View-5-of-14-Action4-Rev-$REV-262144 \
ackpt/View-6-of-14-Action4-Rev-$REV-262144 \
ackpt/View-7-of-14-Action4-Rev-$REV-262144 \
ackpt/View-8-of-14-Action4-Rev-$REV-262144 \
ackpt/View-9-of-14-Action4-Rev-$REV-262144 \
ackpt/View-10-of-14-Action4-Rev-$REV-262144 \
ackpt/View-11-of-14-Action4-Rev-$REV-262144 \
ackpt/View-12-of-14-Action4-Rev-$REV-262144 \
ackpt/View-13-of-14-Action4-Rev-$REV-262144 \
	--sampletouse 262144 \
	--samplebase 524288 > evaluation/IC-A4-Base-524288-Rev-$REV.out
# Final Accuracy 66.2255621889%

exit

./pretrain-d.sh --ferev 4 --elu \
	--ckptdir ckpt/IC-Action4-Rev-4-262144/ --ckptprefix working-try11 \
	--batch 2 --queuemax 64 --threads 1 \
	--iter 262144 \
	--eval \
	--samplein sample/batch2-view14-norgbd-T2-R2-2M/ \
	--viewinitckpt \
ackpt/View-0-of-14-Action4-Rev-4-262144 \
ackpt/View-1-of-14-Action4-Rev-4-262144 \
ackpt/View-2-of-14-Action4-Rev-4-262144 \
ackpt/View-3-of-14-Action4-Rev-4-262144 \
ackpt/View-4-of-14-Action4-Rev-4-262144 \
ackpt/View-5-of-14-Action4-Rev-4-262144 \
ackpt/View-6-of-14-Action4-Rev-4-262144 \
ackpt/View-7-of-14-Action4-Rev-4-262144 \
ackpt/View-8-of-14-Action4-Rev-4-262144 \
ackpt/View-9-of-14-Action4-Rev-4-262144 \
ackpt/View-10-of-14-Action4-Rev-4-262144 \
ackpt/View-11-of-14-Action4-Rev-4-262144 \
ackpt/View-12-of-14-Action4-Rev-4-262144 \
ackpt/View-13-of-14-Action4-Rev-4-262144 \
	--sampletouse 262144 \
	--samplebase 524288
# Final Accuracy 64.6069923134%
exit

REV=6

./pretrain-d.sh --ferev $REV --elu \
	--ckptdir ckpt/IC-Action4-Rev-$REV-262144/ --ckptprefix working-try11 \
	--batch 2 --queuemax 64 --threads 1 \
	--iter 262144 \
	--eval \
	--samplein sample/batch2-view14-norgbd-T2-R2-2M/ \
	--viewinitckpt \
ackpt/View-0-of-14-Action4-Rev-$REV-262144 \
ackpt/View-1-of-14-Action4-Rev-$REV-262144 \
ackpt/View-2-of-14-Action4-Rev-$REV-262144 \
ackpt/View-3-of-14-Action4-Rev-$REV-262144 \
ackpt/View-4-of-14-Action4-Rev-$REV-262144 \
ackpt/View-5-of-14-Action4-Rev-$REV-262144 \
ackpt/View-6-of-14-Action4-Rev-$REV-262144 \
ackpt/View-7-of-14-Action4-Rev-$REV-262144 \
ackpt/View-8-of-14-Action4-Rev-$REV-262144 \
ackpt/View-9-of-14-Action4-Rev-$REV-262144 \
ackpt/View-10-of-14-Action4-Rev-$REV-262144 \
ackpt/View-11-of-14-Action4-Rev-$REV-262144 \
ackpt/View-12-of-14-Action4-Rev-$REV-262144 \
ackpt/View-13-of-14-Action4-Rev-$REV-262144 \
	--sampletouse 262144 \
	--samplebase 524288

exit

