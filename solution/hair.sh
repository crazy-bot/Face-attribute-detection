#----- Uncomment this to train
# CUDA_LAUNCH_BLOCKING=1 \
# CUDA_VISIBLE_DEVICES=1 \
# python -m mains.train_classifier \
#     --dataroot '/data/suparna/workspace/TinyPortraits_thumbnails/' \
#     --classification_type 'haircolor' \
#     --exp_name hair_exp_3 \
#     --batch_size 200 \
#     --epochs 50 \
#     --lr 0.001


#-------Uncomment this to test
# CUDA_LAUNCH_BLOCKING=1 \
# CUDA_VISIBLE_DEVICES=1 \
# python -m mains.test_classifier \
#     --dataroot '/data/suparna/workspace/TinyPortraits_thumbnails/' \
#     --classification_type 'haircolor' \
#     --exp_name hair_exp_3 \
#     --batch_size 200 \
#     --ckpt 'best' \

#-------Uncomment this to predict on unlabelled data
CUDA_LAUNCH_BLOCKING=1 \
CUDA_VISIBLE_DEVICES=1 \
python -m mains.predict \
    --dataroot '/data/suparna/workspace/face_reasearch/smallfaces' \
    --classification_type 'haircolor' \
    --exp_name hair_exp_3 \
    --ckpt 'best' \
