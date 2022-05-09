CUDA_LAUNCH_BLOCKING=1 \
CUDA_VISIBLE_DEVICES=1 \
python -m mains.train_classifier \
    --dataroot '/data/suparna/workspace/TinyPortraits_thumbnails/' \
    --labelfile 'asset/Tiny_Portraits_Attributes.csv' \
    --classification_type 'gender' \
    --exp_name face_exp_2 \
    --batch_size 200 \
    --epochs 50 \
    --lr 0.001

    


