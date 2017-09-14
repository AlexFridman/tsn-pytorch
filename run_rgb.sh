python3 main.py ucf101 RGB \
    /media/e/vsd/data/ucf101_preprocessed/split_01/file_lists/train_rgb.txt \
    /media/e/vsd/data/ucf101_preprocessed/split_01/file_lists/test_rgb.txt \
    --arch BNInception --num_segments 3 \
    --gd 20 --lr 0.001 --lr_steps 30 60 --epochs 80 \
    -b 32 -j 4 \
    --snapshot_pref ucf101_bninception_ 
