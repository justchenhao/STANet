

# train PAM
python ./train.py --save_epoch_freq 1 --angle 15 --dataroot path-to-LEVIR-CD-train --val_dataroot path-to-LEVIR-CD-val --name LEVIR-CDFAp0 --lr 0.001 --model --SA_mode PAM CDFA --batch_size 8 --load_size 256 --crop_size 256 --preprocess rotate_and_crop

