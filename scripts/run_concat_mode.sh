
# concat mode
list=train
lr=0.001
dataset_type=CD_data1,CD_data2,...,
dataset_type=LEVIR_CD1
val_dataset_type=LEVIR_CD1
dataset_mode=concat
name=project_name

python ./train.py --num_threads 4 --display_id 0 --dataset_type $dataset_type --val_dataset_type $val_dataset_type --save_epoch_freq 1 --niter 100 --angle 15 --niter_decay 100  --display_env FAp0 --SA_mode PAM --name $name --lr $lr --model CDFA --batch_size 4 --dataset_mode $dataset_mode --val_dataset_mode $dataset_mode --split $list --load_size 256 --crop_size 256 --preprocess resize_rotate_and_crop