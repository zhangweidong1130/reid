#测试
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 main_dist.py --trainer trainer_dist_vit_kd --datadir /home/zhangweidong/person_reid_dataset/ \
    --data_train train_person_kd --batchid 100 --batchimage 2 --batchtest 100 --test_every 1 --epochs 200 --loss 1.0*KDLoss_vit --random_erasing --data_test kangzhuang_night \
    --margin 0.3 --height 320 --width 128 --num_classes 636748 --model vitbase16_768 --lr 1e-4 --weight_decay 0.04 --optimizer ADAMW --amsgrad --adjust_lr 2e-4 \
    --save 768_vitbase_person_240929_256_kd --resume 0 --pre_train experiment/0_256_vitbase/model.pt --test_all #--load 768_vitbase_person_240929_256_kd 
ps -ef | grep "main_dist" | grep -v grep | awk '{print "kill -9 "$2}' | sh