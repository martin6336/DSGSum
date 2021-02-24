#!/bin/bash

gpu='0,1'
gpu1=0
gpu3=-1
gpu4=3
data_dir='/home/bqw/nlp_data/cnndm/graph_bert/cnndm'
model_dir='../models/abs_gate_cnn'
result_dir='../logs/abs_gate_cnn'
ip=5501
log_dir='../logs/abs_gate_cnn'
train_step=140000
test_log='../logs/abs_gate_valid_cnn'
copy=False
lr_dec=0.2
warm_dec=10000
grad_norm=0
gate=True
max_pos=512
max_tgt_len=200
python train.py  -task abs -mode train -bert_data_path ${data_dir} -dec_dropout 0.2  -model_path ${model_dir} -sep_optim true -lr_bert 0.002 -lr_dec ${lr_dec} -save_checkpoint_steps 2000 -batch_size 280 -train_steps ${train_step} -report_every 50 -accum_count 5 -use_bert_emb true -use_interval true -warmup_steps_bert 20000 -warmup_steps_dec ${warm_dec} -max_pos 512 -visible_gpus ${gpu} -log_file ${log_dir} -copy ${copy} -init_method tcp://localhost:${ip}
#python train.py  -task abs -mode train -bert_data_path ${data_dir} -dec_dropout 0.2  -model_path ${model_dir} -sep_optim true -lr_bert 0.002 -lr_dec ${lr_dec} -save_checkpoint_steps 2000 -batch_size 280 -train_steps ${train_step} -report_every 50 -accum_count 5 -use_bert_emb true -use_interval true -warmup_steps_bert 20000 -warmup_steps_dec ${warm_dec} -max_pos 512 -visible_gpus ${gpu} -log_file ${log_dir} -copy ${copy} -max_grad_norm ${grad_norm} -init_method tcp://localhost:${ip}

#python train.py  -task abs -mode train -bert_data_path ${data_dir} -dec_dropout 0.2  -model_path ${model_dir} -sep_optim true -lr_bert 0.002 -lr_dec ${lr_dec} -save_checkpoint_steps 2000 -batch_size 280 -train_steps ${train_step} -report_every 50 -accum_count 5 -use_bert_emb true -use_interval true -warmup_steps_bert 20000 -warmup_steps_dec ${warm_dec} -max_pos ${max_pos} -visible_gpus ${gpu1},${gpu2} -log_file ${log_dir} -copy ${copy} -max_grad_norm ${grad_norm} -init_method tcp://localhost:${ip} -max_tgt_len ${max_tgt_len}

python train.py -task abs -mode validate -batch_size 3000 -test_batch_size 500 -bert_data_path ${data_dir} -log_file ${test_log} -sep_optim true -use_interval true -visible_gpus ${gpu1} -max_pos ${max_pos} -max_length 200 -alpha 0.95 -min_length 50 -result_path ${result_dir} -model_path ${model_dir} -test_all True -copy ${copy}



