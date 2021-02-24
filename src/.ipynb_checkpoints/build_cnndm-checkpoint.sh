#!/bin/bash
#python preprocess.py -mode format_to_lines -raw_path /home/bqw/nlp_data/cnndm/tokens -save_path /home/bqw/nlp_data/cnndm/graph_json/cnndm -map_path ../urls -lower False -n_cpus 64
 
python preprocess.py -mode format_to_bert -raw_path /home/bqw/nlp_data/cnndm/graph_json/ -save_path /home/bqw/nlp_data/cnndm/graph_bert_long -lower True -n_cpus 64 -log_file ../logs/preprocess.log

