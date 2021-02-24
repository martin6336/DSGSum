#!/bin/bash
#python preprocess.py -mode format_to_lines -raw_path /home/bqw/nlp_data/nyt50/nyt50_token -save_path  /home/bqw/nlp_data/nyt50/nyt50_graph_json/cnndm -n_cpus 40 -use_bert_basic_tokenizer false -map_path ../urls -lower False

python preprocess.py -mode format_to_bert -raw_path /home/bqw/nlp_data/nyt50/nyt50_graph_json -save_path /home/bqw/nlp_data/nyt50/nyt50_graph_bert_view  -lower -n_cpus 60 -log_file ../logs/preprocess.log
