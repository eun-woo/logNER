
# LogNER

LogNER is a log parser that generates a log template through Nested NER 


### Running environment

#### python 3.8
#### regex 2024.9.11
#### torch 2.1.0

### How to run
#### Train
$ python train.py -n 5 --lr 7e-6 --cnn_dim 200 --biaffine_size 400 --n_head 4 -b 8 -d customized_data  --logit_drop 0 --cnn_depth 3 

#### Inference
$ python inference.py -n 5 --lr 7e-6 --cnn_dim 200 --biaffine_size 400 --n_head 4 -b 8 -d inference  --logit_drop 0 --cnn_depth 3 
