%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This section of code are for GEC task adopted from various sources 
% All are edited by Anocha S
% Last edit: Apr 2024

1. GEC using RNN from CS4248 ipynb

> Preparing data
- using same sentence pairs from Liang
- convert2pair.py for json to txt 
- 3.1 Data Preparation.ipynb keep using  spacy.load("en_core_web_trf")
-- src.vocab
-- tgt.vocab
-- vectorized.txt

> Train
- 3.2 RNN MT.ipynb
- using Same RNN architecture as deu-eng
- using
    "vocab_size_encoder": 20006,        # the size of the source vocabulary determines the input size of the encoder embedding
    "vocab_size_decoder": 19591,        # the size of the target vocabulary determines the input size of the decoder embedding
    "embed_size": 300,                           # size of the word embeddings (here the same for encoder and decoder; but not mandatory)
    "rnn_cell": "LSTM",                          # in practice GRU or LSTM will always outperform RNN
    "rnn_hidden_size": 512,                      # size of the hidden state
    "rnn_num_layers": 2,                         # 1 or 2 layers are most common; more rarely sees any benefit
    "rnn_dropout": 0.2,                          # only relevant if rnn_num_layers > 1
    "rnn_encoder_bidirectional": True,           # The encoder can be bidirectional; the decoder can not
    "linear_hidden_sizes": [1024, 2048],         # list of sizes of subsequent hidden layers; can be [] (empty); only relevant for the decoder
    "linear_dropout": 0.2,                       # if hidden linear layers are used, we can also include Dropout; only relevant for the decoder
    "attention": "DOT",                          # Specify if attention should be used; only "DOT" supported; None if no attention
    "teacher_forcing_prob": 0.5,                 # Probability of using Teacher Forcing during training by the decoder
    "special_token_unk": vocab_tgt['<UNK>'],     # Index of special token <UNK>
    "special_token_sos": vocab_tgt['<SOS>'],     # Index of special token <SOS>
    "special_token_eos": vocab_tgt['<EOS>'],     # Index of special token <EOS>
- using CrossEntropyLoss()

> Test
- using ABCN.dev.gold.bea19.json, Awe/valid.m2, test/ABCN.test.bea19.orig
- using Chris' 3.1 - Data Preparation (MT) to create vectorized.txt
- using Chris' 3.2 Test section to test
- getting xxx.out
- using post processing removing duplicate tokens
- using NUS's m2scorer (modified 4 python 3)
>>> python ./m2scorer.py -v ../../ChrisRNN/ABC.train.110-src-tgt-rnn.unique.out ../../ChrisRNN/ABCN.dev.gold.bea
19.4382.m2 > m2scorer.V2110unique.lower.log


Note that: 'en_core_web_trf' is case sensitive
>>> import spacy
>>> nlp = spacy.load('en_core_web_trf')
>>> nlp.vocab.strings['apple']
8566208034543834098
>>> nlp.vocab.strings['Apple']
6418411030699964375


2. Sequence to Edit using Flair, sentence labelling (https://github.com/flairNLP/flair)
- Prepare dataframe for sentence labeling
- Edit train.py
% This source adopt an idea of training ner using flair from Akash Chauhan May 3, 2020
% But replace 'ner' with 'edit' since they are all encoder only for labelling task
% 
% https://github.com/flairNLP/flair.git
% https://medium.com/thecyphy/training-custom-ner-model-using-flair-df1f9ea9c762
%
% Last modified: Anocha 2/4/2024 


