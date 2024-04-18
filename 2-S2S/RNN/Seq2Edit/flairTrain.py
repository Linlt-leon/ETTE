'''
This source adopt an idea of training ner using flair from Akash Chauhan May 3, 2020
But replace 'ner' with 'edit' since they are all encoder only for labelling task

https://github.com/flairNLP/flair.git
https://medium.com/thecyphy/training-custom-ner-model-using-flair-df1f9ea9c762

Last modified: Anocha 2/4/2024
'''
import pandas as pd
from flair.data import Corpus
from flair.datasets import ColumnCorpus

####################
# Reading data files

# define columns
columns = {0 : 'text', 1 : 'edit'}
# directory where the data resides
data_folder = '/home/ano/CS4248/Project/GED_baseline/ged_baselines/token_ged/FlairTrain/'
# initializing the corpus
corpus: Corpus = ColumnCorpus(data_folder, columns,
                              train_file = 'train.txt',
                              test_file = 'validate.txt',
                              dev_file = 'validate.txt')


print(len(corpus.train))
print(corpus.train[0].to_tagged_string('edit'))

####################
# Define tag

# tag to predict
label_type = 'edit'
# make tag dictionary from the corpus
tag_dictionary = corpus.make_label_dictionary(label_type=label_type, add_unk=True)#tag_type=tag_type)
print(tag_dictionary)


from pathlib import Path

from flair.models import SequenceTagger
from flair.data import Sentence

# tagger = SequenceTagger.load("flair/ner-english-fast")

# tagger.predict(Sentence("I live in Vienna."))

path = Path("test-flair")
path.mkdir(exist_ok=True)
# tagger.embeddings.embeddings[0].embedding = tagger.embeddings.embeddings[0].embedding
# tagger.save(path / "flair-ner-model.pt")

# new_tagger = tagger.load(path / "flair-ner-model.pt")


###################
from flair.embeddings import WordEmbeddings, StackedEmbeddings, FlairEmbeddings
from flair.embeddings.base import TokenEmbeddings
from typing import List
embedding_types : List[TokenEmbeddings] = [
        WordEmbeddings('glove'),
        FlairEmbeddings('news-forward'),
        FlairEmbeddings('news-backward'),
        ]
embeddings : StackedEmbeddings = StackedEmbeddings(
                                 embeddings=embedding_types)

####################
# Train SequenceTagger

from flair.models import SequenceTagger
tagger : SequenceTagger = SequenceTagger(hidden_size=256,
                                       embeddings=embeddings,
                                       tag_dictionary=tag_dictionary,
                                       tag_type=label_type,
                                       use_crf=True)
print(tagger)

path = Path("test-flair")
path.mkdir(exist_ok=True)
from flair.trainers import ModelTrainer
trainer : ModelTrainer = ModelTrainer(tagger, corpus)
trainer.train(path,#'resources/taggers/seq2edit',
              learning_rate=0.05,
              mini_batch_size=32,
              max_epochs=20)



from flair.data import Sentence
from flair.models import SequenceTagger

path = Path("test-flair/final-model.pt")
# load the trained model
model = SequenceTagger.load(path)# + 'final-model.pt')
# create example sentence
sentence = Sentence('I love the Berlin')
# predict the tags
model.predict(sentence)
print(sentence.to_tagged_string())