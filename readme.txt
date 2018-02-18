Named Entity Recognition using Deep Maximum Entropy Markov Model


The algorithm finds the named entity within sentences .
It uses Deep Maximum Entropy Markov Model for entity probability and viterbi for the best tags sequence(inference).

The inputs are simply the sentences and their corresponding labels. The dataset consists of sentences in the form of indexes and target labels for each sentence in the form of indexes as well.There are also two dictionaries provided , one to convert the word index to the word and other to convert the label index to the label.


Libraries used :

import subprocess
import argparse
import sys
import gzip 

import torch
import torch.autograd as autograd
import torch.nn as nn
import numpy as np
import pickle as pk
from torch.autograd import Variable

1. Run the program as : python hw2.py or python hw2.py --data {path to the atis.small.pkl.gz file}
2. make sure conlleval.pl is present in the same folder
3. Also i have included my model.pkl in the same folder. Its the trained model which I have trained
