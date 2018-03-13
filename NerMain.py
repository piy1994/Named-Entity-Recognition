# This is the main script to perform Named Entity Recognition using Deep Maximum Entropy Markov model
# WordVecFeature class makes the wordvectors for the words in the sentences and for the tags in the 
# tag sequence.It also generats the training data that appends the present word's wordvector and 
# previous tag word vector.The neural network is used to predict the tag probabilities and 
# Viterbi algorithm is used to find the best tag sequence. A perl scripts evaluates the models 
# accuracy and fscore. The input data is in the form of word indexes and tag indexes and two dictionaries are
# provided to convert these indexes to their repective word and tag
# 

import subprocess
import argparse
import sys
import gzip 

import torch
import torch.autograd as autograd
import torch.nn as nn
#import torch.nn.functional as F
#import torch.optim as optim
import numpy as np
import pickle as pk
#import torchvision.datasets as dsets
#import torchvision.transforms as transforms
from torch.autograd import Variable



class WordVecFeatures():
    def __init__(self,wordvec_dimension,tag_dimension,indx2word,idx2label,label2idx,words2idx):
        self.n_word_vec = wordvec_dimension
        self.n_tag = tag_dimension
        self.idx2word = indx2word
        self.idx2label=idx2label
        self.label2idx = label2idx
        self.words2idx = words2idx
        
        self.embeds_word = nn.Embedding(len(words2idx),self.n_word_vec)
        self.embeds_tag = nn.Embedding(len(label2idx),self.n_tag)
        
        self.one_hot_tag_vec = torch.eye(len(self.idx2label))
        
    
    def idxList2Sent(self,idx_list):
        return ' '.join([self.idx2word[idx] for idx in idx_list])

    
    def idxList2Tags(self,idx_list):
        return [self.idx2label[idx] for idx in idx_list]
    
    def idxToWordVec(self,word_idx):
        lookup_tensor = torch.LongTensor([word_idx])
        word_vec = self.embeds_word(autograd.Variable(lookup_tensor))
        return word_vec

    def idx2TagVec(self,tag_idx):
        lookup_tensor = torch.LongTensor([tag_idx])
        tag_vec = self.embeds_tag(autograd.Variable(lookup_tensor))
        return tag_vec
    

    # generating the input data
    def featGenTrain(self,train_lex,train_y):
        feat_train = torch.FloatTensor(1,self.n_word_vec + self.n_tag).zero_()
        # now this is a dummy thing please remember !!

        feat_train_y = torch.LongTensor([0])
        
        for i in range(len(train_lex)):
            for j in range(len(train_lex[i])):
                word_vec = self.idxToWordVec(train_lex[i][j].item()).data
                if j == 0:
                    prev_tag_vec =torch.FloatTensor(1,self.n_tag).zero_()
                else:
                    prev_tag_vec = self.idx2TagVec(train_y[i][j-1].item()).data
                com_vec = torch.cat((word_vec,prev_tag_vec),1)
                #print feat_train.size()
                #print com_vec.size()
                feat_train = torch.cat((feat_train ,com_vec) , 0 )
                feat_train_y = torch.cat((feat_train_y,
                                          torch.LongTensor([train_y[i][j].item()])),0)
            print "Train {} made".format(i)
        return feat_train,feat_train_y
    
    # making the tag vector
    def makeTagVec(self):
        tag_vec_all = torch.FloatTensor(1,self.n_tag).zero_()
        for i in range(1,len(self.idx2label)):
            curr_tag = self.idx2TagVec(i).data
            tag_vec_all = torch.cat( (tag_vec_all,curr_tag) , 0)
        
        self.tag_vec_all  =tag_vec_all
    
    def testWordFeat(self,word_idx):
        word_vec_test = self.idxToWordVec(word_idx).data.repeat(
                len(self.idx2label),1)
        #print word_vec_test.size()
        return torch.cat( (word_vec_test,self.tag_vec_all) , 1)




class Classifier(object):
    def __init__(self):
        pass

    def train(self):
        """
        Override this method in your class to implement train
        """
        raise NotImplementedError("Train method not implemented")

    def inference(self):
        """
        Override this method in your class to implement inference
        """
        raise NotImplementedError("Inference method not implemented")

def conlleval(p, g, w, filename='tempfile.txt'):
    '''
    INPUT:
    p :: predictions
    g :: groundtruth
    w :: corresponding words

    OUTPUT:
    filename :: name of the file where the predictions
    are written. it will be the input of conlleval.pl script
    for computing the performance in terms of precision
    recall and f1 score
    '''
    out = ''
    for sl, sp, sw in zip(g, p, w):
        out += 'BOS O O\n'
        for wl, wp, ww in zip(sl, sp, sw):
            out += ww + ' ' + wl + ' ' + wp + '\n'
        out += 'EOS O O\n\n'

    f = open(filename, 'w')
    f.writelines(out)
    f.close()

    return get_perf(filename)

def get_perf(filename):
    ''' run conlleval.pl perl script to obtain precision/recall and F1 score '''
    _conlleval = 'conlleval.pl'

    proc = subprocess.Popen(["perl", _conlleval], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    stdout, _ = proc.communicate(open(filename).read())
    for line in stdout.split('\n'):
        if 'accuracy' in line:
            out = line.split()
            break

    precision = float(out[6][:-2])
    recall    = float(out[8][:-2])
    f1score   = float(out[10])

    return (precision, recall, f1score)


def viterbiAlgo(w_vec_test_lst,net):
    back = np.ones((127,len(w_vec_test_lst)),'int32')*-1
    trellis = np.zeros((127,len(w_vec_test_lst)))
    
    trellis[:,0] = net(Variable(w_vec_test_lst[0])).data.numpy()[0]
    #trellis[:,0] = np.random.rand(127)
    
    for t in range(1,len(w_vec_test_lst)):
        transprob = net(Variable(w_vec_test_lst[t])).data.numpy()
        trellis[:,t] = (np.tile(trellis[:, t-1, None],[1, 127]) * transprob).max(0)
        back[:, t] = (np.tile(trellis[:, t-1, None],[1, 127]) * transprob).argmax(0)
    
    
    tokens = [trellis[:,-1].argmax()]
    for i in xrange(trellis.shape[1]-1,0,-1):
        tokens.append(back[tokens[-1],i])
    return tokens[::-1]

# neural network
class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.fc3 = nn.Softmax()
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        #out = self.relu(out)
        out = self.fc3(out)
        return out

def main():

    argparser = argparse.ArgumentParser()
    argparser.add_argument("--data", type=str, default="atis.small.pkl.gz", help="The zipped dataset")

    parsed_args = argparser.parse_args(sys.argv[1:])

    filename = parsed_args.data
    f = gzip.open(filename,'rb')
    torch.manual_seed(1)
    train_set, valid_set, test_set, dicts = pk.load(f)

    train_lex, _, train_y = train_set
    valid_lex, _, valid_y = valid_set
    test_lex,  _,  test_y  = test_set

    idx2label = dict((k,v) for v,k in dicts['labels2idx'].iteritems())
    idx2word  = dict((k,v) for v,k in dicts['words2idx'].iteritems())

    '''
    To have a look what the original data look like, commnet them before your submission
    '''
    #print test_lex[0], map(lambda t: idx2word[t], test_lex[0])
    #print test_y[0], map(lambda t: idx2label[t], test_y[0])

    '''
    My word vec parameters
    '''
    n_word_vec = len(idx2word)/4
    n_tag_vec = len(idx2label)/4
    
    w_vec_obj = WordVecFeatures(n_word_vec,n_tag_vec,indx2word=idx2word,idx2label=idx2label,
                                      label2idx=dicts['labels2idx'],
                                      words2idx= dicts['words2idx'])
    
    w_vec_obj.makeTagVec()
    
    '''
    mkaing the network
    '''
    input_size =  n_word_vec + n_tag_vec
    num_classes = 127
    hidden_size = int((input_size + num_classes)*(2.0/3))
    
    net = Net(input_size, hidden_size, num_classes)
    net.load_state_dict(torch.load('model.pkl'))
    
    '''
    how to get f1 score using my functions, you can use it in the validation and training as well
    '''
    token_main = []
    for words_list in test_lex:
        w_vec_test_lst = []
        for i in words_list:
            w_vec_test_lst.append(w_vec_obj.testWordFeat(i.item()))
        
        token_main.append(viterbiAlgo(w_vec_test_lst,net))
    
    predictions_test = [ map(lambda t: idx2label[t], y) for y in token_main ]
    groundtruth_test = [ map(lambda t: idx2label[t], y) for y in test_y ]
    words_test = [ map(lambda t: idx2word[t], w) for w in test_lex ]
    test_precision, test_recall, test_f1score = conlleval(predictions_test, groundtruth_test, words_test)

    print 'test_precision: {0}, test_recall: {1}, test_f1score: {2}'.\
    format(test_precision, test_recall, test_f1score)



if __name__ == '__main__':
    main()
