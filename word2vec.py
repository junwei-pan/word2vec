import logging
import string
from numpy import exp, dot, zeros, outer, random, dtype, float32 as REAL, empty, ndarray, sum as np_sum

class Word2Vec():
    def __init__(self, sentences, layer_size = 100, window = 5, min_count = 0, model = 0, optimization = 1, alpha = 0.05, negative = 10, cbow_mean = True):
        '''
        `sentences` is a list, each element is a list of words
        `layer_size` is the size of hidden layer
        `window` is the window of context
        `min_count` = ignore all words with count less than this threshold
        `model` 0 for CBOW, 1 for Skip-Gram
        `optimization` 0 for hierarchical softmax, 1 for negative sampling
        `alpha` is the learning rate
        `negative` is the number of negative samples used in negative sampling
        `cbow_mean` indicates whether average projection vector in CBOW
        '''
        self.sentences = sentences
        self.layer_size = layer_size
        self.window = window
        self.min_count = min_count
        self.model = model
        self.optimization = optimization
        self.alpha = alpha
        self.negative = negative
        self.cbow_mean = cbow_mean
        self.build_vocab(sentences)
        self.w_hidden_2_output = zeros((len(self.l_word), self.layer_size), dtype = REAL)
        self.hidden = empty((len(self.l_word), self.layer_size), dtype = REAL)
        for i in xrange(len(self.l_word)):
            self.hidden[i] = (random.rand(self.layer_size) - 0.5) / self.layer_size

    def train(self):
        if self.model == 0:
            # CBOW
            for sentence in self.sentences:
                for pos, word in enumerate(sentence):
                    if word is None:
                        continue
                    # Using words in the random window of the given words to predict this word
                    random_window = random.randint(model.window)
                    start = max(0, pos - random_window)
                    context_indices = [self.index[w] for w in (sentence[start: pos + random_window]) if(w is not None and w != word)]
                    # Projection
                    l1 = np_sum(self.hidden[context_indices], axis = 0)
                    # Mean
                    if self.cbow_mean and context_indices:
                        l1 /= len(context_indices)
                    neu1e = zeros(l1.shape)
                    # negative sampling
                    if self.optimization == 1:
                        # Using the current word as positive samples and #negative other random words as negative samples
                        labels = zeros(self.negative + 1)
                        labels[0] = 1.
                        # The indices of words using as targets, including current word and #negative other random words
                        word_indices = [self.index[word]]
                        while len(word_indices) < self.negative + 1:
                            index_w = random.randint(self.n_word)
                            if index_w != self.index[word]:
                                word_indices.append(index_w)
                        # The weight from hidden layer to output layer
                        l2b = self.w_hidden_2_output[word_indices]
                        # Hidden layer to output layer
                        fb = 1. / (1. + exp(-dot(l1, l2b.T)))
                        # Error * learning rate
                        gb = (labels - fb) * self.alpha
                        # Update weights
                        self.w_hidden_2_output[word_indices] += outer(gb, l1)
                        neu1e += dot(gb, l2b)
                    # Update word vectors
                    self.hidden[context_indices] += neu1e

        elif self.model == 1:
            pass

    def build_vocab(self, sentences):
        '''
        `sentences` is a list of sentence, a sentence is a list of words
        |self.word| is a list of valid words
        |self.index| is a dict, word as key and index as value
        |self.n_word| is the number of valid words
        This function build the vocabulary of all valid words, a word is valid if its count is higher than $min_count
        '''
        d_word = {}
        for sentence in sentences:
            for word in sentence:
                d_word.setdefault(word, 0)
                d_word[word] += 1
        self.l_word = []
        self.index = {}
        i = 0
        for word in d_word.keys():
            if d_word[word] >= self.min_count:
                self.l_word.append(word)
                self.index.setdefault(word, i)
                i += 1
        self.n_word = i

    def store_model(self, path):
        file = open(path, 'w')
        for index, word in enumerate(self.l_word):
            file.write(str(word) + '\t' + ','.join(map(str, [v for v in self.hidden[index]])) + '\n')
        file.close()

class Monolingual():
    def __init__(self, path):
        self.path = path
    def __iter__(self):
        for line in open(path):
            line_no_punctuation = line.strip('\n').translate(string.maketrans("",""), string.punctuation)
            sentence = line_no_punctuation.split(' ')
            yield sentence

if __name__ == '__main__':
    #path = '/Users/jwpan/Downloads/training-monolingual/news.2011.en.shuffled'
    path = 'text_tmp.txt'
    text = Monolingual(path)
    model = Word2Vec(text)
    model.train()
    model.store_model('model.txt')
