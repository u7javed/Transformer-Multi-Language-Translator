#dictionary class that keeps track of all words in a given language as well as assigning a
#token to them

PAD_TOKEN = 0
SOS_TOKEN = 1
EOS_TOKEN = 2

#from pytorch's documentation website on NLP
class Dictionary:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_TOKEN: "PAD", SOS_TOKEN: "SOS", EOS_TOKEN: "EOS"}
        self.n_count = 3

    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_count
            self.word2count[word] = 1
            self.index2word[self.n_count] = word 
            self.n_count += 1
        else:
            self.word2count[word] += 1
