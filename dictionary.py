#dictionary class that keeps track of all words in a given language as well as assigning a
#token to them

#from pytorch's documentation website on NLP
PAD_TOKEN = 0
SOS_TOKEN = 1
EOS_TOKEN = 2

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


#returns a dataloader similar to that of pytorch's that contains input variable 
#and target variable as tuple
def create_batches(input_lang, output_lang, batch_size, device):
    data_loader = []
    for i in range(0, len(input_lang), batch_size):
        seq_length = min(len(input_lang) - batch_size, batch_size)
        input_batch = input_lang[i:i+seq_length][:]
        target_batch = output_lang[i:i+seq_length][:]
        input_tensor = torch.LongTensor(input_batch).to(device)
        target_tensor = torch.LongTensor(target_batch).to(device)
        data_loader.append([input_tensor, target_tensor])

    return data_loader