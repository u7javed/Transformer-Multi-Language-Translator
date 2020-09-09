import re
import unicodedata
import torch

from dictionary import Dictionary

PAD_TOKEN = 0
SOS_TOKEN = 1
EOS_TOKEN = 2

#https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

#load language into lists of sentences where corresponding indeces are translations from
#one language to the other

#reverse controls with language is input and output
#FALSE: lang1 is input and lang2 is output
#TRUE: lang2 is input and lang1 is output
def load_files(lang1, lang2, data_dir, reverse=True, MAX_FILE_SIZE=100000, MAX_LENGTH=60):
    #load first language to list
    lang1_list = []
    lang1_file = open(data_dir + '/' + lang1 + '-' + lang2 + '/' + lang1 + '.txt', 'r', encoding='utf8')
    for i, (line) in enumerate(lang1_file):
        if i < MAX_FILE_SIZE:
            lang1_list.append(line)
        else:
            break

    # load second langauge to list
    lang2_list = []
    lang2_file = open(data_dir + '/' + lang1 + '-' + lang2 + '/' + lang2 + '.txt', 'r', encoding='utf8')
    for i, (line) in enumerate(lang2_file):
        if i < MAX_FILE_SIZE:
            lang2_list.append(line)
        else:
            break

    #preprocess strings
    lang1_normalized = list(map(normalizeString, lang1_list))
    lang2_normalized = list(map(normalizeString, lang2_list))

    lang1_sentences = []
    lang2_sentences = []

    for i in range(len(lang1_normalized)):
        tokens1 = lang1_normalized[i].split(' ')
        tokens2 = lang2_normalized[i].split(' ')
        if len(tokens1) <= MAX_LENGTH and len(tokens2) <= MAX_LENGTH:
            lang1_sentences.append(lang1_normalized[i])
            lang2_sentences.append(lang2_normalized[i])

    del lang1_normalized
    del lang2_normalized

    if reverse:
        input_dic = Dictionary(lang2)
        output_dic = Dictionary(lang1)
        return input_dic, output_dic, lang2_sentences, lang1_sentences
    else:
        input_dic = Dictionary(lang1)
        output_dic = Dictionary(lang2)
        return input_dic, output_dic, lang1_sentences, lang2_sentences

#takes in a sentence and dictionary, and tokenizes based on dictionary
def tokenize(sentence, dictionary, MAX_LENGTH=60):
    split_sentence = [word for word in sentence.split(' ')]
    token = [SOS_TOKEN]
    token += [dictionary.word2index[word] for word in sentence.split(' ')]
    token.append(EOS_TOKEN)
    token += [PAD_TOKEN]*(MAX_LENGTH - len(split_sentence))
    return token

#create dataloader from a batch size and the two language lists
def load_batches(input_lang, output_lang, batch_size, device):
    data_loader = []
    for i in range(0, len(input_lang), batch_size):
        seq_length = min(len(input_lang) - batch_size, batch_size)
        input_batch = input_lang[i:i+seq_length][:]
        target_batch = output_lang[i:i+seq_length][:]
        input_tensor = torch.LongTensor(input_batch).to(device)
        target_tensor = torch.LongTensor(target_batch).to(device)
        data_loader.append([input_tensor, target_tensor])
    return data_loader