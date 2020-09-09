
import torch
import argparse
import pickle

from models import Transformer, Encoder, Decoder
from dictionary import Dictionary
from utilities import *

SOS_TOKEN = 1

def load_dictionary(directory):
    with open(directory, 'rb') as f:
        return pickle.load(f)

def translate_sentence(sentence, input_dic, output_dic, model, device, max_len):
    
    model.eval()
    normalized_sentence = normalizeString(sentence)
    tokens = tokenize(normalized_sentence, input_dic)
    input_tensor = torch.LongTensor(tokens).unsqueeze(0).to(device)
    input_mask = model.make_input_mask(input_tensor)
    
    with torch.no_grad():
        encoded_input = model.encoder(input_tensor, input_mask)

    target_tokens = [SOS_TOKEN]

    for i in range(max_len):

        target_tensor = torch.LongTensor(target_tokens).unsqueeze(0).to(device)
        target_mask = model.make_target_mask(target_tensor)
    
        with torch.no_grad():
            output, attention = model.decoder(target_tensor, encoded_input, target_mask, input_mask)
        
        pred_token = output.argmax(2)[:,-1].item()
        target_tokens.append(pred_token)
        if pred_token == EOS_TOKEN:
            break
    
    target_results = [output_dic.index2word[i] for i in target_tokens]
    
    return ' '.join(target_results[1:-1]), attention

def main():
    #take in arguments
    parser = argparse.ArgumentParser(description='Hyperparameters for training GAN')

    # parameters needed to enhance image
    parser.add_argument('--input_text', type=str, default='Today is a good day.', help='text that will be translated to output language')
    parser.add_argument('--input_lang', type=str, default='english', help='input langauge')
    parser.add_argument('--output_lang', type=str, default='French', help='output language')

    #default hyperparameters 
    parser.add_argument('--MAX_LENGTH', type=int, default=60, help='max number of tokens in input')
    parser.add_argument('--hidden_size', type=int, default=256, help='number of hidden layers in transformer')
    parser.add_argument('--encoder_layers', type=int, default=3, help='number of encoder layers')
    parser.add_argument('--decoder_layers', type=int, default=3, help='number of decoder layers')
    parser.add_argument('--encoder_heads', type=int, default=8, help='number of encoder heads')
    parser.add_argument('--decoder_heads', type=int, default=8, help='number of decoder heads')
    parser.add_argument('--encoder_ff_size', type=int, default=512, help='fully connected input size for encoder')
    parser.add_argument('--decoder_ff_size', type=int, default=512, help='fully connected input size for decoder')
    parser.add_argument('--encoder_dropout', type=float, default=0.1, help='dropout for encoder feed forward')
    parser.add_argument('--decoder_dropout', type=float, default=0.1, help='dropout for decoder feed forward')

    parser.add_argument('--models_directory', type=str, default='saved_models/', help='directory where models are saved')
  

    args = parser.parse_args()

    input_text = args.input_text
    input_lang = args.input_lang
    output_lang = args.output_lang
    models_dir = args.models_directory
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    #hyper parameters
    MAX_LENGTH = args.MAX_LENGTH
    hidden_size = args.hidden_size
    encoder_layers = args.encoder_layers
    decoder_layers = args.decoder_layers
    encoder_heads = args.encoder_heads
    decoder_heads = args.decoder_heads
    encoder_ff_size = args.encoder_ff_size
    decoder_ff_size = args.decoder_ff_size
    encoder_dropout = args.encoder_dropout
    decoder_dropout = args.decoder_dropout

    transformer_location = models_dir + '/' + input_lang + '2' + output_lang + '/'

    #load dictionaries
    input_lang_dic = load_dictionary(transformer_location + 'input_dic.pkl')
    output_lang_dic = load_dictionary(transformer_location + 'output_dic.pkl')

    input_size = input_lang_dic.n_count
    output_size = output_lang_dic.n_count
    
    #define models
    encoder_part = Encoder(input_size, hidden_size, encoder_layers, encoder_heads, encoder_ff_size, encoder_dropout, device)
    decoder_part = Decoder(output_size, hidden_size, decoder_layers, decoder_heads, decoder_ff_size, decoder_dropout, device)

    translator = Transformer(encoder_part, decoder_part, device).to(device)
    translator.load_state_dict(torch.load(transformer_location + 'transformer_model.pt'))

    translation, attention = translate_sentence(input_text, input_lang_dic, output_lang_dic, translator, device, MAX_LENGTH)
    print(input_lang + ': ' + input_text)
    print('\n' + output_lang + ': ' + translation)


if __name__ == "__main__":
    main()