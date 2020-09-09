# Transformer Multi-Language Translator
A multi-langauge translator that utilizes the transformer neural network model described by the paper titled **Attention Is All You Need** in late 2017. A recently rising Natural Language Processing Model shown to often compete with and even out perform LSTMs and GRUs. This Translator uses the Transformer Model is its basis. This project has multiple languages trained. The limit currently are resources thus all languages are paired with English. Please refer the languages section below for more information. This translator has fairly good accuracy considering it was trained on downsized datasets due to resource limitations as well as low epochs. If you have the resources, then you may clone the REPO and train the model on larger datasets as well as more Epochs.

Reference to **Attention Is All You Need** Paper: https://arxiv.org/pdf/1706.03762.pdf

## Model

The Transformer model can be split into to main components, the encoder and the decoder. 

![](data/uploads/transformer_architecture.png)

### Encoder
The input is embedded using nn.Embedding to create an embedding vector to uniquely represent each word as well as closely relate similar words. Position encoding encodes a token based on the position it has in a sequence. The encoder outputs a sequence of context vectors. Unlike the RNN where the token being read is only influenced by the hiddens states of previous tokens, in the transformer model, each token is influenced by all tokens in the sequence. The transformer model in the original paper uses static embedding while current and state-of-the-art transformer NLP models such as BERT use dynamic or learnable positional embeddings. As such, we will use learnable position embeddings. After the input sequence is embedded, it is passed through the Multi-Head Attention layer, which is the promiment aspect of a transformer model. Multiheaded attention takes in a value, key, and query. The Query and key are multiplied and scaled before their product is multiplied by the value. This is known as Scaled Dot-Product Attention as seen in the Diagram Below.

![](data/uploads/multi-head_attention.png)

Multi-Headed attention are multiple scaled dot-product attention layers stacked upon each other and concatenated, then passed through a dense layer. Multi-Headed attention allows the transformer model to jointly deal with multiple subspace representations from different positioned tokens.

### Decoder
The Decoder takes in the target value and applies token and positional encoding to it aswell. The decoder also contains a Multi-Head Attention layer, however the scaled dot-production attention layer in the Multi-Head Attention masks out all values that the softmax activation deems unnecessary or illegal, hence the name Masked Multi-Head Attention Layer. The output of the MMHAL is the query for the next regular Multi-Headed Attention while the key and value are the outputs from the encoder (encoded inputs). The decoder then passes the result through a Feed Forward Layer and then a classification Dense network for final predictions

## Training

### Organization of Data
Please refer to the dataset section for more details on the datasets I'm using. The data files I'm using for training are line-by-line text files meaning each line is training data. Each langauge file comes in pairs. For example, if I want to train the translator to work with English and French, the files I have are located in the ```data/french-english/``` directory and the files present ```english.txt``` and ```french.txt```. Thus, we have the file locations ```data/french-english/english.txt``` and ```data/french-english/french.txt```. This is the organization format I used in this project and the organization applies to all language pairs. Keep in mind, English, is usually the latter in the data/**language-language** directory name as you can see in the data directory.

Python Files
  - train.py
    - an executable python script that takes in parameters including hyperparameters for the transformer model as well as training paramters. While running, the programs save the model weights to a specified directory every epoch and displays training loss. Run this file as follows:
    ```
    python train.py --hyperparameters
    ```
    --hyperparameters are a list of hyperparameters to call in order to properly execute train.py. Each hyperparamter is to be entered in this format:
    ```
    --data_directory data
    ```
    followed by a space to seperate each hyperparameter entered. Please refer to the file to see specific hyperparamters.

## Samples

Since I do not know many of the langauges I have trained on the Transformer model, I will be using more robust translators as comparison such as Google Translate.

### English -> French
**Input:** "It is important to know."  
**Prediction:** "il est important de savoir ."  
**Google Translate:** "Il est important de savoir."  

**Input:** "What should I do?"  
**Prediction:** "que je devrais faire ?"  
**Google Translate:** "Que devrais-je faire?"  
