# Transformer Multi-Language-Translator
A multi-langauge translator that utilizes the transformer neural network model from the paper titled **Attention Is All You Need**. 

## Training

### Organization of Data
Please refer to the dataset section for more details on the datasets I'm using. The data files I'm using for training are line by line files meaning each line is training. Each langauge file comes in pairs. For example, if I want to train the translator to work with English and French, the files I have are located in the ```data/french-english/``` directory and the files present ```english.txt``` and ```french.txt```. Thus, we have the file locations ```data/french-english/english.txt``` and ```data/french-english/french.txt```. This is the organization format I used in this project and the organization applies to all language pairs. Keep in mind, English, is usually the latter in the data/**language-language** directory name as you can see in the data directory.

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
