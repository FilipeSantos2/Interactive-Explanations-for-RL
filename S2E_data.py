import numpy as np 
import torch 
import random as rand
import json

import S2E_data_utils as data_utils

USE_CUDA = torch.cuda.is_available()
DEVICE=torch.device('cuda:0')
seed = 42
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
rand.seed(seed)


def get_data(batch_size=1):
    
    max_length = 11
    y_arr, state_arr, explanation_arr = data_utils.get_data_arrs()  
    processed_explanations_arr, word2index = data_utils.preprocess_explanations_arr(explanation_arr, max_length)
    y_tensor, state_tensor, explanation_tensor = data_utils.get_padded_data(y_arr, state_arr, processed_explanations_arr)
    
    #use this code if retraining and wanting to generate new vocab file
    vocab = data_utils.get_vocabulary(word2index)
    with open("vocab.json", "w") as outfile:
        json.dump(vocab, outfile)
    
    x_tensor = torch.stack((state_tensor, explanation_tensor),axis=1)
        
    #Load the existing vocab
    with open('vocab.json', 'r') as openfile:
        vocab = json.load(openfile)

    training_loader, validation_loader, test_loader, x_test, y_test = data_utils.split_dataset(x_tensor, y_tensor, batch_size)

    return training_loader, validation_loader, test_loader, x_test, y_test, vocab
    

