import numpy as np 
import torch 
import S2E_data_utils as data_utils
import os
import random as rand
import S2E_train_test_utils as train_test_utils
import argparse

from models import *
from S2E_data import *


seed = 42

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
rand.seed(seed)

explanation_list = ['impossible matching',
 'long distance travelled and high CO2 emissions',
 'long distance travelled and long distance travelled with no freight',
 'long distance travelled with no freight and high CO2 emissions',
 'long distance travelled, long distance travelled with no freight, and high CO2 emissions',
 'short distance travelled and low CO2 emissions',
 'short distance travelled and short distance travelled with no freight',
 'short distance travelled with no freight and low CO2 emissions',
 'short distance travelled, short distance travelled with no freight, and low CO2 emissions']
encoded_explanation_list = []

confusion_matrix = {
    "SD_SDF": { "SD_SDF": 0, "SD_LCO2": 0, "SDF_LCO2": 0, "SD_SDF_LCO2": 0, "LD_LDF": 0, "LD_HCO2": 0, "LDF_HCO2": 0, "LD_LDF_HCO2": 0, "null": 0 },
    "SD_LCO2": { "SD_SDF": 0, "SD_LCO2": 0, "SDF_LCO2": 0, "SD_SDF_LCO2": 0, "LD_LDF": 0, "LD_HCO2": 0, "LDF_HCO2": 0, "LD_LDF_HCO2": 0, "null": 0 },
    "SDF_LCO2": { "SD_SDF": 0, "SD_LCO2": 0, "SDF_LCO2": 0, "SD_SDF_LCO2": 0, "LD_LDF": 0, "LD_HCO2": 0, "LDF_HCO2": 0, "LD_LDF_HCO2": 0, "null": 0 },
    "SD_SDF_LCO2": { "SD_SDF": 0, "SD_LCO2": 0, "SDF_LCO2": 0, "SD_SDF_LCO2": 0, "LD_LDF": 0, "LD_HCO2": 0, "LDF_HCO2": 0, "LD_LDF_HCO2": 0, "null": 0 },
    "LD_LDF": { "SD_SDF": 0, "SD_LCO2": 0, "SDF_LCO2": 0, "SD_SDF_LCO2": 0, "LD_LDF": 0, "LD_HCO2": 0, "LDF_HCO2": 0, "LD_LDF_HCO2": 0, "null": 0 },
    "LD_HCO2": { "SD_SDF": 0, "SD_LCO2": 0, "SDF_LCO2": 0, "SD_SDF_LCO2": 0, "LD_LDF": 0, "LD_HCO2": 0, "LDF_HCO2": 0, "LD_LDF_HCO2": 0, "null": 0 },
    "LDF_HCO2": { "SD_SDF": 0, "SD_LCO2": 0, "SDF_LCO2": 0, "SD_SDF_LCO2": 0, "LD_LDF": 0, "LD_HCO2": 0, "LDF_HCO2": 0, "LD_LDF_HCO2": 0, "null": 0 },
    "LD_LDF_HCO2": { "SD_SDF": 0, "SD_LCO2": 0, "SDF_LCO2": 0, "SD_SDF_LCO2": 0, "LD_LDF": 0, "LD_HCO2": 0, "LDF_HCO2": 0, "LD_LDF_HCO2": 0, "null": 0 },
    "null": { "SD_SDF": 0, "SD_LCO2": 0, "SDF_LCO2": 0, "SD_SDF_LCO2": 0, "LD_LDF": 0, "LD_HCO2": 0, "LDF_HCO2": 0, "LD_LDF_HCO2": 0, "null": 0 },
}


def test(PATH, x_test_tensor, y_test_tensor, vocab):
    model = torch.load(PATH)
    model.eval()
    count_ignore = 0
    maxLength=11
    encoded_explanation_list = train_test_utils.encode_explanation_list(vocab, maxLength, explanation_list)
    
    gt_dict = {"null":0, "SD_SDF":0, "SD_LCO2":0, "SDF_LCO2":0, "SD_SDF_LCO2":0, "LD_LDF":0, "LD_HCO2":0, "LDF_HCO2":0, "LD_LDF_HCO2":0}
    
    explanation_predictions = []
    predicted_explanation_embeddings = []
    predicted_state_embeddings = []
    
    recall_at_1 = 0
    recall_at_2 = 0
    recall_at_3 = 0
        
    for i in range(x_test_tensor.size(0)):
        #only care about evaluating the data samples where the the concept arr and state are aligned
        if(y_test_tensor[i] == 1): 
            count_ignore+=1
            continue
        print("test sample: " + str(i))
        #get the original data         
        x_state = x_test_tensor[i,0,:].unsqueeze(0)
        x_explanation = x_test_tensor[i,1,:].unsqueeze(0)             
        
        l2_norm_arr = []
        index_orig = 0
        total_state_embeddings = []
        total_explanation_embeddings = []
        for j, j_explanation in enumerate(explanation_list):
            get_encoded_j_explanation = data_utils.encoding(data_utils.tokenize_explanations([j_explanation]), vocab, maxLength)
            #sample a possible explanation arr to test with the given state
            new_explanation_x = torch.tensor( np.array(get_encoded_j_explanation.copy()) )
            gt_explanation = x_test_tensor[i,1,0:11].tolist()
            if(new_explanation_x.tolist() == gt_explanation):
                index_orig = j #saving which index in the explanation_list corresponds to the gt_explanation 
            mask_state, mask_explanation = train_test_utils.get_masks(x_state, new_explanation_x)
        
            if USE_CUDA:
                x_state = x_state.cuda()
                x_explanation = x_explanation.cuda()
                mask_state = mask_state.cuda()
                mask_explanation = mask_explanation.cuda()
                
                new_explanation_x = new_explanation_x.type(torch.FloatTensor).cuda() #should be an tensor of vocab indexes that make up the explanation
                            
            with torch.no_grad():
                state_embed, explanation_embed = model.forward(x_state, new_explanation_x, mask_state, mask_explanation)
                total_state_embeddings.append(state_embed.cpu().detach().numpy())
                total_explanation_embeddings.append(explanation_embed.cpu().detach().numpy())
            
            #calculate the l2 norm difference between the state and concept_embed using 
            difference =state_embed - explanation_embed
            l2_norm = torch.linalg.norm(difference, dim=1, ord=2)           
            l2_norm_arr.append(l2_norm.cpu().item())

        min_values = sorted(range(len(l2_norm_arr)), key = lambda sub: l2_norm_arr[sub])[:3]
        predicted_index = np.argmin(np.array(l2_norm_arr))
        
        gt_key = train_test_utils.translate_to_key(x_test_tensor[i,1,0:11].tolist(), vocab, explanation_list, maxLength)
        key_at1 = train_test_utils.translate_to_key(encoded_explanation_list[min_values[0]][0].tolist(), vocab, explanation_list, maxLength)
        key_at2 = train_test_utils.translate_to_key(encoded_explanation_list[min_values[1]][0].tolist(), vocab, explanation_list, maxLength)
        key_at3 = train_test_utils.translate_to_key(encoded_explanation_list[min_values[2]][0].tolist(), vocab, explanation_list, maxLength)
        
        gt_dict[gt_key]+=1
        confusion_matrix[gt_key][key_at1] += 1
        if gt_key == key_at1:
            recall_at_1 += 1
        if gt_key == key_at1 or gt_key == key_at2:
            recall_at_2 += 1
        if gt_key == key_at1 or gt_key == key_at2 or gt_key == key_at3:
            recall_at_3 += 1
        
        #FOR TSNE compliation
        explanation_predictions.append(train_test_utils.translate_explanation(explanation_list[predicted_index]))     
        predicted_explanation_embeddings.append(total_explanation_embeddings[predicted_index]) 
        predicted_state_embeddings.append(total_state_embeddings[predicted_index])

    print("concept breakdown: " + str(gt_dict))
    print("\n")
    print("--- Test Results ---")
    print("recall@1: " + str((recall_at_1/float(x_test_tensor.size(0)-count_ignore))*100))
    print("recall@2: " + str((recall_at_2/float(x_test_tensor.size(0)-count_ignore))*100))
    print("recall@3: " + str((recall_at_3/float(x_test_tensor.size(0)-count_ignore))*100))
    
    return predicted_state_embeddings, predicted_explanation_embeddings, explanation_predictions
    

if __name__ == "__main__":
    
    model_folder_name = "test/"
    
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-t", "--tsne", default=False, help="boolean for generate tsne")
    argParser.add_argument("-cm", "--cm", default=True, help="plot confusion matrix")
    args = argParser.parse_args()
    
    training_loader, validation_loader, test_loader, x_test, y_test, vocab = get_data(batch_size=2)
    
    model_files = os.listdir(model_folder_name)
    for file_name in model_files:
        pred_state_embeds, pred_exp_embeds, pred_exps = test(str(model_folder_name) + str(file_name), x_test, y_test, vocab)   
        
        if(args.tsne):
            print("--- Generating TSNE ---")
            train_test_utils.make_TSNE(pred_state_embeds, pred_exps)
            
        if(args.cm):
            print("--- Generating Confusion Matrix ---")
            train_test_utils.make_CM(confusion_matrix)
 


       
        
        
   