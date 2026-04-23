import numpy as np 
import pandas as pd
import S2E_data_utils as data_utils
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt 
import seaborn as sn

def get_masks(x_state, x_explanation):
    mask_state = x_state.ne(99)
    mask_explanation = x_explanation.ne(99)
    
    return mask_state, mask_explanation

def encode_explanation_list(vocab, maxLength, explanation_list):
    encoded_list = []
    for explanation in explanation_list:
        encoded = data_utils.encoding(data_utils.tokenize_explanations([explanation]), vocab, maxLength)
        encoded_list.append(encoded)
    return encoded_list

def translate_to_key(arr, vocab, explanation_list, maxLength):
    #first encode the explanation_list 
    key = "" 
    for explanation in explanation_list:
        encoded = data_utils.encoding(data_utils.tokenize_explanations([explanation]), vocab, maxLength)
        if(encoded[0].tolist() == arr):
            key = translate_explanation(explanation)
    return key


def get_misclassified(x_gt, explanation_pred, vocab, explanation_list, maxLength, domain):
    which_one = translate_to_key(x_gt, vocab, explanation_list, maxLength, domain)
    with_what = translate_to_key(explanation_pred, vocab, explanation_list, maxLength, domain)
    return which_one, with_what


def translate_explanation(arr):
    if(arr == "impossible matching"):
        return "null"
    #elif(arr == "short distance travelled"):
    #    return "SD"
    #elif(arr == "short distance travelled with no freight"):
    #    return "SDF"
    #elif(arr == "low CO2 emissions"):
    #    return "LCO2"
    elif(arr == "short distance travelled and short distance travelled with no freight"):
        return "SD_SDF"
    elif(arr == "short distance travelled and low CO2 emissions"):
        return "SD_LCO2"
    elif(arr == "short distance travelled with no freight and low CO2 emissions"):
        return "SDF_LCO2"
    elif(arr == "short distance travelled, short distance travelled with no freight, and low CO2 emissions"):
        return "SD_SDF_LCO2"
    #elif(arr == "long distance travelled"):
    #    return "LD"
    #elif(arr == "long distance travelled with no freight"):
    #    return "LDF"
    #elif(arr == "high CO2 emissions"):
    #    return "HCO2"
    elif(arr == "long distance travelled and long distance travelled with no freight"):
        return "LD_LDF"
    elif(arr == "long distance travelled and high CO2 emissions"):
        return "LD_HCO2"
    elif(arr == "long distance travelled with no freight and high CO2 emissions"):
        return "LDF_HCO2"
    elif(arr == "long distance travelled, long distance travelled with no freight, and high CO2 emissions"):
        return "LD_LDF_HCO2"
    
def make_CM(dict_):
    
    keys_ = list(dict_.keys())
    arr2 = []
    for i in keys_:
        class_sum = float(sum(dict_[i].values()))
        class_arr = []

        for j in keys_:
            calc_ = dict_[i][j] / class_sum

            class_arr.append(calc_)

        arr2.append(class_arr)
    
    # '{SD}', '{SDF}', '{LCO2}', '{LD}', '{LDF}', '{HCO2}',
    labels = ['{SD, SDF}', '{SD, LCO2}', '{SDF, LCO2}', '{SD, SDF, LCO2}','{LD, LDF}', '{LD, HCO2}', '{LDF, HCO2}', '{LD, LDF, HCO2}', '{null}']
    df_cm2 = pd.DataFrame(np.array(arr2)*100, labels, labels)
    fig, ax = plt.subplots(figsize=(12,12)) 
    sn.set(font_scale=1.0) # for label size
    sn.heatmap(df_cm2, annot=True, annot_kws={"size": 26}, cmap="Greens", fmt='.1f') # font size
    plt.xticks(rotation=45, fontsize=28, horizontalalignment="right", verticalalignment="top", rotation_mode="anchor")
    plt.yticks(rotation=45, fontsize=28, horizontalalignment="right", verticalalignment="bottom", rotation_mode="anchor")
    # plt.show()
    plt.tight_layout()
    plt.savefig("CM-TestSet_Recall1.jpg")
    
    
def make_TSNE(state_embeddings, explanation_predictions):
   
    tsne = TSNE(2, verbose=1)
    new_state_embed = []

    for list_ in state_embeddings:
        new_state_embed.append(list_[0])

    tsne_proj = tsne.fit_transform(new_state_embed)
    plt.figure(figsize=(16,10))
    sn.scatterplot(tsne_proj[:,0], tsne_proj[:,1], hue=explanation_predictions, legend='full')
    plt.savefig("tsne.png")

