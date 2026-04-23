import numpy as np 
import torch 
import torch.nn as nn
import torch.nn.functional as F
import random as rand

seed = 42
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
rand.seed(seed)

class MEM(torch.nn.Module):
    def __init__(self,vocab_size, input_state_embed=64, hidden_state_embed=32, hidden_state_embed2=16, output_state_embed=8, exp_embed=32, output_exp_embed=8):
        super(MEM, self).__init__()
        
        self.state_size = 3
        
        self.FCL_1 = torch.nn.Linear(self.state_size, input_state_embed)
        self.FCL_2 = torch.nn.Linear(input_state_embed, hidden_state_embed)
        self.hidden_2 = torch.nn.Linear(hidden_state_embed, hidden_state_embed2)
        self.output_fc = torch.nn.Linear(hidden_state_embed2, output_state_embed) 
        
        self.relu1 = torch.nn.ReLU()
        self.relu2 = torch.nn.ReLU()
        self.relu3 = torch.nn.ReLU()
        
        self.explanation_embedding = nn.Embedding(vocab_size, exp_embed, padding_idx=0)

        self.lstm = nn.LSTM(exp_embed, output_exp_embed, batch_first=True)
        
    def forward(self, x_state, x_explanation, mask_state, mask_explanation):
        
        x_cs= torch.masked_select(x_state, mask_state)
        x_cs = x_cs.reshape(mask_state.size(0), self.state_size)
        
        x_e = torch.masked_select(x_explanation, mask_explanation)
        x_e = x_e.reshape(x_explanation.size(0), 11)
        
        x_cs_1 = self.relu1(self.FCL_1(x_cs))
        x_cs_2 = self.relu2(self.FCL_2(x_cs_1))
        
        flatten_cs = x_cs_2.reshape((x_cs_2.size(0),-1))
        
        h1 = F.relu(self.hidden_2(flatten_cs))
        state_final = self.output_fc(h1)
        
        x_e = x_e.to(torch.long)
        explanation_out1 = self.explanation_embedding(x_e)
        out_pack, (ht, ct) = self.lstm(explanation_out1)
        ht = ht.squeeze(0)
        explanation_final = self.relu3(ht)
        
        return state_final, explanation_final
    

        

    
