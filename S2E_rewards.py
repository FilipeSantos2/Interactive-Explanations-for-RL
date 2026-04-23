import json
import numpy 
import torch
import torch.nn.functional as F
import S2E_MEM_utils as MEM_utils 
import S2E_data_utils as data_utils
import S2E_train_test_utils as train_test_utils

class Rewards:
    def __init__(self):

        self.null_reward = 0.01
        self.no_reward = 0
        self.good_choice_reward = 20
        self.great_choice_reward = 30
        self.good_rejection_reward = 20
        self.great_rejection_reward = 30

        with open('vocab.json', 'r') as openfile:
            self.vocab = json.load(openfile)
        
        self.MEM_model = torch.load("test/MEM_seed_42.pth", map_location=torch.device('cpu'))
        self.MEM_model.eval()

    def return_reward(self, state, action):
        _, predicted_explanation_to_concept, _ = self.getConcept_MEM(state)
        predicted_concept = predicted_explanation_to_concept
        reward = self.getRewards(predicted_concept, action)

        return reward, predicted_concept


    def getRewards(self, predicted_concept, action):
        #actions: 0 = bad choice, 1 = good choice, according to the agent

        reward = 0

        if (action == 0): #bad choice, according to the agent
            if(predicted_concept == "null"):
                reward = self.null_reward
            elif(predicted_concept == "SD_SDF" or predicted_concept == "SD_LCO2" or predicted_concept == "SDF_LCO2"):
                reward = self.no_reward
            elif(predicted_concept == "SD_SDF_LCO2"):
                reward = self.no_reward
            elif(predicted_concept == "LD_LDF" or predicted_concept == "LD_HCO2" or predicted_concept == "LDF_HCO2"):
                reward = self.good_rejection_reward
            elif(predicted_concept == "LD_LDF_HCO2"):
                reward = self.great_rejection_reward
        elif (action == 1): #good choice, according to the agent
            if(predicted_concept == "null"):
                reward = self.no_reward
            elif(predicted_concept == "SD_SDF" or predicted_concept == "SD_LCO2" or predicted_concept == "SDF_LCO2"):
                reward = self.good_choice_reward
            elif(predicted_concept == "SD_SDF_LCO2"):
                reward = self.great_choice_reward
            elif(predicted_concept == "LD_LDF" or predicted_concept == "LD_HCO2" or predicted_concept == "LDF_HCO2"):
                reward = self.no_reward
            elif(predicted_concept == "LD_LDF_HCO2"):
                reward = self.no_reward

        return reward


    def getConcept_MEM(self, state):
        #convert to tensor and flatten the state
        x_state = state.copy()
        x_state = torch.unsqueeze(torch.Tensor(x_state),0)

        ###Padding######
        x_state = F.pad(x_state, (0, 30-x_state.size(1)), "constant", 99)
        
        ##########################
        maxLength = 11 
        l2_norm_arr = []

        for j, j_explanation in enumerate(MEM_utils.get_explanation_list()):
            get_encoded_j_explanation = data_utils.encoding(data_utils.tokenize_explanations([j_explanation]), self.vocab, maxLength)

            new_explanation_x = torch.tensor( numpy.array( get_encoded_j_explanation.copy() ) )

            mask_state, mask_explanation = train_test_utils.get_masks(x_state, new_explanation_x)

            with torch.no_grad():
                state_embed, explanation_embed = self.MEM_model.forward(x_state, new_explanation_x, mask_state, mask_explanation)
            difference = state_embed - explanation_embed
            l2_norm = torch.linalg.norm(difference, dim=1, ord=2)
            l2_norm_arr.append(l2_norm.cpu())

        predicted_index = numpy.argmin(numpy.array(l2_norm_arr))
        predicted_explanationVec = MEM_utils.get_explanation_list()[predicted_index]
        predicted_explanation_to_concept = MEM_utils.get_shorthand_concept(predicted_explanationVec)
      
        return predicted_explanationVec, predicted_explanation_to_concept, l2_norm


    def getMatchEvaluation(self, state):
        _, predicted_explanation_to_concept, _ = self.getConcept_MEM(state)
        predicted_concept = predicted_explanation_to_concept

        match_evaluation = "null"

        if(predicted_concept == "null"):
            match_evaluation = "null"
        elif(predicted_concept == "SD_SDF" or predicted_concept == "SD_LCO2" or predicted_concept == "SDF_LCO2"):
            match_evaluation = "good"
        elif(predicted_concept == "SD_SDF_LCO2"):
            match_evaluation = "great"
        elif(predicted_concept == "LD_LDF" or predicted_concept == "LD_HCO2" or predicted_concept == "LDF_HCO2"):
            match_evaluation = "bad"
        elif(predicted_concept == "LD_LDF_HCO2"):
            match_evaluation = "terrible"

        explanations = []

        if(predicted_concept == "SD_SDF"):
            explanations.append("-short distance travelled")
            explanations.append("-short distance travelled with no freight")
        elif(predicted_concept == "SD_LCO2"):
            explanations.append("-short distance travelled")
            explanations.append("-low CO2 emissions")
        elif(predicted_concept == "SDF_LCO2"):
            explanations.append("-short distance travelled with no freight")
            explanations.append("-low CO2 emissions")
        elif(predicted_concept == "SD_SDF_LCO2"):
            explanations.append("-short distance travelled")
            explanations.append("-short distance travelled with no freight")
            explanations.append("-low CO2 emissions")
        elif(predicted_concept == "LD_LDF"):
            explanations.append("-long distance travelled")
            explanations.append("-long distance travelled with no freight")
        elif(predicted_concept == "LD_HCO2"):
            explanations.append("-long distance travelled")
            explanations.append("-high CO2 emissions")
        elif(predicted_concept == "LDF_HCO2"):
            explanations.append("-long distance travelled with no freight")
            explanations.append("-high CO2 emissions")
        elif(predicted_concept == "LD_LDF_HCO2"):
            explanations.append("-long distance travelled")
            explanations.append("-long distance travelled with no freight")
            explanations.append("-high CO2 emissions")

        return match_evaluation, explanations, predicted_concept
    
    def get_shorthand_concept(self, predicted_explanationVec):
        predicted_explanation_to_concept = MEM_utils.get_shorthand_concept(predicted_explanationVec)

        return predicted_explanation_to_concept