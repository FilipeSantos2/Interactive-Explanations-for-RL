explanation_list = ["impossible matching", "short distance travelled and short distance travelled with no freight","short distance travelled and low CO2 emissions", "short distance travelled with no freight and low CO2 emissions", "short distance travelled, short distance travelled with no freight, and low CO2 emissions", "long distance travelled and long distance travelled with no freight", "long distance travelled and high CO2 emissions", "long distance travelled with no freight and high CO2 emissions", "long distance travelled, long distance travelled with no freight, and high CO2 emissions"]

def get_explanation_list():
    return explanation_list

def get_shorthand_concept(predicted_explanationVec):

    if(predicted_explanationVec == "impossible matching"):
        predicted_explanation_to_concept = "null"
    elif(predicted_explanationVec == "short distance travelled and short distance travelled with no freight"):
        predicted_explanation_to_concept = "SD_SDF"
    elif(predicted_explanationVec == "short distance travelled and low CO2 emissions"):
        predicted_explanation_to_concept = "SD_LCO2"
    elif(predicted_explanationVec == "short distance travelled with no freight and low CO2 emissions"):
        predicted_explanation_to_concept = "SDF_LCO2"
    elif(predicted_explanationVec == "short distance travelled, short distance travelled with no freight, and low CO2 emissions"):
        predicted_explanation_to_concept = "SD_SDF_LCO2"
    elif(predicted_explanationVec == "long distance travelled and long distance travelled with no freight"):
        predicted_explanation_to_concept = "LD_LDF"
    elif(predicted_explanationVec == "long distance travelled and high CO2 emissions"):
        predicted_explanation_to_concept = "LD_HCO2"
    elif(predicted_explanationVec == "long distance travelled with no freight and high CO2 emissions"):
        predicted_explanation_to_concept = "LDF_HCO2"
    elif(predicted_explanationVec == "long distance travelled, long distance travelled with no freight, and high CO2 emissions"):
        predicted_explanation_to_concept = "LD_LDF_HCO2"

    return predicted_explanation_to_concept

