#####################################################Imports###################################################

from __future__ import print_function

import random
import datetime

from time_slot import TimeSlot
from locations import Locations
from truck import *
from freight import Freight

#Imports for the AI algorithms
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

#S2E
from S2E_rewards import Rewards

#GUI
import streamlit as st
import pandas as pd

###################################################Parameters###################################################

number_of_trucks = 125
number_of_freight = 250

###################################################Time Frame##################################################

if "days_of_the_week" not in st.session_state:

    #Time frame of 1 week
    number_of_days = 7

    start_time = datetime.datetime.now() 
    print("start_time =", start_time)

    delta_time = datetime.timedelta(days=1) 
    delta_time_seconds_day = 24 * 60 * 60 #Seconds in a day
    delta_time_seconds_week = number_of_days * 24 * 60 * 60 #Seconds in a Week

    time_frame = []

    for i in range (number_of_days):
        end_time = start_time + delta_time
        
        time_frame += [start_time]
        start_time = end_time

    #Return start_time back to normal
    start_time -= datetime.timedelta(seconds=delta_time_seconds_week)
    print("start_time =", start_time)
        
    print("time_frame =", time_frame, end = '\n\n')

    days_of_the_week = []
    for i in range(number_of_days):
        days_of_the_week.append(i)

    st.session_state.days_of_the_week = days_of_the_week
    st.session_state.delta_time_seconds_day = delta_time_seconds_day
    st.session_state.delta_time_seconds_week = delta_time_seconds_week
    st.session_state.start_time = start_time


####################################################Locations###################################################

if "locations" not in st.session_state:
    st.session_state.locations = Locations.locations


######################################################Trucks####################################################

def generateRandomTrucks(number_of_trucks):
    trucks = []
    
    #Full list of haulier companies at at https://www.imt-ip.pt/sites/IMTT/Portugues/TransportesRodoviarios/EmpresasLicenciadas/Documents/Empresas_Mercadorias_OrdEmp.pdf
    companies = ["CARKARMA", "CASTRO E MERINO TRANSPORTES", "DESTINOS VIBRANTES, UNIPESSOAL", "DROPLOG", "ESTORILMUDA - TRANSPORTES", "MAIN ROAD CONSTRUCTIONS", "SILVA & AZEVEDO", "TRANSPORTES JOAQUIM MONTEIRO", "TRANSPORTES MALHADA", "TRANSPORTES RAIO", "WINDY TRANSPORT-SUCURSAL EM PORTUGAL"]
    
    truck_types = ["20' Standard - Steel", "40' Standard - Steel", "40' High - Steel", "45' High - Steel", "20' Standard Reefer", "40' High Cube Standard Reefer", "40' High Cube Magnum Reefer", "40' High Cube StarCare", "40' High Cube Super Freezer", "20' Open Top", "40' Open Top High Cube", "40' High Flat Rack Collapsible"]

    for current_id in range(number_of_trucks):
        company = random.choice(companies)
        location = random.choice(st.session_state.locations)

        availability = {}
        
        number_of_time_slots = random.randint(3, 7) #We are considering the time frame of 1 week
        time_ids = random.sample(st.session_state.days_of_the_week, k = number_of_time_slots)

        for time_id in time_ids:
            #from https://stackoverflow.com/questions/553303/how-to-generate-a-random-date-between-two-other-dates?page=1&tab=scoredesc#tab-top
            random_second_start = random.randrange(st.session_state.delta_time_seconds_day) #Random time between the two dates
            new_time_slot_start = time_frame[time_id] + datetime.timedelta(seconds=random_second_start)

            random_second_end = random.randrange(st.session_state.delta_time_seconds_day) #Random time between the two dates
            new_time_slot_end = time_frame[time_id] + datetime.timedelta(seconds=random_second_end)

            if random_second_start < random_second_end: #Start has to come before the end
                new_time_slot = TimeSlot(time_id, new_time_slot_start, new_time_slot_end)
            else: #Start doesn't come before the end, so switch places
                new_time_slot = TimeSlot(time_id, new_time_slot_end, new_time_slot_start)

            availability[time_id] = new_time_slot

        #emission rate in gCO2/t-km; between 58.3 and 307.2 according to https://theicct.org/publication/co2-emissions-from-trucks-in-the-eu-an-analysis-of-the-heavy-duty-co2-standards-baseline-data/
        emission_rate = round(random.uniform(58.3, 307.3), 1)
        
        #Choose type of truck
        #truck_type = random.choice(truck_types)
        truck_type = random.choices(truck_types, cum_weights = [1, 2, 3, 4, 4.8, 5.6, 6.4, 7.2, 8, 9.33, 10.66, 11.99], k = 1)[0]
        
        if (truck_type == "20' Standard - Steel"):
            truck = Dry_20StandardSteel(current_id, company, location, availability, emission_rate)
        elif (truck_type == "40' Standard - Steel"):
            truck = Dry_40StandardSteel(current_id, company, location, availability, emission_rate)
        elif (truck_type == "40' High - Steel"):
            truck = Dry_40HighSteel(current_id, company, location, availability, emission_rate)
        elif (truck_type == "45' High - Steel"):
            truck = Dry_45HighSteel(current_id, company, location, availability, emission_rate)
        elif (truck_type == "20' Standard Reefer"):
            truck = Refrigerated_20StandardReefer(current_id, company, location, availability, emission_rate)
        elif (truck_type == "40' High Cube Standard Reefer"):
            truck = Refrigerated_40HighCubeStandardReefer(current_id, company, location, availability, emission_rate)
        elif (truck_type == "40' High Cube Magnum Reefer"):
            truck = Refrigerated_40HighCubeMagnumReefer(current_id, company, location, availability, emission_rate)
        elif (truck_type == "40' High Cube StarCare"):
            truck = Refrigerated_40HighCubeStarCare(current_id, company, location, availability, emission_rate)
        elif (truck_type == "40' High Cube Super Freezer"):
            truck = Refrigerated_40HighCubeSuperFreezer(current_id, company, location, availability, emission_rate)
        elif (truck_type == "20' Open Top"):
            truck = Special_20OpenTop(current_id, company, location, availability, emission_rate)
        elif (truck_type == "40' Open Top High Cube"):
            truck = Special_40OpenTopHighCube(current_id, company, location, availability, emission_rate)
        elif (truck_type == "40' High Flat Rack Collapsible"):
            truck = Special_40HighFlatRackCollapsible(current_id, company, location, availability, emission_rate)
        
        trucks.append(truck)
        
    return trucks
    
#####################################################Freight####################################################

def generateRandomFreight(number_of_freight):
    freight_list = []
    
    categories = ["Dry", "Refrigerated", "Special"]

    current_id = 0

    for i in range(number_of_freight):
        category = random.choice(categories)
        
        if (category == "Dry"):
            weight = random.uniform(24, 28.87) #28.87 tons = weight capacity of the container with the most capacity considered
            length = random.uniform(3000, 13555) #13556 = length of the container with the most length
            width = random.uniform(2000, 2351) #2352 = width of the widest container
            height = random.uniform(1800, 2697) #2698 = height of the tallest container
        elif (category == "Refrigerated"):
            weight = random.uniform(24, 29.67) #29.67 tons = weight capacity of the container with the most capacity considered
            length = random.uniform(3000, 11587) #11588 = length of the container with the most length
            width = random.uniform(2200, 2289) #2290 = width of the widest container
            height = random.uniform(1700, 2504) #2505 = height of the tallest container
        elif (category == "Special"):
            weight = random.uniform(15, 47.3) #47.3 tons = weight capacity of the container with the most capacity considered
            length = random.uniform(3000, 12031) #12032 = length of the container with the most length
            width = random.uniform(2300, 2373) #2374 = width of the widest container
            height = random.uniform(1700, 2652) #2653 = height of the tallest container
        
        origin = random.choice(st.session_state.locations)
        destination = random.choice(st.session_state.locations)
        #Guarantee that origin is different than the destination
        while (origin == destination):
            destination = random.choice(st.session_state.locations)

        random_second_arrival = random.randrange(st.session_state.delta_time_seconds_week) #Random time between the two dates
        new_arrival = st.session_state.start_time + datetime.timedelta(seconds=random_second_arrival)
        
        random_second_deadline = random.randrange(st.session_state.delta_time_seconds_week) #Random time between the two dates
        new_deadline = st.session_state.start_time + datetime.timedelta(seconds=random_second_deadline)
        
        if random_second_deadline > random_second_arrival: #Arrival needs to happen first
            arrival = new_arrival
            deadline = new_deadline
        else:
            arrival = new_deadline
            deadline = new_arrival
        
        freight = Freight(current_id, category, weight, length, width, height, origin, destination, arrival, deadline)
        freight_list.append(freight)
        
        current_id += 1
    
    return freight_list

##############################################Calculate Statistics#############################################

def calculateStatistics(trucks):
    total_distance_travelled = 0
    total_distance_travelled_with_no_freight = 0
    total_time_spent = datetime.timedelta(seconds=0)
    total_time_spent_with_no_freight = datetime.timedelta(seconds=0)
    total_grams_of_CO2_emitted = 0

    number_of_freight_delivered = 0

    for truck in trucks:
        timeslots = truck.getTimeSlots()

        for day in st.session_state.days_of_the_week:
            slot = timeslots.get(day)

            if ( (slot) and (not slot.isAvailable()) ):
                total_distance_travelled += slot.getDistanceTravelled()
                total_distance_travelled_with_no_freight += slot.getDistanceTravelledWithNoFreight()
                total_time_spent += slot.getTimeSpent()
                total_time_spent_with_no_freight += slot.getTimeSpentWithNoFreight()
                total_grams_of_CO2_emitted += slot.getGramsOfCO2Emitted()

                number_of_freight_delivered += 1
                
    print("total_distance_travelled:", round (total_distance_travelled, 1), "Km")
    print("total_distance_travelled_with_no_freight:", round(total_distance_travelled_with_no_freight, 1), "Km\n")

    print("total_time_spent:", total_time_spent)
    print("total_time_spent_with_no_freight:", total_time_spent_with_no_freight, end = "\n\n")

    total_grams_of_CO2_emitted = round(total_grams_of_CO2_emitted)
    print("total_grams_of_CO2_emitted:", f"{total_grams_of_CO2_emitted:_}", "grams\n")

    print("number_of_freight_delivered:", number_of_freight_delivered)
    print("percentage of freight delivered:", round( (number_of_freight_delivered / number_of_freight * 100), 1), "%")
    print("\n\n")

#####################################AI Algorithm - Reinforcement Learning#####################################

##########################################Create a Custom Environment##########################################


class FreightTruckMatchingEnv(gym.Env):
    """Custom Environment that follows gym interface."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self):
        super().__init__()
        self.action_space = spaces.Discrete(2) #Good time slot or bad time slot

        self.observation_space = spaces.Box(low=-1, high=np.inf,
                                            shape=(3,), dtype=np.float64)
        
        self.S2E_reward = Rewards()

    def what_if_2(self):
        st.title("Truck-Freight Match")

        df = pd.DataFrame(
            [st.session_state.my_values], columns=["Distance Travelled (Km)", "Distance Travelled with no Freight (Km)", "Grams of CO2 Emitted"]
        )

        st.dataframe(df, hide_index=True)

        distance = np.round( float( self.i1 ), 1 )
        distance_no_freight = np.round( float( self.i2 ), 1 )
        grams_CO2 = np.round( float( self.i3 ), 1 )

        #Evaluate the user's inputs
        state = np.log( [distance + 1, distance_no_freight + 1, grams_CO2 + 1] )
        match_evaluation, explanations, _ = self.S2E_reward.getMatchEvaluation(state) 

        if (match_evaluation == "great"):
            colored_word = f":green[{'accepts'}]"
        elif (match_evaluation == "good"):
            colored_word = f":green[{'accepts'}]"
        elif (match_evaluation == "bad"):
            colored_word = f":red[{'rejects'}]"
        elif (match_evaluation == "terrible"):
            colored_word = f":red[{'rejects'}]"
        else:
            colored_word = match_evaluation
            
        st.markdown(f"## The AI {colored_word} because it leads to:")
        for explanation in explanations:
            st.markdown(f"- {explanation}")

        df2 = pd.DataFrame(
            [(distance, distance_no_freight, f"{grams_CO2:_}")], columns=["Distance Travelled (Km)", "Distance Travelled with no Freight (Km)", "Grams of CO2 Emitted"]
        )

        st.dataframe(df2, hide_index=True)

        if st.button("Ok, I understand now!"):
            st.session_state.step = "item_clicked"
            st.rerun()            

        if st.button("What if a value was different?"):
            st.session_state.step = "what_if"
            st.rerun()

    def what_if(self):
        st.title("Truck-Freight Match")

        df = pd.DataFrame(
            [st.session_state.my_values], columns=["Distance Travelled (Km)", "Distance Travelled with no Freight (Km)", "Grams of CO2 Emitted"]
        )

        st.dataframe(df, hide_index=True)

        st.text("Insert your values")

        self.i1 = st.number_input("Distance travelled", min_value=0, max_value=2000, value="min")
        self.i2 = st.number_input("Distance travelled with no freight", min_value=0, max_value=2000, value="min")
        self.i3 = st.number_input("Grams of CO2 emitted", min_value=0, max_value=10000000, value="min")

        if st.button("What if these were the values?"):
            st.session_state.step = "what_if_2"
            st.rerun()

    def training_examples(self):
        st.title("Truck-Freight Match")

        df = pd.DataFrame(
            [st.session_state.my_values], columns=["Distance Travelled (Km)", "Distance Travelled with no Freight (Km)", "Grams of CO2 Emitted"]
        )

        st.dataframe(df, hide_index=True)

        st.text("3 matches classified as similar:")

        #Find 3 training examples with the same concept
        state_arr = np.load("datasets/data/state_aligned.npy", allow_pickle=True)
        explanation_arr = np.load("datasets/data/explanation_aligned.npy", allow_pickle=True)
        
        training_examples = []
        for i in range ( len(state_arr) ):
            explanation = explanation_arr[i]
            
            training_example_concept = self.S2E_reward.get_shorthand_concept(explanation)
            
            if (training_example_concept == self.predicted_concept):
                state = state_arr[i]
                training_examples.append( np.round(np.exp(state) - 1, 1) ) #Retrieve the original value
                
                if ( len(training_examples) >= 3 ):
                    break

        df2 = pd.DataFrame(
            [], columns=["Distance Travelled (Km)", "Distance Travelled with no Freight (Km)", "Grams of CO2 Emitted"]
        )

        for training_example in training_examples:
            df2.loc[len(df2)] = [training_example[0], training_example[1], f"{training_example[2]:_}"]

        st.dataframe(df2, hide_index=True)

        if st.button("Ok, I understand now!"):
            st.session_state.step = "item_clicked"
            st.rerun()            

        if st.button("What if a value was different?"):
            st.session_state.step = "what_if"
            st.rerun()

    def explain_decision(self):
        st.title("Truck-Freight Match")

        df = pd.DataFrame(
            [st.session_state.my_values], columns=["Distance Travelled (Km)", "Distance Travelled with no Freight (Km)", "Grams of CO2 Emitted"]
        )

        st.dataframe(df, hide_index=True)

        label_text = "The model was trained on 150,000 scenarios, many of which are similar to this one. As such, it can predict the most likely outcome of a matching."

        st.text(label_text)

        if st.button("Ok, I understand now!"):
            st.session_state.step = "item_clicked"
            st.rerun()

        if st.button("Can you provide training examples similar to this one?"):
            st.session_state.step = "training_examples"
            st.rerun()

    def why_AI_accept(self):
        st.title("Truck-Freight Match")

        df = pd.DataFrame(
            [st.session_state.my_values], columns=["Distance Travelled (Km)", "Distance Travelled with no Freight (Km)", "Grams of CO2 Emitted"]
        )

        st.dataframe(df, hide_index=True)

        #Get the log values
        distance = np.log(float(st.session_state.my_values["Distance Travelled (Km)"]) + 1)
        distance_no_freight = np.log(float(st.session_state.my_values["Distance Travelled with no Freight (Km)"]) + 1)
        grams_CO2 = np.log(float(st.session_state.my_values["Grams of CO2 Emitted"]) + 1)
        
        state = [distance, distance_no_freight, grams_CO2]
        
        match_evaluation, explanations, self.predicted_concept = self.S2E_reward.getMatchEvaluation(state)
            
        if (match_evaluation == "great"):
            colored_word = f":green[{match_evaluation}]"
        elif (match_evaluation == "good"):
            colored_word = f":green[{match_evaluation}]"
        elif (match_evaluation == "bad"):
            colored_word = f":red[{match_evaluation}]"
        elif (match_evaluation == "terrible"):
            colored_word = f":red[{match_evaluation}]"
        else:
            colored_word = match_evaluation
            
        st.markdown(f"## This matching is {colored_word} because it leads to:")
        for explanation in explanations:
            st.markdown(f"- {explanation}")
        
        if st.button('Ok, I understand now!'):
            st.session_state.step = "item_clicked"
            st.rerun()

        if st.button('How does the model get to its decision?'):
            st.session_state.step = "explain_decision"
            st.rerun()

    def end(self):
        st.title('You have reached the end!')

    def matching(self):
        st.title('Select your Preferred Matching')
        st.markdown(f"## :green[Truck-Freight Matchings Accepted by the AI]")

        st.session_state.df = pd.DataFrame(
            columns=["Distance Travelled (Km)", "Distance Travelled with no Freight (Km)", "Grams of CO2 Emitted"]
        )

        while (st.session_state.current_episode < st.session_state.episodes) and (st.session_state.step == "matching"):
            if st.session_state.done == True: #Move to the next episode
                st.session_state.obs, _ = st.session_state.env.reset()
                st.session_state.done = False

            while not st.session_state.done:
                action, _ = st.session_state.model.predict(st.session_state.obs)
                st.session_state.obs, reward, st.session_state.done, truncated, info = self.step(action)

                if "exit" in st.session_state and st.session_state.exit:
                    st.session_state.exit = False
                    return

            st.session_state.current_episode += 1

            st.session_state.step = "end"
            st.rerun()

    def switch(self):
        st.title('Select your Preferred Matching')

        observations = []
        st.session_state.current_trucks = []
        st.session_state.current_slots = []
                    
        if (st.session_state.accepted_view):
            for slot in st.session_state.bad_slots_truck_list:
                observations.append(slot[2])
                st.session_state.current_trucks.append(slot[1])
                st.session_state.current_slots.append(slot[0])
            
            st.session_state.accepted_view = False
            st.markdown(f"## :red[Truck-Freight Matchings Rejected by the AI]")
        else:
            for slot in st.session_state.good_slots_truck_list:
                observations.append(slot[2])
                st.session_state.current_trucks.append(slot[1])
                st.session_state.current_slots.append(slot[0])
                
            st.session_state.accepted_view = True
            st.markdown(f"## :green[Truck-Freight Matchings Accepted by the AI]")

        st.session_state.df = pd.DataFrame(
            columns=["Distance Travelled (Km)", "Distance Travelled with no Freight (Km)", "Grams of CO2 Emitted"]
        )

        for obs in observations:
            obs = np.round(np.exp(obs) - 1, 1) #Retrieve the original value
            
            new_row = {
                "Distance Travelled (Km)": obs[0],
                "Distance Travelled with no Freight (Km)": obs[1],
                "Grams of CO2 Emitted": f"{obs[2]:_}"
            }
            st.session_state.df.loc[len(st.session_state.df)] = new_row

        st.dataframe(st.session_state.df, hide_index=True, on_select="rerun", selection_mode="single-row", key="obs_table")

        if (st.session_state.accepted_view):
            if st.button("Switch to Rejected Matchings"):
                st.session_state.step = "switch"
        else:
            if st.button("Switch to Accepted Matchings"):
                st.session_state.step = "switch"

    def item_clicked(self):
        st.session_state.index = st.session_state.row_idx #Get the index of the clicked item

        st.session_state.my_values = st.session_state.selected_row #Get the values

        #Get the truck's company
        truck = st.session_state.current_trucks[st.session_state.index]
        truck_company = truck.getCompany()
        
        #Get the time spent travelling
        slot = st.session_state.current_slots[st.session_state.index]
        time_spent_travelling = slot["travel_time"]
        time_spent_travelling_no_freight = slot["time_spent_with_no_freight"]

        st.title('Truck-Freight Match')

        df = pd.DataFrame( #First Table
            [st.session_state.my_values], columns=["Distance Travelled (Km)", "Distance Travelled with no Freight (Km)", "Grams of CO2 Emitted"]
        )

        st.dataframe(df, hide_index=True)

        df2 = pd.DataFrame( #Second Table
            [[time_spent_travelling, time_spent_travelling_no_freight, truck_company]], columns=["Travel Time (HH:mm:ss)", "Travel Time with no Freight (HH:mm:ss)", "Truck Company"]
        )

        st.dataframe(df2, hide_index=True)
        
        truck_location = truck.getLocation()
        freight = st.session_state.freight #Get the current freight
        pickup = freight.getOrigin()
        destination = freight.getDestination()

        df3 = pd.DataFrame( #Third Table
            [[truck_location, pickup, destination]], columns=["Truck Location", "Pickup Location", "Destination"]
        )

        st.dataframe(df3, hide_index=True)

        #Label
        label_text = "## :"
        if (st.session_state.accepted_view):
            label_text += "green[Status: Matching Accepted]"
        else:
            label_text += "red[Status: Matching Rejected]"

        st.markdown(f'{label_text}')

        #Button to select this matching
        if st.button("Select this matching"):
            st.session_state.step = "select_matching"
            st.rerun()

        if (st.session_state.accepted_view):
            text = 'Why did the AI accept?'
        else:
            text = 'Why did the AI reject?'

        #Button with this text
        if st.button(text):
            st.session_state.step = "why_AI_accept"
            st.rerun()

        #Button to go back
        if st.button("Go Back"):
            st.session_state.step = "go_back"
            st.rerun()

    def go_back(self):
        st.title('Select your Preferred Matching')

        observations = []
        st.session_state.current_trucks = []
        st.session_state.current_slots = []
                    
        if (st.session_state.accepted_view):
            for slot in st.session_state.good_slots_truck_list:
                observations.append(slot[2])
                st.session_state.current_trucks.append(slot[1])
                st.session_state.current_slots.append(slot[0])
                
            st.header('Truck-Freight Matchings Accepted by the AI', divider="green")
        else:
            for slot in st.session_state.bad_slots_truck_list:
                observations.append(slot[2])
                st.session_state.current_trucks.append(slot[1])
                st.session_state.current_slots.append(slot[0])
            
            st.header('Truck-Freight Matchings Rejected by the AI', divider="red")

        st.session_state.df = pd.DataFrame(
            columns=["Distance Travelled (Km)", "Distance Travelled with no Freight (Km)", "Grams of CO2 Emitted"]
        )

        for obs in observations:
            obs = np.round(np.exp(obs) - 1, 1) #Retrieve the original value
            
            new_row = {
                "Distance Travelled (Km)": obs[0],
                "Distance Travelled with no Freight (Km)": obs[1],
                "Grams of CO2 Emitted": f"{obs[2]:_}"
            }
            st.session_state.df.loc[len(st.session_state.df)] = new_row

        st.dataframe(st.session_state.df, hide_index=True, on_select="rerun", selection_mode="single-row", key="obs_table")

        if (st.session_state.accepted_view):
            if st.button("Switch to Rejected Matchings"):
                st.session_state.step = "switch"
                st.rerun()
        else:
            if st.button("Switch to Accepted Matchings"):
                st.session_state.step = "switch"
                st.rerun()

    def select_matching(self):
        if (st.session_state.accepted_view):
            slots_truck_list = st.session_state.good_slots_truck_list
        else:
            slots_truck_list = st.session_state.bad_slots_truck_list
        
        best_slot_truck = slots_truck_list[st.session_state.index]

        best_truck = best_slot_truck[1]
        best_slot = best_slot_truck[0]

        freight = st.session_state.freight #Get the current freight
        freight.assignToTruck(best_truck.getTruckId())

        freight_id = st.session_state.freight_ind
        freight_destination = freight.getDestination()
        
        timeslot = best_slot["time_slot"]
        distance_travelled = best_slot["distance_travelled"]
        distance_travelled_with_no_freight = best_slot["distance_travelled_with_no_freight"]
        time_spent = best_slot["travel_time"]
        time_spent_with_no_freight = best_slot["time_spent_with_no_freight"]
        grams_of_CO2_emitted = best_slot["grams_of_CO2_emitted"]
        
        timeslot.freightAssignedToTruck(freight_id, distance_travelled, distance_travelled_with_no_freight, time_spent, time_spent_with_no_freight, grams_of_CO2_emitted)

        #Update the truck's location 
        best_truck.updateLocation(freight_destination)

        ###########################Move to the next freight###########################
                
        st.session_state.slots_truck_list = []
        st.session_state.bad_slots_truck_list = []
        st.session_state.good_slots_truck_list = []

        st.session_state.freight_ind += 1
        st.session_state.truck_ind = 0

        if (st.session_state.freight_ind >= number_of_freight): #Every freight covered -> move to the next day
            st.session_state.day += 1
            st.session_state.freight_ind = 0

            if (st.session_state.day >= number_of_days): #We've went through every day
                st.session_state.done = True
                st.session_state.observation = [-1.0, -1.0, -1.0]

                calculateStatistics(st.session_state.trucks)
            
                st.session_state.observation = np.array(st.session_state.observation)
            
                return
                
        truck = st.session_state.trucks[st.session_state.truck_ind] #Get the current truck
        freight = st.session_state.freight_list[st.session_state.freight_ind] #Get the current freight
        
        if ( not freight.isAssignedToTruck() ):
        
            freight_id = freight.getId()

            freight_category = freight.getCategory()
            freight_weight = freight.getWeight()
            freight_length = freight.getLength()
            freight_width = freight.getWidth()
            freight_height = freight.getHeight()

            freight_arrival = freight.getArrival()
            freight_deadline = freight.getDeadline()
            freight_origin = freight.getOrigin()
            freight_destination = freight.getDestination()
            
            freight_tags = freight.getTags()

            if (truck.containerCanHoldfreight(freight_category, freight_weight, freight_length, freight_width, freight_height, freight_tags)):
                st.session_state.slot = truck.timeSlotAvailableForDelivery(st.session_state.day, freight_arrival, freight_deadline, freight_origin, freight_destination, freight_weight)
                if (st.session_state.slot): #This slot is available
                    
                    #This is the observation
                    st.session_state.observation = [ (np.log(st.session_state.slot["distance_travelled"] + 1)), (np.log(st.session_state.slot["distance_travelled_with_no_freight"] + 1)), (np.log(st.session_state.slot["grams_of_CO2_emitted"] + 1)) ]
                    
                else: #This slot is not available
                    st.session_state.observation = [-1.0, -1.0, -1.0]
                    st.session_state.slot = {}
            else: #The container can't hold the freight
                st.session_state.observation = [-1.0, -1.0, -1.0]
                st.session_state.slot = {}
        else: #The freight is already assigned
            st.session_state.observation = [-1.0, -1.0, -1.0]
            st.session_state.slot = {}
                    
        st.session_state.observation = np.array(st.session_state.observation)

        st.session_state.step = "matching"
        self.matching()

    def switch2(self):
        st.session_state.step = "switch"

    def step(self, action):            
        ###########################Reward the Agent###########################
        
        st.session_state.reward = 0.0
        
        if (st.session_state.slot): #The slot was available
            st.session_state.reward, predicted_concept = self.S2E_reward.return_reward(st.session_state.observation, action)
        
            if (action == 0): #actions: 0 = bad choice, 1 = good choice, according to the agent
                st.session_state.bad_slots_truck_list.append( (st.session_state.slot, st.session_state.trucks[st.session_state.truck_ind], st.session_state.observation) ) #Save it in a list

            elif (action == 1): #actions: 0 = bad choice, 1 = good choice, according to the agent
                st.session_state.good_slots_truck_list.append( (st.session_state.slot, st.session_state.trucks[st.session_state.truck_ind], st.session_state.observation) ) #Save it in a list
            
        ###########################Move to the next truck###########################
        
        st.session_state.truck_ind += 1 
        
        if (st.session_state.truck_ind >= number_of_trucks): #Every truck covered -> assign freight and move to the next one
            
            ###########################Assign this freight to the timeslot###########################
            
            if (st.session_state.good_slots_truck_list): #Check if there is at least one good slot 
                st.session_state.slots_truck_list = st.session_state.good_slots_truck_list
            elif (st.session_state.bad_slots_truck_list): #Else -> Check if there is at least one bad slot 
                st.session_state.slots_truck_list = st.session_state.bad_slots_truck_list
            
            #Check if there is at least one slot 
            if (st.session_state.slots_truck_list): #Slots available -> the user can pick one
                ########Let the user pick a slot

                st.session_state.accepted_view = True
                st.session_state.step = "user_picks_slot"

                observations = []
                st.session_state.current_trucks = []
                st.session_state.current_slots = []

                for slot in st.session_state.good_slots_truck_list:
                    observations.append(slot[2])
                    st.session_state.current_trucks.append(slot[1])
                    st.session_state.current_slots.append(slot[0])

                for obs in observations:
                    obs = np.round(np.exp(obs) - 1, 1) #Retrieve the original value
                    
                    new_row = {
                        "Distance Travelled (Km)": obs[0],
                        "Distance Travelled with no Freight (Km)": obs[1],
                        "Grams of CO2 Emitted": f"{obs[2]:_}"
                    }
                    st.session_state.df.loc[len(st.session_state.df)] = new_row

                st.dataframe(st.session_state.df, hide_index=True, on_select="rerun", selection_mode="single-row", key="obs_table")

                st.session_state.freight = st.session_state.freight_list[st.session_state.freight_ind]

                st.button("Switch to Rejected Matchings", on_click=self.switch2)

                st.session_state.exit = True

                return st.session_state.observation, st.session_state.reward, st.session_state.done, st.session_state.truncated, st.session_state.info
                ########
        
            ###########################Move to the next freight###########################
            
            st.session_state.slots_truck_list = []
            st.session_state.bad_slots_truck_list = []
            st.session_state.good_slots_truck_list = []
        
            st.session_state.freight_ind += 1
            st.session_state.truck_ind = 0
        
            if (st.session_state.freight_ind >= number_of_freight): #Every freight covered -> move to the next day
                st.session_state.day += 1
                st.session_state.freight_ind = 0

                if (st.session_state.day >= number_of_days): #We've went through every day
                    st.session_state.done = True
                    st.session_state.observation = [-1.0, -1.0, -1.0]

                    calculateStatistics(st.session_state.trucks)
                
                    st.session_state.observation = np.array(st.session_state.observation)
                
                    return st.session_state.observation, st.session_state.reward, st.session_state.done, st.session_state.truncated, st.session_state.info
                
        truck = st.session_state.trucks[st.session_state.truck_ind] #Get the current truck
        freight = st.session_state.freight_list[st.session_state.freight_ind] #Get the current freight
        
        if ( not freight.isAssignedToTruck() ):
        
            freight_id = freight.getId()

            freight_category = freight.getCategory()
            freight_weight = freight.getWeight()
            freight_length = freight.getLength()
            freight_width = freight.getWidth()
            freight_height = freight.getHeight()

            freight_arrival = freight.getArrival()
            freight_deadline = freight.getDeadline()
            freight_origin = freight.getOrigin()
            freight_destination = freight.getDestination()
            
            freight_tags = freight.getTags()

            if (truck.containerCanHoldfreight(freight_category, freight_weight, freight_length, freight_width, freight_height, freight_tags)):
                st.session_state.slot = truck.timeSlotAvailableForDelivery(st.session_state.day, freight_arrival, freight_deadline, freight_origin, freight_destination, freight_weight)
                if (st.session_state.slot): #This slot is available
                    
                    #This is the observation
                    st.session_state.observation = [ (np.log(st.session_state.slot["distance_travelled"] + 1)), (np.log(st.session_state.slot["distance_travelled_with_no_freight"] + 1)), (np.log(st.session_state.slot["grams_of_CO2_emitted"] + 1)) ]
                    
                else: #This slot is not available
                    st.session_state.observation = [-1.0, -1.0, -1.0]
                    st.session_state.slot = {}
            else: #The container can't hold the freight
                st.session_state.observation = [-1.0, -1.0, -1.0]
                st.session_state.slot = {}
        else: #The freight is already assigned
            st.session_state.observation = [-1.0, -1.0, -1.0]
            st.session_state.slot = {}
                    
        st.session_state.observation = np.array(st.session_state.observation)
        
        return st.session_state.observation, st.session_state.reward, st.session_state.done, st.session_state.truncated, st.session_state.info

    def reset(self, seed=None, options=None):
        st.session_state.done = False
        st.session_state.reward = 0.0
        st.session_state.observation = [-1.0, -1.0, -1.0]
        
        st.session_state.trucks = generateRandomTrucks(number_of_trucks)
        st.session_state.freight_list = generateRandomFreight(number_of_freight)
        
        st.session_state.day = 0
        st.session_state.freight_ind = 0 #Get the first freight 
        st.session_state.truck_ind = 0 #Get The first truck
        
        st.session_state.slots_truck_list = []
        st.session_state.bad_slots_truck_list = []
        st.session_state.good_slots_truck_list = []
                
        ##############################################The first truck and the first freight

        truck = st.session_state.trucks[st.session_state.truck_ind] #Get the first Truck
        freight = st.session_state.freight_list[st.session_state.freight_ind] #Get the first freight
        
        freight_id = freight.getId()

        freight_category = freight.getCategory()
        freight_weight = freight.getWeight()
        freight_length = freight.getLength()
        freight_width = freight.getWidth()
        freight_height = freight.getHeight()

        freight_arrival = freight.getArrival()
        freight_deadline = freight.getDeadline()
        freight_origin = freight.getOrigin()
        freight_destination = freight.getDestination()
        
        freight_tags = freight.getTags()
        
        st.session_state.slot = {}

        if (truck.containerCanHoldfreight(freight_category, freight_weight, freight_length, freight_width, freight_height, freight_tags)):
            st.session_state.slot = truck.timeSlotAvailableForDelivery(st.session_state.day, freight_arrival, freight_deadline, freight_origin, freight_destination, freight_weight)
            if (st.session_state.slot): #This slot is available
                
                #This is the observation
                st.session_state.observation = [ (np.log(st.session_state.slot["distance_travelled"] + 1)), (np.log(st.session_state.slot["distance_travelled_with_no_freight"] + 1)), (np.log(st.session_state.slot["grams_of_CO2_emitted"] + 1)) ]
                
            else: #This slot is not available
                st.session_state.observation = [-1.0, -1.0, -1.0]
                st.session_state.slot = {}
        else: #The container can't hold the freight
            st.session_state.observation = [-1.0, -1.0, -1.0]
            st.session_state.slot = {}
                
        
        st.session_state.info = {}
        st.session_state.truncated = False
        st.session_state.observation = np.array(st.session_state.observation)
        
        return st.session_state.observation, st.session_state.info

######################Add the environment to the gym registry and make sure it is valid########################

if "env" not in st.session_state:
    gym.register(
        id = "custom/FreightTruckMatching-v1",
        entry_point = FreightTruckMatchingEnv,
        max_episode_steps = number_of_trucks * number_of_freight * number_of_days,
    )

    #env = FreightTruckMatchingEnv()
    st.session_state.env = gym.make("custom/FreightTruckMatching-v1")

    # It will check your custom environment and output additional warnings if needed
    check_env(st.session_state.env)

#################################################Explainability################################################

#################################################Load the Model################################################

if "model" not in st.session_state:
    models_dir = "models/1746025626/"
    model_path = f"{models_dir}/800000.zip"

    st.session_state.model = PPO.load(model_path, env=st.session_state.env)



if "episodes" not in st.session_state:
    st.session_state.episodes = 3

#Was an item from the table clicked?
table = st.session_state.get("obs_table")

if table and table.selection.rows:
    st.session_state.row_idx = table.selection.rows[0] #Get the index clicked
    st.session_state.selected_row = st.session_state.df.iloc[st.session_state.row_idx]

    st.session_state.step = "item_clicked"

#if "step" in st.session_state:
#    print(st.session_state.step)

if "step" not in st.session_state:
    st.session_state.step = "matching"

    st.session_state.current_episode = 0
    st.session_state.done = True

    st.session_state.env.unwrapped.matching()

elif st.session_state == "matching":
    st.session_state.env.unwrapped.matching()

elif st.session_state.step == "switch":
    st.session_state.env.unwrapped.switch()

elif st.session_state.step == "item_clicked":    
    st.session_state.env.unwrapped.item_clicked()

elif st.session_state.step == "go_back":
    st.session_state.env.unwrapped.go_back()

elif st.session_state.step == "select_matching":
    st.session_state.env.unwrapped.select_matching()

elif st.session_state.step == "end":
    st.session_state.env.unwrapped.end()
    
elif st.session_state.step == "why_AI_accept":
    st.session_state.env.unwrapped.why_AI_accept()

elif st.session_state.step == "explain_decision":
    st.session_state.env.unwrapped.explain_decision()

elif st.session_state.step == "training_examples":
    st.session_state.env.unwrapped.training_examples()

elif st.session_state.step == "what_if":
    st.session_state.env.unwrapped.what_if()

elif st.session_state.step == "what_if_2":
    st.session_state.env.unwrapped.what_if_2()

