#Truck categories and types are based on their containers
#See https://www.maersk.com/industry-sectors/dry-cargo
#https://www.maersk.com/industry-sectors/refrigerated-cargo
#https://www.maersk.com/industry-sectors/special-cargo

import datetime

from locations import Locations

import random

class Truck:

    def __init__(self, id, company, location, availability, emission_rate):
        self.id = id
        self.company = company
        
        self.location = location
        self.availability = availability

        self.emission_rate = emission_rate 
        #gCO2/t-km; between 58.3 and 307.2 according to https://theicct.org/publication/co2-emissions-from-trucks-in-the-eu-an-analysis-of-the-heavy-duty-co2-standards-baseline-data/

        self.distance_time_matrix = Locations.distance_time_matrix

        self.tags = []
        prob_of_tail_lift = 0.4

        if random.random() < prob_of_tail_lift:
            self.tags.append("tail lift")

    def getTruckId(self):
        return self.id
    
    def getTimeSlots(self):
        return self.availability

    def getCompany(self):
        return self.company

    def getLocation(self):
        return self.location

    def printAll(self):
        print("id:", self.id)

        print("company:", self.company)
        print("location:", self.location)

        print("availability:")
        for time_id in self.availability:
            self.availability[time_id].printTimeSlot()
        
        print("emission_rate:", self.emission_rate)
        
        print("category:", self.getCategory())
        print("type:", self.getType())

        print("maximum_payload:", self.maximum_payload)
        print("length:", self.length)
        print("width:", self.width)
        print("height:", self.height)

        for tag in self.tags:
            print(f"tag: {tag}")

        print("\n\n")

    def updateLocation(self, new_location):
        self.location = new_location

    #emission rate in gCO2/t-km
    #grams of CO2 = emission rate * tons carried * KMs travelled
    def getGramsOfCO2Produced(self, freight_weight, distance): 
        return self.emission_rate * freight_weight * distance

    def containerCanHoldfreight(self, freight_category, freight_weight, freight_length, freight_width, freight_height, freight_tags):
        #Check if it has the same category and if the tags match 
        if ( (freight_category == self.getCategory()) and (set(freight_tags).issubset(self.tags)) ): 
            #print("freight_weight:", freight_weight)
            #print("maximum_payload:", self.maximum_payload)
            #print("freight_length:", freight_length)
            #print("length:", self.length)
            #print("freight_width:", freight_width)
            #print("width:", self.width)
            #print("freight_height:", freight_height)
            #print("height:", self.height)
            #print("freight_tags:", freight_tags)
            #print("truck_tags:", self.tags, "\n")

            if ( (freight_weight <= self.maximum_payload) and (freight_length < self.length) and (freight_width < self.width) and (freight_height < self.height) ):
                return True
            else:
                return False
        else:
            return False

    def timeSlotAvailableForDelivery(self, day, arrival, deadline, origin, destination, freight_weight):
        truck_location = self.location

        time_slot = self.availability.get(day)
        if (not time_slot):
            return {}

        if (not time_slot.isAvailable()): #Assuming it can only carry one at a time
            return {}
        else:
            start_time = time_slot.getStartTime()
            end_time = time_slot.getEndTime()

            if (start_time >= deadline):
                return {}
            elif (arrival >= end_time):
                return {}
            else:
                if (start_time > arrival): #Simply drive there as soon as the truck is available
                    pickup_time = start_time
                else:
                    #You can drive so that you are there when the freight arrives
                    t1_time = self.distance_time_matrix[(truck_location, origin)][1]
                    pickup_time = arrival - datetime.timedelta(hours=t1_time.hour, minutes=t1_time.minute, seconds=t1_time.second)
                    if (pickup_time < start_time): #The truck wouldn't be available that early
                        pickup_time = start_time

                #From the truck's current location to the origin 
                t1_time = self.distance_time_matrix[(truck_location, origin)][1]
                travel_time_truck_location_origin = datetime.timedelta(hours=t1_time.hour, minutes=t1_time.minute, seconds=t1_time.second)
                #From the origin to the destination
                t2_time = self.distance_time_matrix[(origin, destination)][1]
                travel_time = travel_time_truck_location_origin + datetime.timedelta(hours=t2_time.hour, minutes=t2_time.minute, seconds=t2_time.second)

                delivery_time = pickup_time + travel_time

                if ( (delivery_time > deadline) or (delivery_time > end_time) ):
                    #Doesn't respect the deadline or the truck won't be available that long
                    return {}
                else: #Match found -> collect metrics
                    #Distance travelled
                    distance_travelled = self.distance_time_matrix[(truck_location, origin)][0]
                    distance_travelled += self.distance_time_matrix[(origin, destination)][0]

                    #Distance and time travelled with no freight
                    distance_travelled_with_no_freight = 0
                    time_spent_with_no_freight = datetime.timedelta(seconds=0)

                    if (truck_location != origin): #Has to move there with no freight
                        distance_travelled_with_no_freight += self.distance_time_matrix[(truck_location, origin)][0]
                        time_spent_with_no_freight += travel_time_truck_location_origin

                        grams_of_CO2_emitted = self.getGramsOfCO2Produced(freight_weight, self.distance_time_matrix[(origin, destination)][0]) #Only consider the distance from origin to destination
                    else:
                        grams_of_CO2_emitted = self.getGramsOfCO2Produced(freight_weight, distance_travelled)

                    #print("truck_location:", truck_location)
                    #print("origin:", origin)
                    #print("destination:", destination)
                    #print("travel_time:", travel_time)
                    #print("time_spent_with_no_freight:", time_spent_with_no_freight, end='\n\n')

                    return {"time_slot": time_slot, "distance_travelled": distance_travelled, "distance_travelled_with_no_freight": distance_travelled_with_no_freight, "travel_time": travel_time, "time_spent_with_no_freight": time_spent_with_no_freight, "grams_of_CO2_emitted": grams_of_CO2_emitted}
                    



################################################################################################################

class DryCargoTruck(Truck):

    def __init__(self, id, company, location, availability, emission_rate):
        super().__init__(id, company, location, availability, emission_rate)

        self.width = 2352 

    def getCategory(self):
        return "Dry"

class Dry_20StandardSteel(DryCargoTruck):
    def __init__(self, id, company, location, availability, emission_rate):
        super().__init__(id, company, location, availability, emission_rate)

        self.length = 5898
        self.height = 2393 
        self.maximum_payload = 28.300

    def getType(self):
        return "20' Standard - Steel"
    
class Dry_40StandardSteel(DryCargoTruck):
    def __init__(self, id, company, location, availability, emission_rate):
        super().__init__(id, company, location, availability, emission_rate)

        self.length = 12032 
        self.height = 2393  
        self.maximum_payload = 28.870 

    def getType(self):
        return "40' Standard - Steel"
    
class Dry_40HighSteel(DryCargoTruck):
    def __init__(self, id, company, location, availability, emission_rate):
        super().__init__(id, company, location, availability, emission_rate)

        self.length = 12032  
        self.height = 2698   
        self.maximum_payload = 28.690  

    def getType(self):
        return "40' High - Steel"
    
class Dry_45HighSteel(DryCargoTruck):
    def __init__(self, id, company, location, availability, emission_rate):
        super().__init__(id, company, location, availability, emission_rate)

        self.length = 13556   
        self.height = 2698    
        self.maximum_payload = 27.650   

    def getType(self):
        return "45' High - Steel"

################################################################################################################

class RefrigeratedCargoTruck(Truck):

    def __init__(self, id, company, location, availability, emission_rate):
        super().__init__(id, company, location, availability, emission_rate)

    def getCategory(self):
        return "Refrigerated"
    
class Refrigerated_20StandardReefer(RefrigeratedCargoTruck):
    def __init__(self, id, company, location, availability, emission_rate):
        super().__init__(id, company, location, availability, emission_rate)

        self.length = 5456    
        self.width = 2290   
        self.height = 2188     
        self.maximum_payload = 27.770  

    def getType(self):
        return "20' Standard Reefer"
    
class Refrigerated_40HighCubeStandardReefer(RefrigeratedCargoTruck):
    def __init__(self, id, company, location, availability, emission_rate):
        super().__init__(id, company, location, availability, emission_rate)

        self.length = 11588     
        self.width = 2290    
        self.height = 2470      
        self.maximum_payload = 29.670     

    def getType(self):
        return "40' High Cube Standard Reefer"
    
class Refrigerated_40HighCubeMagnumReefer(RefrigeratedCargoTruck):
    def __init__(self, id, company, location, availability, emission_rate):
        super().__init__(id, company, location, availability, emission_rate)

        self.length = 11578      
        self.width = 2290     
        self.height = 2470       
        self.maximum_payload = 29.380

    def getType(self):
        return "40' High Cube Magnum Reefer"
    
class Refrigerated_40HighCubeStarCare(RefrigeratedCargoTruck):
    def __init__(self, id, company, location, availability, emission_rate):
        super().__init__(id, company, location, availability, emission_rate)

        self.length = 11588       
        self.width = 2290      
        self.height = 2470        
        self.maximum_payload = 29.670       

    def getType(self):
        return "40' High Cube StarCare"

class Refrigerated_40HighCubeSuperFreezer(RefrigeratedCargoTruck):
    def __init__(self, id, company, location, availability, emission_rate):
        super().__init__(id, company, location, availability, emission_rate)

        self.length = 11452        
        self.width = 2246       
        self.height = 2505         
        self.maximum_payload = 28.600        

    def getType(self):
        return "40' High Cube Super Freezer"

################################################################################################################

class SpecialCargoTruck(Truck):

    def __init__(self, id, company, location, availability, emission_rate):
        super().__init__(id, company, location, availability, emission_rate)

    def getCategory(self):
        return "Special"

class Special_20OpenTop(SpecialCargoTruck):
    def __init__(self, id, company, location, availability, emission_rate):
        super().__init__(id, company, location, availability, emission_rate)

        self.length = 5898         
        self.width = 2352        
        self.height = 2348          
        self.maximum_payload = 28.310         

    def getType(self):
        return "20' Open Top"
    
class Special_40OpenTopHighCube(SpecialCargoTruck):
    def __init__(self, id, company, location, availability, emission_rate):
        super().__init__(id, company, location, availability, emission_rate)

        self.length = 12032          
        self.width = 2352         
        self.height = 2653           
        self.maximum_payload = 28.400          

    def getType(self):
        return "40' Open Top High Cube"

class Special_40HighFlatRackCollapsible(SpecialCargoTruck):
    def __init__(self, id, company, location, availability, emission_rate):
        super().__init__(id, company, location, availability, emission_rate)

        self.length = 11652           
        self.width = 2374          
        self.height = 2260            
        self.maximum_payload = 47.300           

    def getType(self):
        return "40' High Flat Rack Collapsible"

################################################################################################################

