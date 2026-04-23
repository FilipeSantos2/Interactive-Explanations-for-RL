import datetime

class TimeSlot():
    def __init__(self, time_id, start_time, end_time):
        self.time_id = time_id #Essentially the day of the week, but 0 corresponds to the first day considered

        self.start_time = start_time
        self.end_time = end_time

        self.freight_id = None
        self.distance_travelled = 0
        self.distance_travelled_with_no_freight = 0
        self.time_spent = datetime.time(0)
        self.time_spent_with_no_freight = datetime.time(0)

        self.grams_of_CO2_emitted = 0

    def getStartTime(self):
        return self.start_time
    
    def getEndTime(self):
        return self.end_time
    
    def getDistanceTravelled(self):
        return self.distance_travelled
    
    def getDistanceTravelledWithNoFreight(self):
        return self.distance_travelled_with_no_freight
    
    def getTimeSpent(self):
        return self.time_spent
    
    def getTimeSpentWithNoFreight(self):
        return self.time_spent_with_no_freight
    
    def getGramsOfCO2Emitted(self):
        return self.grams_of_CO2_emitted
    
    def printTimeSlot(self):
        print("id -", self.time_id, "- from", self.start_time, "to", self.end_time)
        print("freight_id:", self.freight_id)

    def isAvailable(self):
        if (self.freight_id == None):
            return True
        else:
            return False
    
    def freightAssignedToTruck(self, freight_id, distance_travelled, distance_travelled_with_no_freight, time_spent, time_spent_with_no_freight, grams_of_CO2_emitted):
        self.freight_id = freight_id
        self.distance_travelled = distance_travelled
        self.distance_travelled_with_no_freight = distance_travelled_with_no_freight
        self.time_spent = time_spent
        self.time_spent_with_no_freight = time_spent_with_no_freight

        self.grams_of_CO2_emitted = grams_of_CO2_emitted