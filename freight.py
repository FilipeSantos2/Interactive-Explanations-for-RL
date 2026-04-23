import random

class Freight:

    def __init__(self, id, category, weight, length, width, height, origin, destination, arrival, deadline):
        self.id = id

        self.category = category #Dry, Refrigerated, Special
        self.weight = weight
        self.length = length
        self.width = width
        self.height = height

        self.origin = origin
        self.destination = destination

        self.arrival = arrival
        self.deadline = deadline

        self.assigned_to_truck = None

        self.tags = []
        prob_of_tail_lift = 0.1

        if random.random() < prob_of_tail_lift:
            self.tags.append("tail lift")

    def printAll(self):
        print("id:", self.id)

        print("category:", self.category)
        print("weight:", self.weight)
        print("length:", self.length)
        print("width:", self.width)
        print("height:", self.height)

        print("origin:", self.origin)
        print("destination:", self.destination)

        print("arrival:", self.arrival)
        print("deadline:", self.deadline)

        for tag in self.tags:
            print(f"tag: {tag}")

        print("\n\n")

    def getId(self):
        return self.id

    def getCategory(self):
        return self.category
    
    def getWeight(self):
        return self.weight
    
    def getLength(self):
        return self.length
    
    def getWidth(self):
        return self.width
    
    def getHeight(self):
        return self.height

    def getArrival(self):
        return self.arrival
    
    def getDeadline(self):
        return self.deadline
    
    def getOrigin(self):
        return self.origin
    
    def getDestination(self):
        return self.destination
    
    def getTags(self):
        return self.tags

    def isAssignedToTruck(self):
        if (self.assigned_to_truck == None):
            return False 
        else:
            return True
        
    def assignToTruck(self, truck_id):
        self.assigned_to_truck = truck_id