import random

def infect(a_float:float) -> bool:
    """Takes a float giving the infection prob. for the disease, and
    randomly returns T or F, indicating if an infection has occurred."""
    #If/else for value of the generated float.
    if random.uniform(0, 1) < a_float:
        return True
    else:
        return False


def recover(a_float:float) -> bool:
    """Takes a float giving the recovery prob for a person
    infected with the disease, and randomly returns T or F, indicating if the person
    has recovered."""
    ##If/else for value of the generated float.
    if random.uniform(0, 1) < a_float:
        return True
    else:
        return False


def contact_indices(pop_size:int, source:int, contact_range:int) -> list:
    """Determines which people come into contact with an
    infected person, and returns the list of indices of these people."""
    indices = []
    #Checks to see if contact range + index of the infected is within the bounds of the list
    for x in range(1, contact_range + 1):
        #Makes sure value is less than pop. size.
        if source + x < pop_size:
            indices.append(source + x)
        #Makes sure value is at least 0.
        if source - x >= 0:
            indices.append(source - x)
    return indices

def apply_recoveries(population:list, recover_prob:float) -> list:
    """Iterates through the list population once, and for each infected person, uses the function recover to determine
    whether or not the infected person recovers today."""
    #Iterates through the list using indexs.
    for x in range(len(population)):
        #If the value at index x is 'I'
        if population[x] == "I":
            #Use recover fct. Returns true = recovery; false = still infected
            if recover(recover_prob):
                population[x] = "R"
    return population


def contact(population:list, source:int, contact_range:int, infect_chance:float) -> list:
    """Uses the contact indices to find the indices of the people that source comes into contact with. If it's S,
    we use the function infect to see of they're infected."""
    #Get the people that were in contact with the infected
    contacts = contact_indices(len(population), source, contact_range)
    #For every person in contacts, if they're S, use infect to determien whether they're infected or not.
    for x in contacts:
        if population[x] == "S":
            if infect(infect_chance):
                population[x] = "I"
    return population


def apply_contacts(population:list, contact_range:int, infect_chance:float) -> list:
    """Uses the function contact to put each of the currently infected people in contact with
    neighbors, randomly infects them."""
    #Create a list of currently infected
    infected = []
    for x in range(len(population)):
        if population[x] == "I":
            infected.append(x)
    #Puts each infected in contact with their neighbors         
    for x in infected:
        total_infected = contact(population, x, contact_range, infect_chance)
    return population



def population_SIR_counts(population:list) -> dict:
    """Counts the number of people who are susceptible,
    infected, and recovered, and returns these counts in a dictionary."""
    #Create a dictionary with values of 0
    counts = {'susceptible': 0, 'infected': 0, 'recovered': 0}
    #Three if statements; depending on what X is, add one to values in our dictioanry
    for x in population:
        if x == 'S':
            counts['susceptible'] += 1
        elif x == 'I':
            counts['infected'] += 1
        elif x == 'R':
            counts['recovered'] += 1
    return counts


def simulate_day(population:list, contact_range:int, infect_chance:float, recover_chance:float) -> None:
    """Uses the function apply recoveries to simulate infected people recovering, and
    it uses the function apply contacts to simulate infected contact."""
    apply_recoveries(population, recover_chance)
    apply_contacts(population, contact_range, infect_chance)


def initialize_population(pop_size:int) -> list:
    """Sets up our pop for the simulation."""
    #Get a certain sized population
    population = ['S'] * pop_size
    #Sets one person as infected
    population[0] = 'I'
    return population


def simulate_disease(pop_size:int, contact_range:int, infect_chance:float, recover_chance:float) -> list:
    """While we have at least one infected, we simulate disease and make changes to counts; which
    contains the information of who is S, I, and R."""
    #Get pop size
    population = initialize_population(pop_size)
    #Set up our counts of S, I , and R
    counts = population_SIR_counts(population)
    #Put counts into a list, all_counts is a lsit of dictionaries
    all_counts = [counts]
    #While I > 1, we simulate infection and recovery while making changes to counts.
    while counts['infected'] > 0:
        simulate_day(population, contact_range, infect_chance, recover_chance)
        counts = population_SIR_counts(population)
        all_counts.append(counts)
    return all_counts


def peak_infections(all_counts:list) -> int:
    """Essentialy creates a variable that always shows the highest number of infections."""
    max_infections = 0
    #Iterate through all days
    for day in all_counts:
        #If the number of infected of one day is greater than the max number of infected, change new max_infections.
        if day['infected'] > max_infections:
            max_infections = day['infected']
    return max_infections

        
def display_results(all_counts:list) -> None:
    """Prints the number of days and the number of S, I, R per day until there's no infected."""
    #Get num of days of the simulation
    num_days = len(all_counts)
    #Set up columns Day, Susceptible, Infected, Recovered
    print("Day".rjust(12) + "Susceptible".rjust(12) + "Infected".rjust(12) + "Recovered".rjust(12))
    #Get the values for each column under the right column
    for day in range(num_days):
        line = str(day).rjust(12)
        line += str(all_counts[day]["susceptible"]).rjust(12)
        line += str(all_counts[day]["infected"]).rjust(12)
        line += str(all_counts[day]["recovered"]).rjust(12)
        print(line)
    #Print out the highest number of infections
    print("\nPeak Infections: {}".format(peak_infections(all_counts)))

    





            
            
            
            
        
        



    
    
    
