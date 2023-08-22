#Open the data file
data_file = open("fatal-police-shootings-data.csv", 'r')
#Get a list of all the lines in the file
data_lines = data_file.readlines()
#Create a dictionary
database = {}
#For loop using the number of lines
for row in range(1, len(data_lines)):
    #Line = "row" index of datalines for whatever number row is
    line = data_lines[row]
    #Get rid of the commas from line
    entries = line.split(',')
    #Create a dictionary 
    db_entry = {}
    #Set up the keys and values
    db_entry["name"] = entries[11]
    db_entry["date"] = entries[1]
    db_entry["armed_with"] = entries[4]
    db_entry["age"] = entries[12]
    db_entry["gender"] = entries[13]
    db_entry["race"] = entries[14]
    db_entry["state"] = entries[7]
    #Set entry id equal to the very first line in entries
    entry_id = int(entries[0])
    #db_entry is now a value and entry_id is the key
    database[entry_id] = db_entry
#Print the name of the subject with id# 1694
print (database[1694]["name"])

#For every entry in the database
for entry_id in database:
    #Print the names of subjects from MN
    if database[entry_id]["state"] == "MN":
        print(database[entry_id]["name"])


race_counts = {}
#For every entry_id of database
for entry_id in database:
    #If the race exists, add 1 to that race
    race = database[entry_id]["race"]
    if race in race_counts:
        race_counts[race] += 1
    #If theres none than it's a 0   
    else:
        race_counts[race] = 0
print (race_counts)
        
#Use get method to get the number of times "B" appears       
black_shootings = race_counts.get("B")
#Total number of shootings
total_shootings = len(database)
#Divide
fraction_black_shootings = black_shootings/total_shootings
print (fraction_black_shootings)


unarmed_selection = {}
for entry_id in database:
    #If armed_with contains unarmed
    if database[entry_id]["armed_with"] == "unarmed":
        #New dict becoems a copy of that specific case.
        unarmed_selection[entry_id] = database[entry_id].copy()


unarmed_race_counts = {}
for entry in unarmed_selection:
    race = unarmed_selection[entry]["race"]
    #If race exists we than add 1 for each time it appears.
    if race in unarmed_race_counts:
        unarmed_race_counts[race] += 1
    #If there's none than it's a 0
    else:
        unarmed_race_counts[race] = 0


#Total unarmed shootings   
total_unarmed_shootings = len(unarmed_selection)
#Get method to get cases of a black subject
total_black_unarmed_shootings = unarmed_race_counts.get("B")
#Calculate fraction
fraction_black_unarmed_shootings = total_black_unarmed_shootings/total_unarmed_shootings
print (fraction_black_unarmed_shootings)       
      



