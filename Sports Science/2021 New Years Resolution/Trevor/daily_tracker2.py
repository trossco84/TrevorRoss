import pandas as pd
import numpy as np 
import datetime
import warnings


warnings.filterwarnings("ignore")
#creating the dataset and features to be tracked

#Feature Set
##Did You eat Breakfast? (Binary)
##Did You have Coffee? (Binary)
##Breakfast Type: (Categorical, [VGood, Good, Ok, Bad, Terrible])
##Breakfast Calories: (Integer)
##Coffee Calories: (Integer)
##Did You have Lunch? (Binary)
##Lunch Type: (Categorical, [VGood, Good, Ok, Bad, Terrible])
##Lunch Calories: (Integer)
##Did You have Dinner? (Binary)
##Dinner Type: (Categorical, [VGood, Good, Ok, Bad, Terrible])
##Dinner Calories: (Integer)
##Did You Exercise? (Binary)
##What Kind of Exercise? (Categorical, [Weights, Cardio, Both])
##Did You Have a Good Workout Today? (Categorical, [I had a good workout today, I had an Ok workout, Bad Day])
##Did You Exercise More Than Once? (Binary)

#creating the dataframe
# cols = "timestamp,breakfast_binary, coffee_binary, breakfast_type, breakfast_calories, coffee_calories, lunch_binary, lunch_type, lunch_calories, dinner_binary, dinner_type, dinner_calories, exercise_binary, exercise_type, exercise_level, exercise_multiple".split(",")

# trdf = pd.DataFrame(columns=cols)
# trdf.set_index('timestamp',inplace=True)
# trdf.head()

trdf = pd.read_csv('/Users/tross/WILD/RossCo/MyRepo/TrevorRoss/Sports Science/2021 New Years Resolution/Trevor/data.csv',index_col=0)
morningdf = trdf[trdf.columns[0:5]]
lunchdf = trdf[trdf.columns[5:8]]
dinnerdf = trdf[trdf.columns[8:11]]
exercisedf = trdf[trdf.columns[11:15]]


#creating the classes
class questions:
    def __init__(self, name, qstring, kind, vrange):
        self.name = name
        self.qstring = qstring
        self.kind = kind
        self.vrange = vrange
        self.question = {"name":name,"qstring":qstring,"kind":kind,"vrange":vrange}
    
    def __getitem__(self,key):
        return self.question[key]
    
    def __iter__(self):
        return self
    

#creating the class items
breakfast_binary = questions('breakfast_binary','Did You eat Breakfast?','binary',["No","Yes"])
coffee_binary = questions('coffee_binary','Did You have Coffee?','binary',["No","Yes"])
lunch_binary = questions('lunch_binary','Did You have Lunch?','binary',["No","Yes"])
dinner_binary = questions('dinner_binary','Did You have Dinner?','binary',["No","Yes"])
exercise_binary1 = questions('exercise_binary1','Did You Exercise?','binary',["No","Yes"])
exercise_binary2 = questions('exercise_binary2','Did You Exercise More Than Once?','binary',["No","Yes"])

breakfast_category = questions('breakfast_category','Breakfast Type:','categorical',['VGood', 'Good', 'Ok', 'Bad', 'Terrible'])
lunch_category = questions('lunch_category','Lunch Type:','categorical',['VGood', 'Good', 'Ok', 'Bad', 'Terrible'])
dinner_category = questions('dinner_category','Dinner Type:','categorical',['VGood', 'Good', 'Ok', 'Bad', 'Terrible'])
exercise_category1 = questions('exercise_category1','What Kind of Exercise?','categorical',['Weights', 'Cardio', 'Both'])
exercise_category2 = questions('exercise_category2','Did You Have a Good Workout Today?','categorical',['I had a good workout today', 'I had an Ok workout', 'Bad Day'])

breakfast_calories = questions('breakfast_calories','Breakfast Calories:','numerical',range(0,100000))
coffee_calories = questions('coffee_calories','Coffee Calories:','numerical',range(0,100000))
lunch_calories = questions('lunch_calories','Lunch Calories:','numerical',range(0,100000))
dinner_calories = questions('dinner_calories','Dinner Calories:','numerical',range(0,100000))


#data collection
##next steps: add functions for lunch, dinner, exercise, collect all, and combination
def collect_morning(tstamp):
    timestamp = tstamp
    bbin = ask_question(breakfast_binary)
    cbin = ask_question(coffee_binary)
    if bbin == "1":
        btype = ask_question(breakfast_category)
        bcal = ask_question(breakfast_calories)
    else:
        btype = np.nan
        bcal = 0
    
    if cbin == "1":
        ccal = ask_question(coffee_calories)
    else:
        ccal = 0
    
    morning_values = [bbin,cbin,btype,bcal,ccal]
    morningdf.loc[str(timestamp)] = morning_values

def collect_lunch(tstamp):
    timestamp = tstamp
    lbin = ask_question(lunch_binary)
    if lbin == "1":
        ltype = ask_question(lunch_category)
        lcal = ask_question(lunch_calories)
    else:
        ltype = np.nan
        lcal = 0
    lunch_values = [lbin,ltype,lcal]
    lunchdf.loc[str(timestamp)] = lunch_values

def collect_dinner(tstamp):
    timestamp = tstamp
    dbin = ask_question(dinner_binary)
    if dbin == "1":
        dtype = ask_question(dinner_category)
        dcal = ask_question(dinner_calories)
    else:
        dtype = np.nan
        dcal = 0
    dinner_values = [dbin,dtype,dcal]
    dinnerdf.loc[str(timestamp)] = dinner_values

def collect_exercise(tstamp):
    timestamp = tstamp
    ebin1 = ask_question(exercise_binary1)
    if ebin1 == "1":
        etype = ask_question(exercise_category1)
        escale = ask_question(exercise_category2)
        ebin2 = ask_question(exercise_binary2)
    else:
        etype = np.nan
        escale = np.nan
        ebin2 = "0"
    exercise_values = [ebin1,etype,escale,ebin2]
    exercisedf.loc[str(timestamp)] = exercise_values


def ask_question(question):
    option_count = int(len(question.vrange))
    print()
    if question.kind == 'binary':
        print(question.qstring)
        print()
        print("options:")
        for i in range(0,option_count):
            print(f'{i}: {question.vrange[i]}')
        decision = input("Select Option:")
    elif question.kind == 'categorical':
        print(question.qstring)
        print()
        print("options:")
        catdict = {}
        for i in range(0,option_count):
            catdict[str(i)] = question.vrange[i]
            print(f'{i}: {question.vrange[i]}')
        dictlocation = input("Select Option:")
        decision = catdict[dictlocation]
    else:
        decision = int(input(question.qstring))
    return decision

def data_retrieval(tstamp):
    create_data()
    timestamp = tstamp
    trdf = pd.read_csv('/Users/tross/WILD/RossCo/MyRepo/TrevorRoss/Sports Science/2021 New Years Resolution/Trevor/data.csv',index_col=0)
    todays_calories = trdf.loc[str(timestamp)]['daily_calories']
    available = 1200 - todays_calories
    print(f'Current Calories: {todays_calories}')
    print(f'Calories Available: {available}')


def what_day():
    year = int(input("year?"))
    month = int(input("month?"))
    day = int(input("day?"))
    return datetime.date(year,month,day)

def interaction(tstamp):
    print()
    intentions = {0:"Input Morning",1:"Input Lunch",2:"Input Dinner",3:"Input Exercise",4:"Check Calories"}
    valid_options = "0 1 2 3 4".split()
    
    for item in intentions:
        print(f'{item}: {intentions[item]}')
    
    while True:
        intent = input("What would you like to do?")
        if intent not in valid_options:
            print("Please Enter a Valid Option!")
        else:
            break

    if intent == "0":
        collect_morning(tstamp)
    elif intent =="1":
        collect_lunch(tstamp)
    elif intent == "2":
        collect_dinner(tstamp)
    elif intent == "3":
        collect_exercise(tstamp)
    elif intent == "4":
        data_retrieval(tstamp)

def initial_options():
    i_options = {0:"Input Data",1:"Change Date",2:"Check Calories"}
    v_i = "0 1 2".split()
    
    for item in i_options:
        print(f'{item}: {i_options[item]}')
    
    while True:
        intent = input("What would you like to do?")
        if intent not in v_i:
            print("Please Enter a Valid Option!")
        else:
            break
    return intent

def create_data():
    trdf = morningdf.join([lunchdf,dinnerdf,exercisedf])
    trdf.loc[:,'daily_calories'] = trdf.sum(numeric_only=True, axis=1)
    trdf.to_csv('/Users/tross/WILD/RossCo/MyRepo/TrevorRoss/Sports Science/2021 New Years Resolution/Trevor/data.csv',index=True)

def initial_display(tstamp):
    init = initial_options()
    if init == "0":
        interaction(tstamp)
        keep_going_check(tstamp)
    elif init == "1":
        tstamp = what_day()
        keep_going_check(tstamp)
    elif init == "2":
        data_retrieval(tstamp)
        keep_going_check(tstamp)

def run_application():
    print("Welcome! Did you have a good workout today?")
    print()
    today = datetime.date.today()
    tstmp = str(today)
    print(f"Today's Date: {today}")
    print()
    initial_display(tstmp)

def keep_going_check(tstmp):
    if input("Are you done?") in ["Yes","yes","Y","y"]:
        end_program()
    else:
        print()
        initial_display(tstmp)

def end_program():
    create_data()
    print("Have a good day!")

def main():
    run_application()

if __name__ == "__main__":
    main()