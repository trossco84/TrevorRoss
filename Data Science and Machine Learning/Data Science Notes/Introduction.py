##DETERMINING NETWORK METRIC DEGREE CENTRALITY##

import numpy as np 
import pandas as pd 

## creates a list for each user's id and respective name

user_ids = [0,1,2,3,4,5,6,7,8,9]
user_names = "Hero Dunn Sue Chi Thor Clive Hicks Devin Kate Klein".split()

## creates the users list of dictionaries
users = []
# for counter in user_ids:
#     temp_dict = {"id" : user_ids[counter], "name" : user_names[counter]}
#     users.append(temp_dict)

# dictList = [{k:v for k,v in zip(keys, n)} for n in names]
users = [{ "id":uid, "name":uname} for uid,uname in zip(user_ids,user_names)]
users

# creating the two lists for the tuple zip
f1 = [0,0,1,1,2,3,4,5,5,6,7,8]
f2 = [1,2,2,3,3,4,5,6,7,8,8,9]
#zipping the lists into tuples and then creating a list of tuples
friendship_pairs = list(zip(f1,f2))
friendship_pairs

# creating the dictionary of friendships
#creating the dictionary with keys = uids
friendships = {user["id"]: [] for user in users}
friendships

#setting the dictionary's value = a list of the user's friends
for i,j in friendship_pairs:
    friendships[i].append(j)
    friendships[j].append(i)

#creating a function which counts the number of friends for each user
#not great value because you need to call it with the index of the user you want from the user list
# def number_of_friends(user):
#     user_id = user["id"]
#     friend_ids = friendships[user_id]
#     return len(friend_ids)

# number_of_friends(users[0])

#retooled the function to allow the name of the user to be passed to the function, capitalization doesn't matter
# def number_of_friends(user_str):
#     user_index = [user["id"] for user in users if user["name"].lower()==user_str.lower()][0]
#     friend_ids = friendships[user_index]
#     return len(friend_ids)

# number_of_friends("Hero")

#this function only works if the user inputs the user's name
# retooling the function to allow an integer, dictionary index or string to be passed through
def number_of_friends(input):
    if (type(input) == str):
        user_index = [user["id"] for user in users if user["name"].lower()==input.lower()][0]
        friend_ids = friendships[user_index]
    elif (type(input) == dict):
        friend_ids = friendships[input["id"]]
    elif (type(input) == int):
        friend_ids = friendships[input]
    else:
        return 'Please Enter Valid Input (name, id, or user dictionary)'
    return len(friend_ids)

number_of_friends("Hero")
number_of_friends(users[0])
number_of_friends(0)

#calcualte total number of connections, the number of users, and the average number of connections
total_connections = sum(number_of_friends(user) for user in users)
number_of_users = len(users)
avg_connections = total_connections/number_of_users

#determining the most connected people (those with the largest number of friends)
num_friends_by_id = [(user["id"],number_of_friends(user)) for user in users]
num_friends_by_id.sort(key=lambda id_and_friends: id_and_friends[1],reverse=True)

l1=[]
l2=[]
l3=[]
for tup in num_friends_by_id:
    if tup[1]==1:
        l1.append(users[tup[0]]['name'])
    elif tup[1]==2:
        l2.append(users[tup[0]]['name'])
    elif tup[1]==3:
        l3.append(users[tup[0]]['name'])

# print(f"Users with 3 Friends: {l3}")
# print(f"Users with 2 Friends: {l2}")
# print(f"Users with 1 Friend: {l1}")

#Creating a Recommendation System based on Interests and Mutual Connections
#foaf means friend of friend
#Mutual Friends
#username = input("What is your name?")

def mutual_friends(input2):
    if (type(input2)==str):
        user_index = [user["id"] for user in users if user["name"].lower()==input2.lower()][0]
    elif (type(input2)==int):
        user_index = input2
    elif (type(input2)==dict):
        user_index = input2['id']
    else:
        return "Please Enter Valid Input (name, id, or user dictionary)"
    print("Your Mutual Friends Are: ")
    return foaf(user_index)

def foaf(ui):
    mf_list = []
    for friend in friendships[ui]:
        for mf in friendships[friend]:
            if mf not in friendships[ui]:
                if users[mf] not in mf_list:
                    if mf != ui:
                        mf_list.append(users[mf])
    return mf_list

#print(mutual_friends(username))

#Interests

interests = [(0, "Hadoop"), (0, "Big Data"), (0, "HBase"), (0, "Java"),(0, "Spark"), (0, "Storm"), (0, "Cassandra"),
(1, "NoSQL"), (1, "MongoDB"), (1, "Cassandra"), (1, "HBase"),
(1, "Postgres"), (2, "Python"), (2, "scikit-learn"), (2, "scipy"),
(2, "numpy"), (2, "statsmodels"), (2, "pandas"), (3, "R"), (3, "Python"),
(3, "statistics"), (3, "regression"), (3, "probability"),
(4, "machine learning"), (4, "regression"), (4, "decision trees"),
(4, "libsvm"), (5, "Python"), (5, "R"), (5, "Java"), (5, "C++"),
(5, "Haskell"), (5, "programming languages"), (6, "statistics"),
(6, "probability"), (6, "mathematics"), (6, "theory"),
(7, "machine learning"), (7, "scikit-learn"), (7, "Mahout"),
(7, "neural networks"), (8, "neural networks"), (8, "deep learning"),
(8, "Big Data"), (8, "artificial intelligence"), (9, "Hadoop"),
(9, "Java"), (9, "MapReduce"), (9, "Big Data")
]

names_list = [user["name"] for user in users]
ids_list = [user["id"] for user in users]
dicts_list = [user for user in users]

def users_name(input_user):
    if (type(input_user)==str):
        if input_user in names_list:
            uname = input_user
        else:
            return "Please Enter a Valid Name for User"
    elif (type(input_user)==int):
        if input_user in ids_list:
            uname = [user["name"] for user in users if user["id"]==input_user][0]
        else:
            return "Please Enter a Valid ID for User"
    elif (type(input_user)==dict):
        if input_user in dicts_list:
            uname = input_user["name"]
        else:
            return "Please Enter a Valid User for User"
    else:
        return "Please Enter Valid Input (name, id, or user dictionary)"
    return uname

def users_id(input_user):
    if (type(input_user)==str):
        if input_user in names_list:
            uid = [user["id"] for user in users if user["name"]==input_user][0]
        else:
            return "Please Enter a Valid Name"
    elif (type(input_user)==int):
        if input_user in ids_list:
            uid = input_user
        else:
            return "Please Enter a Valid ID"
    elif (type(input_user)==dict):
        if input_user in dicts_list:
            uid = input_user["id"]
        else:
            return "Please Enter a Valid User"
    else:
        return "Please Enter Valid Input (name, id, or user dictionary)"
    return uid

users_id("Hero")
#simple function to find data scientists names who like a certain interest
def data_scientists_who_like(target_interest):
    return[users_name(user_id) for user_id,user_interest in interests if user_interest == target_interest]


users_interests = []
check_list = []
for interest_tuple in interests:
    user_id = interest_tuple[0]
    interest = interest_tuple[1]
    if user_id in check_list:
        current_interests = list(users_interests[user_id][users_name(user_id)])
        current_interests.append(interest)
        users_interests[user_id][users_name(user_id)] = current_interests
    else:
        check_list.append(user_id)
        users_interests.append({(users_name(user_id)):[interest]})

def retrieve_users_interests(input_user2):
    if type(input_user2)==str:
        if input_user2 in names_list:
            interest_list = list(users_interests[users_id(input_user2)].values())[0]
        else:
            return "Please Enter a Valid Name"
    elif (type(input_user2)==int):
        if input_user2 in ids_list:
            interest_list = list(users_interests[input_user2].values())[0]
        else:
            return "Please Enter a Valid ID"
    elif (type(input_user2)==dict):
        if input_user2 in dicts_list:
            interest_list = list(users_interests[users_id(input_user2)].values())[0]
        else:
            return "Please Enter a Valid User"
    else:
        return "Please Enter Valid Input (name, id, or user dictionary)"
    return interest_list


def recommend_based_on_interests(input_user3):
    uid = users_id(input_user3)
    print("People Who You Share Interests With:")
    print()
    for user_interest in retrieve_users_interests(uid):
        shared_users = []
        for user in users:
            if user_interest in retrieve_users_interests(user):
                shared_users.append(user["name"])
        print(f'{user_interest}: ',', '.join(shared_users))

recommend_based_on_interests(users[0])
