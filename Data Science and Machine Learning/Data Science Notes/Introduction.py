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
username = input("What is your name?")

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

print(mutual_friends(username))
