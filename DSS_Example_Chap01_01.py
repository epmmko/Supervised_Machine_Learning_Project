users = [
        { "id":0, "name": "Hero"},
        { "id":1, "name": "Dunn"},
        { "id":2, "name": "Sue"},
        { "id":3, "name": "Chi"},
        { "id":4, "name": "Thor"},
        { "id":5, "name": "Clive"},
        { "id":6, "name": "Hicks"},
        { "id":7, "name": "Devin"},
        { "id":8, "name": "Kate"},
        { "id":9, "name": "Klein"}
        ]
friendships=[(0,1),(0,2),(1,2),(1,3),(2,3),(3,4),(4,5),(5,6),(5,7),(6,8),(7,8),(8,9)]
for user in users:
    user["friends"]=[]

for i,j in friendships:
    # this works because users[i] is the user whose id is 1
    users[i]["friends"].append(users[j]) #add i as a friend of j
    users[j]["friends"].append(users[i]) #add j as a friend of i
    
def number_of_friends(user):
    """How many friends does _user_ have?"""
    return len(user["friends"]) # length of friends_ids list

total_connections=sum(number_of_friends(user) for user in users)
num_users=len(users)                            # length of users list
avg_connections=total_connections/num_users     # 2.4

# Create a list (user_id, num_friends)
num_friends_by_id=[(user["id"],number_of_friends(user)) for user in users]

print(sorted(num_friends_by_id,key=lambda aux:aux[1],reverse=True))

def friends_of_friends_ids_bad(user):
    #"foaf" is short for friend of a friend"
    return [foaf["id"]
            for friend in user["friends"]       #for each user's friends
            for foaf in friend["friends"]]      #get each of _their_ friends

print(friends_of_friends_ids_bad(users[0]))

from collections import Counter
def not_the_same(user,other_user):
    """two users are not the same if they have different ids"""
    return user["id"] != other_user["id"]

def not_friends(user,other_user):
    """other_user is not a friend if he's not in user["friends"];
    that is, if he's not the same as all the people in user["friends"]"""
    return all(not_the_same(friend,other_user)
               for friend in user["friends"])
def friends_of_friends_ids(user):
    return Counter(foaf["id"]
                   for friend in user["friends"]            # for each of my friends
                   for foaf in friend["friends"]            # count *their* friends
                   if not_the_same(user,foaf)               # who aren't me
                   and not_friends(user,foaf))              # who aren't my friends

print(friends_of_friends_ids(users[3]))