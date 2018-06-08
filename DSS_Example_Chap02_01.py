# Whitespace Formating
for i in [1,2,3,4,5]:
    print(i)
    for j in [1,2,3,4,5]:
        print(j)
        print(i+j)
    print(i)
print("done looping")

list_of_list=[[1,2,3],
              [4,5,6],
              [7,8,9]]

from collections import defaultdict, Counter
lookup=defaultdict(int)
my_counter=Counter()
a=5//2

# Functions
def double(x):
    return x*2

def apply_to_one(f):
    return f(1)

my_double=double
x=apply_to_one(my_double)
y=apply_to_one(lambda x:x+4)

#Strings
single_quoted_string='data science'
double_quoted_string="data science"
tab_string="\t"
print(0/0)

# List
integer_list=[1,2,3]
heterogeneous_list=["string",0.1,True]
list_of_list=[integer_list,heterogeneous_list,[]]
list_length=len(integer_list)
list_sum=sum(integer_list)

print(0 in [1,2,3])
x,y,z=integer_list

# Tuples
def sum_and_prod(x,y):
    return (x+y),(x*y)
sp=sum_and_prod(2,3)

x,y=1,2
x,y=y,x

# Dictionaries
empty_dict={}
empty_dict2=dict()
grades={"Joel":80,"Tim":95}

tweet={
       "user":"joelgrus",
       "text":"Data science is awesome",
       "retweet_count":100,
       "hashtags":["#data","#science","#datascience","#awesome","#yolo"]}
print(tweet.keys())
tweet_values=list(tweet.values())
print("joelgrus" in tweet.values())

#Defaultdict
dd_list=defaultdict(list)
dd_list[2].append(1)

dd_dict=defaultdict(dict)
dd_dict["Joel"]["City"]="Seattle"

dd_pair=defaultdict(lambda:[0,0])
dd_pair[2][1]=1

#Counter
c=Counter([0,1,2,0])

#Sets
a=set('abracadabra')
s={1,2,3,4,2,1}
print(s)
aux=set()
aux.add(1)
aux.add(2)
aux.add(1)
print(len(aux))
print(aux)
x={'a','b','c'}
print(x)

item_list=[1,2,3,1,2,3]
num_items=len(item_list)
item_set=list(set(item_list))
print(set(item_list))

# Control Flow
x=3
parity="even" if x % 2 == 0 else "odd"

x=0
while x<10:
    print(x, "is less than 10")
    x +=1

for x in range(10):
    if x==3:
        continue
    if x==5:
        break
    print(x)

#Truthiness
all([True,1,{3}])
all([True,1,{}])
any([True,1,{}])
all([])
any([])

# Sorting
x=[4,1,2,3]
y=sorted(x)
x.sort()

x=sorted([-4,1,-2,3],key=abs,reverse=True)

#List Comprehensions
even_numbers=[x for x in range(5) if x%2 ==0]
squares=[x*x for x in range(5)]
even_squares=[x*x for x in even_numbers]

square_dict={x:x*x for x in range(5)}
square_set={x*x for x in [1,-1]}
print(square_set)

pairs=[(x,y)
        for x in range(10)
        for y in range(5)]
aux=list(range(3,10))
increasing_pairs=[(x,y)
                    for x in range(10)
                    for y in range(x+1,10)]

# Randomness
import random
four_uniform_randoms=[random.random() for _ in range(4)]

up_to_ten=list(range(2,10))
random.shuffle(up_to_ten)
print(up_to_ten)

lottery_numbers=list(range(60))
winning_numbers=random.sample(lottery_numbers,6)
four_with_replacement=[random.choice(range(10))
                        for _ in range(4)]

#Functional tools
from functools import partial
def double(x):
    return 2*x

xs=[1,2,3,4]
twice_xs=[double(x) for x in xs]
twice_xs2=list(map(double,xs))
list_doubler=partial(map,double)
twice_xs3=list(list_doubler(xs))

def multiply(x,y): return x*y
products=list(map(multiply,[1,2],[4,5]))

def is_even(x): return x%2==0
x_evens=list(filter(is_even,xs))
list_evener=partial(filter,is_even)
x_evens2=list(list_evener(xs))

# Zip
list1=["a","b","c"]
list2=[1,2,3]
a=list(zip(list1,list2))
letters,numbers=zip(*a)
