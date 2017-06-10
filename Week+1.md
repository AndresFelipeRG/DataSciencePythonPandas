
---

_You are currently looking at **version 1.1** of this notebook. To download notebooks and datafiles, as well as get help on Jupyter notebooks in the Coursera platform, visit the [Jupyter Notebook FAQ](https://www.coursera.org/learn/python-data-analysis/resources/0dhYG) course resource._

---

# The Python Programming Language: Functions


```python
x = 1
y = 2
x + y
```




    3




```python
x
```




    1



<br>
`add_numbers` is a function that takes two numbers and adds them together.


```python
def add_numbers(x, y):
    return x + y

add_numbers(1, 2)
add_numbers(1, 3)
```




    4



<br>
`add_numbers` updated to take an optional 3rd parameter. Using `print` allows printing of multiple expressions within a single cell.


```python
def add_numbers(x,y,z=None):
    if (z==None):
        return x+y
    else:
        return x+y+z

print(add_numbers(1, 2))
print(add_numbers(1, 2, 3))
def substract_numbers2(x, y, z = None):
    if(z == None):
        return x - y
    else:
        return x - y - z
print(substract_numbers2(1,2,3))
print(substract_numbers2(1,2))
```

    3
    6
    -4
    -1


<br>
`add_numbers` updated to take an optional flag parameter.


```python
def add_numbers(x, y, z=None, flag=False):
    if (flag):
        print('Flag is true!')
    if (z==None):
        return x + y
    else:
        return x + y + z
    
print(add_numbers(1, 2, flag=True))

def substract_numbers(x, y, z = None, flag = False):
    if(flag):
        print('Flag is false!')
    if( z == None):
        return x - y
    else:
        return x-y-z
```

    Flag is true!
    3


<br>
Assign function `add_numbers` to variable `a`.


```python
def add_numbers(x,y):
    return x+y

a = add_numbers
a(1,2)
z = add_numbers
z(1,3)
```




    4



<br>
# The Python Programming Language: Types and Sequences

<br>
Use `type` to return the object's type.


```python
type('This is a string')
```


```python
type(None)
```


```python
type(1)
```


```python
type(1.0)
```


```python
type(add_numbers)
```

<br>
Tuples are an immutable data structure (cannot be altered).


```python
x = (1, 'a', 2, 'b')
type(x)

```


    ---------------------------------------------------------------------------

    AttributeError                            Traceback (most recent call last)

    <ipython-input-8-f5cad1717823> in <module>()
          1 x = (1, 'a', 2, 'b')
          2 type(x)
    ----> 3 x.append(2)
    

    AttributeError: 'tuple' object has no attribute 'append'


<br>
Lists are a mutable data structure.


```python
x = [1, 'a', 2, 'b']
type(x)
```




    list



<br>
Use `append` to append an object to a list.


```python
x.append(3.3)
x.append(4)
print(x)
```

    [1, 'a', 2, 'b', 3.3, 4]


<br>
This is an example of how to loop through each item in the list.


```python
for item in x:
    print(item)
```

<br>
Or using the indexing operator:


```python
i=0
while( i != len(x) ):
    print(x[i])
    i = i + 1
```

<br>
Use `+` to concatenate lists.


```python
[1,2] + [3,4]
```

<br>
Use `*` to repeat lists.


```python
[1]*3
```

<br>
Use the `in` operator to check if something is inside a list.


```python
1 in [1, 2, 3]
```

<br>
Now let's look at strings. Use bracket notation to slice a string.


```python
x = 'This is a string'

print(x[0]) #first character
print(x[0:1]) #first character, but we have explicitly set the end character
print(x[0:2]) #first two characters
print([1,3]+['a','a'])
```

    T
    T
    Th
    [1, 3, 'a', 'a']


<br>
This will return the last element of the string.


```python
x[-1]
```

<br>
This will return the slice starting from the 4th element from the end and stopping before the 2nd element from the end.


```python
x[-4:-2]
```




    [2, 'b']



<br>
This is a slice from the beginning of the string and stopping before the 3rd element.


```python
x[:3]
```

<br>
And this is a slice starting from the 3rd element of the string and going all the way to the end.


```python
x[3:]
```


```python
firstname = 'Christopher'
lastname = 'Brooks'

print(firstname + ' ' + lastname)
print(firstname*3)
print('Chris' in firstname)

```

    Christopher Brooks
    ChristopherChristopherChristopher
    True


<br>
`split` returns a list of all the words in a string, or a list split on a specific character.


```python
firstname = 'Christopher Arthur Hansen Brooks'.split(' ')[0] # [0] selects the first element of the list
lastname = 'Christopher Arthur Hansen Brooks'.split(' ')[-1] # [-1] selects the last element of the list
print(firstname)
print(lastname)
```

<br>
Make sure you convert objects to strings before concatenating.


```python
'Chris' + 2
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-14-1623ac76de6e> in <module>()
    ----> 1 'Chris' + 2
    

    TypeError: Can't convert 'int' object to str implicitly



```python
'Chris' + str(2)
```




    'Chris2'



<br>
Dictionaries associate keys with values.


```python
x = {'Christopher Brooks': 'brooksch@umich.edu', 'Bill Gates': 'billg@microsoft.com'}
x['Christopher Brooks'] # Retrieve a value by using the indexing operator

```


```python
x['Kevyn Collins-Thompson'] = None
x['Kevyn Collins-Thompson']
```

<br>
Iterate over all of the keys:


```python
for name in x:
    print(x[name])
```

<br>
Iterate over all of the values:


```python
for email in x.values():
    print(email)
```

<br>
Iterate over all of the items in the list:


```python
for name, email in x.items():
    print(name)
    print(email)
```

<br>
You can unpack a sequence into different variables:


```python
x = ('Christopher', 'Brooks', 'brooksch@umich.edu')
fname, lname, email = x
```


```python
fname
```




    'Christopher'




```python
lname
```




    'Brooks'



<br>
Make sure the number of values you are unpacking matches the number of variables being assigned.


```python
x = ('Christopher', 'Brooks', 'brooksch@umich.edu', 'Ann Arbor')
fname, lname, email, z = x
```

<br>
# The Python Programming Language: More on Strings


```python
print('Chris' + 2)
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-21-82ccfdd3d5d3> in <module>()
    ----> 1 print('Chris' + 2)
    

    TypeError: Can't convert 'int' object to str implicitly



```python
print('Chris' + str(2))
```

    Chris2


<br>
Python has a built in method for convenient string formatting.


```python
sales_record = {
'price': 3.24,
'num_items': 4,
'person': 'Chris'}

sales_statement = '{} bought {} item(s) at a price of {} each for a total of {}'

print(sales_statement.format(sales_record['person'],
                             sales_record['num_items'],
                             sales_record['price'],
                             sales_record['num_items']*sales_record['price']))

```

    Chris bought 4 item(s) at a price of 3.24 each for a total of 12.96



```python
age_record = {
    'age': 12,
    'name': 'Andres'
    
}
statement = 'My name is {} and I am {} years old'
print(statement.format(age_record['name'],age_record['age']))
```

    My name is Andres and I am 12 years old


<br>
# Reading and Writing CSV files

<br>
Let's import our datafile mpg.csv, which contains fuel economy data for 234 cars.

* mpg : miles per gallon
* class : car classification
* cty : city mpg
* cyl : # of cylinders
* displ : engine displacement in liters
* drv : f = front-wheel drive, r = rear wheel drive, 4 = 4wd
* fl : fuel (e = ethanol E85, d = diesel, r = regular, p = premium, c = CNG)
* hwy : highway mpg
* manufacturer : automobile manufacturer
* model : model of car
* trans : type of transmission
* year : model year


```python
import csv

%precision 2

with open('mpg.csv') as csvfile:
    mpg = list(csv.DictReader(csvfile))
    
mpg[:3] # The first three dictionaries in our list.
```




    [{'': '1',
      'class': 'compact',
      'cty': '18',
      'cyl': '4',
      'displ': '1.8',
      'drv': 'f',
      'fl': 'p',
      'hwy': '29',
      'manufacturer': 'audi',
      'model': 'a4',
      'trans': 'auto(l5)',
      'year': '1999'},
     {'': '2',
      'class': 'compact',
      'cty': '21',
      'cyl': '4',
      'displ': '1.8',
      'drv': 'f',
      'fl': 'p',
      'hwy': '29',
      'manufacturer': 'audi',
      'model': 'a4',
      'trans': 'manual(m5)',
      'year': '1999'},
     {'': '3',
      'class': 'compact',
      'cty': '20',
      'cyl': '4',
      'displ': '2',
      'drv': 'f',
      'fl': 'p',
      'hwy': '31',
      'manufacturer': 'audi',
      'model': 'a4',
      'trans': 'manual(m6)',
      'year': '2008'}]



<br>
`csv.Dictreader` has read in each row of our csv file as a dictionary. `len` shows that our list is comprised of 234 dictionaries.


```python
len(mpg)
```




    234



<br>
`keys` gives us the column names of our csv.


```python
mpg[0].keys()
```




    dict_keys(['', 'class', 'displ', 'cyl', 'manufacturer', 'hwy', 'cty', 'model', 'trans', 'drv', 'year', 'fl'])



<br>
This is how to find the average cty fuel economy across all cars. All values in the dictionaries are strings, so we need to convert to float.


```python
sum(float(d['cty']) for d in mpg) / len(mpg)
```

<br>
Similarly this is how to find the average hwy fuel economy across all cars.


```python
sum(float(d['hwy']) for d in mpg) / len(mpg)
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-1-f7c5699e8c30> in <module>()
    ----> 1 sum(float(d['hwy']) for d in mpg) / len(mpg)
    

    NameError: name 'mpg' is not defined


<br>
Use `set` to return the unique values for the number of cylinders the cars in our dataset have.


```python
cylinders = set(d['cyl'] for d in mpg)
cylinders
```




    {'4', '5', '6', '8'}



<br>
Here's a more complex example where we are grouping the cars by number of cylinder, and finding the average cty mpg for each group.


```python
CtyMpgByCyl = []

for c in cylinders: # iterate over all the cylinder levels
    summpg = 0
    cyltypecount = 0
    for d in mpg: # iterate over all dictionaries
        if d['cyl'] == c: # if the cylinder level type matches,
            summpg += float(d['cty']) # add the cty mpg
            cyltypecount += 1 # increment the count
    CtyMpgByCyl.append((c, summpg / cyltypecount)) # append the tuple ('cylinder', 'avg mpg')

CtyMpgByCyl.sort(key=lambda x: x[0])
CtyMpgByCyl
```




    [('4', 21.01), ('5', 20.50), ('6', 16.22), ('8', 12.57)]



<br>
Use `set` to return the unique values for the class types in our dataset.


```python
vehicleclass = set(d['class'] for d in mpg) # what are the class types
vehicleclass
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-1-e8ee153325ee> in <module>()
    ----> 1 vehicleclass = set(d['class'] for d in mpg) # what are the class types
          2 vehicleclass


    NameError: name 'mpg' is not defined


<br>
And here's an example of how to find the average hwy mpg for each class of vehicle in our dataset.


```python
HwyMpgByClass = []

for t in vehicleclass: # iterate over all the vehicle classes
    summpg = 0
    vclasscount = 0
    for d in mpg: # iterate over all dictionaries
        if d['class'] == t: # if the cylinder amount type matches,
            summpg += float(d['hwy']) # add the hwy mpg
            vclasscount += 1 # increment the count
    HwyMpgByClass.append((t, summpg / vclasscount)) # append the tuple ('class', 'avg mpg')

HwyMpgByClass.sort(key=lambda x: x[1])
HwyMpgByClass
```

<br>
# The Python Programming Language: Dates and Times


```python
import datetime as dt
import time as tm
```


```python
import datetime as dt
import time as tm
```

<br>
`time` returns the current time in seconds since the Epoch. (January 1st, 1970)


```python
tm.time()
```




    1496799876.93



<br>
Convert the timestamp to datetime.


```python
dtnow = dt.datetime.fromtimestamp(tm.time())
dtnow
```




    datetime.datetime(2017, 6, 7, 1, 44, 55, 256537)



<br>
Handy datetime attributes:


```python
dtnow.year, dtnow.month, dtnow.day, dtnow.hour, dtnow.minute, dtnow.second # get year, month, day, etc.from a datetime
```

<br>
`timedelta` is a duration expressing the difference between two dates.


```python
delta = dt.timedelta(days = 100) # create a timedelta of 100 days
delta
```




    datetime.timedelta(100)



<br>
`date.today` returns the current local date.


```python
today = dt.date.today()
```


```python
today - delta # the date 100 days ago
```




    datetime.date(2017, 2, 27)




```python
today > today-delta # compare dates
```




    True



<br>
# The Python Programming Language: Objects and map()

<br>
An example of a class in python:


```python
class Person:
    department = 'School of Information' #a class variable
    grade = '4.0' # grade of the student
    
    def set_name(self, new_name): #a method
        self.name = new_name
    def set_location(self, new_location):
        self.location = new_location
```


```python
person = Person()
person.set_name('Christopher Brooks')
person.set_location('Ann Arbor, MI, USA')
print('{} live in {} and works in the department {}'.format(person.name, person.location, person.department))
```

    Christopher Brooks live in Ann Arbor, MI, USA and works in the department School of Information


<br>
Here's an example of mapping the `min` function between two lists.


```python
store1 = [10.00, 11.00, 12.34, 2.34]
store2 = [9.00, 11.10, 12.34, 2.01]
cheapest = map(min, store1, store2) #apply to both inputs
cheapest
```




    <map at 0x7f4fb031dc18>



<br>
Now let's iterate through the map object to see the values.


```python
for item in cheapest:
    print(item)
```

    9.0
    11.0
    12.34
    2.01


<br>
# The Python Programming Language: Lambda and List Comprehensions

<br>
Here's an example of lambda that takes in three parameters and adds the first two.


```python
my_function = lambda a, b, c : a + b
```


```python
type(lambda x: x + 1)
```




    function




```python
my_function(1, 2, 3)
```


```python
people = ['Dr. Christopher Brooks', 'Dr. Kevyn Collins-Thompson', 'Dr. VG Vinod Vydiswaran', 'Dr. Daniel Romero']
numbers = [1,2,3,4]
def split_title_and_name(person):
    return person.split()[0] + ' ' + person.split()[-1]

#option 1
for person in people:
    print(split_title_and_name(person) == (lambda x: x.split()[0] + ' ' + x.split()[-1])(person))
#option 2
list(map(split_title_and_name, people)) == list(map(lambda person: person.split()[0] + ' ' + person.split()[-1], people))
```

    True
    True
    True
    True





    True



<br>
Let's iterate from 0 to 999 and return the even numbers.


```python
my_list = []
for number in range(0, 1000):
    if number % 2 == 0:
        my_list.append(number)
my_list
```

<br>
Now the same thing but with list comprehension.


```python
my_list = [number for number in range(0,1000) if number % 2 == 0] # faster than loop
my_list
```




    [0,
     2,
     4,
     6,
     8,
     10,
     12,
     14,
     16,
     18,
     20,
     22,
     24,
     26,
     28,
     30,
     32,
     34,
     36,
     38,
     40,
     42,
     44,
     46,
     48,
     50,
     52,
     54,
     56,
     58,
     60,
     62,
     64,
     66,
     68,
     70,
     72,
     74,
     76,
     78,
     80,
     82,
     84,
     86,
     88,
     90,
     92,
     94,
     96,
     98,
     100,
     102,
     104,
     106,
     108,
     110,
     112,
     114,
     116,
     118,
     120,
     122,
     124,
     126,
     128,
     130,
     132,
     134,
     136,
     138,
     140,
     142,
     144,
     146,
     148,
     150,
     152,
     154,
     156,
     158,
     160,
     162,
     164,
     166,
     168,
     170,
     172,
     174,
     176,
     178,
     180,
     182,
     184,
     186,
     188,
     190,
     192,
     194,
     196,
     198,
     200,
     202,
     204,
     206,
     208,
     210,
     212,
     214,
     216,
     218,
     220,
     222,
     224,
     226,
     228,
     230,
     232,
     234,
     236,
     238,
     240,
     242,
     244,
     246,
     248,
     250,
     252,
     254,
     256,
     258,
     260,
     262,
     264,
     266,
     268,
     270,
     272,
     274,
     276,
     278,
     280,
     282,
     284,
     286,
     288,
     290,
     292,
     294,
     296,
     298,
     300,
     302,
     304,
     306,
     308,
     310,
     312,
     314,
     316,
     318,
     320,
     322,
     324,
     326,
     328,
     330,
     332,
     334,
     336,
     338,
     340,
     342,
     344,
     346,
     348,
     350,
     352,
     354,
     356,
     358,
     360,
     362,
     364,
     366,
     368,
     370,
     372,
     374,
     376,
     378,
     380,
     382,
     384,
     386,
     388,
     390,
     392,
     394,
     396,
     398,
     400,
     402,
     404,
     406,
     408,
     410,
     412,
     414,
     416,
     418,
     420,
     422,
     424,
     426,
     428,
     430,
     432,
     434,
     436,
     438,
     440,
     442,
     444,
     446,
     448,
     450,
     452,
     454,
     456,
     458,
     460,
     462,
     464,
     466,
     468,
     470,
     472,
     474,
     476,
     478,
     480,
     482,
     484,
     486,
     488,
     490,
     492,
     494,
     496,
     498,
     500,
     502,
     504,
     506,
     508,
     510,
     512,
     514,
     516,
     518,
     520,
     522,
     524,
     526,
     528,
     530,
     532,
     534,
     536,
     538,
     540,
     542,
     544,
     546,
     548,
     550,
     552,
     554,
     556,
     558,
     560,
     562,
     564,
     566,
     568,
     570,
     572,
     574,
     576,
     578,
     580,
     582,
     584,
     586,
     588,
     590,
     592,
     594,
     596,
     598,
     600,
     602,
     604,
     606,
     608,
     610,
     612,
     614,
     616,
     618,
     620,
     622,
     624,
     626,
     628,
     630,
     632,
     634,
     636,
     638,
     640,
     642,
     644,
     646,
     648,
     650,
     652,
     654,
     656,
     658,
     660,
     662,
     664,
     666,
     668,
     670,
     672,
     674,
     676,
     678,
     680,
     682,
     684,
     686,
     688,
     690,
     692,
     694,
     696,
     698,
     700,
     702,
     704,
     706,
     708,
     710,
     712,
     714,
     716,
     718,
     720,
     722,
     724,
     726,
     728,
     730,
     732,
     734,
     736,
     738,
     740,
     742,
     744,
     746,
     748,
     750,
     752,
     754,
     756,
     758,
     760,
     762,
     764,
     766,
     768,
     770,
     772,
     774,
     776,
     778,
     780,
     782,
     784,
     786,
     788,
     790,
     792,
     794,
     796,
     798,
     800,
     802,
     804,
     806,
     808,
     810,
     812,
     814,
     816,
     818,
     820,
     822,
     824,
     826,
     828,
     830,
     832,
     834,
     836,
     838,
     840,
     842,
     844,
     846,
     848,
     850,
     852,
     854,
     856,
     858,
     860,
     862,
     864,
     866,
     868,
     870,
     872,
     874,
     876,
     878,
     880,
     882,
     884,
     886,
     888,
     890,
     892,
     894,
     896,
     898,
     900,
     902,
     904,
     906,
     908,
     910,
     912,
     914,
     916,
     918,
     920,
     922,
     924,
     926,
     928,
     930,
     932,
     934,
     936,
     938,
     940,
     942,
     944,
     946,
     948,
     950,
     952,
     954,
     956,
     958,
     960,
     962,
     964,
     966,
     968,
     970,
     972,
     974,
     976,
     978,
     980,
     982,
     984,
     986,
     988,
     990,
     992,
     994,
     996,
     998]




```python
def times_tables():
    lst = []
    for i in range(10):
        for j in range (10):
            lst.append(i*j)
    return lst

times_tables() == [j*i for i in range(10) for j in range(10)]
my_list = [j-i for j in range(100) for i in range(10)]
my_list
```




    [0,
     -1,
     -2,
     -3,
     -4,
     -5,
     -6,
     -7,
     -8,
     -9,
     1,
     0,
     -1,
     -2,
     -3,
     -4,
     -5,
     -6,
     -7,
     -8,
     2,
     1,
     0,
     -1,
     -2,
     -3,
     -4,
     -5,
     -6,
     -7,
     3,
     2,
     1,
     0,
     -1,
     -2,
     -3,
     -4,
     -5,
     -6,
     4,
     3,
     2,
     1,
     0,
     -1,
     -2,
     -3,
     -4,
     -5,
     5,
     4,
     3,
     2,
     1,
     0,
     -1,
     -2,
     -3,
     -4,
     6,
     5,
     4,
     3,
     2,
     1,
     0,
     -1,
     -2,
     -3,
     7,
     6,
     5,
     4,
     3,
     2,
     1,
     0,
     -1,
     -2,
     8,
     7,
     6,
     5,
     4,
     3,
     2,
     1,
     0,
     -1,
     9,
     8,
     7,
     6,
     5,
     4,
     3,
     2,
     1,
     0,
     10,
     9,
     8,
     7,
     6,
     5,
     4,
     3,
     2,
     1,
     11,
     10,
     9,
     8,
     7,
     6,
     5,
     4,
     3,
     2,
     12,
     11,
     10,
     9,
     8,
     7,
     6,
     5,
     4,
     3,
     13,
     12,
     11,
     10,
     9,
     8,
     7,
     6,
     5,
     4,
     14,
     13,
     12,
     11,
     10,
     9,
     8,
     7,
     6,
     5,
     15,
     14,
     13,
     12,
     11,
     10,
     9,
     8,
     7,
     6,
     16,
     15,
     14,
     13,
     12,
     11,
     10,
     9,
     8,
     7,
     17,
     16,
     15,
     14,
     13,
     12,
     11,
     10,
     9,
     8,
     18,
     17,
     16,
     15,
     14,
     13,
     12,
     11,
     10,
     9,
     19,
     18,
     17,
     16,
     15,
     14,
     13,
     12,
     11,
     10,
     20,
     19,
     18,
     17,
     16,
     15,
     14,
     13,
     12,
     11,
     21,
     20,
     19,
     18,
     17,
     16,
     15,
     14,
     13,
     12,
     22,
     21,
     20,
     19,
     18,
     17,
     16,
     15,
     14,
     13,
     23,
     22,
     21,
     20,
     19,
     18,
     17,
     16,
     15,
     14,
     24,
     23,
     22,
     21,
     20,
     19,
     18,
     17,
     16,
     15,
     25,
     24,
     23,
     22,
     21,
     20,
     19,
     18,
     17,
     16,
     26,
     25,
     24,
     23,
     22,
     21,
     20,
     19,
     18,
     17,
     27,
     26,
     25,
     24,
     23,
     22,
     21,
     20,
     19,
     18,
     28,
     27,
     26,
     25,
     24,
     23,
     22,
     21,
     20,
     19,
     29,
     28,
     27,
     26,
     25,
     24,
     23,
     22,
     21,
     20,
     30,
     29,
     28,
     27,
     26,
     25,
     24,
     23,
     22,
     21,
     31,
     30,
     29,
     28,
     27,
     26,
     25,
     24,
     23,
     22,
     32,
     31,
     30,
     29,
     28,
     27,
     26,
     25,
     24,
     23,
     33,
     32,
     31,
     30,
     29,
     28,
     27,
     26,
     25,
     24,
     34,
     33,
     32,
     31,
     30,
     29,
     28,
     27,
     26,
     25,
     35,
     34,
     33,
     32,
     31,
     30,
     29,
     28,
     27,
     26,
     36,
     35,
     34,
     33,
     32,
     31,
     30,
     29,
     28,
     27,
     37,
     36,
     35,
     34,
     33,
     32,
     31,
     30,
     29,
     28,
     38,
     37,
     36,
     35,
     34,
     33,
     32,
     31,
     30,
     29,
     39,
     38,
     37,
     36,
     35,
     34,
     33,
     32,
     31,
     30,
     40,
     39,
     38,
     37,
     36,
     35,
     34,
     33,
     32,
     31,
     41,
     40,
     39,
     38,
     37,
     36,
     35,
     34,
     33,
     32,
     42,
     41,
     40,
     39,
     38,
     37,
     36,
     35,
     34,
     33,
     43,
     42,
     41,
     40,
     39,
     38,
     37,
     36,
     35,
     34,
     44,
     43,
     42,
     41,
     40,
     39,
     38,
     37,
     36,
     35,
     45,
     44,
     43,
     42,
     41,
     40,
     39,
     38,
     37,
     36,
     46,
     45,
     44,
     43,
     42,
     41,
     40,
     39,
     38,
     37,
     47,
     46,
     45,
     44,
     43,
     42,
     41,
     40,
     39,
     38,
     48,
     47,
     46,
     45,
     44,
     43,
     42,
     41,
     40,
     39,
     49,
     48,
     47,
     46,
     45,
     44,
     43,
     42,
     41,
     40,
     50,
     49,
     48,
     47,
     46,
     45,
     44,
     43,
     42,
     41,
     51,
     50,
     49,
     48,
     47,
     46,
     45,
     44,
     43,
     42,
     52,
     51,
     50,
     49,
     48,
     47,
     46,
     45,
     44,
     43,
     53,
     52,
     51,
     50,
     49,
     48,
     47,
     46,
     45,
     44,
     54,
     53,
     52,
     51,
     50,
     49,
     48,
     47,
     46,
     45,
     55,
     54,
     53,
     52,
     51,
     50,
     49,
     48,
     47,
     46,
     56,
     55,
     54,
     53,
     52,
     51,
     50,
     49,
     48,
     47,
     57,
     56,
     55,
     54,
     53,
     52,
     51,
     50,
     49,
     48,
     58,
     57,
     56,
     55,
     54,
     53,
     52,
     51,
     50,
     49,
     59,
     58,
     57,
     56,
     55,
     54,
     53,
     52,
     51,
     50,
     60,
     59,
     58,
     57,
     56,
     55,
     54,
     53,
     52,
     51,
     61,
     60,
     59,
     58,
     57,
     56,
     55,
     54,
     53,
     52,
     62,
     61,
     60,
     59,
     58,
     57,
     56,
     55,
     54,
     53,
     63,
     62,
     61,
     60,
     59,
     58,
     57,
     56,
     55,
     54,
     64,
     63,
     62,
     61,
     60,
     59,
     58,
     57,
     56,
     55,
     65,
     64,
     63,
     62,
     61,
     60,
     59,
     58,
     57,
     56,
     66,
     65,
     64,
     63,
     62,
     61,
     60,
     59,
     58,
     57,
     67,
     66,
     65,
     64,
     63,
     62,
     61,
     60,
     59,
     58,
     68,
     67,
     66,
     65,
     64,
     63,
     62,
     61,
     60,
     59,
     69,
     68,
     67,
     66,
     65,
     64,
     63,
     62,
     61,
     60,
     70,
     69,
     68,
     67,
     66,
     65,
     64,
     63,
     62,
     61,
     71,
     70,
     69,
     68,
     67,
     66,
     65,
     64,
     63,
     62,
     72,
     71,
     70,
     69,
     68,
     67,
     66,
     65,
     64,
     63,
     73,
     72,
     71,
     70,
     69,
     68,
     67,
     66,
     65,
     64,
     74,
     73,
     72,
     71,
     70,
     69,
     68,
     67,
     66,
     65,
     75,
     74,
     73,
     72,
     71,
     70,
     69,
     68,
     67,
     66,
     76,
     75,
     74,
     73,
     72,
     71,
     70,
     69,
     68,
     67,
     77,
     76,
     75,
     74,
     73,
     72,
     71,
     70,
     69,
     68,
     78,
     77,
     76,
     75,
     74,
     73,
     72,
     71,
     70,
     69,
     79,
     78,
     77,
     76,
     75,
     74,
     73,
     72,
     71,
     70,
     80,
     79,
     78,
     77,
     76,
     75,
     74,
     73,
     72,
     71,
     81,
     80,
     79,
     78,
     77,
     76,
     75,
     74,
     73,
     72,
     82,
     81,
     80,
     79,
     78,
     77,
     76,
     75,
     74,
     73,
     83,
     82,
     81,
     80,
     79,
     78,
     77,
     76,
     75,
     74,
     84,
     83,
     82,
     81,
     80,
     79,
     78,
     77,
     76,
     75,
     85,
     84,
     83,
     82,
     81,
     80,
     79,
     78,
     77,
     76,
     86,
     85,
     84,
     83,
     82,
     81,
     80,
     79,
     78,
     77,
     87,
     86,
     85,
     84,
     83,
     82,
     81,
     80,
     79,
     78,
     88,
     87,
     86,
     85,
     84,
     83,
     82,
     81,
     80,
     79,
     89,
     88,
     87,
     86,
     85,
     84,
     83,
     82,
     81,
     80,
     90,
     89,
     88,
     87,
     86,
     85,
     84,
     83,
     82,
     81,
     91,
     90,
     89,
     88,
     87,
     86,
     85,
     84,
     83,
     82,
     92,
     91,
     90,
     89,
     88,
     87,
     86,
     85,
     84,
     83,
     93,
     92,
     91,
     90,
     89,
     88,
     87,
     86,
     85,
     84,
     94,
     93,
     92,
     91,
     90,
     89,
     88,
     87,
     86,
     85,
     95,
     94,
     93,
     92,
     91,
     90,
     89,
     88,
     87,
     86,
     96,
     95,
     94,
     93,
     92,
     91,
     90,
     89,
     88,
     87,
     97,
     96,
     95,
     94,
     93,
     92,
     91,
     90,
     89,
     88,
     98,
     97,
     96,
     95,
     94,
     93,
     92,
     91,
     90,
     89,
     99,
     98,
     97,
     96,
     95,
     94,
     93,
     92,
     91,
     90]



<br>
# The Python Programming Language: Numerical Python (NumPy)


```python
import numpy as np
```

<br>
## Creating Arrays

Create a list and convert it to a numpy array


```python
mylist = [1, 2, 3]
x = np.array(mylist)
x
```




    array([1, 2, 3])



<br>
Or just pass in a list directly


```python
y = np.array([4, 5, 6])
y
```




    array([4, 5, 6])



<br>
Pass in a list of lists to create a multidimensional array.


```python
m = np.array([[7, 8, 9], [10, 11, 12]])
m
```




    array([[ 7,  8,  9],
           [10, 11, 12]])



<br>
Use the shape method to find the dimensions of the array. (rows, columns)


```python
m.shape
```




    (2, 3)



<br>
`arange` returns evenly spaced values within a given interval.


```python
n = np.arange(0, 30, 2) # start at 0 count up by 2, stop before 30
n
```




    array([ 0,  2,  4,  6,  8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28])



<br>
`reshape` returns an array with the same data with a new shape.


```python
n = n.reshape(3, 5) # reshape array to be 3x5
n
```




    array([[ 0,  2,  4,  6,  8],
           [10, 12, 14, 16, 18],
           [20, 22, 24, 26, 28]])



<br>
`linspace` returns evenly spaced numbers over a specified interval.


```python
o = np.linspace(0, 4, 9) # return 9 evenly spaced values from 0 to 4
o
```




    array([ 0. ,  0.5,  1. ,  1.5,  2. ,  2.5,  3. ,  3.5,  4. ])



<br>
`resize` changes the shape and size of array in-place.


```python
o.resize(3, 3)
o
```




    array([[ 0. ,  0.5,  1. ],
           [ 1.5,  2. ,  2.5],
           [ 3. ,  3.5,  4. ]])




```python
o[::4]
o.reshape(9)
```




    array([ 0. ,  0.5,  1. ,  1.5,  2. ,  2.5,  3. ,  3.5,  4. ])



<br>
`ones` returns a new array of given shape and type, filled with ones.


```python
np.ones((3, 2))
```




    array([[ 1.,  1.],
           [ 1.,  1.],
           [ 1.,  1.]])



<br>
`zeros` returns a new array of given shape and type, filled with zeros.


```python
np.zeros((2, 3))
```




    array([[ 0.,  0.,  0.],
           [ 0.,  0.,  0.]])



<br>
`eye` returns a 2-D array with ones on the diagonal and zeros elsewhere.


```python
np.eye(3)
```




    array([[ 1.,  0.,  0.],
           [ 0.,  1.,  0.],
           [ 0.,  0.,  1.]])



<br>
`diag` extracts a diagonal or constructs a diagonal array.


```python
np.diag(y)
```




    array([[4, 0, 0],
           [0, 5, 0],
           [0, 0, 6]])



<br>
Create an array using repeating list (or see `np.tile`)


```python
np.array([1, 2, 3] * 3)
```




    array([1, 2, 3, 1, 2, 3, 1, 2, 3])



<br>
Repeat elements of an array using `repeat`.


```python
np.repeat([1, 2, 3], 3)
```




    array([1, 1, 1, 2, 2, 2, 3, 3, 3])



<br>
#### Combining Arrays


```python
p = np.ones([2, 3], int)
p
```




    array([[1, 1, 1],
           [1, 1, 1]])



<br>
Use `vstack` to stack arrays in sequence vertically (row wise).


```python
np.vstack([p, 2*p])
```




    array([[1, 1, 1],
           [1, 1, 1],
           [2, 2, 2],
           [2, 2, 2]])



<br>
Use `hstack` to stack arrays in sequence horizontally (column wise).


```python
np.hstack([p, 2*p])
```




    array([[1, 1, 1, 2, 2, 2],
           [1, 1, 1, 2, 2, 2]])



<br>
## Operations

Use `+`, `-`, `*`, `/` and `**` to perform element wise addition, subtraction, multiplication, division and power.


```python
print(x + y) # elementwise addition     [1 2 3] + [4 5 6] = [5  7  9]
print(x - y) # elementwise subtraction  [1 2 3] - [4 5 6] = [-3 -3 -3]
```

    [5 7 9]
    [-3 -3 -3]



```python
print(x * y) # elementwise multiplication  [1 2 3] * [4 5 6] = [4  10  18]
print(x / y) # elementwise divison         [1 2 3] / [4 5 6] = [0.25  0.4  0.5]
```

    [ 4 10 18]
    [ 0.25  0.4   0.5 ]



```python
print(x**2) # elementwise power  [1 2 3] ^2 =  [1 4 9]
```

    [1 4 9]


<br>
**Dot Product:**  

$ \begin{bmatrix}x_1 \ x_2 \ x_3\end{bmatrix}
\cdot
\begin{bmatrix}y_1 \\ y_2 \\ y_3\end{bmatrix}
= x_1 y_1 + x_2 y_2 + x_3 y_3$


```python
x.dot(y) # dot product  1*4 + 2*5 + 3*6
```




    32



<br>
Let's look at transposing arrays. Transposing permutes the dimensions of the array.


```python
z = np.array([y, y**2])
print(len(z)) # number of rows of array
```

    2



```python
z
```




    array([[ 4,  5,  6],
           [16, 25, 36]])




```python
z = np.array([y, y**2])
z
```

<br>
The shape of array `z` is `(2,3)` before transposing.


```python
z.shape
```




    (2, 3)



<br>
Use `.T` to get the transpose.


```python
z.T
```




    array([[ 4, 16],
           [ 5, 25],
           [ 6, 36]])



<br>
The number of rows has swapped with the number of columns.


```python
z.T.shape
```




    (3, 2)



<br>
Use `.dtype` to see the data type of the elements in the array.


```python
z.dtype
```




    dtype('int64')



<br>
Use `.astype` to cast to a specific type.


```python
z = z.astype('f')
z.dtype
```




    dtype('float32')



<br>
## Math Functions

Numpy has many built in math functions that can be performed on arrays.


```python
a = np.array([-4, -2, 1, 3, 5])
```


```python
a.sum()
```




    3




```python
a.max()
```




    5




```python
a.min()
```




    -4




```python
a.mean()
```




    0.59999999999999998




```python
a.std()
```




    3.2619012860600183



<br>
`argmax` and `argmin` return the index of the maximum and minimum values in the array.


```python
a.argmax()
```




    4




```python
a.argmin()
```




    0



<br>
## Indexing / Slicing


```python
s = np.arange(13)**2
s
```




    array([  0,   1,   4,   9,  16,  25,  36,  49,  64,  81, 100, 121, 144])



<br>
Use bracket notation to get the value at a specific index. Remember that indexing starts at 0.


```python
s[0], s[4], s[-1]
```




    (0, 16, 144)



<br>
Use `:` to indicate a range. `array[start:stop]`


Leaving `start` or `stop` empty will default to the beginning/end of the array.


```python
s[1:5]
```




    array([ 1,  4,  9, 16])



<br>
Use negatives to count from the back.


```python
s[-4:]
```




    array([ 81, 100, 121, 144])



<br>
A second `:` can be used to indicate step-size. `array[start:stop:stepsize]`

Here we are starting 5th element from the end, and counting backwards by 2 until the beginning of the array is reached.


```python
s[-5::-2]
```




    array([64, 36, 16,  4,  0])



<br>
Let's look at a multidimensional array.


```python
r = np.arange(36)
r.resize((6, 6))
r
```




    array([[ 0,  1,  2,  3,  4,  5],
           [ 6,  7,  8,  9, 10, 11],
           [12, 13, 14, 15, 16, 17],
           [18, 19, 20, 21, 22, 23],
           [24, 25, 26, 27, 28, 29],
           [30, 31, 32, 33, 34, 35]])



<br>
Use bracket notation to slice: `array[row, column]`


```python
r[2, 2]
```




    14



<br>
And use : to select a range of rows or columns


```python
r[3, 3:6]
```




    array([21, 22, 23])



<br>
Here we are selecting all the rows up to (and not including) row 2, and all the columns up to (and not including) the last column.


```python
r[:2, :-1]
```




    array([[ 0,  1,  2,  3,  4],
           [ 6,  7,  8,  9, 10]])



<br>
This is a slice of the last row, and only every other element.


```python
r[-1, ::2]
```




    array([30, 32, 34])



<br>
We can also perform conditional indexing. Here we are selecting values from the array that are greater than 30. (Also see `np.where`)


```python
r[r > 30]
```




    array([31, 32, 33, 34, 35])



<br>
Here we are assigning all values in the array that are greater than 30 to the value of 30.


```python
r[r > 30] = 30
r
```




    array([[ 0,  1,  2,  3,  4,  5],
           [ 6,  7,  8,  9, 10, 11],
           [12, 13, 14, 15, 16, 17],
           [18, 19, 20, 21, 22, 23],
           [24, 25, 26, 27, 28, 29],
           [30, 30, 30, 30, 30, 30]])



<br>
## Copying Data

Be careful with copying and modifying arrays in NumPy!


`r2` is a slice of `r`


```python
r2 = r[:3,:3] #r2 has a reference to r.
r2
```




    array([[ 0,  1,  2],
           [ 6,  7,  8],
           [12, 13, 14]])



<br>
Set this slice's values to zero ([:] selects the entire array)


```python
r2[:] = 0
r2
```




    array([[0, 0, 0],
           [0, 0, 0],
           [0, 0, 0]])



<br>
`r` has also been changed!


```python
r
```




    array([[ 0,  0,  0,  3,  4,  5],
           [ 0,  0,  0,  9, 10, 11],
           [ 0,  0,  0, 15, 16, 17],
           [18, 19, 20, 21, 22, 23],
           [24, 25, 26, 27, 28, 29],
           [30, 30, 30, 30, 30, 30]])



<br>
To avoid this, use `r.copy` to create a copy that will not affect the original array


```python
r_copy = r.copy()
r_copy
```




    array([[ 0,  0,  0,  3,  4,  5],
           [ 0,  0,  0,  9, 10, 11],
           [ 0,  0,  0, 15, 16, 17],
           [18, 19, 20, 21, 22, 23],
           [24, 25, 26, 27, 28, 29],
           [30, 30, 30, 30, 30, 30]])



<br>
Now when r_copy is modified, r will not be changed.


```python
r_copy[:] = 10
print(r_copy, '\n')
print(r)
```

    [[10 10 10 10 10 10]
     [10 10 10 10 10 10]
     [10 10 10 10 10 10]
     [10 10 10 10 10 10]
     [10 10 10 10 10 10]
     [10 10 10 10 10 10]] 
    
    [[ 0  0  0  3  4  5]
     [ 0  0  0  9 10 11]
     [ 0  0  0 15 16 17]
     [18 19 20 21 22 23]
     [24 25 26 27 28 29]
     [30 30 30 30 30 30]]


<br>
### Iterating Over Arrays

Let's create a new 4 by 3 array of random numbers 0-9.


```python
test = np.random.randint(0, 10, (4,3))
test
```




    array([[3, 2, 5],
           [0, 6, 3],
           [2, 0, 0],
           [7, 9, 5]])



<br>
Iterate by row:


```python
for row in test:
    print(row)
```

    [3 2 5]
    [0 6 3]
    [2 0 0]
    [7 9 5]


<br>
Iterate by index:


```python
for i in range(len(test)):
    print(test[i])
```

    [3 2 5]
    [0 6 3]
    [2 0 0]
    [7 9 5]


<br>
Iterate by row and index:


```python
for i, row in enumerate(test):
    print('row', i, 'is', row)
```

    row 0 is [3 2 5]
    row 1 is [0 6 3]
    row 2 is [2 0 0]
    row 3 is [7 9 5]


<br>
Use `zip` to iterate over multiple iterables.


```python
test2 = test**2
test2
```




    array([[ 9,  4, 25],
           [ 0, 36,  9],
           [ 4,  0,  0],
           [49, 81, 25]])




```python
for i, j in zip(test, test2):
    print(i,'+',j,'=',i+j)
```

    [3 2 5] + [ 9  4 25] = [12  6 30]
    [0 6 3] + [ 0 36  9] = [ 0 42 12]
    [2 0 0] + [4 0 0] = [6 0 0]
    [7 9 5] + [49 81 25] = [56 90 30]



```python

```
