
# coding: utf-8

# ---
# 
# _You are currently looking at **version 1.0** of this notebook. To download notebooks and datafiles, as well as get help on Jupyter notebooks in the Coursera platform, visit the [Jupyter Notebook FAQ](https://www.coursera.org/learn/python-data-analysis/resources/0dhYG) course resource._
# 
# ---

# # The Series Data Structure

# In[5]:

import pandas as pd
get_ipython().magic('pinfo pd.Series')


# In[10]:

animals = ['Tiger', 'Bear', 'Moose']
pd.Series(animals) # list and dictionary.


# In[11]:

numbers = [1, 2, 3]
pd.Series(numbers)


# In[14]:

animals = ['Tiger', 'Bear', None]
pd.Series(animals)


# In[15]:

numbers = [1, 2, None] #None = NaN
pd.Series(numbers)


# In[17]:

import numpy as np
np.nan == None #nan is not None


# In[19]:

np.nan == np.nan


# In[20]:

np.isnan(np.nan)


# In[21]:

sports = {'Archery': 'Bhutan',
          'Golf': 'Scotland',
          'Sumo': 'Japan',
          'Taekwondo': 'South Korea'}
s = pd.Series(sports)
s


# In[22]:

s.index


# In[23]:

s = pd.Series(['Tiger', 'Bear', 'Moose'], index=['India', 'America', 'Canada'])
s


# In[24]:

sports = {'Archery': 'Bhutan',
          'Golf': 'Scotland',
          'Sumo': 'Japan',
          'Taekwondo': 'South Korea'}
s = pd.Series(sports, index=['Golf', 'Sumo', 'Hockey'])
s


# # Querying a Series

# In[25]:

sports = {'Archery': 'Bhutan',
          'Golf': 'Scotland',
          'Sumo': 'Japan',
          'Taekwondo': 'South Korea'}
s = pd.Series(sports)
s


# In[26]:

s.iloc[3]


# In[27]:

s.loc['Golf']


# In[ ]:

s[3]


# In[28]:

s['Golf']


# In[29]:

sports = {99: 'Bhutan',
          100: 'Scotland',
          101: 'Japan',
          102: 'South Korea'}
s = pd.Series(sports)


# In[32]:

s[0] #This won't call s.iloc[0] as one might expect, it generates an error instead


# In[33]:

s = pd.Series([100.00, 120.00, 101.00, 3.00])
s


# In[34]:

total = 0
for item in s:
    total+=item
print(total) # slow


# In[1]:

import numpy as np

total = np.sum(s)
print(total) #fast by vectorization


# In[8]:

#this creates a big series of random numbers
s = pd.Series(np.random.randint(0,1000,10000))
s.head()


# In[9]:

len(s)


# In[10]:

get_ipython().run_cell_magic('timeit', '-n 100', 'summary = 0\nfor item in s:\n    summary+=item')


# In[11]:

get_ipython().run_cell_magic('timeit', '-n 100', 'summary = np.sum(s)')


# In[42]:

s+=2 #adds two to each item in s using broadcasting.. FASTER
s.head()


# In[43]:

for label, value in s.iteritems():
    s.set_value(label, value+2)
s.head()


# In[12]:

get_ipython().run_cell_magic('timeit', '-n 10', 's = pd.Series(np.random.randint(0,1000,10000))\nfor label, value in s.iteritems():\n    s.loc[label]= value+2')


# In[ ]:

get_ipython().run_cell_magic('timeit', '-n 10', 's = pd.Series(np.random.randint(0,1000,10000))\ns+=2')


# In[13]:

s = pd.Series([1, 2, 3])
s.loc['Animal'] = 'Bears'
s


# In[14]:

original_sports = pd.Series({'Archery': 'Bhutan',
                             'Golf': 'Scotland',
                             'Sumo': 'Japan',
                             'Taekwondo': 'South Korea'})
cricket_loving_countries = pd.Series(['Australia',
                                      'Barbados',
                                      'Pakistan',
                                      'England'], 
                                   index=['Cricket',
                                          'Cricket',
                                          'Cricket',
                                          'Cricket'])
all_countries = original_sports.append(cricket_loving_countries)


# In[15]:

original_sports


# In[16]:

cricket_loving_countries


# In[17]:

all_countries


# In[18]:

all_countries.loc['Cricket']


# # The DataFrame Data Structure

# In[1]:

import pandas as pd
purchase_1 = pd.Series({'Name': 'Chris',
                        'Item Purchased': 'Dog Food',
                        'Cost': 22.50})
purchase_2 = pd.Series({'Name': 'Kevyn',
                        'Item Purchased': 'Kitty Litter',
                        'Cost': 2.50})
purchase_3 = pd.Series({'Name': 'Vinod',
                        'Item Purchased': 'Bird Seed',
                        'Cost': 5.00})
df = pd.DataFrame([purchase_1, purchase_2, purchase_3], index=['Store 1', 'Store 1', 'Store 2'])
df.head()


# In[22]:

df.loc['Store 2']


# In[23]:

type(df.loc['Store 2'])


# In[24]:

df.loc['Store 1']


# In[25]:

df.loc['Store 1', 'Cost']


# In[26]:

df.T


# In[27]:

df.T.loc['Cost'] # since Cost needs to be in the row section to use loc


# In[28]:

df['Cost']


# In[30]:

df.loc['Store 1']['Cost']


# In[31]:

df.loc[:,['Name', 'Cost']]


# In[30]:

import pandas as pd

purchase_1 = pd.Series({'Name': 'Chris',
                        'Item Purchased': 'Dog Food',
                        'Cost': 22.50})
purchase_2 = pd.Series({'Name': 'Kevyn',
                        'Item Purchased': 'Kitty Litter',
                        'Cost': 2.50})
purchase_3 = pd.Series({'Name': 'Vinod',
                        'Item Purchased': 'Bird Seed',
                        'Cost': 5.00})

df = pd.DataFrame([purchase_1, purchase_2, purchase_3], index=['Store 1', 'Store 1', 'Store 2'])
df


# In[32]:

df['Item Purchased']


# In[ ]:

df.drop('Store 1')


# In[38]:

df


# In[39]:

copy_df = df.copy()
copy_df = copy_df.drop('Store 1')
copy_df


# In[40]:

get_ipython().magic('pinfo copy_df.drop')


# In[41]:

del copy_df['Name']
copy_df


# In[42]:

df['Location'] = None
df


# In[10]:

purchase_1 = pd.Series({'Name': 'Chris',
                        'Item Purchased': 'Dog Food',
                        'Cost': 22.50})
purchase_2 = pd.Series({'Name': 'Kevyn',
                        'Item Purchased': 'Kitty Litter',
                        'Cost': 2.50})
purchase_3 = pd.Series({'Name': 'Vinod',
                        'Item Purchased': 'Bird Seed',
                        'Cost': 5.00})

df = pd.DataFrame([purchase_1, purchase_2, purchase_3], index=['Store 1', 'Store 1', 'Store 2'])


df['Cost'] *= 0.8
df['Cost2'] = 3
df['Cost2'] = df['Cost2']-2
print(df)


# # Dataframe Indexing and Loading

# In[11]:

costs = df['Cost']
costs


# In[14]:

costs+=2
costs


# In[5]:

df


# In[15]:

get_ipython().system('cat olympics.csv')


# In[31]:

df = pd.read_csv('olympics.csv')
df.head()


# In[39]:

df = pd.read_csv('olympics.csv', index_col = 0, skiprows=1)
df.head()


# In[40]:

df.columns


# In[41]:

for col in df.columns:
    if col[:2]=='01':
        df.rename(columns={col:'Gold' + col[4:]}, inplace=True)
        # inplace = True. Pandas updates directly.
    if col[:2]=='02':
        df.rename(columns={col:'Silver' + col[4:]}, inplace=True)
    if col[:2]=='03':
        df.rename(columns={col:'Bronze' + col[4:]}, inplace=True)
    if col[:1]=='№':
        df.rename(columns={col:'#' + col[1:]}, inplace=True) 

df.head()


# # Querying a DataFrame

# In[42]:

df['Gold'] > 0


# In[9]:

only_gold = df.where(df['Gold'] > 0) #nan doesnt meet conditionk
only_gold.head()


# In[10]:

only_gold['Gold'].count()


# In[11]:

df['Gold'].count()


# In[12]:

only_gold = only_gold.dropna()
only_gold.head()


# In[13]:

only_gold = df[df['Gold'] > 0]
only_gold.head()


# In[14]:

len(df[(df['Gold'] > 0) | (df['Gold.1'] > 0)]) ## won in winter or summer


# In[16]:

df[(df['Gold.1'] > 0) & (df['Gold'] == 0)] ## won in winter and not in summer


# In[18]:

purchase_1 = pd.Series({'Name': 'Chris',
                        'Item Purchased': 'Dog Food',
                        'Cost': 22.50})
purchase_2 = pd.Series({'Name': 'Kevyn',
                        'Item Purchased': 'Kitty Litter',
                        'Cost': 2.50})
purchase_3 = pd.Series({'Name': 'Vinod',
                        'Item Purchased': 'Bird Seed',
                        'Cost': 5.00})

df = pd.DataFrame([purchase_1, purchase_2, purchase_3], index=['Store 1', 'Store 1', 'Store 2'])


df['Name'][df['Cost']>3] ## indexing returns indexes or rows numbers..not the columns


# # Indexing Dataframes

# In[43]:

df.head()


# In[44]:

df['country'] = df.index   #make a column country with index of df which is the name of the countries
df = df.set_index('Gold')  # set gold as index
df.head()


# In[46]:

df = df.reset_index()  #create default index
df.head()


# In[47]:

df = pd.read_csv('census.csv')
df.head()


# In[49]:

df['SUMLEV'].unique() #get possible values in the column SUMLEV


# In[51]:

df=df[df['SUMLEV'] == 50]
df.head()


# In[53]:

columns_to_keep = ['STNAME',
                   'CTYNAME',
                   'BIRTHS2010',
                   'BIRTHS2011',
                   'BIRTHS2012',
                   'BIRTHS2013',
                   'BIRTHS2014',
                   'BIRTHS2015',
                   'POPESTIMATE2010',
                   'POPESTIMATE2011',
                   'POPESTIMATE2012',
                   'POPESTIMATE2013',
                   'POPESTIMATE2014',
                   'POPESTIMATE2015']
df = df[columns_to_keep]
df.head()


# In[54]:

df = df.set_index(['STNAME', 'CTYNAME'])
df.head()


# In[55]:

df.loc['Michigan', 'Washtenaw County']


# In[57]:

df.loc[ [('Michigan', 'Washtenaw County'),
         ('Michigan', 'Wayne County')] ]  # need to pass a list of indexes


# In[58]:

purchase_1 = pd.Series({'Name': 'Chris',
                        'Item Purchased': 'Dog Food',
                        'Cost': 22.50})
purchase_2 = pd.Series({'Name': 'Kevyn',
                        'Item Purchased': 'Kitty Litter',
                        'Cost': 2.50})
purchase_3 = pd.Series({'Name': 'Vinod',
                        'Item Purchased': 'Bird Seed',
                        'Cost': 5.00})

df2 = pd.DataFrame([purchase_1, purchase_2, purchase_3], index=['Store 1', 'Store 1', 'Store 2'])


df2 = df2.set_index([df2.index, 'Name'])
df2.index.names = ['Location', 'Name']
df2 = df2.append(pd.Series(data={'Cost': 3.00, 'Item Purchased': 'Kitty Food'}, name=('Store 2', 'Kevyn')))
df2


# # Missing values

# In[65]:

df = pd.read_csv('log.csv')
df


# In[70]:

get_ipython().magic('pinfo df.fillna')


# In[67]:

df = df.set_index('time')
df = df.sort_index()
df


# In[71]:

df = df.reset_index()
df = df.set_index(['time', 'user'])
df


# In[72]:

df = df.fillna(method='ffill')
df.head()


# In[ ]:



