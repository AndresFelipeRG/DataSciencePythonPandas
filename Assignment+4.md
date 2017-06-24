
---

_You are currently looking at **version 1.1** of this notebook. To download notebooks and datafiles, as well as get help on Jupyter notebooks in the Coursera platform, visit the [Jupyter Notebook FAQ](https://www.coursera.org/learn/python-data-analysis/resources/0dhYG) course resource._

---


```python
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
```

# Assignment 4 - Hypothesis Testing
This assignment requires more individual learning than previous assignments - you are encouraged to check out the [pandas documentation](http://pandas.pydata.org/pandas-docs/stable/) to find functions or methods you might not have used yet, or ask questions on [Stack Overflow](http://stackoverflow.com/) and tag them as pandas and python related. And of course, the discussion forums are open for interaction with your peers and the course staff.

Definitions:
* A _quarter_ is a specific three month period, Q1 is January through March, Q2 is April through June, Q3 is July through September, Q4 is October through December.
* A _recession_ is defined as starting with two consecutive quarters of GDP decline, and ending with two consecutive quarters of GDP growth.
* A _recession bottom_ is the quarter within a recession which had the lowest GDP.
* A _university town_ is a city which has a high percentage of university students compared to the total population of the city.

**Hypothesis**: University towns have their mean housing prices less effected by recessions. Run a t-test to compare the ratio of the mean price of houses in university towns the quarter before the recession starts compared to the recession bottom. (`price_ratio=quarter_before_recession/recession_bottom`)

The following data files are available for this assignment:
* From the [Zillow research data site](http://www.zillow.com/research/data/) there is housing data for the United States. In particular the datafile for [all homes at a city level](http://files.zillowstatic.com/research/public/City/City_Zhvi_AllHomes.csv), ```City_Zhvi_AllHomes.csv```, has median home sale prices at a fine grained level.
* From the Wikipedia page on college towns is a list of [university towns in the United States](https://en.wikipedia.org/wiki/List_of_college_towns#College_towns_in_the_United_States) which has been copy and pasted into the file ```university_towns.txt```.
* From Bureau of Economic Analysis, US Department of Commerce, the [GDP over time](http://www.bea.gov/national/index.htm#gdp) of the United States in current dollars (use the chained value in 2009 dollars), in quarterly intervals, in the file ```gdplev.xls```. For this assignment, only look at GDP data from the first quarter of 2000 onward.

Each function in this assignment below is worth 10%, with the exception of ```run_ttest()```, which is worth 50%.


```python
# Use this dictionary to map state names to two letter acronyms
states = {'OH': 'Ohio', 'KY': 'Kentucky', 'AS': 'American Samoa', 'NV': 'Nevada', 'WY': 'Wyoming', 'NA': 'National', 'AL': 'Alabama', 'MD': 'Maryland', 'AK': 'Alaska', 'UT': 'Utah', 'OR': 'Oregon', 'MT': 'Montana', 'IL': 'Illinois', 'TN': 'Tennessee', 'DC': 'District of Columbia', 'VT': 'Vermont', 'ID': 'Idaho', 'AR': 'Arkansas', 'ME': 'Maine', 'WA': 'Washington', 'HI': 'Hawaii', 'WI': 'Wisconsin', 'MI': 'Michigan', 'IN': 'Indiana', 'NJ': 'New Jersey', 'AZ': 'Arizona', 'GU': 'Guam', 'MS': 'Mississippi', 'PR': 'Puerto Rico', 'NC': 'North Carolina', 'TX': 'Texas', 'SD': 'South Dakota', 'MP': 'Northern Mariana Islands', 'IA': 'Iowa', 'MO': 'Missouri', 'CT': 'Connecticut', 'WV': 'West Virginia', 'SC': 'South Carolina', 'LA': 'Louisiana', 'KS': 'Kansas', 'NY': 'New York', 'NE': 'Nebraska', 'OK': 'Oklahoma', 'FL': 'Florida', 'CA': 'California', 'CO': 'Colorado', 'PA': 'Pennsylvania', 'DE': 'Delaware', 'NM': 'New Mexico', 'RI': 'Rhode Island', 'MN': 'Minnesota', 'VI': 'Virgin Islands', 'NH': 'New Hampshire', 'MA': 'Massachusetts', 'GA': 'Georgia', 'ND': 'North Dakota', 'VA': 'Virginia'}
```


```python
import pandas as pd
import numpy as np
def get_list_of_university_towns():
    '''Returns a DataFrame of towns and the states they are in from the 
    university_towns.txt list. The format of the DataFrame should be:
    DataFrame( [ ["Michigan", "Ann Arbor"], ["Michigan", "Yipsilanti"] ], 
    columns=["State", "RegionName"]  )
    
    The following cleaning needs to be done:

    1. For "State", removing characters from "[" to the end.
    2. For "RegionName", when applicable, removing every character from " (" to the end.
    3. Depending on how you read the data, you may need to remove newline character '\n'. '''
    
    university = pd.read_table('university_towns.txt', header = None)
    university.replace(to_replace=r'\[.*\]', value='', inplace=True, regex=True)
    data = pd.DataFrame(columns = ["State","RegionName"])
   
    state = "Alabama"
    for x in university.index:
        
        if university.loc[x][0] in states.values():
            state = university.loc[x][0]
            continue
        else:
            s = university.loc[x][0]
            regionName = s
            res = s.find("(")
            if res > 0:
                regionName = s[0:res]
            if res < 0:
                regionName = s[0:len(s)]
           
            if regionName[-1] == " ":
                regionName = regionName[0:len(regionName)-1]
            element = pd.Series({'State': state,'RegionName': regionName})
            data = data.append(element, ignore_index = True)
            
            
        
        
    
    
    return data

```


```python

```


```python
def get_recession_start():
    
    '''Returns the year and quarter of the recession start time as a 
    string value in a format such as 2005q3'''
    gdp = pd.read_excel('gdplev.xls', header = None)
    gdp = gdp[gdp.columns[4:7]]
    gdp = gdp[220:]
    indexes = gdp.index
    gdp["decrease"] = 0
    starting = ""
    for x in range(1, len(indexes)):
        if (x == 0):
            gdp.loc[indexes[x],"decrease"] = 0
            continue
        if(gdp.loc[indexes[x],6] < gdp.loc[indexes[x-1],6]):
            gdp.loc[indexes[x],"decrease"] = 1
            continue
        if(gdp.loc[indexes[x],6] == gdp.loc[indexes[x-1],6]):
            gdp.loc[indexes[x],"decrease"] = 0
            continue
        if(gdp.loc[indexes[x],6] > gdp.loc[indexes[x-1],6]):
            gdp.loc[indexes[x],"decrease"] = 2
            continue
    for x in range(1, len(indexes)-2):
        if(gdp.loc[indexes[x],"decrease"] == 1 and gdp.loc[indexes[x +1],"decrease"] == 1):
                   starting = gdp.loc[indexes[x],4] 
                   break
        
    return starting

```


```python
def get_recession_end():
    '''Returns the year and quarter of the recession end time as a 
    string value in a format such as 2005q3'''
    gdp = pd.read_excel('gdplev.xls', header = None)
    gdp = gdp[gdp.columns[4:7]]
    gdp = gdp[220:]
    indexes = gdp.index
    gdp["decrease"] = 0
    starting = ""
    ending = ""
    for x in range(1, len(indexes)):
        if (x == 0):
            gdp.loc[indexes[x],"decrease"] = 0
            continue
        if(gdp.loc[indexes[x],6] < gdp.loc[indexes[x-1],6]):
            gdp.loc[indexes[x],"decrease"] = 1
            continue
        if(gdp.loc[indexes[x],6] == gdp.loc[indexes[x-1],6]):
            gdp.loc[indexes[x],"decrease"] = 0
            continue
        if(gdp.loc[indexes[x],6] > gdp.loc[indexes[x-1],6]):
            gdp.loc[indexes[x],"decrease"] = 2
            continue
    for x in range(1, len(indexes)-2):
        if(gdp.loc[indexes[x],"decrease"] == 1 and gdp.loc[indexes[x +1],"decrease"] == 1):
                   starting = x
                   break
    for x in range(x+2, len(indexes)-3):
         if(gdp.loc[indexes[x],"decrease"] == 2 and gdp.loc[indexes[x +1],"decrease"] == 2):
                    ending = gdp.loc[indexes[x+1],4]
                    break
    return ending

```


```python
def get_recession_bottom():
    '''Returns the year and quarter of the recession bottom time as a 
    string value in a format such as 2005q3'''
    
    gdp = pd.read_excel('gdplev.xls', header = None)
    gdp = gdp[gdp.columns[4:7]]
    gdp = gdp[220:]
    indexes = gdp.index
    gdp["decrease"] = 0
    starting = 0
    ending = 0
    minimum = 10000000000
    minquart = ""
    for x in range(1, len(indexes)):
        if (x == 0):
            gdp.loc[indexes[x],"decrease"] = 0
            continue
        if(gdp.loc[indexes[x],6] < gdp.loc[indexes[x-1],6]):
            gdp.loc[indexes[x],"decrease"] = 1
            continue
        if(gdp.loc[indexes[x],6] == gdp.loc[indexes[x-1],6]):
            gdp.loc[indexes[x],"decrease"] = 0
            continue
        if(gdp.loc[indexes[x],6] > gdp.loc[indexes[x-1],6]):
            gdp.loc[indexes[x],"decrease"] = 2
            continue
    for x in range(1, len(indexes)-2):
        if(gdp.loc[indexes[x],"decrease"] == 1 and gdp.loc[indexes[x +1],"decrease"] == 1):
                   starting = x
                   break
    for x in range(x+2, len(indexes)-3):
         if(gdp.loc[indexes[x],"decrease"] == 2 and gdp.loc[indexes[x +1],"decrease"] == 2):
                    ending = x+1
                    break
    for x in range(starting, ending + 1):
        if(gdp.loc[indexes[x],6] < minimum):
            minimum = gdp.loc[indexes[x],6] 
            minquart = gdp.loc[indexes[x],4]
            
         
    return minquart

```


```python

```


```python
def convert_housing_data_to_quarters():
    '''Converts the housing data to quarters and returns it as mean 
    values in a dataframe. This dataframe should be a dataframe with
    columns for 2000q1 through 2016q3, and should have a multi-index
    in the shape of ["State","RegionName"].
    
    Note: Quarters are defined in the assignment description, they are
    not arbitrary three month periods.
    
    The resulting dataframe should have 67 columns, and 10,730 rows.
    
    '''
    def fun2(row):
        states = {'OH': 'Ohio', 'KY': 'Kentucky', 'AS': 'American Samoa', 'NV': 'Nevada', 'WY': 'Wyoming', 'NA': 'National', 'AL': 'Alabama', 'MD': 'Maryland', 'AK': 'Alaska', 'UT': 'Utah', 'OR': 'Oregon', 'MT': 'Montana', 'IL': 'Illinois', 'TN': 'Tennessee', 'DC': 'District of Columbia', 'VT': 'Vermont', 'ID': 'Idaho', 'AR': 'Arkansas', 'ME': 'Maine', 'WA': 'Washington', 'HI': 'Hawaii', 'WI': 'Wisconsin', 'MI': 'Michigan', 'IN': 'Indiana', 'NJ': 'New Jersey', 'AZ': 'Arizona', 'GU': 'Guam', 'MS': 'Mississippi', 'PR': 'Puerto Rico', 'NC': 'North Carolina', 'TX': 'Texas', 'SD': 'South Dakota', 'MP': 'Northern Mariana Islands', 'IA': 'Iowa', 'MO': 'Missouri', 'CT': 'Connecticut', 'WV': 'West Virginia', 'SC': 'South Carolina', 'LA': 'Louisiana', 'KS': 'Kansas', 'NY': 'New York', 'NE': 'Nebraska', 'OK': 'Oklahoma', 'FL': 'Florida', 'CA': 'California', 'CO': 'Colorado', 'PA': 'Pennsylvania', 'DE': 'Delaware', 'NM': 'New Mexico', 'RI': 'Rhode Island', 'MN': 'Minnesota', 'VI': 'Virgin Islands', 'NH': 'New Hampshire', 'MA': 'Massachusetts', 'GA': 'Georgia', 'ND': 'North Dakota', 'VA': 'Virginia'}
        val = row["State"]
        row["State"] = states[val]
        
          
        return row
        
    
            
    
   
    data = pd.read_csv("City_Zhvi_AllHomes.csv")
    columns = [ 'State', 'RegionName']
    columns2 = [str(year)+'-0'+str(month) if month <10 else str(year)+'-'+str(month) for year in range(2000,2016) for month in range(1,13)]
    columns3 = ["2016"+'-0'+str(month) if month <10 else "2016"+'-'+str(month) for month in range(1,9)]
    
    missing = data[columns]
    
    
    data = data[columns2+columns3]
    data = data.rename(columns=pd.to_datetime)
    cols = data.columns
    sel_cols = cols[cols > '2000']
    data = data[sel_cols]
    
 

    data = data.resample('Q',axis=1).mean().rename(columns=lambda x: '{:}q{:}'.format(x.year, [1, 2,3,4][(x.quarter-1)%4]))
    data[columns] = missing
    data = data.apply(fun2, axis = 1)
    data.set_index(['State', 'RegionName'],inplace = True)
    return data

```


```python

```


```python
def run_ttest():
    '''First creates new data showing the decline or growth of housing prices
    between the recession start and the recession bottom. Then runs a ttest
    comparing the university town values to the non-university towns values, 
    return whether the alternative hypothesis (that the two groups are the same)
    is true or not as well as the p-value of the confidence. 
    
    Return the tuple (different, p, better) where different=True if the t-test is
    True at a p<0.01 (we reject the null hypothesis), or different=False if 
    otherwise (we cannot reject the null hypothesis). The variable p should
    be equal to the exact p value returned from scipy.stats.ttest_ind(). The
    value for better should be either "university town" or "non-university town"
    depending on which has a lower mean price ratio (which is equivilent to a
    reduced market loss).'''
   
    start = get_recession_start()
    before_start = "2008q2"
    ending = get_recession_bottom()
    u = get_list_of_university_towns()
    u.set_index(["State","RegionName"], inplace = True)
    housing_data = convert_housing_data_to_quarters()
    array = [True if x >= before_start and x <= ending else False for x in housing_data.columns]
    data = housing_data[housing_data.columns[array]]
    u_data = pd.merge(u, data, how = 'inner', left_index = True, right_index = True)
    data_indices = data.index
    u_data_indices = u_data.index
    diff = data_indices.difference(u_data_indices)
    no_u_data = data.loc[diff]
    
    
    u_data["Price"] = np.divide(u_data[before_start], u_data[ending])
    no_u_data["Price"] = np.divide(no_u_data[before_start], no_u_data[ending])
    p_val = ttest_ind(u_data["Price"].values.astype('float'),no_u_data["Price"].values.astype('float'),nan_policy='omit')
    different = p_val.pvalue < 0.01
    pv = p_val.pvalue
    better = "university town" if np.mean(u_data["Price"]) < np.mean(no_u_data["Price"]) else "non-university town"
    return  (different, pv, better)

```


```python

```
