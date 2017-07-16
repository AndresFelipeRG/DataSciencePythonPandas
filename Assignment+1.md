
---

_You are currently looking at **version 1.1** of this notebook. To download notebooks and datafiles, as well as get help on Jupyter notebooks in the Coursera platform, visit the [Jupyter Notebook FAQ](https://www.coursera.org/learn/python-machine-learning/resources/bANLa) course resource._

---

# Assignment 1 - Introduction to Machine Learning

For this assignment, you will be using the Breast Cancer Wisconsin (Diagnostic) Database to create a classifier that can help diagnose patients. First, read through the description of the dataset (below).


```python
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()

print(cancer.DESCR) # Print the data set description
```

    Breast Cancer Wisconsin (Diagnostic) Database
    =============================================
    
    Notes
    -----
    Data Set Characteristics:
        :Number of Instances: 569
    
        :Number of Attributes: 30 numeric, predictive attributes and the class
    
        :Attribute Information:
            - radius (mean of distances from center to points on the perimeter)
            - texture (standard deviation of gray-scale values)
            - perimeter
            - area
            - smoothness (local variation in radius lengths)
            - compactness (perimeter^2 / area - 1.0)
            - concavity (severity of concave portions of the contour)
            - concave points (number of concave portions of the contour)
            - symmetry 
            - fractal dimension ("coastline approximation" - 1)
    
            The mean, standard error, and "worst" or largest (mean of the three
            largest values) of these features were computed for each image,
            resulting in 30 features.  For instance, field 3 is Mean Radius, field
            13 is Radius SE, field 23 is Worst Radius.
    
            - class:
                    - WDBC-Malignant
                    - WDBC-Benign
    
        :Summary Statistics:
    
        ===================================== ====== ======
                                               Min    Max
        ===================================== ====== ======
        radius (mean):                        6.981  28.11
        texture (mean):                       9.71   39.28
        perimeter (mean):                     43.79  188.5
        area (mean):                          143.5  2501.0
        smoothness (mean):                    0.053  0.163
        compactness (mean):                   0.019  0.345
        concavity (mean):                     0.0    0.427
        concave points (mean):                0.0    0.201
        symmetry (mean):                      0.106  0.304
        fractal dimension (mean):             0.05   0.097
        radius (standard error):              0.112  2.873
        texture (standard error):             0.36   4.885
        perimeter (standard error):           0.757  21.98
        area (standard error):                6.802  542.2
        smoothness (standard error):          0.002  0.031
        compactness (standard error):         0.002  0.135
        concavity (standard error):           0.0    0.396
        concave points (standard error):      0.0    0.053
        symmetry (standard error):            0.008  0.079
        fractal dimension (standard error):   0.001  0.03
        radius (worst):                       7.93   36.04
        texture (worst):                      12.02  49.54
        perimeter (worst):                    50.41  251.2
        area (worst):                         185.2  4254.0
        smoothness (worst):                   0.071  0.223
        compactness (worst):                  0.027  1.058
        concavity (worst):                    0.0    1.252
        concave points (worst):               0.0    0.291
        symmetry (worst):                     0.156  0.664
        fractal dimension (worst):            0.055  0.208
        ===================================== ====== ======
    
        :Missing Attribute Values: None
    
        :Class Distribution: 212 - Malignant, 357 - Benign
    
        :Creator:  Dr. William H. Wolberg, W. Nick Street, Olvi L. Mangasarian
    
        :Donor: Nick Street
    
        :Date: November, 1995
    
    This is a copy of UCI ML Breast Cancer Wisconsin (Diagnostic) datasets.
    https://goo.gl/U2Uwz2
    
    Features are computed from a digitized image of a fine needle
    aspirate (FNA) of a breast mass.  They describe
    characteristics of the cell nuclei present in the image.
    
    Separating plane described above was obtained using
    Multisurface Method-Tree (MSM-T) [K. P. Bennett, "Decision Tree
    Construction Via Linear Programming." Proceedings of the 4th
    Midwest Artificial Intelligence and Cognitive Science Society,
    pp. 97-101, 1992], a classification method which uses linear
    programming to construct a decision tree.  Relevant features
    were selected using an exhaustive search in the space of 1-4
    features and 1-3 separating planes.
    
    The actual linear program used to obtain the separating plane
    in the 3-dimensional space is that described in:
    [K. P. Bennett and O. L. Mangasarian: "Robust Linear
    Programming Discrimination of Two Linearly Inseparable Sets",
    Optimization Methods and Software 1, 1992, 23-34].
    
    This database is also available through the UW CS ftp server:
    
    ftp ftp.cs.wisc.edu
    cd math-prog/cpo-dataset/machine-learn/WDBC/
    
    References
    ----------
       - W.N. Street, W.H. Wolberg and O.L. Mangasarian. Nuclear feature extraction 
         for breast tumor diagnosis. IS&T/SPIE 1993 International Symposium on 
         Electronic Imaging: Science and Technology, volume 1905, pages 861-870,
         San Jose, CA, 1993.
       - O.L. Mangasarian, W.N. Street and W.H. Wolberg. Breast cancer diagnosis and 
         prognosis via linear programming. Operations Research, 43(4), pages 570-577, 
         July-August 1995.
       - W.H. Wolberg, W.N. Street, and O.L. Mangasarian. Machine learning techniques
         to diagnose breast cancer from fine-needle aspirates. Cancer Letters 77 (1994) 
         163-171.
    


The object returned by `load_breast_cancer()` is a scikit-learn Bunch object, which is similar to a dictionary.


```python
cancer.keys()
```




    dict_keys(['feature_names', 'data', 'target_names', 'target', 'DESCR'])



### Question 0 (Example)

How many features does the breast cancer dataset have?

*This function should return an integer.*


```python
# You should write your whole answer within the function provided. The autograder will call
# this function and compare the return value against the correct solution value
def answer_zero():
    # This function returns the number of features of the breast cancer dataset, which is an integer. 
    # The assignment question description will tell you the general format the autograder is expecting
    return len(cancer['feature_names'])

# You can examine what your function returns by calling it in the cell. If you have questions
# about the assignment formats, check out the discussion forums for any FAQs
answer_zero() 
```




    30



### Question 1

Scikit-learn works with lists, numpy arrays, scipy-sparse matrices, and pandas DataFrames, so converting the dataset to a DataFrame is not necessary for training this model. Using a DataFrame does however help make many things easier such as munging data, so let's practice creating a classifier with a pandas DataFrame. 



Convert the sklearn.dataset `cancer` to a DataFrame. 

*This function should return a `(569, 31)` DataFrame with * 

*columns = *

    ['mean radius', 'mean texture', 'mean perimeter', 'mean area',
    'mean smoothness', 'mean compactness', 'mean concavity',
    'mean concave points', 'mean symmetry', 'mean fractal dimension',
    'radius error', 'texture error', 'perimeter error', 'area error',
    'smoothness error', 'compactness error', 'concavity error',
    'concave points error', 'symmetry error', 'fractal dimension error',
    'worst radius', 'worst texture', 'worst perimeter', 'worst area',
    'worst smoothness', 'worst compactness', 'worst concavity',
    'worst concave points', 'worst symmetry', 'worst fractal dimension',
    'target']

*and index = *

    RangeIndex(start=0, stop=569, step=1)


```python
def answer_one():
    
   
    cancerd = cancer.target.reshape(569,1)
    answer  = np.hstack([cancer.data, cancerd])
    newdata = pd.DataFrame(answer, index = range(0,569), columns =['mean radius', 'mean texture', 'mean perimeter', 'mean area',
'mean smoothness', 'mean compactness', 'mean concavity',
'mean concave points', 'mean symmetry', 'mean fractal dimension',
'radius error', 'texture error', 'perimeter error', 'area error',
'smoothness error', 'compactness error', 'concavity error',
'concave points error', 'symmetry error', 'fractal dimension error',
'worst radius', 'worst texture', 'worst perimeter', 'worst area',
'worst smoothness', 'worst compactness', 'worst concavity',
'worst concave points', 'worst symmetry', 'worst fractal dimension',
'target'] )
    return newdata
    


```

### Question 2
What is the class distribution? (i.e. how many instances of `malignant` (encoded 0) and how many `benign` (encoded 1)?)

*This function should return a Series named `target` of length 2 with integer values and index =* `['malignant', 'benign']`


```python
def answer_two():
    cancerdf = answer_one()
    
    # Your code here
    m = len(cancerdf[cancerdf["target"] == 0])
    b = len(cancerdf[cancerdf["target"] == 1])
    data = {"malignant": m, "benign": b}
    s = pd.Series(data, index = ['malignant', 'benign'])
    return s# Return your answer


answer_two()
```




    malignant    212
    benign       357
    dtype: int64



### Question 3
Split the DataFrame into `X` (the data) and `y` (the labels).

*This function should return a tuple of length 2:* `(X, y)`*, where* 
* `X` *has shape* `(569, 30)`
* `y` *has shape* `(569,)`.


```python
def answer_three():
    cancerdf = answer_one()
    cols = ['mean radius', 'mean texture', 'mean perimeter', 'mean area',
'mean smoothness', 'mean compactness', 'mean concavity',
'mean concave points', 'mean symmetry', 'mean fractal dimension',
'radius error', 'texture error', 'perimeter error', 'area error',
'smoothness error', 'compactness error', 'concavity error',
'concave points error', 'symmetry error', 'fractal dimension error',
'worst radius', 'worst texture', 'worst perimeter', 'worst area',
'worst smoothness', 'worst compactness', 'worst concavity',
'worst concave points', 'worst symmetry', 'worst fractal dimension']
    X = cancerdf[cols]
    y = cancerdf["target"]
    # Your code here
    
    return X, y

```

### Question 4
Using `train_test_split`, split `X` and `y` into training and test sets `(X_train, X_test, y_train, and y_test)`.

**Set the random number generator state to 0 using `random_state=0` to make sure your results match the autograder!**

*This function should return a tuple of length 4:* `(X_train, X_test, y_train, y_test)`*, where* 
* `X_train` *has shape* `(426, 30)`
* `X_test` *has shape* `(143, 30)`
* `y_train` *has shape* `(426,)`
* `y_test` *has shape* `(143,)`


```python
from sklearn.model_selection import train_test_split

def answer_four():
    X, y = answer_three()
    
    # Your code here
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    return X_train, X_test, y_train, y_test
```

### Question 5
Using KNeighborsClassifier, fit a k-nearest neighbors (knn) classifier with `X_train`, `y_train` and using one nearest neighbor (`n_neighbors = 1`).

*This function should return a * `sklearn.neighbors.classification.KNeighborsClassifier`.


```python
from sklearn.neighbors import KNeighborsClassifier

def answer_five():
    X_train, X_test, y_train, y_test = answer_four()
    
    knn = KNeighborsClassifier(n_neighbors = 1)
    # Your code here
    
    return knn.fit(X_train, y_train)
```

### Question 6
Using your knn classifier, predict the class label using the mean value for each feature.

Hint: You can use `cancerdf.mean()[:-1].values.reshape(1, -1)` which gets the mean value for each feature, ignores the target column, and reshapes the data from 1 dimension to 2 (necessary for the precict method of KNeighborsClassifier).

*This function should return a numpy array either `array([ 0.])` or `array([ 1.])`*


```python
def answer_six():
    cancerdf = answer_one()
    means = cancerdf.mean()[:-1].values.reshape(1, -1)
    classifier = answer_five()
    predictions = classifier.predict(means)
    
   
    
    return predictions

```

### Question 7
Using your knn classifier, predict the class labels for the test set `X_test`.

*This function should return a numpy array with shape `(143,)` and values either `0.0` or `1.0`.*


```python
def answer_seven():
    X_train, X_test, y_train, y_test = answer_four()
    knn = answer_five()
    predictions = knn.predict(X_test)
    # Your code here
    
    return predictions# Return your answer

```

### Question 8
Find the score (mean accuracy) of your knn classifier using `X_test` and `y_test`.

*This function should return a float between 0 and 1*


```python
def answer_eight():
    X_train, X_test, y_train, y_test = answer_four()
    knn = answer_five()
    
  
    
    return knn.score(X_test, y_test)
```

### Optional plot

Try using the plotting function below to visualize the differet predicition scores between training and test sets, as well as malignant and benign cells.


```python
def accuracy_plot():
    import matplotlib.pyplot as plt

    %matplotlib notebook

    X_train, X_test, y_train, y_test = answer_four()

    # Find the training and testing accuracies by target value (i.e. malignant, benign)
    mal_train_X = X_train[y_train==0]
    mal_train_y = y_train[y_train==0]
    ben_train_X = X_train[y_train==1]
    ben_train_y = y_train[y_train==1]

    mal_test_X = X_test[y_test==0]
    mal_test_y = y_test[y_test==0]
    ben_test_X = X_test[y_test==1]
    ben_test_y = y_test[y_test==1]

    knn = answer_five()

    scores = [knn.score(mal_train_X, mal_train_y), knn.score(ben_train_X, ben_train_y), 
              knn.score(mal_test_X, mal_test_y), knn.score(ben_test_X, ben_test_y)]


    plt.figure()

    # Plot the scores as a bar chart
    bars = plt.bar(np.arange(4), scores, color=['#4c72b0','#4c72b0','#55a868','#55a868'])

    # directly label the score onto the bars
    for bar in bars:
        height = bar.get_height()
        plt.gca().text(bar.get_x() + bar.get_width()/2, height*.90, '{0:.{1}f}'.format(height, 2), 
                     ha='center', color='w', fontsize=11)

    # remove all the ticks (both axes), and tick labels on the Y axis
    plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='on')

    # remove the frame of the chart
    for spine in plt.gca().spines.values():
        spine.set_visible(False)

    plt.xticks([0,1,2,3], ['Malignant\nTraining', 'Benign\nTraining', 'Malignant\nTest', 'Benign\nTest'], alpha=0.8);
    plt.title('Training and Test Accuracies for Malignant and Benign Cells', alpha=0.8)
```


```python
# Uncomment the plotting function to see the visualization, 
# Comment out the plotting function when submitting your notebook for grading

accuracy_plot() 
```


    <IPython.core.display.Javascript object>



<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAABQAAAAPACAYAAABq3NR5AAAAAXNSR0IArs4c6QAAQABJREFUeAHs3QecXFXB/vGT3nuDEJIQCKRAICSEJhBAOogggmBDLK9YEdRXrNjwtSE2UNT3jw3Fgq+AFKUjvVepISSBhDTSGyn/55nMnZxM7rTd2dnZ2d/J58k9t9/7vWXunJ3dCYGCAAIIIIAAAggggAACCCCAAAIIIIAAAggggAACCCCAAAIIIIAAAggggAACCCCAAAIIIIAAAggggAACCCCAAAIIIIAAAggggAACCCCAAAIIIIAAAggggAACCCCAAAIIIIAAAggggAACCCCAAAIIIIAAAggggAACCCCAAAIIIIAAAggggAACCCCAAAIIIIAAAggggAACCCCAAAIIIIAAAggggAACCCCAAAIIIIAAAggggAACCCCAAAIIIIAAAggggAACCCCAAAIIIIAAAggggAACCCCAAAIIIIAAAggggAACCCCAAAIIIIAAAggggAACCCCAAAIIIIAAAggggAACCCCAAAIIIIAAAggggAACCCCAAAIIIIAAAggggAACCCCAAAIIIIAAAggggAACCCCAAAIIIIAAAggggAACCCCAAAIIIIAAAggggAACCCCAAAIIIIAAAggggAACCCCAAAIIIIAAAggggAACCCCAAAIIIIAAAggggAACCCCAAAIIIIAAAggggAACCCCAAAIIIIAAAggggAACCCCAAAIIIIAAAggggAACCCCAAAIIIIAAAggggAACCCCAAAIIIIAAAggggAACCCCAAAIIIIAAAggggAACCCCAAAIIIIAAAggggAACCCCAAAIIIIAAAggggAACCCCAAAIIIIAAAggggAACCCCAAAIIIIAAAggggAACCCCAAAIIIIAAAggggAACCCCAAAIIIIAAAggggAACCCCAAAIIIIAAAggggAACCCCAAAIIIIAAAggggAACCCCAAAIIIIAAAggggAACCCCAAAIIIIAAAggggAACCCCAAAIIIIAAAggggAACCCCAAAIIIIAAAggggAACCCCAAAIIIIAAAggggAACCCCAAAIIIIAAAggggAACCCCAAAIIIIAAAggggAACCCCAAAIIIIAAAggggAACCCCAAAIIIIAAAggggAACCCCAAAIIIIAAAggggAACCCCAAAIIIIAAAggggAACCCCAAAIIIIAAAggggAACCCCAAAIIIIAAAggggAACCCCAAAIIIIAAAggggAACCCCAAAIIIIAAAggggAACCCCAAAIIIIAAAggggAACCCCAAAIIIIAAAggggAACCCCAAAIIIIAAAggggAACCCCAAAIIIIAAAggggAACCCCAAAIIIIAAAggggAACCCCAAAIIIIAAAggggAACCCCAAAIIIIAAAggggAACCCCAAAIIIIAAAggggAACCCCAAAIIIIAAAggggAACCCCAAAIIIIAAAggggAACCCCAAAIIIIAAAggggAACCCCAAAIIIIAAAggggAACCCCAAAIIIIAAAggggAACCCCAAAIIIIAAAggggAACCCCAAAIIIIAAAggggAACCCCAAAIIIIAAAggggAACCCCAAAIIIIAAAggggAACCCCAAAIIIIAAAggggAACCCCAAAIIIIAAAggggAACCCCAAAIIIIAAAggggAACCCCAAAIIIIAAAggggAACCCCAAAIIIIAAAggggAACCCCAAAIIIIAAAggggAACCCCAAAIIIIAAAggggAACCCCAAAIIIIAAAggggAACCCCAAAIIIIAAAggggAACCCCAAAIIIIAAAggggAACCCCAAAIIIIAAAggggAACCCCAAAIIIIAAAggggAACCCCAAAIIIIAAAggggAACCCCAAAIIIIAAAggggAACCCCAAAIIIIAAAggggAACCCCAAAIIIIAAAggggAACCCCAAAIIIIAAAggggAACCCCAAAIIIIAAAggggAACCCCAAAIIIIAAAggggAACCCCAAAIIIIAAAggggAACCCCAAAIIIIAAAggggAACCCCAAAIIIIAAAggggAACCCCAAAIIIIAAAggggAACCCCAAAIIIIAAAggggAACCCCAAAIIIIAAAggggAACCCCAAAIIIIAAAggggAACCCCAAAIIIIAAAggggAACCCCAAAIIIIAAAggggAACCCCAAAIIIIAAAggggAACCCCAAAIIIIAAAggggAACCCCAAAIIIIAAAggggAACCCCAAAIIIIAAAggggAACCCCAAAIIIIAAAggggAACCCCAAAIIIIAAAggggAACCCCAAAIIIIAAAggggAACCCCAAAIIIIAAAggggAACCCCAAAIIIIAAAggggAACCCCAAAIIIIAAAggggAACCCCAAAIIIIAAAggggAAC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BHjldh5JgAABAgQIECBAYE0Fhn9Y44abblhy29cP1skr9VYzvnDRl0d/9fE3jM6+9Lypm9l1l11HT7rfsaOnHv7E0W677DZ1naVmHnefY0Y777TtR5AMHa++4ZqlHmI5AQIECBAgsAYCrgBcA0SbIECAAAECBAgQILAjC9x59z1HLzjm50cPPvDw0VXXfWd04iffNDr96/kR4EtGu++6++jQvQ8e/VAEf4846CGjp8Tt/fc7dPS/PvB/RlcNrkBcav93Gu00Ou4+R8+v9r6vfnh+2gQBAgQIECBQK+AKwFpfWydAgAABAgQIECCwbIFrb7xuft3lXGW3++BKvPzdfCsZuY0XPfHX58O/F773paN3nvm+Uf5+wZtuuXnud/597sIvjV7ywVeO3v3lk+da3G/f+4ye84hnbFe7hxz43aP999x37jHfit9vmNs0CBAgQIAAgfUREACuj7MuBAgQIECAAAECBJYU+M7gI7F3iavylhp55d5kDB87mbec2yfd99jRPe9y97lV3/al944uuOqimQ/7uzP+X1z1d/Xc8mPudeTorne8y8x1Fy54/KHHzM866aunzk+bIECAAAECBOoFBID1xjoQIECAAAECBAgQWJbAN6781vx6B+y13/z0rIkD9rx1neFjZ60/bX5+rHcyzrjg85PJqbfX3XT96MyLz5pbtvPOO4/uu8/BU9dbOHPP3fcYPeoeD52bffPNN49O9vHfhUTuEyBAgACBUgEBYCmvjRMgQIAAAQIECBBYvsB5l319fuXDlgjX7nqHO4/223OfufUvu/aK0ZXXXTX/2O2Z2PtOd51ffTl/lGN4peHwj5bMb2TKxGMPPmo0+bjypyJkvOSay6asZRYBAgQIECBQJSAArJK1XQIECBAgQIAAAQLbKZDh2GQ87G4PmkxOvX34QQ+en//Jb352fnp7J6694dbfO7jvHtsCxcW2sf9gnSvjD4YsZzw+/vrvZPj470TCLQECBAgQWD8BAeD6WetEgAABAgQIECBAYFGBz1905ujSay6fWyf/Iu999r7X1PV32mmn0Q/e7/Hzy0497/T56e2dOO/yW686fOzBj1r04Qfutf/ovvEHQHLkR3nPuvTcRdfPhbkPk/24/Nor468Ln7HkY6xAgAABAgQIrK2AAHBtPW2NAAECBAgQIECAwIoFbrnlltGbP/fO+cc//3uePbpLfNR34fjpI542H6p98aKvjM644AsLV5m7f+whR4/e+IxXzdXvPf4FU9c55dzT5ucfd59Hj4ZX680viIn8gx8vOOa5o1133mVu9se/+ZnRd8Z/EGS43sLp4fY+cO5H5/6y8MJ13CdAgAABAgRqBXat3bytEyBAgAABAgQIECCwPQL/fvYpo6Pu+dDRQ+/2wNG973rQ6A+f9N9HOe/8yy8Y7XWHPUaPufejRg/Y/75zm8y/yPvq0/9+ezZ/u3U//a0vjD78tY+PHh1/1XfnnXYe/eejnjl63CHfM3el3revvmy0+667jQ7b++DRY2PeXvHHPHJcEb9v8G8/9ZbbbWvhjN123nX0H+L5Tsb7zv7QZNItAQIECBAgsI4CAsB1xNaKAAECBAgQIECAwFICN99y8+iPP/Tq0X89+jmjI+9xxCj/SMfTH/SU2z3s4qsvGb381NeOzr/im7dbtr0zXvGRE0fX3HDt6AmHPmbuoQ864P6jrGnj61dcMHr5h187+tZVF01bfJt5R93zYRFa7jk378yLzx7lYw0CBAgQIEBg/QUEgOtvriMBAgQIECBAgACBRQWuvfG60UtPedXokQcdMcqP8eZfBL7LHe88ujZCugu+c/HoY+d/cvRvZ31wLrRbdEPLXHjjzTeO/uK014/e9eWTR/kx4MP3O3R04J77je60251GueyK+N19Z1963ui0+P19p8bVgjfdfNOytjz8+K8//rEsMisRIECAAIESAQFgCauNEiBAgAABAgQIEFi9wOnf+PQoa6Xj/ed8ZJS13HHuZeeP/uaTb1ru6kuu9+L3/9mS61iBAAECBAgQqBfwR0DqjXUgQIAAAQIECBAgQIAAAQIECBAg0CYgAGyj15gAAQIECBAgQIAAAQIECBAgQIBAvYAAsN5YBwIECBAgQIAAAQIECBAgQIAAAQJtAgLANnqNCRAgQIAAAQIECBAgQIAAAQIECNQLCADrjXUgQIAAAQIECBAgQIAAAQIECBAg0CYgAGyj15gAAQIECBAgQIAAAQIECBAgQIBAvYAAsN5YBwIECBAgQIAAAQIECBAgQIAAAQJtAgLANnqNCRAgQIAAAQIECBAgQIAAAQIECNQLCADrjXUgQIAAAQIECBAgQIAAAQIECBAg0CYgAGyj15gAAQIECBAgQIAAAQIECBAgQIBAvYAAsN5YBwIECBAgQIAAAQIECBAgQIAAAQJtAgLANnqNCRAgQIAAAQIECBAgQIAAAQIECNQLCADrjXUgQIAAAQIECBAgQIAAAQIECBAg0CYgAGyj15gAAQIECBAgQIAAAQIECBAgQIBAvYAAsN5YBwIECBAgQIAAAQIECBAgQIAAAQJtAgLANnqNCRAgQIAAAQIECBAgQIAAAQIECNQLCADrjXUgQIAAAQIECBAgQIAAAQIECBAg0CYgAGyj15gAAQIECBAgQIAAAQIECBAgQIBAvYAAsN5YBwIECBAgQIAAAQIECBAgQIAAAQJtAgLANnqNCRAgQIAAAQIECBAgQIAAAQIECNQLCADrjXUgQIAAAQIECBAgQIAAAQIECBAg0CYgAGyj15gAAQIECBAgQIAAAQIECBAgQIBAvYAAsN5YBwIECBAgQIAAAQIECBAgQIAAAQJtAgLANnqNCRAgQIAAAQIECBAgQIAAAQIECNQLCADrjXUgQIAAAQIECBAgQIAAAQIECBAg0CYgAGyj15gAAQIECBAgQIAAAQIECBAgQIBAvYAAsN5YBwIECBAgQIAAAQIECBAgQIAAAQJtAgLANnqNCRAgQIAAAQIECBAgQIAAAQIECNQLCADrjXUgQIAAAQIECBAgQIAAAQIECBAg0CYgAGyj15gAAQIECBAgQIAAAQIECBAgQIBAvYAAsN5YBwIECBAgQIAAAQIECBAgQIAAAQJtAgLANnqNCRAgQIAAAQIECBAgQIAAAQIECNQLCADrjXUgQIAAAQIECBAgQIAAAQIECBAg0CYgAGyj15gAAQIECBAgQIAAAQIECBAgQIBAvYAAsN5YBwIECBAgQIAAAQIECBAgQIAAAQJtAgLANnqNCRAgQIAAAQIECBAgQIAAAQIECNQLCADrjXUgQIAAAQIECBAgQIAAAQIECBAg0CYgAGyj15gAAQIECBAgQIAAAQIECBAgQIBAvYAAsN5YBwIECBAgQIAAAQIECBAgQIAAAQJtAgLANnqNCRAgQIAAAQIECBAgQIAAAQIECNQLCADrjXUgQIAAAQIECBAgQIAAAQIECBAg0CYgAGyj15gAAQIECBAgQIAAAQIECBAgQIBAvYAAsN5YBwIECBAgQIAAAQIECBAgQIAAAQJtAgLANnqNCRAgQIAAAQIECBAgQIAAAQIECNQLCADrjXUgQIAAAQIECBAgQIAAAQIECBAg0CYgAGyj15gAAQIECBAgQIAAAQIECBAgQIBAvYAAsN5YBwIECBAgQIAAAQIECBAgQIAAAQJtAgLANnqNCRAgQIAAAQIECBAgQIAAAQIECNQLCADrjXUgQIAAAQIECBAgQIAAAQIECBAg0CYgAGyj15gAAQIECBAgQIAAAQIECBAgQIBAvYAAsN5YBwIECBAgQIAAAQIECBAgQIAAAQJtAgLANnqNCRAgQIAAAQIECBAgQIAAAQIECNQLCADrjXUgQIAAAQIECBAgQIAAAQIECBAg0CYgAGyj15gAAQIECBAgQIAAAQIECBAgQIBAvYAAsN5YBwIECBAgQIAAAQIECBAgQIAAAQJtAgLANnqNCRAgQIAAAQIECBAgQIAAAQIECNQLCADrjXUgQIAAAQIECBAgQIAAAQIECBAg0CYgAGyj15gAAQIECBAgQIAAAQIECBAgQIBAvYAAsN5YBwIECBAgQIAAAQIECBAgQIAAAQJtAgLANnqNCRAgQIAAAQIECBAgQIAAAQIECNQLCADrjXUgQIAAAQIECBAgQIAAAQIECBAg0CYgAGyj15gAAQIECBAgQIAAAQIECBAgQIBAvYAAsN5YBwIECBAgQIAAAQIECBAgQIAAAQJtAgLANnqNCRAgQIAAAQIECBAgQIAAAQIECNQLCADrjXUgQIAAAQIECBAgQIAAAQIECBAg0CYgAGyj15gAAQIECBAgQIAAAQIECBAgQIBAvYAAsN5YBwIECBAgQIAAAQIECBAgQIAAAQJtAgLANnqNCRAgQIAAAQIECBAgQIAAAQIECNQLCADrjXUgQIAAAQIECBAgQIAAAQIECBAg0CYgAGyj15gAAQIECBAgQIAAAQIECBAgQIBAvYAAsN5YBwIECBAgQIAAAQIECBAgQIAAAQJtAgLANnqNCRAgQIAAAQIECBAgQIAAAQIECNQLCADrjXUgQIAAAQIECBAgQIAAAQIECBAg0CYgAGyj15gAAQIECBAgQIAAAQIECBAgQIBAvYAAsN5YBwIECBAgQIAAAQIECBAgQIAAAQJtAgLANnqNCRAgQIAAAQIECBAgQIAAAQIECNQLCADrjXUgQIAAAQIECBAgQIAAAQIECBAg0CYgAGyj15gAAQIECBAgQIAAAQIECBAgQIBAvYAAsN5YBwIECBAgQIAAAQIECBAgQIAAAQJtAgLANnqNCRAgQIAAAQIECBAgQIAAAQIECNQLCADrjXUgQIAAAQIECBAgQIAAAQIECBAg0CYgAGyj15gAAQIECBAgQIAAAQIECBAgQIBAvYAAsN5YBwIECBAgQIAAAQIECBAgQIAAAQJtAgLANnqNCRAgQIAAAQIECBAgQIAAAQIECNQLCADrjXUgQIAAAQIECBAgQIAAAQIECBAg0CYgAGyj15gAAQIECBAgQIAAAQIECBAgQIBAvYAAsN5YBwIECBAgQIAAAQIECBAgQIAAAQJtAgLANnqNCRAgQIAAAQIECBAgQIAAAQIECNQLCADrjXUgQIAAAQIECBAgQIAAAQIECBAg0CYgAGyj15gAAQIECBAgQIAAAQIECBAgQIBAvYAAsN5YBwIECBAgQIAAAQIECBAgQIAAAQJtAgLANnqNCRAgQIAAAQIECBAgQIAAAQIECNQLCADrjXUgQIAAAQIECBAgQIAAAQIECBAg0CYgAGyj15gAAQIECBAgQIAAAQIECBAgQIBAvYAAsN5YBwIECBAgQIAAAQIECBAgQIAAAQJtAgLANnqNCRAgQIAAAQIECBAgQIAAAQIECNQLCADrjXUgQIAAAQIECBAgQIAAAQIECBAg0CYgAGyj15gAAQIECBAgQIAAAQIECBAgQIBAvYAAsN5YBwIECBAgQIAAAQIECBAgQIAAAQJtAgLANnqNCRAgQIAAAQIECBAgQIAAAQIECNQLCADrjXUgQIAAAQIECBAgQIAAAQIECBAg0CYgAGyj15gAAQIECBAgQIAAAQIECBAgQIBAvYAAsN5YBwIECBAgQIAAAQIECBAgQIAAAQJtAgLANnqNCRAgQIAAAQIECBAgQIAAAQIECNQLCADrjXUgQIAAAQIECBAgQIAAAQIECBAg0CYgAGyj15gAAQIECBAgQIAAAQIECBAgQIBAvYAAsN5YBwIECBAgQIAAAQIECBAgQIAAAQJtAgLANnqNCRAgQIAAAQIECBAgQIAAAQIECNQLCADrjXUgQIAAAQIECBAgQIAAAQIECBAg0CYgAGyj15gAAQIECBAgQIAAAQIECBAgQIBAvYAAsN5YBwIECBAgQIAAAQIECBAgQIAAAQJtAgLANnqNCRAgQIAAAQIECBAgQIAAAQIECNQLCADrjXUgQIAAAQIECBAgQIAAAQIECBAg0CYgAGyj15gAAQIECBAgQIAAAQIECBAgQIBAvYAAsN5YBwIECBAgQIAAAQIECBAgQIAAAQJtAgLANnqNCRAgQIAAAQIECBAgQIAAAQIECNQLCADrjXUgQIAAAQIECBAgQIAAAQIECBAg0CYgAGyj15gAAQIECBAgQIAAAQIECBAgQIBAvYAAsN5YBwIECBAgQIAAAQIECBAgQIAAAQJtAgLANnqNCRAgQIAAAQIECBAgQIAAAQIECNQLCADrjXUgQIAAAQIECBAgQIAAAQIECBAg0CYgAGyj15gAAQIECBAgQIAAAQIECBAgQIBAvYAAsN5YBwIECBAgQIAAAQIECBAgQIAAAQJtAgLANnqNCRAgQIAAAQIECBAgQIAAAQIECNQLCADrjXUgQIAAAQIECBAgQIAAAQIECBAg0CYgAGyj15gAAQIECBAgQIAAAQIECBAgQIBAvYAAsN5YBwIECBAgQIAAAQIECBAgQIAAAQJtAgLANnqNCRAgQIAAAQIECBAgQIAAAQIECNQLCADrjXUgQIAAAQIECBAgQIAAAQIECBAg0CYgAGyj15gAAQIECBAgQIAAAQIECBAgQIBAvYAAsN5YBwIECBAgQIAAAQIECBAgQIAAAQJtAgLANnqNCRAgQIAAAQIECBAgQIAAAQIECNQLCADrjXUgQIAAAQIECBAgQIAAAQIECBAg0CYgAGyj15gAAQIECBAgQIAAAQIECBAgQIBAvYAAsN5YBwIECBAgQIAAAQIECBAgQIAAAQJtAgLANnqNCRAgQIAAAQIECBAgQIAAAQIECNQLCADrjXUgQIAAAQIECBAgQIAAAQIECBAg0CYgAGyj15gAAQIECBAgQIAAAQIECBAgQIBAvYAAsN5YBwIECBAgQIAAAQIECBAgQIAAAQJtAgLANnqNCRAgQIAAAQIECBAgQIAAAQIECNQLCADrjXUgQIAAAQIECBAgQIAAAQIECBAg0CYgAGyj15gAAQIECBAgQIAAAQIECBAgQIBAvYAAsN5YBwIECBAgQIAAAQIECBAgQIAAAQJtAgLANnqNCRAgQIAAAQIECBAgQIAAAQIECNQLCADrjXUgQIAAAQIECBAgQIAAAQIECBAg0CYgAGyj15gAAQIECBAgQIAAAQIECBAgQIBAvYAAsN5YBwIECBAgQIAAAQIECBAgQIAAAQJtAgLANnqNCRAgQIAAAQIECBAgQIAAAQIECNQLCADrjXUgQIAAAQIECBAgQIAAAQIECBAg0CYgAGyj15gAAQIECBAgQIAAAQIECBAgQIBAvYAAsN5YBwIECBAgQIAAAQIECBAgQIAAAQJtAgLANnqNCRAgQIAAAQIECBAgQIAAAQIECNQLCADrjXUgQIAAAQIECBAgQIAAAQIECBAg0CYgAGyj15gAAQIECBAgQIAAAQIECBAgQIBAvYAAsN5YBwIECBAgQIAAAQIECBAgQIAAAQJtAgLANnqNCRAgQIAAAQIECBAgQIAAAQIECNQLCADrjXUgQIAAAQIECBAgQIAAAQIECBAg0CYgAGyj15gAAQIECBAgQIAAAQIECBAgQIBAvYAAsN5YBwIECBAgQIAAAQIECBAgQIAAAQJtAgLANnqNCRAgQIAAAQIECBAgQIAAAQIECNQLCADrjXUgQIAAAQIECBAgQIAAAQIECBAg0CYgAGyj15gAAQIECBAgQIAAAQIECBAgQIBAvYAAsN5YBwIECBAgQIAAAQIECBAgQIAAAQJtAgLANnqNCRAgQIAAAQIECBAgQIAAAQIECNQLCADrjXUgQIAAAQIECBAgQIAAAQIECBAg0CYgAGyj15gAAQIECBAgQIAAAQIECBAgQIBAvYAAsN5YBwIECBAgQIAAAQIECBAgQIAAAQJtAgLANnqNCRAgQIAAAQIECBAgQIAAAQIECNQLCADrjXUgQIAAAQIECBAgQIAAAQIECBAg0CYgAGyj15gAAQIECBAgQIAAAQIECBAgQIBAvYAAsN5YBwIECBAgQIAAAQIECBAgQIAAAQJtAgLANnqNCRAgQIAAAQIECBAgQIAAAQIECNQLCADrjXUgQIAAAQIECBAgQIAAAQIECBAg0CYgAGyj15gAAQIECBAgQIAAAQIECBAgQIBAvYAAsN5YBwIECBAgQIAAAQIECBAgQIAAAQJtAgLANnqNCRAgQIAAAQIECBAgQIAAAQIECNQLCADrjXUgQIAAAQIECBAgQIAAAQIECBAg0CYgAGyj15gAAQIECBAgQIAAAQIECBAgQIBAvYAAsN5YBwIECBAgQIAAAQIECBAgQIAAAQJtAgLANnqNCRAgQIAAAQIECBAgQIAAAQIECNQLCADrjXUgQIAAAQIECBAgQIAAAQIECBAg0CYgAGyj15gAAQIECBAgQIAAAQIECBAgQIBAvYAAsN5YBwIECBAgQIAAAQIECBAgQIAAAQJtAgLANnqNCRAgQIAAAQIECBAgQIAAAQIECNQLCADrjXUgQIAAAQIECBAgQIAAAQIECBAg0CYgAGyj15gAAQIECBAgQIAAAQIECBAgQIBAvYAAsN5YBwIECBAgQIAAAQIECBAgQIAAAQJtAgLANnqNCRAgQIAAAQIECBAgQIAAAQIECNQLCADrjXUgQIAAAQIECBAgQIAAAQIECBAg0CYgAGyj15gAAQIECBAgQIAAAQIECBAgQIBAvYAAsN5YBwIECBAgQIAAAQIECBAgQIAAAQJtAgLANnqNCRAgQIAAAQIECBAgQIAAAQIECNQLCADrjXUgQIAAAQIECBAgQIAAAQIECBAg0CYgAGyj15gAAQIECBAgQIAAAQIECBAgQIBAvYAAsN5YBwIECBAgQIAAAQIECBAgQIAAAQJtAgLANnqNCRAgQIAAAQIECBAgQIAAAQIECNQLCADrjXUgQIAAAQIECBAgQIAAAQIECBAg0CYgAGyj15gAAQIECBAgQIAAAQIECBAgQIBAvYAAsN5YBwIECBAgQIAAAQIECBAgQIAAAQJtAgLANnqNCRAgQIAAAQIECBAgQIAAAQIECNQLCADrjXUgQIAAAQIECBAgQIAAAQIECBAg0CYgAGyj15gAAQIECBAgQIAAAQIECBAgQIBAvYAAsN5YBwIECBAgQIAAAQIECBAgQIAAAQJtAgLANnqNCRAgQIAAAQIECBAgQIAAAQIECNQLCADrjXUgQIAAAQIECBAgQIAAAQIECBAg0CYgAGyj15gAAQIECBAgQIAAAQIECBAgQIBAvYAAsN5YBwIECBAgQIAAAQIECBAgQIAAAQJtAgLANnqNCRAgQIAAAQIECBAgQIAAAQIECNQLCADrjXUgQIAAAQIECBAgQIAAAQIECBAg0CYgAGyj15gAAQIECBAgQIAAAQIECBAgQIBAvYAAsN5YBwIECBAgQIAAAQIECBAgQIAAAQJtAgLANnqNCRAgQIAAAQIECBAgQIAAAQIECNQLCADrjXUgQIAAAQIECBAgQIAAAQIECBAg0CYgAGyj15gAAQIECBAgQIAAAQIECBAgQIBAvYAAsN5YBwIECBAgQIAAAQIECBAgQIAAAQJtAgLANnqNCRAgQIAAAQIECBAgQIAAAQIECNQLCADrjXUgQIAAAQIECBAgQIAAAQIECBAg0CYgAGyj15gAAQIECBAgQIAAAQIECBAgQIBAvYAAsN5YBwIECBAgQIAAAQIECBAgQIAAAQJtAgLANnqNCRAgQIAAAQIECBAgQIAAAQIECNQLCADrjXUgQIAAAQIECBAgQIAAAQIECBAg0CYgAGyj15gAAQIECBAgQIAAAQIECBAgQIBAvYAAsN5YBwIECBAgQIAAAQIECBAgQIAAAQJtAgLANnqNCRAgQIAAAQIECBAgQIAAAQIECNQLCADrjXUgQIAAAQIECBAgQIAAAQIECBAg0CYgAGyj15gAAQIECBAgQIAAAQIECBAgQIBAvYAAsN5YBwIECBAgQIAAAQIECBAgQIAAAQJtAgLANnqNCRAgQIAAAQIECBAgQIAAAQIECNQLCADrjXUgQIAAAQIECBAgQIAAAQIECBAg0CYgAGyj15gAAQIECBAgQIAAAQIECBAgQIBAvYAAsN5YBwIECBAgQIAAAQIECBAgQIAAAQJtAgLANnqNCRAgQIAAAQIECBAgQIAAAQIECNQLCADrjXUgQIAAAQIECBAgQIDym9JrAABAAElEQVQAAQIECBAg0CYgAGyj15gAAQIECBAgQIAAAQIECBAgQIBAvYAAsN5YBwIECBAgQIAAAQIECBAgQIAAAQJtAgLANnqNCRAgQIAAAQIECBAgQIAAAQIECNQLCADrjXUgQIAAAQIECBAgQIAAAQIECBAg0CYgAGyj15gAAQIECBAgQIAAAQIECBAgQIBAvYAAsN5YBwIECBAgQIAAAQIECBAgQIAAAQJtAgLANnqNCRAgQIAAAQIECBAgQIAAAQIECNQLCADrjXUgQIAAAQIECBAgQIAAAQIECBAg0CYgAGyj15gAAQIECBAgQIAAAQIECBAgQIBAvYAAsN5YBwIECBAgQIAAAQIECBAgQIAAAQJtAgLANnqNCRAgQIAAAQIECBAgQIAAAQIECNQLCADrjXUgQIAAAQIECBAgQIAAAQIECBAg0CYgAGyj15gAAQIECBAgQIAAAQIECBAgQIBAvYAAsN5YBwIECBAgQIAAAQIECBAgQIAAAQJtAgLANnqNCRAgQIAAAQIECBAgQIAAAQIECNQLCADrjXUgQIAAAQIECBAgQIAAAQIECBAg0CYgAGyj15gAAQIECBAgQIAAAQIECBAgQIBAvYAAsN5YBwIECBAgQIAAAQIECBAgQIAAAQJtAgLANnqNCRAgQIAAAQIECBAgQIAAAQIECNQLCADrjXUgQIAAAQIECBAgQIAAAQIECBAg0CYgAGyj15gAAQIECBAgQIAAAQIECBAgQIBAvYAAsN5YBwIECBAgQIAAAQIECBAgQIAAAQJtAgLANnqNCRAgQIAAAQIECBAgQIAAAQIECNQLCADrjXUgQIAAAQIECBAgQIAAAQIECBAg0CYgAGyj15gAAQIECBAgQIAAAQIECBAgQIBAvYAAsN5YBwIECBAgQIAAAQIECBAgQIAAAQJtAgLANnqNCRAgQIAAAQIECBAgQIAAAQIECNQLCADrjXUgQIAAAQIECBAgQIAAAQIECBAg0CYgAGyj15gAAQIECBAgQIAAAQIECBAgQIBAvYAAsN5YBwIECBAgQIAAAQIECBAgQIAAAQJtAgLANnqNCRAgQIAAAQIECBAgQIAAAQIECNQLCADrjXUgQIAAAQIECBAgQIAAAQIECBAg0CYgAGyj15gAAQIECBAgQIAAAQIECBAgQIBAvYAAsN5YBwIECBAgQIAAAQIECBAgQIAAAQJtAgLANnqNCRAgQIAAAQIECBAgQIAAAQIECNQLCADrjXUgQIAAAQIECBAgQIAAAQIECBAg0CYgAGyj15gAAQIECBAgQIAAAQIECBAgQIBAvYAAsN5YBwIECBAgQIAAAQIECBAgQIAAAQJtAgLANnqNCRAgQIAAAQIECBAgQIAAAQIECNQLCADrjXUgQIAAAQIECBAgQIAAAQIECBAg0CYgAGyj15gAAQIECBAgQIAAAQIECBAgQIBAvYAAsN5YBwIECBAgQIAAAQIECBAgQIAAAQJtAgLANnqNCRAgQIAAAQIECBAgQIAAAQIECNQLCADrjXUgQIAAAQIECBAgQIAAAQIECBAg0CYgAGyj15gAAQIECBAgQIAAAQIECBAgQIBAvYAAsN5YBwIECBAgQIAAAQIECBAgQIAAAQJtAgLANnqNCRAgQIAAAQIECBAgQIAAAQIECNQLCADrjXUgQIAAAQIECBAgQIAAAQIECBAg0CYgAGyj15gAAQIECBAgQIAAAQIECBAgQIBAvYAAsN5YBwIECBAgQIAAAQIECBAgQIAAAQJtAgLANnqNCRAgQIAAAQIECBAgQIAAAQIECNQLCADrjXUgQIAAAQIECBAgQIAAAQIECBAg0CYgAGyj15gAAQIECBAgQIAAAQIECBAgQIBAvYAAsN5YBwIECBAgQIAAAQIECBAgQIAAAQJtAgLANnqNCRAgQIAAAQIECBAgQIAAAQIECNQLCADrjXUgQIAAAQIECBAgQIAAAQIECBAg0CYgAGyj15gAAQIECBAgQIAAAQIECBAgQIBAvYAAsN5YBwIECBAgQIAAAQIECBAgQIAAAQJtAgLANnqNCRAgQIAAAQIECBAgQIAAAQIECNQLCADrjXUgQIAAAQIECBAgQIAAAQIECBAg0CYgAGyj15gAAQIECBAgQIAAAQIECBAgQIBAvYAAsN5YBwIECBAgQIAAAQIECBAgQIAAAQJtAgLANnqNCRAgQIAAAQIECBAgQIAAAQIECNQLCADrjXUgQIAAAQIECBAgQIAAAQIECBAg0CYgAGyj15gAAQIECBAgQIAAAQIECBAgQIBAvYAAsN5YBwIECBAgQIAAAQIECBAgQIAAAQJtAgLANnqNCRAgQIAAAQIECBAgQIAAAQIECNQLCADrjXUgQIAAAQIECBAgQIAAAQIECBAg0CYgAGyj15gAAQIECBAgQIAAAQIECBAgQIBAvYAAsN5YBwIECBAgQIAAAQIECBAgQIAAAQJtAgLANnqNCRAgQIAAAQIECBAgQIAAAQIECNQLCADrjXUgQIAAAQIECBAgQIAAAQIECBAg0CYgAGyj15gAAQIECBAgQIAAAQIECBAgQIBAvYAAsN5YBwIECBAgQIAAAQIECBAgQIAAAQJtAgLANnqNCRAgQIAAAQIECBAgQIAAAQIECNQLCADrjXUgQIAAAQIECBAgQIAAAQIECBAg0CYgAGyj15gAAQIECBAgQIAAAQIECBAgQIBAvYAAsN5YBwIECBAgQIAAAQIECBAgQIAAAQJtAgLANnqNCRAgQIAAAQIECBAgQIAAAQIECNQLCADrjXUgQIAAAQIECBAgQIAAAQIECBAg0CYgAGyj15gAAQIECBAgQIAAAQIECBAgQIBAvYAAsN5YBwIECBAgQIAAAQIECBAgQIAAAQJtAgLANnqNCRAgQIAAAQIECBAgQIAAAQIECNQLCADrjXUgQIAAAQIECBAgQIAAAQIECBAg0CYgAGyj15gAAQIECBAgQIAAAQIECBAgQIBAvYAAsN5YBwIECBAgQIAAAQIECBAgQIAAAQJtAgLANnqNCRAgQIAAAQIECBAgQIAAAQIECNQLCADrjXUgQIAAAQIECBAgQIAAAQIECBAg0CYgAGyj15gAAQIECBAgQIAAAQIECBAgQIBAvYAAsN5YBwIECBAgQIAAAQIECBAgQIAAAQJtAgLANnqNCRAgQIAAAQIECBAgQIAAAQIECNQLCADrjXUgQIAAAQIECBAgQIAAAQIECBAg0CYgAGyj15gAAQIECBAgQIAAAQIECBAgQIBAvYAAsN5YBwIECBAgQIAAAQIECBAgQIAAAQJtAgLANnqNCRAgQIAAAQIECBAgQIAAAQIECNQLCADrjXUgQIAAAQIECBAgQIAAAQIECBAg0CYgAGyj15gAAQIECBAgQIAAAQIECBAgQIBAvYAAsN5YBwIECBAgQIAAAQIECBAgQIAAAQJtAgLANnqNCRAgQIAAAQIECBAgQIAAAQIECNQLCADrjXUgQIAAAQIECBAgQIAAAQIECBAg0CYgAGyj15gAAQIECBAgQIAAAQIECBAgQIBAvYAAsN5YBwIECBAgQIAAAQIECBAgQIAAAQJtAgLANnqNCRAgQIAAAQIECBAgQIAAAQIECNQLCADrjXUgQIAAAQIECBAgQIAAAQIECBAg0CYgAGyj15gAAQIECBAgQIAAAQIECBAgQIBAvYAAsN5YBwIECBAgQIAAAQIECBAgQIAAAQJtAgLANnqNCRAgQIAAAQIECBAgQIAAAQIECNQLCADrjXUgQIAAAQIECBAgQIAAAQIECBAg0CYgAGyj15gAAQIECBAgQIAAAQIECBAgQIBAvYAAsN5YBwIECBAgQIAAAQIECBAgQIAAAQJtAgLANnqNCRAgQIAAAQIECBAgQIAAAQIECNQLCADrjXUgQIAAAQIECBAgQIAAAQIECBAg0CYgAGyj15gAAQIECBAgQIAAAQIECBAgQIBAvYAAsN5YBwIECBAgQIAAAQIECBAgQIAAAQJtAgLANnqNCRAgQIAAAQIECBAgQIAAAQIECNQLCADrjXUgQIAAAQIECBAgQIAAAQIECBAg0CYgAGyj15gAAQIECBAgQIAAAQIECBAgQIBAvYAAsN5YBwIECBAgQIAAAQIECBAgQIAAAQJtAgLANnqNCRAgQIAAAQIECBAgQIAAAQIECNQLCADrjXUgQIAAAQIECBAgQIAAAQIECBAg0CYgAGyj15gAAQIECBAgQIAAAQIECBAgQIBAvYAAsN5YBwIECBAgQIAAAQIECBAgQIAAAQJtAgLANnqNCRAgQIAAAQIECBAgQIAAAQIECNQLCADrjXUgQIAAAQIECBAgQIAAAQIECBAg0CYgAGyj15gAAQIECBAgQIAAAQIECBAgQIBAvYAAsN5YBwIECBAgQIAAAQIECBAgQIAAAQJtAgLANnqNCRAgQIAAAQIECBAgQIAAAQIECNQLCADrjXUgQIAAAQIECBAgQIAAAQIECBAg0CYgAGyj15gAAQIECBAgQIAAAQIECBAgQIBAvYAAsN5YBwIECBAgQIAAAQIECBAgQIAAAQJtAgLANnqNCRAgQIAAAQIECBAgQIAAAQIECNQLCADrjXUgQIAAAQIECBAgQIAAAQIECBAg0CYgAGyj15gAAQIECBAgQIAAAQIECBAgQIBAvYAAsN5YBwIECBAgQIAAAQIECBAgQIAAAQJtAgLANnqNCRAgQIAAAQIECBAgQIAAAQIECNQLCADrjXUgQIAAAQIECBAgQIAAAQIECBAg0CYgAGyj15gAAQIECBAgQIAAAQIECBAgQIBAvYAAsN5YBwIECBAgQIAAAQIECBAgQIAAAQJtAgLANnqNCRAgQIAAAQIECBAgQIAAAQIECNQLCADrjXUgQIAAAQIECBAgQIAAAQIECBAg0CYgAGyj15gAAQIECBAgQIAAAQIECBAgQIBAvYAAsN5YBwIECBAgQIAAAQIECBAgQIAAAQJtAgLANnqNCRAgQIAAAQIECBAgQIAAAQIECNQLCADrjXUgQIAAAQIECBAgQIAAAQIECBAg0CYgAGyj15gAAQIECBAgQIAAAQIECBAgQIBAvYAAsN5YBwIECBAgQIAAAQIECBAgQIAAAQJtAgLANnqNCRAgQIAAAQIECBAgQIAAAQIECNQLCADrjXUgQIAAAQIECBAgQIAAAQIECBAg0CYgAGyj15gAAQIECBAgQIAAAQIECBAgQIBAvYAAsN5YBwIECBAgQIAAAQIECBAgQIAAAQJtAgLANnqNCRAgQIAAAQIECBAgQIAAAQIECNQLCADrjXUgQIAAAQIECBAgQIAAAQIECBAg0CYgAGyj15gAAQIECBAgQIAAAQIECBAgQIBAvYAAsN5YBwIECBAgQIAAAQIECBAgQIAAAQJtAgLANnqNCRAgQIAAAQIECBAgQIAAAQIECNQLCADrjXUgQIAAAQIECBAgQIAAAQIECBAg0CYgAGyj15gAAQIECBAgQIAAAQIECBAgQIBAvYAAsN5YBwIECBAgQIAAAQIECBAgQIAAAQJtAgLANnqNCRAgQIAAAQIECBAgQIAAAQIECNQLCADrjXUgQIAAAQIECBAgQIAAAQIECBAg0CYgAGyj15gAAQIECBAgQIAAAQIECBAgQIBAvYAAsN5YBwIECBAgQIAAAQIECBAgQIAAAQJtAgLANnqNCRAgQIAAAQIECBAgQIAAAQIECNQLCADrjXUgQIAAAQIECBAgQIAAAQIECBAg0CYgAGyj15gAAQIECBAgQIAAAQIECBAgQIBAvYAAsN5YBwIECBAgQIAAAQIECBAgQIAAAQJtAgLANnqNCRAgQIAAAQIECBAgQIAAAQIECNQLCADrjXUgQIAAAQIECBAgQIAAAQIECBAg0CYgAGyj15gAAQIECBAgQIAAAQIECBAgQIBAvYAAsN5YBwIECBAgQIAAAQIECBAgQIAAAQJtAgLANnqNCRAgQIAAAQIECBAgQIAAAQIECNQLCADrjXUgQIAAAQIECBAgQIAAAQIECBAg0CYgAGyj15gAAQIECBAgQIAAAQIECBAgQIBAvYAAsN5YBwIECBAgQIAAAQIECBAgQIAAAQJtAgLANnqNCRAgQIAAAQIECBAgQIAAAQIECNQLCADrjXUgQIAAAQIECBAgQIAAAQIECBAg0CYgAGyj15gAAQIECBAgQIAAAQIECBAgQIBAvYAAsN5YBwIECBAgQIAAAQIECBAgQIAAAQJtAgLANnqNCRAgQIAAAQIECBAgQIAAAQIECNQLCADrjXUgQIAAAQIECBAgQIAAAQIECBAg0CYgAGyj15gAAQIECBAgQIAAAQIECBAgQIBAvYAAsN5YBwIECBAgQIAAAQIECBAgQIAAAQJtAgLANnqNCRAgQIAAAQIECBAgQIAAAQIECNQLCADrjXUgQIAAAQIECBAgQIAAAQIECBAg0CYgAGyj15gAAQIECBAgQIAAAQIECBAgQIBAvYAAsN5YBwIECBAgQIAAAQIECBAgQIAAAQJtAgLANnqNCRAgQIAAAQIECBAgQIAAAQIECNQLCADrjXUgQIAAAQIECBAgQIAAAQIECBAg0CYgAGyj15gAAQIECBAgQIAAAQIECBAgQIBAvYAAsN5YBwIECBAgQIAAAQIECBAgQIAAAQJtAgLANnqNCRAgQIAAAQIECBAgQIAAAQIECNQLCADrjXUgQIAAAQIECBAgQIAAAQIECBAg0CYgAGyj15gAAQIECBAgQIAAAQIECBAgQIBAvYAAsN5YBwIECBAgQIAAAQIECBAgQIAAAQJtAgLANnqNCRAgQIAAAQIECBAgQIAAAQIECNQLCADrjXUgQIAAAQIECBAgQIAAAQIECBAg0CYgAGyj15gAAQIECBAgQIAAAQIECBAgQIBAvYAAsN5YBwIECBAgQIAAAQIECBAgQIAAAQJtAgLANnqNCRAgQIAAAQIECBAgQIAAAQIECNQLCADrjXUgQIAAAQIECBAgQIAAAQIECBAg0CYgAGyj15gAAQIECBAgQIAAAQIECBAgQIBAvYAAsN5YBwIECBAgQIAAAQIECBAgQIAAAQJtAgLANnqNCRAgQIAAAQIECBAgQIAAAQIECNQLCADrjXUgQIAAAQIECBAgQIAAAQIECBAg0CYgAGyj15gAAQIECBAgQIAAAQIECBAgQIBAvYAAsN5YBwIECBAgQIAAAQIECBAgQIAAAQJtAgLANnqNCRAgQIAAAQIECBAgQIAAAQIECNQLCADrjXUgQIAAAQIECBAgQIAAAQIECBAg0CYgAGyj15gAAQIECBAgQIAAAQIECBAgQIBAvYAAsN5YBwIECBAgQIAAAQIECBAgQIAAAQJtAgLANnqNCRAgQIAAAQIECBAgQIAAAQIECNQLCADrjXUgQIAAAQIECBAgQIAAAQIECBAg0CYgAGyj15gAAQIECBAgQIAAAQIECBAgQIBAvYAAsN5YBwIECBAgQIAAAQIECBAgQIAAAQJtAgLANnqNCRAgQIAAAQIECBAgQIAAAQIECNQLCADrjXUgQIAAAQIECBAgQIAAAQIECBAg0CYgAGyj15gAAQIECBAgQIAAAQIECBAgQIBAvYAAsN5YBwIECBAgQIAAAQIECBAgQIAAAQJtAgLANnqNCRAgQIAAAQIECBAgQIAAAQIECNQLCADrjXUgQIAAAQIECBAgQIAAAQIECBAg0CYgAGyj15gAAQIECBAgQIAAAQIECBAgQIBAvYAAsN5YBwIECBAgQIAAAQIECBAgQIAAAQJtAgLANnqNCRAgQIAAAQIECBAgQIAAAQIECNQLCADrjXUgQIAAAQIECBAgQIAAAQIECBAg0CYgAGyj15gAAQIECBAgQIAAAQIECBAgQIBAvYAAsN5YBwIECBAgQIAAAQIECBAgQIAAAQJtAgLANnqNCRAgQIAAAQIECBAgQIAAAQIECNQLCADrjXUgQIAAAQIECBAgQIAAAQIECBAg0CYgAGyj15gAAQIECBAgQIAAAQIECBAgQIBAvYAAsN5YBwIECBAgQIAAAQIECBAgQIAAAQJtAgLANnqNCRAgQIAAAQIECBAgQIAAAQIECNQLCADrjXUgQIAAAQIECBAgQIAAAQIECBAg0CYgAGyj15gAAQIECBAgQIAAAQIECBAgQIBAvYAAsN5YBwIECBAgQIAAAQIECBAgQIAAAQJtAgLANnqNCRAgQIAAAQIECBAgQIAAAQIECNQLCADrjXUgQIAAAQIECBAgQIAAAQIECBAg0CYgAGyj15gAAQIECBAgQIAAAQIECBAgQIBAvYAAsN5YBwIECBAgQIAAAQIECBAgQIAAAQJtAgLANnqNCRAgQIAAAQIECBAgQIAAAQIECNQLCADrjXUgQIAAAQIECBAgQIAAAQIECBAg0CYgAGyj15gAAQIECBAgQIAAAQIECBAgQIBAvYAAsN5YBwIECBAgQIAAAQIECBAgQIAAAQJtAgLANnqNCRAgQIAAAQIECBAgQIAAAQIECNQLCADrjXUgQIAAAQIECBAgQIAAAQIECBAg0CYgAGyj15gAAQIECBAgQIAAAQIECBAgQIBAvYAAsN5YBwIECBAgQIAAAQIECBAgQIAAAQJtAgLANnqNCRAgQIAAAQIECBAgQIAAAQIECNQLCADrjXUgQIAAAQIECBAgQIAAAQIECBAg0CYgAGyj15gAAQIECBAgQIAAAQIECBAgQIBAvYAAsN5YBwIECBAgQIAAAQIECBAgQIAAAQJtAgLANnqNCRAgQIAAAQIECBAgQIAAAQIECNQLCADrjXUgQIAAAQIECBAgQIAAAQIECBAg0CYgAGyj15gAAQIECBAgQIAAAQIECBAgQIBAvYAAsN5YBwIECBAgQIAAAQIECBAgQIAAAQJtAgLANnqNCRAgQIAAAQIECBAgQIAAAQIECNQLCADrjXUgQIAAAQIECBAgQIAAAQIECBAg0CYgAGyj15gAAQIECBAgQIAAAQIECBAgQIBAvYAAsN5YBwIECBAgQIAAAQIECBAgQIAAAQJtAgLANnqNCRAgQIAAAQIECBAgQIAAAQIECNQLCADrjXUgQIAAAQIECBAgQIAAAQIECBAg0CYgAGyj15gAAQIECBAgQIAAAQIECBAgQIBAvYAAsN5YBwIECBAgQIAAAQIECBAgQIAAAQJtAgLANnqNCRAgQIAAAQIECBAgQIAAAQIECNQLCADrjXUgQIAAAQIECBAgQIAAAQIECBAg0CYgAGyj15gAAQIECBAgQIAAAQIECBAgQIBAvYAAsN5YBwIECBAgQIAAAQIECBAgQIAAAQJtAgLANnqNCRAgQIAAAQIECBAgQIAAAQIECNQLCADrjXUgQIAAAQIECBAgQIAAAQIECBAg0CYgAGyj15gAAQIECBAgQIAAAQIECBAgQIBAvYAAsN5YBwIECBAgQIAAAQIECBAgQIAAAQJtAgLANnqNCRAgQIAAAQIECBAgQIAAAQIECNQLCADrjXUgQIAAAQIECBAgQIAAAQIECBAg0CYgAGyj15gAAQIECBAgQIAAAQIECBAgQIBAvYAAsN5YBwIECBAgQIAAAQIECBAgQIAAAQJtAgLANnqNCRAgQIAAAQIECBAgQIAAAQIECNQLCADrjXUgQIAAAQIECBAgQIAAAQIECBAg0CYgAGyj15gAAQIECBAgQIAAAQIECBAgQIBAvYAAsN5YBwIECBAgQIAAAQIECBAgQIAAAQJtAgLANnqNCRAgQIAAAQIECBAgQIAAAQIECNQLCADrjXUgQIAAAQIECBAgQIAAAQIECBAg0CYgAGyj15gAAQIECBAgQIAAAQIECBAgQIBAvYAAsN5YBwIECBAgQIAAAQIECBAgQIAAAQJtAgLANnqNCRAgQIAAAQIECBAgQIAAAQIECNQLCADrjXUgQIAAAQIECBAgQIAAAQIECBAg0CYgAGyj15gAAQIECBAgQIAAAQIECBAgQIBAvYAAsN5YBwIECBAgQIAAAQIECBAgQIAAAQJtAgLANnqNCRAgQIAAAQIECBAgQIAAAQIECNQLCADrjXUgQIAAAQIECBAgQIAAAQIECBAg0CYgAGyj15gAAQIECBAgQIAAAQIECBAgQIBAvYAAsN5YBwIECBAgQIAAAQIECBAgQIAAAQJtAgLANnqNCRAgQIAAAQIECBAgQIAAAQIECNQLCADrjXUgQIAAAQIECBAgQIAAAQIECBAg0CYgAGyj15gAAQIECBAgQIAAAQIECBAgQIBAvYAAsN5YBwIECBAgQIAAAQIECBAgQIAAAQJtAgLANnqNCRAgQIAAAQIECBAgQIAAAQIECNQLCADrjXUgQIAAAQIECBAgQIAAAQIECBAg0CYgAGyj15gAAQIECBAgQIAAAQIECBAgQIBAvYAAsN5YBwIECBAgQIAAAQIECBAgQIAAAQJtAgLANnqNCRAgQIAAAQIECBAgQIAAAQIECNQLCADrjXUgQIAAAQIECBAgQIAAAQIECBAg0CYgAGyj15gAAQIECBAgQIAAAQIECBAgQIBAvYAAsN5YBwIECBAgQIAAAQIECBAgQIAAAQJtAgLANnqNCRAgQIAAAQIECBAgQIAAAQIECNQLCADrjXUgQIAAAQIECBAgQIAAAQIECBAg0CYgAGyj15gAAQIECBAgQIAAAQIECBAgQIBAvYAAsN5YBwIECBAgQIAAAQIECBAgQIAAAQJtAgLANnqNCRAgQIAAAQIECBAgQIAAAQIECNQLCADrjXUgQIAAAQIECBAgQIAAAQIECBAg0CYgAGyj15gAAQIECBAgQIAAAQIECBAgQIBAvYAAsN5YBwIECBAgQIAAAQIECBAgQIAAAQJtAgLANnqNCRAgQIAAAQIECBAgQIAAAQIECNQLCADrjXUgQIAAAQIECBAgQIAAAQIECBAg0CYgAGyj15gAAQIECBAgQIAAAQIECBAgQIBAvYAAsN5YBwIECBAgQIAAAQIECBAgQIAAAQJtAgLANnqNCRAgQIAAAQIECBAgQIAAAQIECNQLCADrjXUgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQGAHFfiheN6nj+ugKftwwnjZ26YsM4sAga0rcELser53eG/Yut8D9nz5As61y7eyJgEC2wROiBvnWd8NBAjMCezKgQCBHULgyHiWfzl4plfH9PdHXTuYN23yDjHzPVF7Dhb+Qkx/fHDfJAECO4bAwveBhc/6mphxUdRnozJQOy3KIEBg+QILjzHn2uXbWZPAZhBY+B6wcJ+cZxeKuE+AwA4lsPMO9Ww9WQIEJgJ7xMRxkzuL3B4by4bh3yKrWrQBBSZXVT5vAz43T2njCdwpntK9o54c9aqoF0U5zweCQWCFAs61K4TbwR7mXLuDvWCNT9d5thFfawIEVi/gCsDVG9oCgfUWuD4a7h6VP+S/e4nmTxkvnzxmidXLFp8QW84yCBBYG4E3x2beNNjUTjF9l6gjoo6P2icqj/9vRb0yaqOOE+KJZRkENprA5LzpXLvRXhnPh8D6CDjPro+zLgQIrKOAKwPWEVsrAmsk8P7xdo6O230X2WYGALlOjpPnvvpCgMBmEbgkduSsQX0lpj8RdWLUL0ZleJHjJ6N2m5vyhQCB7RFwrt0eLesS2HwCzrOb7zW1RwS2vIAAcMt/CwDYAQU+Es/521F5/D5pkeefy3aJynU/ush6FhEgsLkEzo7dOWW8S/kRxkPG024IEFi+gHPt8q2sSWCrCTjPbrVX3P4S2CQCPgK8SV5Iu7GlBG6Ovf3XqPyYX37E7++jpo3Jx3/zY8L5mMXGYbHwuKiHRx0atXfUjVEXR306Kj8G8ZmolY4T4oFPjfpmVP4Vw1njcbHgx6MeEJW/ZyU/vphXYbw+KoPMt0XdPertUSdEDUdu9/fGM344bi+I+o9R2fc+UXkV1PlR74lKs2ujpo0MVvOXQD8m6oiog6P2ispf/PyNqI9FvSEqtz9rvDoWPCIqr8h6XtQBUT8TlfuX09n781H5PE6NWjgm+zmZn9vIGo5pBsPlpre2wNcHu7/YFYDfHev9aNQjo/aP2inqwqjTov4u6ryoaWOtjrcTYuM7ynvDNAfzNq+Ac61zbX53O9du3mN8tXvmPHur4Fr9m+DWLZoiQKBEQABYwmqjBMoF3hEdMgA8PCoDu7OjhiPn5Q/2OXLdXG/WyLBr+BeGJ+tlaHCvcWWYeGLUn0dVjd+MDWf4Nxz3jjvPjMrfwfQrwwVLTN8xludzPWrBeveN+1n5x1F+MSpDvYXjuTFjYdiW62QIeP9xPT1ufyfqpKilxkNjhT+O+q7BirvH9NHj+tO4fd1gmUkCayGQQflkXDCZGNxm0P2rUT8VlaHfcORxl5UB+kuj/ilqsbGa422x7U6WbZT3hsnzcbt1BJxrF3+tV3PsO9cubmvpxhdwnp3+Gq3mfWH6Fs0lQGDNBHZdsy3ZEAEC6ynwpWiWoV8GfRnOvSJqODIwy5G/I+zMqMUCwPyYcAZhp0TlVT/nRH0nKn+HYG7/J6PyHznPjjov6q1Raz2eFRuchH951d+JUXmFXAZlj4766aiXReU/KpYzfjtWenBU/s/9v0V9O+puUdknr+p7UNR/ipoWaOb74sVRGe59Jur8qOujDozKMC/Dvz2ifj8qn9dXo2aN/WJBhn83R+Vr9KmoG6IeFvXzUXeO+qWoD0Xl6zkZz4+J3aL+cTzjzXH7pvH05ObKyYRbAgsEDon7jx3Py+/hS8bTw5v/Fncmx9wnYjqPlfxez6tTM+jO/2DI4/+FUXk8fCBq1ljN8TZrm5P5G+m9YfKc3G4dAefaxV/r1Rz7zrWL21q6sQUOiafnPDv9NVrN+8L0LZpLgMCaCeTJ1yBAYMcUyB/YfyXqB6L+POqWqBx5Nc8Pzk1tu/pvPDnzJgPCDAynBUofjvlvjHp51PdEZWiVfTPQWquxb2zoF8cb+1rc/lzUZeP7efPJqFOi/jIqQ7HljAz5fjfqnYOVvxjTp0a9LuqwqKdFvSrqpqjh+Oe48+qoG4czYzof//6oN0SdGHVAVD7X7DNr3DsWfDMqw8YLBytluJn1mqhdovIjmH8UNRkZtA5HBjgZ5hoEJgIZ0Of38WTkcZ+B8kOiMpi+Q9RVUX8StXDksTwJ//5nTP/LghXye/OdUXl16qOiMiz8UNTCYyVmzY3VHG+TbUy73WjvDdOeo3mbX8C5dvZrvJpj37l2tqslG0PAeXb9/w2+MV55z4LAJhbYeRPvm10jsNkF3h07mEFcXpl25GBnczrn5bJcZ6mRYduVi6yUV6xlEJAjrwQ8fG5q7b48NTaVV/rlyKvlhuHf3Mz48umohVfATZZNu31fzMwAY+G4PmZMrqq7a0znFU4LxzdixsLwb7hOBnmvG884Nm4zeFls/GEsHIZ/k3U/FROfHd95+GSmWwLLFHh6rJffy5PKYDoD5V+J2jvqLVHPivpM1MLx7PGMPE7+ZeHC8f08Vl42ns7j/pHj6Wk3qznepm1vMm+jvTdMnpfbrSXgXDv79V7Nse9cO9vVko0h4Dy77XVYz3+Db4xX3rMgsIkFBICb+MW1a5teIEOl08d7+eTB3k6mc1mus70jw7i7RWU4llcYDa8yiruj++WXNRx5NVKODP5OnZua/uXt02dPnbtY8PnFwSPuMZieNblnLDgoauhx7XjlybJZj81gNa9enDW+MF6wnOcxaxvmE1gokOf274/KH17yeB6O/J7N/yTI8d5tNzO/5sfbJ4F8Xukza6zl8TbssdHfG4bP1fTmFXCunf3aruWxPzmfOtfO9rZk4wg4z85+LdbyfWF2F0sIEFiRgI8Ar4jNgwhsGIF3xDM5KuqJUS8dP6vvHd9uT2B2p3jMT0ZlaJCBX/7DZtb4rlkLVjh/EjCeGY+/eZFtfCWW5dWIy/kY8DmLbOfywbI9BtPDybzi6ZlRj43K6cVGenx9xgpfi/mL7dMV48fNeh4zNms2gbmPqb96gUN+7PdeUU+O+umo46MeGPVLUddG5cgreCfH9x/EdNZyxr6LrHTOIsuWc7zNevhGfG+Y9VzN39wCzrXTX99zps+em7ucY9+5dhFAi9oF8hzrPLvtZVjPf4O3v/CeAIHNLDD5IWAz76N9I7CZBU6Kncsf7PN/zvPjqMdFZZh0TVQuW844KFbKjxE+Pyqv7lvqfeGOsc5ajvy9ZTku3XYz82sGaZPAbOZK4wWTsGPaercMZu4ymJ5MHhMTb4r6iailwr98TIYus8ZizyMfMwkHlzKftX3zCQwFros7+Y/0P4t6yXjBw+L2OePpvNlnML09k4sd94t9ny91vC32HDbae8Niz9WyzS2Q51Pn2tu/xqs59p1rb+9pzsYXcJ5d+jVazfvC0lu3BgECqxJwBeCq+DyYQLvA1fEM8geTH4x6StTk99HlvAwBlzP+R6yUIWD+oP62qH+N+mpUfvQvr7jL+RlQfSxqs4+8mu/3ozLsSNvXR3046vyoq6LSI0f+YYRXzU3daj6+64bAhhDI3+33y1F3ifrhqFdG5RiGzXn13xlzc5f+kh9nNwhsVQHn2rV95Z1r19bT1noEnGd73HUlQGAVAgLAVeB5KIENIpAfTcoA8OjB88l5yxmHxEp5hVCOv46ahARzMwZfMkSoGhks5McL916iQQYXkyuCllh1xYufOOjxGzE9K/Ss9Fjxk/dAAgOBvLr0vKgHR+0XlX/05vJxxc3cyP+lP2s8vRFvNtJ7w0b08ZzWV8C5du28nWvXztKW+gScZ/vsdSZAYIUCwysBVrgJDyNAoFkgQ6qLo/LjrFkXRZ0WtZxx6GCl9wymF04+cOGMNbw/CSDuH9tc7D3pvrF89zXsO21Th41n5keNZ4V/uUqlx/gpuCGwaoF8P5iMyfSZMSOv6s3x0G03G/brRnpv2LBInti6CTjXrh21c+3aWdpSr8Dk3JrPYjLtPNv7muhOgMAiAov9sL3IwywiQGADCeT/QL4z6vpx5fTkd8vF5KJjeBVw/iGQWePHZi1Yg/mTsDI/EnTMItt76iLL1mrR5B9vGTTOen/Mjwc/ea0aLrGdfE1zVAef27r4upkE8vt0EvDn7yy6bLxz+bs2PzOe/oG4XerK2/GqLTcb6b2hBUDTDSXgXLt2L4dz7dpZ2lKfgPNsn73OBAisUGDWD7gr3JyHESDQJJC/9D/Ds6xXbMdzOG+w7g8NpoeTT487xw5nrPH022N7k6Dr12M6g8CF44iY8eMLZxbcn3jkP+q+d8r28z3zd6L2n7KsYlZe2ZnjnttufCWwbIHnxZqTP1CTv8dy+J8Crx1vZc+4fWnUYh+tz/D5J6I6QuiN9N4QBAaBuT+w41y7+m8E59rVG9pCv4DzbP9r4BkQILCdAsOrf7bzoVYnQGATCHwp9iE/ZndY1I9GZRCQVxBm8HRAVF7p9sSoM6KqPi54UWz7NVHPj7pX1OujToz6fFSGDo+O+pmoXC+vUswrliYfYYzJNR3/FlvL55F9T4g6POqjUVdFHRr1jKgHRFV6xObnR/Y5KOpxUfn65P1JWPqdmL4kytiaAvvEbudxOxwZ+OUx9JSoDCly5PfLX8xN3frlQzH5D1E/FfWIqDdHvSXqU1F5pWAeZ7mdh0c9Pip/5+UwjIu76zLymN8o7w3rssOabFoB59rbvrTOtbf1cG9jCjjPru+/wTfmd4FnRWCTCQgAN9kLancIbKdABmm/E5UBQf6Q/33jipv58ZWY+s2od8/PWfuJE2OTd4/KkOtuUb8VNRwZSuS8PxzPnIRgw3XWYvrC2MhLon47KkPAnx1X3MyP98TUP0e9cn5O3cTrYtMZwOZzeeGCNhnInLBgnrtbR+DpsatZi438uG8e33kMLxx/EjOuiHpu1L5Rz4uaNfIvit80a2Hx/BNj+xvhvaF4N21+kws41972BXauva2HextTwHl22+uyXv8G35jfBZ4VgU0mIADcZC+o3SGwAoEz4zHHR/1c1DFR+fHWq6O+FpX/S//GqOujKkf+cPQHUadE/UTUA6PyY7jfisqrlf42Kn9g2DMqx1Xbbkq+vjW2ek7Us6Lyqse8KjL/8ZNOb4tKkyOj1mNkz3xdJs8l/zc6w0CDwDSBG2JmhnpnReVxk9+veX/ayGPuNVHviMofch4Zdc+ovaKujbogKq9a+kjUSVHXRXWMjfTe0LH/em4egXw/Pz7KuXbba+pcu3m+t7fSnjjP1v4bfCt9L9lXAgQIECBAYBGBA2LZ6eP6kUXWs4gAga0l4L1ha73e9rZWwPFU62vrBHZEAe8LO+Kr5jkTmCKQv9DeIECAwI4g8AODJ/mZwbRJAgS2toD3hq39+tv7tRVwPK2tp60R2AwC3hc2w6toHwiEgADQtwEBAhtBIP/owH6LPJH8Yxz5u8pyfCHq7LkpXwgQ2OwC3hs2+yts/9ZTwPG0ntp6EdgxBLwv7Bivk2dJYE0E/A7ANWG0EQIEVimwdzw+/xLpyVGnRp0bdX1U/j7CY6LyI7/5F07z94H97yiDAIGtIeC9YWu8zvZyfQQcT+vjrAuBHUnA+8KO9Gp5rgRWKbDTKh/v4QQIEFgLgYNiI/kLwRcb+YuXXxyVf7TAIEBgawh4b9gar7O9XB8Bx9P6OOtCYEcS8L6wI71aniuBVQrsssrHezgBAgTWQuCa2Ej+5dKroyZXJu82vn9O3L4r6nejPhFlECCwdQS8N2yd19qe1gs4nuqNdSCwowl4X9jRXjHPlwABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQILEPgzrHO6eM6fhnrr3SVPx73+PuVbsDjCBCYE3DM+kYgQIAAAQIECBAgsAMJTP7a5g70lD1VAgRmCBwU8986Y9n2zH7k9qxsXQIEVizgmF0xnQcS2BIC3iO2xMtsJzeRgGN2E72YdoXAZhTYeTPulH0iQIAAAQIECBAgQGBDCKzXFcMbYmc9CQKbQMAxuwleRLtAYJqAKwCnqZhHYMcUuDCe9jMWeer/OF72+bh90SLrdSy6Mpqux5WHv96xc3oSmCHgmJ0BM5jtmB1gmNxyAjvye8SWe7HsMIEQcMz6NiBAYEMLCAA39MvjyRHYLoEbY+2zlvGIa5e53jI2ZRUCBFYh4JhdBZ6HEtgCAt4jtsCLbBc3lYBjdlO9nHaGwOYT8BHgzfea2iMCBAgQIECAAAECBAgQIECAAAEC8wKuAJynMEGAwEAg/1rusVFnRh0fdfeon4p6TNQBUXeKyvm5PMd3RT0hKj/Ge3jUgVG7RF0e9aWo90S9O+qmqGkjf9fISeMFfxK3C/9Kb/b6tfHyx8ftNVE/FvWUqIOj8j8zzovKHm+IuiFq2li4X8N1pj2Hh8cKud9HRN016pKoj0a9NurrUYuNPWPhz0bl8z0oKp/z2VH/FJXP87ioP4rKkft35tyULwRWJrDwe9sx65hd2XeSR212gZ1iB78v6vujHhi1d1Sen/IcenLUG6Oujpo1DosFz4h6RNSBUfmzxGVRl0Z9LurDUe+PmpzvT47pvaImI8/lk/P5ZN4/xES+hxkECNxewDF7exNzCBBYoYAAcIVwHkZgCwkcFfv6sqjhP+AX7v4/z1i+X8zPyuDwR6NeEHVF1GpGBnV/GpWh3HBk8JiVvX45alYIGIuWNZ4da/2XqAwXJyN/2PnhqCdE5bL8fYrTxr1i5qui7jZYeIeYfsS4Hhe37x0sM0lgLQUcs7dqOmZvtTBFYJ8gyKDtIQsodov7Dx7Xj8ftr0Z9OWrh+JGY8cKo/A++4dg/7mTdP+ppUU+OujDKIEBgdQKO2dX5eTQBAgsEBIALQNwlQOA2Anll30uibon6i6iPR10flUHbMMjLkOyTUR+KyivZ8kqAO0YdFJWB2ZFRD406IerXolYzXhwPflBUXkmXVw1eEnXPqOdE5fPKqxCPj/qbqJWO740HHhH12ag3Rp0TlVc95hUTeeVhhqEnRD0jKm2GI/f7FVGT8O/fY/rtURdFpUf+cJXbyWmDwFoLOGYds2v9PWV7m0Ngj9iN10QdHJW/p+xtUR+LyqvZ87w1OXdmaJ7nsDyP5vl1Mu4RE78VleFfhnv5h8W+EHV5VD4+t3tk1LFRw/FzcefOUX81npnn5neOpyc3l00m3BIgMC/gmJ2nMEGAwFoJ7LpWG7IdAgQ2pcABsVf5j/tnR50XNRmfm0yMb38mbofLJ4szFHxHVP4gkcFfXvn2gKj8oWGlI69S+I2oDww2kB8z/nDUG6ImIdtqAsAM/94T9dtRN0dNRgagV0c9M+rQqPxh5/So4XhW3MlAMserxzV3J758MeqkqBdFPTnKILDWAo5Zx+xaf0/Z3uYQ+OXYjQzpMtT7haivRg3HJ+LOv0b9ddR+UT8f9dKoyTguJvJKwTwnPjfqG1HDcUbceWtUhhbDK/CzTwaAk/HtmDhrcsctAQIzBRyzM2ksIEBgpQI7r/SBHkeAwJYRyBBrWrg3BFhq+T/EypMfFo4bPnAF03k13QemPC6DubeM5+fVd5Mr8KasuuSsK2ONF0cNw7/Jg14/mYjbhw+mczLfU582nndu3P7f8fTw5pa487Ko7GEQqBBwzN5W1TF7Ww/3tp7A3rHLPzLe7ZfH7cLwbyKS563Xje/k79jNc9pk7DueuCBuJ+fzybLhbZ6LhwHgcJlpAgSWJ+CYXZ6TtQgQ2E6B4Yl9Ox9qdQIEtojAu7dzP3eK9fPqgYOjDhvXoXH7ragc99t2s+Kv71rkkcMrC/PjSisdH4wH5g8x00ZevXDheMHCHofE/Pw9SDnSbVqAmMuuinp/ThgECgQcs7dFdcze1sO9rSdwdOzy7lF5Tjppid3PKwFz5JV8eQ6fjIvHE/+/vfsNlays4wCO2SYrZZgQZtKCii80Q0HNXbKF3RUzIgQpFUEpNSEU/IMR9CKEfFFQ9CYIQUpF9E0iVpAQogSiUKSlgqUpi5YS4r/EzSz7/q7n7J6ZO3Nn7p07c/fOfB743vN3zszzufvs3HnOmeccnekZ7UpTAgSmIqDNToXVQQkQ+CACAgQIrCBQHV2vr7C9u2l3Furqtxrrb2t3Q998jVE2SXl+hQd3xyWsu/CutTw/4oH1PPVVy/qA1C0ndBa6nZGd1ftnn8rcl/YvmSGwPgLa7GBHbXawi7WLIVBDb1SpE/91gmvcUlf9/bXZ+beZfjOp8f5+kjya1LEeS55Jhp3wyiaFAIFVCmizqwSzOwEC4wnoABzPyV4EFlVgnK+pHhqcm5M9YyLVh4dJyr4VHtz9ADLJFc4rPUc9ffs8Vfdu+Uhn4dXO/KBZg54PUrFuUgFtdrCgNjvYxdrFEPjYGqvZfb+ukws3JjcldbztTTJZGtLikUzvTapjUCFAYDIBbXYyP48mQGCIgA7AITBWEyCwJPDfMRwuzj5t598Tmb87qavb6utC1ZHWfvD+UeY/nygECExPQJudnq0jE9isAu0JsXdSgbqJ1bjlH3071s22aizBes/fkZyW1JAfdfLrnCYPZPqd5D+JQoDA2gS02bW5eRQBAiMEdACOALKZAIGRAvW13yp/SS5PhnVAfLR2mvPSvfrqyBF1nfSr0CMObzOBoQLa7AEabfaAhbn5FXi9qVqNA1idesPGuB1H4O3s9Msmtf+nkrOTC5Njkl3J15JbEoUAgbUJaLNrc/MoAgRGCLRnF0bsZjMBAgQGCtRXYOuP/yp11n9Y51+dbJj05h/1HAd7ebbzAtvxWzqremZP6lmyQGA2Atpsr7M22+thaT4Fnu5Uq8bpXc+yNwe7M7ksaTst2m8FtM/zXjtjSoDAWALa7FhMdiJAYLUCOgBXK2Z/AgS6AtWZUHf9rbL1/cnAn+dl7eEDt8zXyudSnfZOiV/I/LD/Yz+cbTvnq+pqs0kEtNneX5Q22+thaT4FHk613m2qVsN2TKPUuLftDUP6r3B/p/OEWzrzZgkQGCygzQ52sZYAgQkFhn04nfCwHk6AwIII1B/1Lzd13Z3poE6+47P+umafeZ/8LxWsQdCrbEuuWJpb/uNbWdW9YcjyPawhMB0BbbbXVZvt9bA0nwJ1A4/7mqrtyPSqEdX8eLb336W+vubb37HXPUzdtODEZsXfuxsyX//vvNGsO7Zvm0UCBJYLaLPLTawhQGAdBIwBuA6IDkFgwQV+nfp/Pak/6n+W3J7UVTV1ReBZyUXJe8kzyQnJvJfbUsEvJsck30iOS2q8pFeSWveV5IzkyeTkRCEwawFttldcm+31sDSfAj9Oterrv3VS7spke1KdgvXe/O/kiKTeo2v9Z5M/JL9K2nJ+Zr6f1I1AHk3+ltQYmnVFe3X81Xt9HaPKL96f9Pz8U5Y+l5yb1Hy9B7Y3CqnjvJYoBAgcENBmD1iYI0BgnQR0AK4TpMMQWGCBW1P305PPJPXB4qakW2qw8W8nFySL0AFYA6Rfnfw0qaso9jTJZH+p8RLvT+rDVJX68KUQmJWANtsrrc32eliaT4F6L66Ov5uT6uT7dJNMBpa3Bqz9UNbtbDJg89LJvjuyoU569ZfqaK+rD+ubAt/t23hXln/Yt84igUUX0GYX/V+A+hOYgoAOwCmgOiSBBROozqv6OtHFSY17ty2pr9X9M3k4qT/sX0yqA3BRyt5U9KvJpcmupK7825fUFRN1xUV9OOp+vepfWVYIzEpAm10urc0uN7Fm/gTeSJWuSc5MzktOTY5KDkveTF5I/pz8Lvl90i3fy8KDSZ3wq5N99bgjk3eTl5LHknuSp5JB5Y9ZeUVySXJKUo/dkigECAwX0GaH29hCgAABAgQIENg0AtfmldYHrIeSQzbNq/ZCCSyugDa7uL97NSdAgAABAgQIbHqBD2z6GqgAAQIENp/AoXnJu5uXXeMg1RiJCgECB6+ANnvw/m68MgIECBAgQIAAgTEEdACOgWQXAgQIrFLgk9l/2P+vdbXf9cknmmN2B1lvVpkQIDBjAW12xuCejgABAgQIECBAYLYCxgCcrbdnI0BgMQTqbohnJ/cnjyevJDV4eo2b9OWkbphSpa7++83SnB8ECGykgDa7kfqemwABAgQIECBAYOoCOgCnTuwJCBBYUIFjU+/LV6j709l2Q1I3TFEIENh4AW12438HXgEBAgQIECBAgMCUBGpMG4UAAQIE1leg7ihad0Fub+7Rnmx5NevqTog/T36QvJUoBAhsCu5ZJQAAAl5JREFUvIA2u/G/A6+AAAECBAgQIECAAAECBAgQIECAAAECBAgQIECAAAECBAgQIECAAAECBAgQIECAAAECBAgQIECAAAECBAgQIECAAAECBAgQIECAAAECBAgQIECAAAECBAgQIECAAAECBAgQIECAAAECBAgQIECAAAECBAgQIECAAAECBAgQIECAAAECBAgQIECAAAECBAgQIECAAAECBAgQIECAAAECBAgQIECAAAECBAgQIECAAAECBAgQIECAAAECBAgQIECAAAECBAgQIECAAAECBAgQIECAAAECBAgQIECAAAECBAgQIECAAAECBAgQIECAAAECBAgQIECAAAECBAgQIECAAAECBAgQIECAAAECBAgQIECAAAECBAgQIECAAAECBAgQIECAAAECBAgQIECAAAECBAgQIECAAAECBAgQIECAAAECBAgQIECAAAECBAgQIECAAAECBAgQIECAAAECBAgQIECAAAECBAgQIECAAAECBAgQIECAAAECBAgQIECAAAECBAgQIECAAAECBAgQIECAAAECBAgQIECAAAECBAgQIECAAAECBAgQIECAAAECBAgQIECAAAECBAgQIECAAAECBAgQIECAAAECBAgQIECAAAECBAgQIECAAAECBAgQIECAAAECBAgQIECAAAECBAgQIECAAAECBAgQIECAAAECBAgQIECAAAECBAgQIECAAAECBAgQIECAAAECBAgQIECAAAECBAgQIECAAAECBAgQIECAAAECBAgQIECAAAECBAgQIECAAAECBAgQIECAAAECBGYq8H+hqb4xSIFQdwAAAABJRU5ErkJggg==" width="640">



```python

```
