# -*- coding: utf-8 -*-
#Classification Model
#Please run only specific parts of the code based on requirement.
#This  file has code to do create many Classification models.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Importing and merging Datasets
dataset1 = pd.read_csv('Datasets/Accidents_2014.csv', low_memory=False) # , parse_dates=['Date' , 'Time']
dataset2 = pd.read_csv('Datasets/Accidents_2015.csv', low_memory=False) # , parse_dates=['Date' , 'Time']
dataset3 = pd.read_csv('Datasets/Accidents_2016.csv', low_memory=False) # , parse_dates=['Date' , 'Time']
dataset_A = pd.concat([dataset1, dataset2, dataset3])

#Importing and merging Datasets
dataset4 = pd.read_csv('Datasets/Vehicles_2014.csv', low_memory=False) # , parse_dates=['Date' , 'Time']
dataset5 = pd.read_csv('Datasets/Vehicles_2015.csv', low_memory=False) # , parse_dates=['Date' , 'Time']
dataset6 = pd.read_csv('Datasets/Vehicles_2016.csv', low_memory=False) # , parse_dates=['Date' , 'Time']
dataset_V = pd.concat([dataset4, dataset5, dataset6])

dataset = dataset_A.set_index('Accident_Index').join(dataset_V.set_index('Accident_Index'))

# Valuising time as number (from Epoch)
dataset['Date'] =  pd.to_datetime(dataset['Date'], format='%d/%m/%Y')
dataset['Time'] =  pd.to_datetime(dataset['Time'], format='%H:%M')
dataset['Date'] =  pd.to_numeric(dataset['Date'], downcast = 'unsigned')
dataset['Time'] =  pd.to_numeric(dataset['Time'], downcast = 'unsigned')

# Taking care of missing values with most frequent values
dataset = dataset.apply(lambda x:x.fillna(x.value_counts().index[0]))

# Dropping least relevant features and preparing Training set and Test Set
X = dataset.drop(['Accident_Severity','LSOA_of_Accident_Location', 'Local_Authority_(Highway)', '1st_Road_Number', '2nd_Road_Number', 'Local_Authority_(District)',
                  'Vehicle_Reference', 'Towing_and_Articulation', 'Vehicle_Manoeuvre',	'Vehicle_Location-Restricted_Lane',	'Junction_Location','1st_Point_of_Impact',	'Was_Vehicle_Left_Hand_Drive?',	
                  'Journey_Purpose_of_Driver', 'Propulsion_Code', 'Age_of_Vehicle',	'Driver_IMD_Decile','Driver_Home_Area_Type'
                  ], axis = 1).values

# Independent variable
y = dataset.iloc[:, 5].values

# Use this to deal with only specific features, say Latitudes and Longitudes
#X = dataset.iloc[:, [33,43]].values

# Taking care of missing data in categorical columns
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = -1, strategy = 'most_frequent', axis = 0)
imputer = imputer.fit(X[:,[9,11,12,24,25,31,32,34]])
(X[:,[9,11,12,24,25,31,32,34]]) = imputer.transform(X[:,[9,11,12,24,25,31,32,34]])

# Taking care of missing data in rest of the columns
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = -1, strategy = 'mean', axis = 0)
imputer = imputer.fit(X)
(X) = imputer.transform(X)

# Encoding categorical data
# Encoding the Independent Variable
# Label Encoding is not necessary as the dataset is already label encoded
categorical = [9,11,12,24,25,31,32,34]   # column number of categorical values 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
onehotencoder = OneHotEncoder(categorical_features = categorical)
X = onehotencoder.fit_transform(X).toarray()

# Encoding the Dependent Variable
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# Applying PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = None)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_


'''
# Fitting classifier to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0, n_jobs = -1)
classifier.fit(X_train, y_train)

'''
'''
# Fitting Kernel SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)

# Fitting Decision Tree Classification to the Training set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)
'''
'''
# Fitting SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)

# Fitting K-NN to the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)
'''

# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)


# Predicting the Test set results
y_pred = classifier.predict(X_test)

"""
# Building the optimal model by Backward Elimination
import statsmodels.formula.api as sm
X = np.append(arr = np.ones((dataset.shape[0],1)).astype(int), values = X, axis = 1)
X_opt = X[:, [0,1,2,3,4,5,6,7,8,9,10,
              11,12,13,14,15,16,17,18,
              23,24,25,26,27,28,29,30,
              31,32,33,34,35,36,37,38,39,40,
              41,41,43,44,45]]
regressor_OLS =sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
"""

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
den = sum(sum(cm))
num = cm[0][0]+cm[1][1]+cm[2][2]
efficiency = num/den

"""
# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
accuracies.mean()
accuracies.std()

# Applying Grid Search to find the best model and the best parameters
from sklearn.model_selection import GridSearchCV
parameters = [{'C': [1, 10, 100, 1000], 'kernel': ['linear']},
              {'C': [1, 10, 100, 1000], 'kernel': ['rbf'], 'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}]
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
grid_search = grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_
"""

# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green', 'blue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green', 'blue'))(i), label = j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('Age_Of_Driver')
plt.ylabel('Sex_Of_Driver')
plt.legend()
plt.show()

# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green', 'blue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green', 'blue'))(i), label = j)
plt.title('Logistic Regression (Test set)')
plt.xlabel('Age_Of_Driver')
plt.ylabel('Sex_Of_Driver')
plt.legend()
plt.show()