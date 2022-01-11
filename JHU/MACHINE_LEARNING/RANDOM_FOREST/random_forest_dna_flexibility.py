# Author: Basilio Cieza Huaman
# email: bciezah@gmail.com


# Importing libraries

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, RandomTreesEmbedding
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split 
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
import matplotlib.pyplot as plt


# Loading data 

mydata = pd.read_csv("sorted_dna_seq_F26_all_data.csv") # comando para subir la data
mydata.head(3) # comando para visualizar la data



# Variable dependent/independent

var_dep = mydata["F"] # cogemos la columna de nombre insuranceclaim
var_ind = mydata.drop(["sequence","F"],axis=1) # cogemos TODAS las columnas menos la que dice insuranceclaim (drop)
var_ind.head(3)


# Normalizing data
var_ind_norm = StandardScaler().fit_transform(var_ind)
print("ready")


# Split in test and training
x_train, x_test, y_train, y_test = train_test_split(var_ind_norm, var_dep, test_size=0.25, random_state=0)
print("ready")


# Creat model and train it
clf=RandomForestRegressor(max_depth=4,n_estimators=100)
clf.fit(x_train, y_train) # entrena el modelo (data de entrenamie)
print("ready")


#Predicting
predictions = clf.predict(x_test)


#Plotting predicted versus observed (test data)
plt.plot(predictions,y_test,'o')
