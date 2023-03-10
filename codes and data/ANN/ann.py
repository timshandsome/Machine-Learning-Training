# Artificial Neural Network

# Installing Tensorflow and Keras
 
# 1. On Mac: open "Terminal"
#    On Windows: open "Anaconda Prompt"  

# 2. Type:
# conda install tensorflow
# conda install -c conda-forge keras
# conda update --all


# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1]) #國家
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2]) #性別
#onehotencoder = OneHotEncoder(categorical_features = [1])
#onehotencoder = OneHotEncoder(categories='auto')
onehotencoder = ColumnTransformer([('encoder', OneHotEncoder(), [1])], remainder='passthrough')
#X = onehotencoder.fit_transform(X).toarray()
X = onehotencoder.fit_transform(X.tolist())
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential #初始化神經網路
from keras.layers import Dense #新增神經網路層

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))

#神經元建議數量: 輸入維度與輸出維度之平均
#kernel_initializer:初始化權重 / uniform:隨機初始化(基本)

# Adding the second hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid')) 

#二元輸出:sigmoid

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#優化器(ptimizer):最小化Loss
#loss:binary_crossentropy(二種分類結果)/categorical_crossentropy(三種(含)以上分類結果)
#性能評估器(metrics):以acc進行評估

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100) 

#batch_size:批量學習之數據量，與效率有關
#batch_size與epochs目前尚無較佳建議值

# Part 3 - Making the predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)



