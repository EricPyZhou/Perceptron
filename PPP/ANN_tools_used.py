import pandas as pd
import os
from sklearn.preprocessing import StandardScaler 
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix


fileDir = os.path.dirname(os.path.abspath(__file__))
print(fileDir)
trainseeds = fileDir + '\\trainSeeds.csv'
testseeds = fileDir + '\\testSeeds.csv'

df = pd.read_csv(trainseeds,header=None)
df_test = pd.read_csv(testseeds,header=None)

X_train = df.iloc[:,4:6]
y_train = df.iloc[:,-1]

X_test = df_test.iloc[:,4:6]
y_test = df_test.iloc[:,-1]

scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# 'hidden_layer_sizes= 5,5' means the network has two hidden layers and each of them contains 5 neurons
mlp = MLPClassifier(hidden_layer_sizes=(5,5), max_iter=2000)
mlp.fit(X_train,y_train.values.ravel())

# Predict the result based on the testSeeds
predictions = mlp.predict(X_test)
print(predictions)

# shows analytical report
print(classification_report(y_test,predictions))  