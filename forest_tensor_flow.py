# -*- coding: utf-8 -*-
"""Forest Tensor Flow .ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1dWTn9Rkg-v99FjkIms_6vg6elD-nKJ8m
"""

import tensorflow as tf
import numpy as np
import decisionTree
import random
from scipy import stats

from sklearn.metrics import f1_score

numTrees = 10
maxDepth = 10


airlineData =  pd.read_csv("AirlineReduced")
    
airlineData = airlineData.loc[:, ~airlineData.columns.isin(['DepTime',
                                                        'ArrTime', 
                                                        'CRSArrTime',
                                                        'CRSDepTime',
                                                        'ActualElapsedTime',
                                                        'ArrTimeInMins',
                                                        'ArrDelay'])]
    
X_train, X_test, y_train, y_test = train_test_split(airlineData.loc[:, ~airlineData.columns.isin(['IsDelayed'])], airlineData['IsDelayed'], test_size=0.75, random_state=42)
xTrain, xTest, yTrain, yTest = np.array(X_train),np.array(X_test),np.array(y_train),np.array(y_test)

currSess = tf.InteractiveSession()
import random

def splitData(xTrain,yTrain,numTrees):
  listOfRandomIndexes = [random.sample(list(range(yTrain.size)),int((2/3)*yTrain.size)) for i in range(numTrees)]
  return listOfRandomIndexes


def tree_fit_predict(xTrain,yTrain,xTest,index):
  model = decisionTree.DecisionTree(maxDepth = maxDepth,verbose = True)
#   new = tf.gather(xTrain,index)
#   print(new)
# #   index = np.array(index)
# #   index = index.astype(int)
# #   print(index.dtype)
  result  =  model.fit(xTrain[index],yTrain[index]).predict(xTest)
  return result

indices = splitData(xTrain,yTrain,numTrees)

xTrainTensor = tf.placeholder(tf.float32)
yTrainTensor = tf.placeholder(tf.float32)
xTestTensor = tf.placeholder(tf.float32)
indexTensor = tf.placeholder(tf.int32)


inputTensor = [xTrainTensor,yTrainTensor,xTestTensor,indexTensor]
tree_fit_predict_tensor = tf.py_func(tree_fit_predict, inputTensor, tf.float32)

def predFromForest(numTrees):
  AllPreds = []
  for i in range(numTrees):
    eachTreePreds = tree_fit_predict_tensor.eval(feed_dict = {xTrainTensor:xTrain,yTrainTensor:yTrain,xTestTensor:xTest,indexTensor:indices[i]})
    AllPreds.append(eachTreePreds)

  AllPreds = np.array(AllPreds).T
  MajorityPreds = stats.mode(AllPreds,axis = 1)
  return MajorityPreds[0]

yPreds = predFromForest(numTrees)
accuracy = f1_score(yTest, yPreds)

print(accuracy)