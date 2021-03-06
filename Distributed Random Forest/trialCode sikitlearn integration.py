# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 23:59:40 2019

"""

from pyspark import SparkConf, SparkContext

import numpy as np
import random
from sklearn.model_selection import train_test_split

import time 
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier
import pandas as pd

def run(sc):
    def zero_matrix(n, m):
        return np.zeros(n*m, dtype = int).reshape(n, m)
	
    def vote_increment(y_est):	
        increment = zero_matrix(y_est.size, n_ys)
        increment[np.arange(y_est.size), y_est] = 1
        return increment # test point x class matrix with 1s marking the estimator prediction
    
    ##data cleaning part
    airlineData =  pd.read_csv("AirlineReduced")
    
    
    airlineData = airlineData.loc[:, ~airlineData.columns.isin(['DepTime',
                                                            'ArrTime', 
                                                            'CRSArrTime',
                                                            'CRSDepTime',
                                                            'ActualElapsedTime',
                                                            'ArrTimeInMins',
                                                            'ArrDelay'])]
    
    X_train, X_test, y_train, y_test = train_test_split(airlineData.loc[:, ~airlineData.columns.isin(['IsDelayed'])], airlineData['IsDelayed'], test_size=0.25, random_state=42)
    X_train, X_test, y_train, y_test = np.array(X_train),np.array(X_test),np.array(y_train),np.array(y_test)
    
    
    
    
    n_test = X_test.shape[0]
    n_ys = np.unique(y_train).size
    
    ###broad cast variables
    X_train_BC = sc.broadcast(X_train)
    X_test_BC = sc.broadcast(X_test)
    y_train_BC = sc.broadcast(y_train)
    y_test_BC = sc.broadcast(y_test)
    
    
    model = DecisionTreeClassifier()
	# Partition the training data into random sub-samples with replacement.
    listOfRanomIndexes = [random.sample(list(range(y_train.size)),int((2/3)*y_train.size)) for i in range(10)]
    samples = sc.parallelize(listOfRanomIndexes)
    #samples = sc.parallelize(Bootstrap(y.size))
	# Train a model for each sub-sample and apply it to the test data.
    vote_tally = samples.map(lambda index:model.fit(X_train_BC.value[index], y_train_BC.value[index]).predict(X_test_BC.value)).map(vote_increment).fold(zero_matrix(n_test, n_ys), np.add) # Take the learner majority vote.
    
    y_estimate_vote = np.argmax(vote_tally, axis = 1)
    return f1_score(y_test_BC.value, y_estimate_vote)

if __name__ == '__main__':
    start = time.time()
#    conf = (SparkConf().set("spark.driver.maxResultSize", "4g"))
    SparkContext.setSystemProperty("spark.driver.maxResultSize", "5g")
    print (run(SparkContext("local[*]", "Boost")))
    print("Time elapsed: ", time.time()- start)