import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from config import *

def dataDivider(df,trainFraction):
    # split Class to default and nonDefault.
    df.rename(columns = {'Class':'default'}, inplace=True) 
    df.loc[df.default == 0, 'nonDefault'] = 1
    df.loc[df.default == 1, 'nonDefault'] = 0
    # segment df to dataframe of only default and nonDefault observations.
    Default = df[df.default == 1]
    NonDefault = df[df.nonDefault == 1]
    # set trainSetFeature equal to trainFraction of the observations that defaulted.
    # concatenate majority fraction of the nondefaulted observations to trainSetFeature.
    trainSetFeature = Default.sample(frac=trainFraction)
    trainSetFeature = pd.concat([trainSetFeature, NonDefault.sample(frac = trainFraction)], axis = 0)
    # Put leftover data in testSetFeature.
    testSetFeature = df.loc[~df.index.isin(trainSetFeature.index)]
    # Shuffle the dataframes.
    trainSetFeature = shuffle(trainSetFeature)
    testSetFeature = shuffle(testSetFeature)
    # Add our target classes to trainSetTarget and testSetTarget.
    trainSetTarget = trainSetFeature.default
    trainSetTarget = pd.concat([trainSetTarget, trainSetFeature.nonDefault], axis=1)
    testSetTarget = testSetFeature.default
    testSetTarget = pd.concat([testSetTarget, testSetFeature.nonDefault], axis=1)
    # Drop target classes from trainSetFeature and testSetFeature.
    trainSetFeature = trainSetFeature.drop(['default','nonDefault'], axis = 1)
    testSetFeature = testSetFeature.drop(['default','nonDefault'], axis = 1)
    features = trainSetFeature.columns.values
    # Prepare the features to be fed into model by normalization
    for feature in features: 
        mean, std = df[feature].mean(), df[feature].std()
        trainSetFeature.loc[:, feature] = (trainSetFeature[feature] - mean) / std
        testSetFeature.loc[:, feature] = (testSetFeature[feature] - mean) / std
    div = int(len(testSetTarget)/2) # Divide testing set into validation and testing sets
    # Convert df into np array
    featureTrain = trainSetFeature.values
    targetTrain = trainSetTarget.values
    featureTrainValid = testSetFeature.values[:div]
    targetTrainValid = testSetTarget.values[:div]
    featureTestSet = testSetFeature.values[div:]
    targetTestSet = testSetTarget.values[div:]
    return [featureTrain, targetTrain, featureTrainValid, targetTrainValid, featureTestSet, targetTestSet]