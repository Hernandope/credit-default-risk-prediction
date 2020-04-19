import argparse
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import tensorflow as tf

from config import *
from preprocessor import dataDivider
from postprocessor import *
from model_classes import *

def init_args():
    """
    To initialize arguments during execution
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-m','--model', type=str, help='Specify the model type to use (tapered, custom)', required=True)
    parser.add_argument('-a','--action', type=str, help='Specify the action to be taken with the model (demo,train,run)', required=True)
    parser.add_argument('-d','--dropout', type=bool, help='Option to include node dropout when training/optimize', default=False)
    parser.add_argument('-iw''--inWeightDir', type=str, help='The directory path of saved model checkpoint to restore', default=None)
    # parser.add_argument('-i','--inData', type=str, help='The directory path of input data', default='credit_card.csv')
    # parser.add_argument('--multiplier', type=int, help='Option maintains fixed ratio of nodes between layers, otherwise Optimal 29,20,10 is used', default=None)
    # parser.add_argument('--outWeightDir', type=str, help='The directory path of output model checkpoint')
    # parser.add_argument('--name', type=str, help='Specify the name of the saved model', default='saved_model')
    # procOutput = predictFromTestSet(model.x, model.y, featureTestSet, procOutput)

    return parser.parse_args()

def main():

    ########### Read data ###########
    df = pd.read_csv(inData)

    ########### Pre-process Data ###########
    dataSet = dataDivider(df,trainFraction)

    featureTrain = dataSet[0]
    targetTrain = dataSet[1]
    featureTrainValid = dataSet[2]
    targetTrainValid = dataSet[3]
    featureTestSet = dataSet[4]
    targetTestSet = dataSet[5]
    numSample = targetTrain.shape[0]

    ########### Model Definition ###########
    # Define number of nodes in each hidden layer
    if args.model == 'custom':
        model = customModel(29,
                            learningRate,
                            args.dropout,
                            hidLayer1,
                            hidLayer2,
                            hidLayer3)

    elif args.model == 'tapered':
        model = taperedModel(29,
                            learningRate,
                            args.dropout,
                            multiplier)
    else:
        print("##### Error no model specified")

    ########### Quickly optimize the network/model ###########
    # Declare lists to record model performance summary
    accRecord = [] 
    costRecord = [] 
    valAccRecord = [] 
    valCostRecord = [] 
    stopEarlyCounter = 0 

    model.train(numSample, featureTrain, targetTrain, featureTrainValid, targetTrainValid)

    ### plot the accuracy and cost summaries 
    plotTrainResults(model.accRecord, model.valAccRecord, model.costRecord, model.valCostRecord)

    if args.action == 'demo':
        ########### Post-process data ###########

        ### predict and save result as csv
        procOutput = model.predict(featureTestSet, df)
        procOutput.to_csv(r'{}/after_{}'.format(outWeightDir,outCsvName), index = False)
        
        ### find the output stats
        refOutput = getClassList(targetTestSet)
        comparisonStats = getComparison(refOutput,procOutput.Class)

    elif args.action == 'run':
        ########### Read data ###########
        featureTestSet = pd.read_csv(runData)
        ########### Post-process data ###########

        ### predict and save result as csv
        procOutput = model.predict(featureTestSet, df)
        procOutput.to_csv(r'after_{}'.format(outCsvName), index = False)
        
        ### find the output stats
        refOutput = getClassList(targetTestSet)
        comparisonStats = getComparison(refOutput,procOutput.Class)

    model.sess.close()

if __name__ == '__main__':
    # initialize arguments 
    args = init_args()
    main()