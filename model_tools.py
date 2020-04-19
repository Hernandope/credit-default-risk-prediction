import argparse
import math
import os
import os.path as ops
import time
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import tensorflow as tf

from config import *
from preprocessor import dataDivider
from postprocessor import *

from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

def init_args():
    """
    To initialize arguments during execution
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--inData', type=str, help='The directory path of input data', default='credit_card.csv')
    parser.add_argument('--multiplier', type=int, help='Option maintains fixed ratio of nodes between layers, otherwise Optimal 29,20,10 is used', default=None)
    parser.add_argument('--name', type=str, help='Specify the name of the saved model', default='saved_model')
    parser.add_argument('--inWeightDir', type=str, help='The directory path of saved model checkpoint to restore', default=None)
    parser.add_argument('--dropout', type=bool, help='Option to include node dropout when training', default=False)
    # parser.add_argument('--action', type=str, help='Specify the action to be taken with the model', default=None)
    # parser.add_argument('--outWeightDir', type=str, help='The directory path of output model checkpoint')
    # parser.add_argument('--model', type=str, help='Specify the model type to use (tapered, custom)', default="custom")

    return parser.parse_args()

def main():
    ########### Read data ###########
    df = pd.read_csv(args.inData)

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
    if args.multiplier == None:
        hiddenNodes1 = numInputFeatures
        hiddenNodes2 = 20
        hiddenNodes3 = 10
        print('########## Selected custom network/model')
    else:
        hiddenNodes1 = numInputFeatures
        hiddenNodes2 = round(hiddenNodes1 * args.multiplier)
        hiddenNodes3 = round(hiddenNodes2 * args.multiplier)
        print('########## Selected tapered network/model')


    pkeep = tf.placeholder(tf.float32)

    # input
    x = tf.placeholder(tf.float32, [None, numInputFeatures])
    # hidden layer 1
    W1 = tf.Variable(tf.truncated_normal([numInputFeatures, hiddenNodes1],
                                         stddev = 0.1))
    b1 = tf.Variable(tf.zeros([hiddenNodes1]))
    y1 = tf.nn.sigmoid(tf.matmul(x, W1) + b1)
    # hidden layer 2
    W2 = tf.Variable(tf.truncated_normal([hiddenNodes1, hiddenNodes2],
                                         stddev = 0.1))
    b2 = tf.Variable(tf.zeros([hiddenNodes2]))
    y2 = tf.nn.sigmoid(tf.matmul(y1, W2) + b2)
    # hidden layer 3
    W3 = tf.Variable(tf.truncated_normal([hiddenNodes2, hiddenNodes3],
                                         stddev = 0.1)) 
    b3 = tf.Variable(tf.zeros([hiddenNodes3]))
    y3 = tf.nn.sigmoid(tf.matmul(y2, W3) + b3)

    # Add node dropout to training
    if args.dropout == True:
        y3 = tf.nn.dropout(y3, pkeep)
        print('########## Dropout will be incorporated to training, pkeep {}'\
                .format(str(trainingDropout)))
        
    # hidden layer 4
    W4 = tf.Variable(tf.truncated_normal([hiddenNodes3, 2], stddev = 0.1)) 
    b4 = tf.Variable(tf.zeros([2]))
    y4 = tf.nn.softmax(tf.matmul(y3, W4) + b4)
    # output
    y = y4
    y_ = tf.placeholder(tf.float32, [None, 2])
    # Cost function: Cross Entropy
    cost = -tf.reduce_sum(y_ * tf.log(y))
    # Model optimized with adamOptimizer
    optimizer = tf.train.AdamOptimizer(learningRate).minimize(cost)
    # Output node with highest softmax value is the correct class prediction
    correctPrediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correctPrediction, tf.float32))


    ########### Train the network/model ###########
    # Declare lists to record model performance summary
    accRecord = [] 
    costRecord = [] 
    valAccRecord = [] 
    valCostRecord = [] 
    stopEarlyCounter = 0 

    # Set tf saver
    saver = tf.train.Saver()
    if not ops.exists(outWeightDir):
        os.makedirs(outWeightDir)
    train_start_time = time.strftime('%Y-%m-%d-%H-%M-%S', 
                                    time.localtime(time.time()))
    modelName = '{}_{:s}.ckpt'.format(args.name,str(train_start_time))
    modelSavePath = ops.join(outWeightDir, modelName)

    # Initialize variables and tensorflow session
    sess = tf.InteractiveSession()
    tf.train.write_graph(graph_or_graph_def=sess.graph, logdir='',
                        name='{:s}/saved_model.pb'.format(outWeightDir))

    # Restore model to continue training from previous checkpoint if saved weights is given
    if inWeightDir is None:
        print('########## Training from scratch')
        init = tf.global_variables_initializer()
        sess.run(init)
    else:
        print('########## Restoring model from last model checkpoint {:s}'.format(inWeightDir))
        saver.restore(sess=sess, save_path=inWeightDir)

    for epoch in range(trainingEpochs): 
        for batch in range(int(numSample/batchSize)):
            # Take chunks of data to be trained
            batchFeature = featureTrain[batch*batchSize : (1+batch)*batchSize]
            batchTarget = targetTrain[batch*batchSize : (1+batch)*batchSize]
            sess.run([optimizer], feed_dict={x: batchFeature, 
                                            y_: batchTarget,
                                            pkeep: trainingDropout})

        # Print logs after every displayStep epochs
        if (epoch) % displayStep == 0:
            trainingAccuracy, newCost = sess.run([accuracy, cost], feed_dict={x: featureTrain, 
                                                                            y_: targetTrain,
                                                                            pkeep: trainingDropout})

            validationAccuracy, validNewCost = sess.run([accuracy, cost], feed_dict={x: featureTrainValid, 
                                                                                y_: targetTrainValid,
                                                                                pkeep: trainingDropout})

            print ("Epoch:", epoch,
                "Acc =", "{:.5f}".format(trainingAccuracy), 
                "Cost =", "{:.5f}".format(newCost),
                "ValAcc =", "{:.5f}".format(validationAccuracy), 
                "ValCost = ", "{:.5f}".format(validNewCost))

            # Record the results of the model
            accRecord.append(trainingAccuracy)
            costRecord.append(newCost)
            valAccRecord.append(validationAccuracy)
            valCostRecord.append(validNewCost)

            if epoch % saveEpoch == 0:
                saver.save(sess=sess, save_path=modelSavePath, global_step=epoch)

            # If the model does not improve after 15 epochs, training is stopped and save.
            if validationAccuracy < max(valAccRecord) and epoch > 100:
                stopEarlyCounter += 1
                if stopEarlyCounter == stopEarlyEpoch:
                    saver.save(sess=sess, save_path=modelSavePath, global_step=epoch)
                    break
            else:
                stopEarlyCounter = 0
    print('\n ########## Model training has Completed!\n')

    ### plot the accuracy and cost summaries 
    plotTrainResults(accRecord, valAccRecord, costRecord, valCostRecord)

    ########### Post-process data ###########
    featureCols = df.columns.values[:numInputFeatures]
    procOutput = getFeatureTestSetDf(featureTestSet, featureCols)

    ### predict and save result as csv
    procOutput = predictFromTestSet(x, y,featureTestSet, procOutput)
    procOutput.to_csv(r'{}/after_{}'.format(outWeightDir,outCsvName), index = False)
    
    ### find the output stats
    refOutput = getClassList(targetTestSet)
    comparisonStats = getComparison(refOutput,procOutput.Class)

    sess.close()

if __name__ == '__main__':
    # initialize arguments 
    args = init_args()
    main()