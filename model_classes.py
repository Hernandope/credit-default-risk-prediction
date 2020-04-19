import os
import os.path as ops
import time
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np 
import tensorflow as tf

from config import *
from postprocessor import *

class customModel:
    def __init__(self, inputNodes, learningRate, dropout, hiddenNodes1, hiddenNodes2, hiddenNodes3):
        self.inputNodes = inputNodes
        self.learningRate = learningRate
        self.dropout = dropout
        self.hiddenNodes1 = hiddenNodes1
        self.hiddenNodes2 = hiddenNodes2
        self.hiddenNodes3 = hiddenNodes3
        self.pkeep = tf.placeholder(tf.float32)
        # input
        self.x = tf.placeholder(tf.float32, [None, self.inputNodes])
        # hidden layer 1
        self.W1 = tf.Variable(tf.truncated_normal([self.inputNodes, self.hiddenNodes1],
                                            stddev = 0.1))
        self.b1 = tf.Variable(tf.zeros([self.hiddenNodes1]))
        self.y1 = tf.nn.sigmoid(tf.matmul(self.x, self.W1) + self.b1)
        # hidden layer 2
        self.W2 = tf.Variable(tf.truncated_normal([self.hiddenNodes1, self.hiddenNodes2],
                                            stddev = 0.1))
        self.b2 = tf.Variable(tf.zeros([self.hiddenNodes2]))
        self.y2 = tf.nn.sigmoid(tf.matmul(self.y1, self.W2) + self.b2)
        # hidden layer 3
        self.W3 = tf.Variable(tf.truncated_normal([self.hiddenNodes2, self.hiddenNodes3],
                                            stddev = 0.1)) 
        self.b3 = tf.Variable(tf.zeros([self.hiddenNodes3]))
        self.y3 = tf.nn.sigmoid(tf.matmul(self.y2, self.W3) + self.b3)
        # Add node dropout to training
        if self.dropout == True:
            self.y3 = tf.nn.dropout(self.y3, self.pkeep)
            print('########## Dropout will be incorporated to training, pkeep {}'\
                    .format(str(trainingDropout)))
        # hidden layer 4
        self.W4 = tf.Variable(tf.truncated_normal([self.hiddenNodes3, 2], stddev = 0.1)) 
        self.b4 = tf.Variable(tf.zeros([2]))
        self.y4 = tf.nn.softmax(tf.matmul(self.y3, self.W4) + self.b4)
        # output
        self.y = self.y4
        self.y_ = tf.placeholder(tf.float32, [None, 2])
        # Cost function: Cross Entropy
        self.cost = -tf.reduce_sum(self.y_ * tf.log(self.y))
        # Model optimized with adamOptimizer
        self.optimizer = tf.train.AdamOptimizer(self.learningRate).minimize(self.cost)
        # Output node with highest softmax value is the correct class prediction
        self.correctPrediction = tf.equal(tf.argmax(self.y,1), tf.argmax(self.y_,1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correctPrediction, tf.float32))
    
    def train(self, numSample, featureTrain, targetTrain, featureTrainValid, targetTrainValid):
        # Declare lists to record model performance summary
        self.accRecord = [] 
        self.costRecord = [] 
        self.valAccRecord = [] 
        self.valCostRecord = [] 
        self.stopEarlyCounter = 0 
        
        # Set tf saver
        self.saver = tf.train.Saver()
        if not ops.exists(outWeightDir):
            os.makedirs(outWeightDir)
        self.train_start_time = time.strftime('%Y-%m-%d-%H-%M-%S', 
                                        time.localtime(time.time()))
        self.modelName = '{}_{:s}.ckpt'.format(modelName,str(self.train_start_time))
        self.modelSavePath = ops.join(outWeightDir, self.modelName)

        # Initialize variables and tensorflow session
        self.sess = tf.InteractiveSession()
        tf.train.write_graph(graph_or_graph_def=self.sess.graph, logdir='',
                            name='{:s}/saved_model.pb'.format(outWeightDir))

        # Restore model to continue training from previous checkpoint if saved weights is given
        if inWeightDir is None:
            print('########## Training from scratch')
            self.sess.run(tf.global_variables_initializer())
        else:
            print('########## Restoring model from last model checkpoint {:s}'.format(inWeightDir))
            self.saver.restore(sess=self.sess, save_path=inWeightDir)

        for epoch in range(trainingEpochs): 
            for batch in range(int(numSample/batchSize)):
                # Take chunks of data to be trained
                batchFeature = featureTrain[batch*batchSize : (1+batch)*batchSize]
                batchTarget = targetTrain[batch*batchSize : (1+batch)*batchSize]
                self.sess.run([self.optimizer], feed_dict={self.x: batchFeature, 
                                                            self.y_: batchTarget,
                                                            self.pkeep: trainingDropout})

            # Print logs after every displayStep epochs
            if (epoch) % displayStep == 0:
                self.trainingAccuracy, self.newCost = self.sess.run([self.accuracy, self.cost], feed_dict={self.x: featureTrain, 
                                                                                            self.y_: targetTrain,
                                                                                            self.pkeep: trainingDropout})

                self.validationAccuracy, self.validNewCost = self.sess.run([self.accuracy, self.cost], feed_dict={self.x: featureTrainValid, 
                                                                                                    self.y_: targetTrainValid,
                                                                                                    self.pkeep: trainingDropout})

                print ("Epoch:", epoch,
                    "Acc =", "{:.5f}".format(self.trainingAccuracy), 
                    "Cost =", "{:.5f}".format(self.newCost),
                    "ValAcc =", "{:.5f}".format(self.validationAccuracy), 
                    "ValCost = ", "{:.5f}".format(self.validNewCost))

                # Record the results of the model
                self.accRecord.append(self.trainingAccuracy)
                self.costRecord.append(self.newCost)
                self.valAccRecord.append(self.validationAccuracy)
                self.valCostRecord.append(self.validNewCost)

                if epoch % saveEpoch == 0:
                    self.saver.save(sess=self.sess, save_path=self.modelSavePath, global_step=epoch)

                # If the model does not improve after 15 epochs, training is stopped and save.
                if self.validationAccuracy < max(self.valAccRecord) and epoch > 100:
                    self.stopEarlyCounter += 1
                    if self.stopEarlyCounter == stopEarlyEpoch:
                        self.saver.save(sess=self.sess, save_path=self.modelSavePath, global_step=epoch)
                        break
                else:
                    self.stopEarlyCounter = 0
        print('\n ########## Model training has Completed!\n')

    def predict(self, featureTestSet, df):
        self.featureCols = df.columns.values[:numInputFeatures]
        self.procOutput = getFeatureTestSetDf(featureTestSet, self.featureCols)
        ### Model inference
        self.start = time.time()
        self.getRisk = self.y
        self.rawOutput = self.getRisk.eval({self.x: featureTestSet, self.pkeep: 1})  
        self.timeElapsed = time.time() - self.start
        ### add columns to output data frame
        self.procOutput['Class'] = getClassList(self.rawOutput)
        print('\n ########## Model inference completed in {} ms\n'.format(self.timeElapsed))
        return self.procOutput        


class taperedModel:
    """
    The tapered model has its subsequent layers multiplied by a fixed ratio
    """
    def __init__(self, inputNodes, learningRate, dropout, multiplier):
        self.inputNodes = inputNodes
        self.learningRate = learningRate
        self.dropout = dropout
        self.multiplier = multiplier
        self.hiddenNodes1 = inputNodes
        self.hiddenNodes2 = round(self.hiddenNodes1 * self.multiplier)
        self.hiddenNodes3 = round(self.hiddenNodes2 * self.multiplier)
        self.pkeep = tf.placeholder(tf.float32)
        # input
        self.x = tf.placeholder(tf.float32, [None, self.inputNodes])
        # hidden layer 1
        self.W1 = tf.Variable(tf.truncated_normal([self.inputNodes, self.hiddenNodes1],
                                            stddev = 0.1))
        self.b1 = tf.Variable(tf.zeros([self.hiddenNodes1]))
        self.y1 = tf.nn.sigmoid(tf.matmul(self.x, self.W1) + self.b1)
        # hidden layer 2
        self.W2 = tf.Variable(tf.truncated_normal([self.hiddenNodes1, self.hiddenNodes2],
                                            stddev = 0.1))
        self.b2 = tf.Variable(tf.zeros([self.hiddenNodes2]))
        self.y2 = tf.nn.sigmoid(tf.matmul(self.y1, self.W2) + self.b2)
        # hidden layer 3
        self.W3 = tf.Variable(tf.truncated_normal([self.hiddenNodes2, self.hiddenNodes3],
                                            stddev = 0.1)) 
        self.b3 = tf.Variable(tf.zeros([self.hiddenNodes3]))
        self.y3 = tf.nn.sigmoid(tf.matmul(self.y2, self.W3) + self.b3)
        # Add node dropout to training
        if self.dropout == True:
            self.y3 = tf.nn.dropout(self.y3, self.pkeep)
            print('########## Dropout will be incorporated to training, pkeep {}'\
                    .format(str(trainingDropout)))
        # hidden layer 4
        self.W4 = tf.Variable(tf.truncated_normal([self.hiddenNodes3, 2], stddev = 0.1)) 
        self.b4 = tf.Variable(tf.zeros([2]))
        self.y4 = tf.nn.softmax(tf.matmul(self.y3, self.W4) + self.b4)
        # output
        self.y = self.y4
        self.y_ = tf.placeholder(tf.float32, [None, 2])
        # Cost function: Cross Entropy
        self.cost = -tf.reduce_sum(self.y_ * tf.log(self.y))
        # Model optimized with adamOptimizer
        self.optimizer = tf.train.AdamOptimizer(self.learningRate).minimize(self.cost)
        # Output node with highest softmax value is the correct class prediction
        self.correctPrediction = tf.equal(tf.argmax(self.y,1), tf.argmax(self.y_,1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correctPrediction, tf.float32))
    
    def train(self, numSample, featureTrain, targetTrain, featureTrainValid, targetTrainValid):
        # Declare lists to record model performance summary
        self.accRecord = [] 
        self.costRecord = [] 
        self.valAccRecord = [] 
        self.valCostRecord = [] 
        self.stopEarlyCounter = 0 
        
        # Set tf saver
        self.saver = tf.train.Saver()
        if not ops.exists(outWeightDir):
            os.makedirs(outWeightDir)
        self.train_start_time = time.strftime('%Y-%m-%d-%H-%M-%S', 
                                        time.localtime(time.time()))
        self.modelName = '{}_{:s}.ckpt'.format(modelName,str(self.train_start_time))
        self.modelSavePath = ops.join(outWeightDir, self.modelName)

        # Initialize variables and tensorflow session
        self.sess = tf.InteractiveSession()
        tf.train.write_graph(graph_or_graph_def=self.sess.graph, logdir='',
                            name='{:s}/saved_model.pb'.format(outWeightDir))

        # Restore model to continue training from previous checkpoint if saved weights is given
        if inWeightDir is None:
            print('########## Training from scratch')
            self.sess.run(tf.global_variables_initializer())
        else:
            print('########## Restoring model from last model checkpoint {:s}'.format(inWeightDir))
            self.saver.restore(sess=self.sess, save_path=inWeightDir)

        for epoch in range(trainingEpochs): 
            for batch in range(int(numSample/batchSize)):
                # Take chunks of data to be trained
                batchFeature = featureTrain[batch*batchSize : (1+batch)*batchSize]
                batchTarget = targetTrain[batch*batchSize : (1+batch)*batchSize]
                self.sess.run([self.optimizer], feed_dict={self.x: batchFeature, 
                                                            self.y_: batchTarget,
                                                            self.pkeep: trainingDropout})

            # Print logs after every displayStep epochs
            if (epoch) % displayStep == 0:
                self.trainingAccuracy, self.newCost = self.sess.run([self.accuracy, self.cost], feed_dict={self.x: featureTrain, 
                                                                                            self.y_: targetTrain,
                                                                                            self.pkeep: trainingDropout})

                self.validationAccuracy, self.validNewCost = self.sess.run([self.accuracy, self.cost], feed_dict={self.x: featureTrainValid, 
                                                                                                    self.y_: targetTrainValid,
                                                                                                    self.pkeep: trainingDropout})

                print ("Epoch:", epoch,
                    "Acc =", "{:.5f}".format(self.trainingAccuracy), 
                    "Cost =", "{:.5f}".format(self.newCost),
                    "ValAcc =", "{:.5f}".format(self.validationAccuracy), 
                    "ValCost = ", "{:.5f}".format(self.validNewCost))

                # Record the results of the model
                self.accRecord.append(self.trainingAccuracy)
                self.costRecord.append(self.newCost)
                self.valAccRecord.append(self.validationAccuracy)
                self.valCostRecord.append(self.validNewCost)

                if epoch % saveEpoch == 0:
                    self.saver.save(sess=self.sess, save_path=self.modelSavePath, global_step=epoch)

                # If the model does not improve after 15 epochs, training is stopped and save.
                if self.validationAccuracy < max(self.valAccRecord) and epoch > 100:
                    self.stopEarlyCounter += 1
                    if self.stopEarlyCounter == stopEarlyEpoch:
                        self.saver.save(sess=self.sess, save_path=self.modelSavePath, global_step=epoch)
                        break
                else:
                    self.stopEarlyCounter = 0
        print('\n ########## Model training has Completed!\n')

    def predict(self, featureTestSet, df):
        self.featureCols = df.columns.values[:numInputFeatures]
        self.procOutput = getFeatureTestSetDf(featureTestSet, self.featureCols)
        ### Model inference
        self.start = time.time()
        self.getRisk = self.y
        self.rawOutput = self.getRisk.eval({self.x: featureTestSet, self.pkeep: 1})  
        self.timeElapsed = time.time() - self.start
        ### add columns to output data frame
        self.procOutput['Class'] = getClassList(self.rawOutput)
        print('\n ########## Model inference completed in {} ms\n'.format(self.timeElapsed))
        return self.procOutput        