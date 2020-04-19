### This file declares variables for training and execution of model

# Parameters

### Model parameters
modelName = 'saved_model'
## tapered model parameter
multiplier = 4 #Specify the fixed ratio for layer increase/decrease in tapered model
## tapered model parameter
hidLayer1 = 25 #Specify the node number in hidden layer of custom model
hidLayer2 = 20 #Specify the node number in hidden layer of custom model
hidLayer3 = 10 #Specify the node number in hidden layer of custom model


### Data read and save parameters
numInputFeatures = 29 #depends on the number of features input into the network
outWeightDir = '/mnt/d/Jul/school/NUS/aisg/TAPERED_drop_M4'
inWeightDir = None
inData = 'credit_card.csv'
runData = 'before_predict.csv'
outCsvName = 'predict.csv' 
# inWeightDir = '/mnt/d/Jul/school/NUS/aisg/saved_model_ndrop_292010_3/saved_original_2020-04-18-19-33-18.ckpt-115'

### Model training parameters
trainingDropout = 0.9
trainFraction= 0.8 #Take x fraction of dataset for training, split the rest by half for test and val set
trainingEpochs =1000 #Maximum number of epoch in training
saveEpoch = 10 #save every x epoch
stopEarlyEpoch = 15 #Early stopping after no improvement in x epoch
displayStep = 1 #Loss is displayed after x epoch
batchSize = 2048 
learningRate = 0.01




