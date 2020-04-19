import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd

from config import *


# Plot the accuracy and cost summaries 
def plotTrainResults(accRecord, valAccRecord, costRecord, valCostRecord):
    f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10,4))
    ax1.plot(accRecord, label='acc')
    ax1.plot(valAccRecord, label='validation acc')
    ax1.set_title('Model Accuracy')
    ax2.plot(costRecord, label='cost')
    ax2.plot(valCostRecord, label='validation cost')
    ax2.set_title('Model Cost')
    plt.xlabel('Epochs')
    ax1.legend()
    ax2.legend()
    saveDir = '{}/summary.png'.format(outWeightDir)
    plt.savefig(saveDir)
    plt.show()
    print('\n ########## Summary plot has been saved in {}'.format(saveDir))


# Turn output of 2 nodes from the network into a single class
def getClassList(raw):
    outClass = []
    for i,row in enumerate(raw):
        if row[0] > row[1]:
            outClass.append(True)
        elif row[1] > row[0]:
            outClass.append(False)
        else:
            # print("error in row{}".format(i))
            # import pdb;pdb.set_trace()
            outClass.append(False)
    return  outClass

def getFeatureTestSetDf(featureTestSet,featureCols):
    procOutput = pd.DataFrame(data=featureTestSet,
                    index=range(0,len(featureTestSet)),
                    columns=featureCols)
    procOutput.to_csv(r'{}/before_{}'.format(outWeightDir,outCsvName), index = False)
    print('\n ########## featureTestSet has been saved in {}'.format(outWeightDir))
    return procOutput

### Deprecated function, not used anymore
def predictFromTestSet(x, y, featureTestSet, procOutput):
    getRisk = y
    rawOutput = getRisk.eval({x: featureTestSet})
    procOutput['Class'] = getClassList(rawOutput)
    return procOutput

def getComparison(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)): 
        if y_actual[i]==y_hat[i]==1:
            TP += 1
        if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
            FP += 1
        if y_actual[i]==y_hat[i]==0:
            TN += 1
        if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
            FN += 1

    print ("\nPerformance stats on test set"
        "\nSensitivity =", "{:.5f}".format(TP/(TP+FN)),
        "\nSpecificity =", "{:.5f}".format(TN/(TN+FP)), 
        "\nPrecision =", "{:.5f}".format(TP/(TP+FP)),
        "\nNegative predictive value =", "{:.5f}".format(TN/(TN+FN)),
        "\nFalse positive rate =", "{:.5f}".format(FP/(FP+TN)), 
        "\nFalse negative rate =", "{:.5f}".format(FN/(TP+FN)), 
        "\nFalse discovery rate = ", "{:.5f}".format(FP/(TP+FP)),
        "\nAccuracy = ", "{:.5f}".format((TP+TN)/(TP+FP+FN+TN)))
    
    # ### Write reults to a text file
    # text_file = open("test_performance.txt", "w")
    # text_file.write(result)
    # text_file.close()

    return [TP, FP, TN, FN]