# End-to-End ML Pipeline Documentation
#### Problem statement:
Using the dataset specified on page 7 (under the heading ‘Data’), design and create a simple
machine learning pipeline that will ingest/process the entailed dataset and feed it into a deep
learning model of your choice, returning metric outputs. You are expected to use either the
Pytorch or Tensorflow deep learning framework for this section. DO NOTuse a high level API such
as Keras or FastAI.

Rules:
- Structured as Python modules/classes with well-defined functions
- Relevant training/evaluation metric(s) outputs to be generated upon completion
- For Python, use only versions 3.6.7/3.6.8
- DO NOTinclude the data file in your submission

#### Theory:
##### Data processing
The dataset provided gave 29 features with 540198 rows, with a single boolean target attribute `Class`. This means that the network is supposed to have 29 nodes as its first input layer. The `Class` is split into 2 boolean attributes, `default` and `nonDefault` to allow 2 nodes which gives more precision in network prediction. This allows a feature to be related to defaulters and non defaulters. The data frames are then separated into defaults and non defaults, where a fraction  (`trainFraction=0.8`) from each group are taken and shuffled together for test set. similarly, the leftover data are split by half and shuffled to form validation set and test set for training. The training and validation data sets are also normalized before fed into the network for training to increase network efficiency.

The `Class` column is then reintroduced from the network output to for performance measurement of
the network.

##### Model/network design
There are 2 model designs explored in this project, tapered and custom models both with 4 hidden layers. Size of first and 4th hidden layer are fixed at 29 and 2 due to data constraints. The first model explored is tapered model, where each subsequent hidden layer is multiplied by a fixed ratio `m`. When `m` > 1 It results in a large cone-like tapered network shape, where next hidden layer is bigger than the previous. Model generally converges quickly in about 100 training epoch. When `m` < 1 It results in a small reverse cone-like tapered network shape, where next hidden layer is smaller than the previous. Model generally converges later in about 200 training epoch. In the last hidden layer, softmax activation function is used to allow for more varied result of the output node. 

Unfortunately model with larger hidden layer run the risk of having trivial mapping in the nodes, so model with smaller hidden layer like `m` < 1 is preferred over models with large middle layer, as smaller subsequent hiddel layer would allow the network allows feature extraction from input Nodes after training. Also, there's more noise in cost and accuracy in larger networks.

Using custom model constructor allows hidden layers of the model to be manually specified. It was found after experimentation that using 25,20,10 hidden layer sizes allows for faster and more accurate accuracy performance on test data set as compared to tapered model with `m` < 1.

The dataset provided gave 29 features with 540198 rows, with a single boolean target attribute `Class`. This means that the network is supposed to have 29 nodes as its first input layer. The `Class` is split into 2 boolean attributes, `default` and `nonDefault` to allow 2 nodes which gives more precision in network prediction. This allows a feature to be related to defaulters and non defaulters. 

##### Evaluation
Overall, larger model requires higher amount of train iteration using dropout while training allows the model to converge relatively quick, and slightly improves the netwrok's prediction performance. With the tapered model, the larger the `m` when `m >1`, accuracy increases but runs the risk of trivial mapping, and network training is considerably slower due to larger size of network.
Custom network with hidden layer 25,20,10 is by far the most ideal to use. 

During training, gradient vanishing/explosion could occur, resulting in the cost and Valcost to be `NaN` in such cases, early stopping is necessary, max training iteration `trainingEpochs` must be specified, and learning rate can be changed. 

#### Requirements:
Tried and tested with `Python 3.6.8` and following packages:
- numpy==1.18.2
- sklearn==0.0
- tensorflow==1.14.0
- tensorflow-estimator==1.14.0
- pandas==1.0.3
- matplotlib==3.2.1

#### Folder structure:
folder contains:
- `run_model.py` : main script contains functions to run, demo, or train model
- `config.py`: declares all the important parameters 
- `model_classes.py`: contains model definition and functions to train and predict model
- `preprocessor.py`:contains functions to preprocess data
- `postprocessor.py`: contains functions to postprocess data

All results (weight, output plot, csv of test data before and after processing is stored in directory `outWeightDir`, example is given in the file)
if `demo` is run, expect a beautiful graph of training summary `summary.png`, `before_predict.csv`, `after_predict.csv` in directory `outWeightDir`.

#### How to run:
VERY IMPORTANT: Before running, please open config.py to specify `outWeightDir`, `inData`, `inData`

Install requirements to run:
```
pip3 install -r requirements.txt 
```
Train (optimize) and execute model normally:
```
$ python3 run_model.py -m <MODEL> -a <ACTION>
```
`<MODEL>` and `<ACTION>` argument specifies the type of model to be run as well as the type of action to be done.

`<MODEL>` options: 
-  `tapered`: optimizes the model, saves it in directory `outWeightDir` and use model to predict test set data segmented from `inData`. Specify `multiplier` parameter in `config.py` to adjust network size.
- `custom`: Use custom model `multiplier` parameter in `config.py` to adjust network size. Specify `hidLayer1`, `hidLayer2`, `hidLayer3` parameter in `config.py` to adjust network size.

`<ACTION>` options: 
-  `demo`: optimizes the model, saves it in directory `outWeightDir` and use model to predict test set data segmented from `inData`.
- `train`: optimizes the model and then saves weight and summary plot in directory `outWeightDir`.
- `run`: optimizes the model, saves it in directory `outWeightDir` and use model to predict `runData`.
 
Train (optimize) and execute model with dropout:
```
$ python3 run_model.py -m <MODEL> -a <ACTION> -d True
```
Use the format above to run the script in order to allow drop out to occur in the third hidden layer of the model, in order to get a better model.

Specifiying `--inWeightDir` and the directory and name to saved model checkpoint allows you to open saved model, restore graph, and start training from the frozen checkpoint. Saved optimized model would be able to be directly used to predict data (specify epoch as low), saving time.


#### Appendix B: Experimental results:

###CUSTOM_ndrop (29,20,10)
- Performance stats on test set
- Sensitivity = 0.99992
- Specificity = 0.99905
- Precision = 0.99895
- Negative predictive value = 0.99993
- False positive rate = 0.00095
- False negative rate = 0.00008
- False discovery rate =  0.00105
- Accuracy= 0.99946
- epoch  = 1000
- 0.019558429718017578 ms


###CUSTOM_drop (29,20,10)
- Performance stats on test set
- Sensitivity = 0.99992
- Specificity = 0.99874
- Precision = 0.99859
- Negative predictive value = 0.99993
- False positive rate = 0.00126
- False negative rate = 0.00008
- False discovery rate =  0.00141
- Accuracy= 0.99930
- Epoch = 630
- 0.02557206153869629 ms

###TAPERED_ndrop_m3
- Performance stats on test set
- Sensitivity = 0.99984
- Specificity = 0.99675
- Precision = 0.99643
- Negative predictive value = 0.99986
- False positive rate = 0.00325
- False negative rate = 0.00016
- False discovery rate =  0.00357
- Accuracy =  0.99822
- 0.13595867156982422 ms
- epoch 50

###TAPERED_drop_m3
- Performance stats on test set
- Sensitivity = 1.00000
- Specificity = 0.99648
- Precision = 0.99611
- Negative predictive value = 1.00000
- False positive rate = 0.00352
- False negative rate = 0.00000
- False discovery rate =  0.00389
- Accuracy =  0.99815
- 0.32816100120544434 ms
- epoch 50


###TAPERED_ndrop_m0.5
- Performance stats on test set
- Sensitivity = 0.99937
- Specificity = 0.99678
- Precision = 0.99640
- Negative predictive value = 0.99944
- False positive rate = 0.00322
- False negative rate = 0.00063
- False discovery rate =  0.00360
- Accuracy =  0.99800
- 0.025522708892822266 ms
- epoch 100


###TAPERED_drop_m0.5
- Performance stats on test set
- Sensitivity = 0.99980
- Specificity = 0.99902
- Precision = 0.99891
- Negative predictive value = 0.99982
- False positive rate = 0.00098
- False negative rate = 0.00020
- False discovery rate =  0.00109
- Accuracy =  0.99939
- 0.02213001251220703 ms
- epoch: 185


###TAPERED_ndrop_m4
- Performance stats on test set
- Sensitivity = 0.99988
- Specificity = 0.99800
- Precision = 0.99777
- Negative predictive value = 0.99989
- False positive rate = 0.00200
- False negative rate = 0.00012
- False discovery rate =  0.00223
- Accuracy =  0.99889
- 0.2351241111755371 ms
- epoch: 164

###TAPERED_drop_m4


###CUSTOM_ndrop_fives
- Performance stats on test set
- Sensitivity = 0.99996
- Specificity = 0.99524
- Precision = 0.99476
- Negative predictive value = 0.99996
- False positive rate = 0.00476
- False negative rate = 0.00004
- False discovery rate =  0.00524
- Accuracy =  0.99748
- 0.023156404495239258 ms
- epoch = 115


###CUSTOM_drop_fives (25,20,10)
- Performance stats on test set
- Sensitivity = 0.99866
- Specificity = 0.99776
- Precision = 0.99749
- Negative predictive value = 0.99881
- False positive rate = 0.00224
- False negative rate = 0.00134
- False discovery rate =  0.00251
- Accuracy =  0.99819
- 0.03167462348937988 ms
- epoch = 151
