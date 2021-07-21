# Promotech-CNN
Bacteria Promoter Prediction Tool using Convolution Neural Networks. This tool was built using Python 3.8.2.

The code uses a single class that is able to read fasta files to retrieve bacteria sequences for processing.
The architecture of the Convolution Neural Network was guided by the BPNet architecture found at https://www.nature.com/articles/s41588-021-00782-6, however, this solution does not use as many layers and have several tweeks in architecture.


You may run the code from scratch using the following instructions or import the H5 file and start predicting. The output from the predicitons are written to the execution directory.

#########################Code for setting up and testing ###############################
from PromotechCNN import PromotechCNN
code = PromotechCNN()

print("Setting up the data...")
code.createTrainFastaFiles("/TrainingData/40nt-sequences/bacteria-1-10-ratio")
code.buildData()

print("------RUNNING EPOCH 100------")
code.buildBPNet(100)

print("Running Predictions...")
code.predictValidation("/ValidationData/40nt-sequences/bacteria-1-1-ratio")

###########################################################################################
