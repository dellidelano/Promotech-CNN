#Main function to load module and make predictions
#Author: Delano Thomas
#Email; dmthomas@mun.ca
#COMP6999 - Researh Project
#July 7, 2021

#Setup the environment library
from PromotechCNN import PromotechCNN
import sys
from tensorflow import keras
from keras.models import load_model


###Get Input argument list
inputArg = sys.argv

###Parse input to ensure correct number of argument was entered
if len(inputArg) != 3:
    print("The number of arguments is not correct. \nThe format is: PromotechCNNPredict.py Promotech-CNN-20210720.h5 negative.fasta")

else:

    ###Process input arguments to get input file names
    h5_File,fasta_File=inputArg[1],inputArg[2]

    #Load the Origial Class to use functions
    code = PromotechCNN()

    #Load the module
    print("Loading the Model")
    code.model = load_model(h5_File)

    #summarize model.
    print("Model Summary----")
    code.model.summary()

    #Run prediction
    print("Making prediciton on fasta file", fasta_File)
    code.predictSequenceFile(fasta_File)


