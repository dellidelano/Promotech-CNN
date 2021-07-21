#Creating a class to handle the Convolution Neural Network
#Author: Delano Thomas
#Email; dmthomas@mun.ca
#COMP6999 - Researh Project
#June 5, 2021

#Setup the environment library
import pandas as pd
import numpy as np
import os

#CNN Library
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.python.eager.monitoring import Metric
from tensorflow.keras.regularizers import l1

#Create the CNN Class
class PromotechCNN:

    #Initialize the Module
    def __init__(self):

        self.model = keras.models.Model()
        self.homeDir = os.getcwd()

        #Directory for training data
        #self.trainDir = self.homeDir + "/TrainingData/40nt-sequences"
        self.trainDir = self.homeDir + ""
        self.ValidateDir = self.homeDir + ""

        #Fasta file name given main directorty
        self.trainNegativeFile = []
        self.trainPositiveFile = []

        #Train and Validate data
        self.trainData_X = []
        self.trainData_Y = []

        #Track the prediction input
        self.predictionData = []
        self.trainingData = []

        #Split training and test data
        self.X_train = []
        self.X_test = []
        self.y_train = []
        self.y_test = []

    #Convert the string into 4by40 numpy array
    def convertToOneHot(self, strData):

        data = np.array(strData)

        oneHotData = np.empty((40,4), dtype=np.uint8)
        pos =0

        for item in data:

            if(item == 'A'):
                itemList = list('1000')

            elif(item == 'C'):
                itemList = list('0100')

            elif(item == 'G'):
                itemList = list('0010')

            elif(item == 'T'):
                itemList = list('0001')

            else:
                return []

            oneHotData[pos] = itemList
            pos +=1

        return(oneHotData)


    #Convert 4by40 numpy array to sequence of ACGT
    def convertToSequence(self, data):

        outputStr = ""
        for code in data:

            strData = ""
            for item in code:
                strData += str(item)

            if(strData=="1000"):
                outputStr += "A"

            elif(strData=="0100"):
                outputStr += "C"

            elif(strData=="0010"):
                outputStr += "G"

            elif(strData=="0001"):
                outputStr += "T"

        return outputStr


    #Given a path, compile the list of Training fasta files
    def createTrainFastaFiles(self, trainPath):

        self.trainDir += trainPath
        trainFiles = os.listdir(self.trainDir)

        for fol in trainFiles:
            self.trainNegativeFile.append(self.trainDir + "/" + fol + "/" + "negative.fasta")
            self.trainPositiveFile.append(self.trainDir + "/" + fol + "/" + "positive.fasta")


    #Use list of fasta files and create arrays to be used for training and testing
    def makeDataArray(self, filename, val):

        Data_X = []
        Data_Y = []

        #Positive file
        with open(filename, 'r') as readFile:
            inputLines = readFile.readlines()

            #Go through the list and chose only tetra values
            for line in inputLines:

                #Check to be sure the line is valid
                if(line[0] != '>'):

                    #Clear list to remove any extra character
                    if(len(line) > 40):
                        line = line.strip('\n')
                        #print(line)

                        #Convert the Sequence to 4by40 one-hot array
                        line = self.convertToOneHot(list(line))

                        #Returned data is fine and has no error in the sequence
                        #If there was an error in the sequence then the return is empty
                        if(len(line)>0):
                            Data_X.append(line)
                            Data_Y.append(val)

        return [Data_X, Data_Y]


    #Build the training and validation data from the files supplied
    def buildData(self):

        #---Train Negative
        for file in self.trainNegativeFile:
            trainInfo = self.makeDataArray(file, 0)
            self.trainData_X.extend(trainInfo[0])
            self.trainData_Y.extend(trainInfo[1])
            print("Getting data from", file)

        #---Train Positive
        for file in self.trainPositiveFile:
            trainInfo = self.makeDataArray(file, 1)
            self.trainData_X.extend(trainInfo[0])
            self.trainData_Y.extend(trainInfo[1])
            print("Getting data from", file)

        print("Training data count", len(self.trainData_X))


    ###Build and train the BPNet CCN Network
    def buildBPNet(self, epoch):

        #Splitting the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(np.array(self.trainData_X), np.array(self.trainData_Y), test_size=0.25, random_state=42, shuffle=True, stratify=self.trainData_Y)

        ###Input Shape
        inputLayer = keras.layers.Input(shape=(40,4))

        ###Input Layer
        x = keras.layers.Conv1D(64, kernel_size=40, padding='same', activity_regularizer=l1(0.001))(inputLayer)
        x = keras.layers.LeakyReLU(alpha=0.2)(x)

        ###Nine dilated layers
        for i in range(1, 6):
            conv_x = keras.layers.Conv1D(64, kernel_size=3, padding='same', activity_regularizer=l1(0.001), dilation_rate=2**i)(x)
            conv_x = keras.layers.LeakyReLU(alpha=0.2)(conv_x)
            x =  keras.layers.add([conv_x, x])
        bottleneck = x

        #Final layer - Break from code
        cx = keras.layers.GlobalAveragePooling1D()(bottleneck)
        outputLayer = keras.layers.Dense(2, activation='softmax')(cx)

        ###Create the model
        self.model = keras.models.Model(inputLayer, outputLayer)

        #Compile the model
        self.model.compile(keras.optimizers.Adam(learning_rate=0.0001),
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                        metrics=['accuracy'])

        print("Printing the model summary:\\n", self.model.summary())

        #Callback for early stopping
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)

        #Training the model
        #history = self.model.fit(np.array(self.trainData_X), np.array(self.trainData_Y), epochs=epoch, verbose=0, shuffle=True, callbacks=[es])
        history = self.model.fit(self.X_train, self.y_train, epochs=epoch, verbose=0, validation_data=(self.X_test, self.y_test), shuffle=True, callbacks=[es])
        print(history.history)


    #Predict the test from training data
    def predictTestData(self, outputFile):

        #Define Softmax function for probability result
        probability_model = tf.keras.Sequential([self.model, keras.layers.Softmax()])

        #Make prediction for test data in training set
        predicted_action = probability_model.predict(self.X_test)

        #Write the prediction to file
        pos = 0
        with open(outputFile, 'w') as writeFile:
            writeFile.write("Seq,Base,Neg_C,Pos_C\n")
            for item in predicted_action:
                #choice = item.tolist().index(np.amax(item))
                writeFile.write(self.convertToSequence(self.X_test[pos]) + "," + str(self.y_test[pos]) + ","  + str(item[0]) + "," + str(item[1]) + "\n")
                pos += 1


    def predictData(self, inputFile):

        print("Making Predictions on", inputFile)

        inputDataConverted = []
        inputDataSequence = []

        #Read the contents of the input file and prepare the data
        with open(inputFile, 'r') as readFile:
            inputLines = readFile.readlines()

            #Go through the list and chose only tetra values
            for line in inputLines:

                #Check to be sure the line is valid
                if(line[0] != '>'):

                    #Clear list to remove any extra character
                    if(len(line) > 40):
                        line = line.strip('\n')

                        #Convert the Sequence to 4by40 one-hot array
                        lineConv = self.convertToOneHot(list(line))

                        #Returned data is fine and has no error in the sequence
                        #If there was an error in the sequence then the return is empty
                        if(len(line)>0):
                            inputDataConverted.append(lineConv)
                            inputDataSequence.append(line)

        predicted_action = self.model.predict(np.array(inputDataConverted))
        return [inputDataSequence, predicted_action]



        #Read input folder to gather input Fasta files
    def predictValidation(self, validatePath):

        self.ValidateDir += validatePath
        valFiles = os.listdir(self.ValidateDir)
        fastaFiles = []

        for fol in valFiles:
            fastaFiles.append(self.ValidateDir + "/" + fol + "/" + "negative.fasta")
            fastaFiles.append(self.ValidateDir + "/" + fol + "/" + "positive.fasta")

        #Predicting the files
        mainPredictionOutput = []

        print("Files to process\n:::", valFiles)

        #Read two files at a time, negative and positive fasta
        i = 0;
        while(i < len(fastaFiles)):
            
            fastaFile = valFiles[int(i/2)] + "_prediction.csv"
            
            prediciton1 = self.predictData(fastaFiles[i])
            prediciton2 = self.predictData(fastaFiles[i+1])
            i+=2

            sequence1 = prediciton1[0]
            sequence2 = prediciton2[0]
            pred1 = prediciton1[1]
            pred2 = prediciton2[1]

            with open(fastaFile, 'w') as writeFile:
                writeFile.write("Seq,Base,Neg_C,Pos_C,Pred\n")

                #Negative file
                pos = 0
                for item in pred1:
                    choice = item.tolist().index(np.amax(item))
                    writeFile.write(sequence1[pos] + ",0," + str(item[0]) + "," + str(item[1]) +  "," + str(choice) + "\n")
                    mainPredictionOutput.append(sequence1[pos] + ",0," + str(item[0]) + "," + str(item[1]) +  "," + str(choice) + "\n")
                    pos +=1

                #Positive file
                pos = 0
                for item in pred2:
                    choice = item.tolist().index(np.amax(item))
                    writeFile.write(sequence2[pos] + ",1," + str(item[0]) + "," + str(item[1]) +  "," + str(choice) + "\n")
                    mainPredictionOutput.append(sequence2[pos] + ",1," + str(item[0]) + "," + str(item[1]) +  "," + str(choice) + "\n")
                    pos +=1

        with open("All_bacteria_prediction.csv", 'w') as writeFile:
            writeFile.write("Seq,Base,Neg_C,Pos_C,Pred\n")
            for line in mainPredictionOutput:
                writeFile.write(line)
            

