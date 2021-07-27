This Project is Bacteria Promoter Prediction Tool built using Convolution Neural Networks.

The body of the code was built using a Class module that read fasta files to retrieve bacteria sequences. Once read, the sequences are converted to one-hot-coding and passed to the CNN for building and training. Our CNN architecture was guided by the BPNet architecture found at https://www.nature.com/articles/s41588-021-00782-6, however, this solution does not use as many layers and have several tweeks in architecture.
A link to the report for this particular architecture will be shared at a later date.

Environment:

	-- Operating System: Linux
	-- Python: 3.8.2
	-- tensorflow: 2.5.0
	-- numpy: 1.2.0
	-- pandas: 1.2.4


About the Files:

	-- Promotech-CNN-20210720.h5 - H5 module built from the trained network. This is loaded during predictions
	-- PromotechCNN.py	- Base Class module that is used to build and train a netwok if required. It handles promoter sequence conversion before they are fed into the Neural Network. The main prediction method is processed here.
	-- PromotechCNNPedict.py - Main file to load CNN module and run predictions on the input file

Making Predictions:
- Predictions are made after loading the H5 data file and then running the prediction on an input promoter sequence file.

		-- You may run predictions on fasta files or files with only one 40nt sequence per line.
			python PromotechCNNPredict.py Promotech-CNN-20210720.h5 negative.fasta


Training the Network from Scratch:
- The model was built for training on a Linux environment, you may change line 115 and 116 to run on Windows - the '/' should be switched to '\\'.
- If you intend to rebuild the network from scratch, you may run the commands below - this is ideal for new training data.

		-- Import the Python Class.
			from PromotechCNN import PromotechCNN
			code = PromotechCNN()
		
		-- Setup the training data(Path to directory with training data is used).
			code.createTrainFastaFiles("/TrainingData/40nt-sequences/bacteria-1-10-ratio")
			code.buildData()

		-- Build and Train the network with 100 epochs, early stopping is in place so the network might stop before 100 epochs.
			code.buildBPNet(100)

		-- Predict the validation data. Used to build AUPRC and AUROC graphs
			code.predictValidation("/ValidationData/40nt-sequences/bacteria-1-1-ratio")

		-- Run prediction on some promoter sequence.
			code.predictSequenceFile("BACILLUS_negative.fasta")

