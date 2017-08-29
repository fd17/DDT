import configparser
from ddt_utils import makedir, listdir_fullpath, save_history
from keras.models import Sequential, load_model
from keras.losses import mean_squared_error
from keras.layers import Dense, Activation, Dropout
from keras.callbacks import ModelCheckpoint
from keras.constraints import max_norm
from keras import backend as K

class Trainer:

	tr_inputPath = ""
	tr_outputPath = ""

	inputs = []
	outputs = []

	epochs = 10
	batch_size = 128

	input_dim = 0	
	neurons = 0
	layers = 0
	init = ""
	activation = ""

	lr = False
	momentum = False
	decay = False

	DO_reg = False
	DO_val = 0.0

	model = Sequential()

	def __init__(self, inipath = None):

		configpath = "example.ini"
		if(inipath):
			configpath = inipath


		config=configparser.ConfigParser()
		config.read(configpath)

		self.tr_inputPath = config["Paths"].get("tr_inputPath")
		self.tr_outputPath = config["Paths"].get("tr_outputPath")
		self.epochs = int(config["Training"].get("epochs"))
		self.batch_size = int(config["Training"].get("batchsize"))
		self.model_loss = str(config["Training"].get("loss_function"))
		self.model_optimizer = str(config["Training"].get("optimizer"))
		self.layers = int(config["Network"].get("layers"))
		self.neurons = int(config["Network"].get("neurons"))
		self.init = str(config["Network"].get("init"))
		self.activation = str(config["Network"].get("activation"))
		self.DO_reg = config["Network"].getboolean("DO_regularization")
		self.DO_val = float(config["Network"].get("DO_percentage"))

		lr = config["Training"].get("learningrate")
		decay = config["Training"].get("decay")
		momentum = config["Training"].get("momentum")
		if lr:
			self.lr = float(lr)
		if decay:
			self.decay = float(decay)
		if momentum:
			self.momentum = float(momentum)

		makedir(self.tr_outputPath)



	def generate_model(self):
		layers = self.layers
		neurons = self.neurons
		if layers<1 or neurons < 1:
			print "Error: Check number of neurons and layers"
			return
		if len(self.inputs) == 0:
			print "Error: No Inputs yet"

		model = Sequential()
		init = self.init
		act = self.activation
		in_dim = self.input_dim
		do_val = self.DO_val
		# add first layer
		if self.DO_reg:
			model.add(Dropout(do_val, input_shape=(in_dim,)))
			model.add(Dense(neurons, kernel_initializer=init, activation=act))
			model.add(Dropout(do_val))

		else:
			model.add(Dense(neurons,input_dim=in_dim, kernel_initializer=init, activation=act))

		# add remaining hidden layers
		for i in xrange(layers-1):
			model.add(Dense(neurons, kernel_initializer=init, activation=act))
			if(self.DO_reg):
				model.add(Dropout(do_val))

		#add output layer
		model.add(Dense(1, kernel_initializer = init, activation = act))
		mloss = self.model_loss
		optimizer = self.model_optimizer
		model.compile(loss=mloss,optimizer=optimizer)

		if self.lr:
			K.set_value(model.optimizer.lr, self.lr)
		if self.momentum:
			K.set_value(model.optimizer.momentum, self.momentum)
		if self.decay:
			K.set_value(model.optimizer.decay, self.decay)

		self.model = model

	def generate_inputs(self):
		files = listdir_fullpath(self.tr_inputPath)
		print "found %d file(s) with training data" % len(files)
		entries = []
		for eventFile in files:
			nlines = 0
			with open(eventFile, "r") as infile: 
				for line in infile:
					nlines+=1
					linelist = line.split(",")[0:-1]
					floats   = map(float, linelist)
					input   = floats
					realmass  = float(line.split(",")[-1])
					self.inputs.append(input)
					self.outputs.append(realmass)
			entries.append(nlines)
		self.input_dim = len(self.inputs[0])
		return entries

	def generate_inputs_normalized(self):
		maxEntries = min(self.generate_inputs())
		self.inputs = []
		self.outputs = []
		files = listdir_fullpath(self.tr_inputPath)
		for eventFile in files:
			nlines = 0
			with open(eventFile, "r") as infile: 
				for line in infile:
					if nlines > maxEntries:
						break
					nlines+=1
					linelist = line.split(",")[0:-1]
					floats   = map(float, linelist)
					input   = floats
					realmass  = float(line.split(",")[-1])
					self.inputs.append(input)
					self.outputs.append(realmass)

	def n_trainingEvents(self):
		return len(self.outputs)


	def start_training(self):
		cppath = self.tr_outputPath+"/{epoch:02d}.hdf5"
		print "Saving training  checkpoints to"+self.tr_outputPath+"/{epoch}.hdf5"
		epochs = self.epochs
		batch_size = self.batch_size
		cb = ModelCheckpoint(cppath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
		history = self.model.fit(self.inputs,self.outputs, batch_size=batch_size, epochs = epochs, verbose=1, callbacks=[cb])
		self.model.save(self.tr_outputPath+"/"+str(epochs)+"hdf5")
		hist_path = self.tr_outputPath+"/history.txt"
		save_history(history,hist_path)

