import os

def makedir(directory):
	try:
		os.makedirs(directory)
	except OSError as e:			
		#print e
		print "Error making directories."
		#raise

def listdir_fullpath(d):
	return [os.path.join(d, f) for f in os.listdir(d)]

def save_history(history, path):
	ofile = open(path,"w+")
	for line in history.history['loss']:
	    ofile.write(str(line)+"\n")
	ofile.close()