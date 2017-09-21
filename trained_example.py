from keras.models import Sequential, load_model
import numpy as np
model_path = "TrainingCheckpoints/DO_trained.hdf5"
model=load_model(model_path)

inputs = []
#input labels:
# first 3 values determine decay type
# (1,0,0) both leptoinc
# (0,1,0) mixed
# (0,0,1) both hadronic
# then
# p1pt, p1eta, p1phi, p1E, p1m
# p2pt, p2wta, p2phi, p2E, p2m
# MET
# phi_MET

# example inputs for 125 GeV Higgs decay
line = (0,1,0,70.699005127,0.943030714989,-1.73818576336,12.432457617,0.000510999991093,28.8855628967,0.000565979687963,0.906305015087,5.60208343102,1.58043241501,37.1358261108,-1.37983822823)
inputs.append(line)
inputs = np.array(inputs) 

# feed inputs to model   
model_prediction = model.predict(inputs)

# adjust model overestimation
adjustment for overestimation
for i in xrange(len(model_prediction)):
    model_prediction[i] -= 115 - 0.60 * 125

# print estimated mass
print model_prediction