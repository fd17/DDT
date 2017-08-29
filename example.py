from DDT import Training
import time

trainer = Training.Trainer("example1.ini")

start = time.clock()
trainer.generate_inputs_normalized()
trainer.generate_model()
trainer.start_training()
print "elapsed time:", time.clock()-start
