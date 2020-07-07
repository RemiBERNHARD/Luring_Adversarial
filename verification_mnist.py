import sys
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"

from keras.models import load_model
from keras.utils import np_utils
from keras.datasets import mnist
from keras import backend 
import tensorflow as tf
import numpy as np

from utils_func import metrics, clip_adv, clip_adv_l2, agree_func, comp_func, comp_func_transfer
from cleverhans.utils_keras import KerasModelWrapper
from cleverhans.attacks import FastGradientMethod, ProjectedGradientDescent, MomentumIterativeMethod



#Params FGSM
fgsm_params = {'eps': float(sys.argv[1]),
               'clip_min': 0.,
               'clip_max': 1.
               }
#PARAMS PGD
pgd_params = {'eps': float(sys.argv[1]),
              'eps_iter': 0.01,
              'nb_iter': 1000,
              'ord': np.inf,
               'clip_min': 0.,
               'clip_max': 1.,
               'rand_init': True
               }
#PARAMS MIM
mim_params = {'eps': float(sys.argv[1]),
              'eps_iter': 0.01,
              'nb_iter': 1000,
              'ord': np.inf,
               'clip_min': 0.,
               'clip_max': 1.
               }
#PARAMS MIML2
miml2_params = {'eps': 30.0,
              'eps_iter': 0.01,
              'nb_iter': 1000,
              'ord': 2,
               'clip_min': 0.,
               'clip_max': 1.         
               }    

    
    
#################################    
####Load data set####
#################################
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)

sess = tf.Session()
backend.set_session(sess)
backend._LEARNING_PHASE = tf.constant(0)
backend.set_learning_phase(0)



#################################    
####Build models####
#################################
model = load_model("models/MNIST_float.h5")
model_stacked = load_model("models/MNIST_stacked.h5")
model_auto = load_model("models/MNIST_auto.h5")
model_ce = load_model("models/MNIST_ce.h5")
model_rob = load_model("models/MNIST_luring.h5")

print("Performances of models on test set:")
print("model_base: "+ str(model.evaluate(X_test, Y_test, verbose=0)))
print("model_stacked: " + str(model_stacked.evaluate(X_test, Y_test, verbose=0)))
print("model_auto: " + str(model_auto.evaluate(X_test, Y_test, verbose=0)))
print("model_ce: " + str(model_ce.evaluate(X_test, Y_test, verbose=0)))
print("model_luring: " + str(model_rob.evaluate(X_test, Y_test, verbose=0)))


#Get common success indices
pred_base = np.argmax(model.predict(X_test), axis = 1)
pred_stacked = np.argmax(model_stacked.predict(X_test), axis = 1)
pred_auto = np.argmax(model_auto.predict(X_test), axis = 1)
pred_ce = np.argmax(model_ce.predict(X_test), axis = 1)
pred_rob = np.argmax(model_rob.predict(X_test), axis = 1)   

success_indices_base = np.equal(pred_base, y_test)
success_indices_stacked = np.equal(pred_stacked, y_test)           
success_indices_auto = np.equal(pred_auto, y_test)           
success_indices_ce = np.equal(pred_ce, y_test)           
success_indices_rob = np.equal(pred_rob, y_test)

success_indices = np.arange(0,len(X_test))[(success_indices_stacked == success_indices_auto) & 
                           (success_indices_auto == success_indices_ce) &
                           (success_indices_ce == success_indices_rob) &
                           (success_indices_rob == True) & (success_indices_base == success_indices_stacked)]

indices_test = np.random.choice(success_indices, 5000, replace=False)

c = 0
for i in range(len(X_test)):
    if (pred_base[i] == pred_stacked[i]) & (pred_stacked[i] == y_test[i]):
        c = c+1
print("Agreement test set stacked:" + str(c))

c = 0
for i in range(len(X_test)):
    if (pred_base[i] == pred_auto[i]) & (pred_auto[i] == y_test[i]):
        c = c+1
print("Agreement test set auto:" + str(c))

c = 0
for i in range(len(X_test)):
    if (pred_base[i] == pred_ce[i]) & (pred_ce[i] == y_test[i]):
        c = c+1
print("Agreement test set ce:" + str(c))

c = 0
for i in range(len(X_test)):
    if (pred_base[i] == pred_rob[i]) & (pred_rob[i] == y_test[i]):
        c = c+1
print("Agreement test set luring:" + str(c))    



#################################    
####Perform attacks####
#################################
wrap_stacked = KerasModelWrapper(model_stacked)
wrap_auto = KerasModelWrapper(model_auto)
wrap_ce = KerasModelWrapper(model_ce)
wrap_rob = KerasModelWrapper(model_rob)

####################################
    #FGSM  
print("\n\n")    
print("FGSM")    
fgsm_stacked = FastGradientMethod(wrap_stacked, sess=sess)
fgsm_auto = FastGradientMethod(wrap_auto, sess=sess)
fgsm_ce = FastGradientMethod(wrap_ce, sess=sess)
fgsm_rob = FastGradientMethod(wrap_rob, sess=sess)

X_adv_stacked = fgsm_stacked.generate_np(X_test[indices_test], **fgsm_params)
X_adv_auto = fgsm_auto.generate_np(X_test[indices_test], **fgsm_params)
X_adv_ce = fgsm_ce.generate_np(X_test[indices_test], **fgsm_params)
X_adv_rob = fgsm_rob.generate_np(X_test[indices_test], **fgsm_params)

comp_func(X_adv_stacked, X_adv_auto, X_adv_ce, X_adv_rob, indices_test, pred_base, pred_stacked, pred_auto, pred_ce, pred_rob)
#comp_func_transfer(X_adv_stacked, indices_test, pred_base, pred_stacked, model_stacked, model)
#comp_func_transfer(X_adv_auto, indices_test, pred_base, pred_auto, model_auto, model)
#comp_func_transfer(X_adv_ce, indices_test, pred_base, pred_ce, model_ce, model)
#comp_func_transfer(X_adv_rob, indices_test, pred_base, pred_rob, model_rob, model)



###################################
    #PGD  
print("\n\n")    
print("PGD")    
pgd_stacked = ProjectedGradientDescent(wrap_stacked, sess=sess)
pgd_auto = ProjectedGradientDescent(wrap_auto, sess=sess)
pgd_ce = ProjectedGradientDescent(wrap_ce, sess=sess)
pgd_rob = ProjectedGradientDescent(wrap_rob, sess=sess)

X_adv_stacked = pgd_stacked.generate_np(X_test[indices_test], **pgd_params)
X_adv_auto = pgd_auto.generate_np(X_test[indices_test], **pgd_params)
X_adv_ce = pgd_ce.generate_np(X_test[indices_test], **pgd_params)
X_adv_rob = pgd_rob.generate_np(X_test[indices_test], **pgd_params)

comp_func(X_adv_stacked, X_adv_auto, X_adv_ce, X_adv_rob, indices_test, pred_base, pred_stacked, pred_auto, pred_ce, pred_rob)
#comp_func_transfer(X_adv_stacked, indices_test, pred_base, pred_stacked, model_stacked, model)
#comp_func_transfer(X_adv_auto, indices_test, pred_base, pred_auto, model_auto, model)
#comp_func_transfer(X_adv_ce, indices_test, pred_base, pred_ce, model_ce, model)
#comp_func_transfer(X_adv_rob, indices_test, pred_base, pred_rob, model_rob, model)



####################################
    #MIM
print("\n\n")        
print("MIM")    
mim_stacked = MomentumIterativeMethod(wrap_stacked, sess=sess)
mim_auto = MomentumIterativeMethod(wrap_auto, sess=sess)
mim_ce = MomentumIterativeMethod(wrap_ce, sess=sess)
mim_rob = MomentumIterativeMethod(wrap_rob, sess=sess)

X_adv_stacked = mim_stacked.generate_np(X_test[indices_test], **mim_params)
X_adv_auto = mim_auto.generate_np(X_test[indices_test], **mim_params)
X_adv_ce = mim_ce.generate_np(X_test[indices_test], **mim_params)
X_adv_rob = mim_rob.generate_np(X_test[indices_test], **mim_params)

comp_func(X_adv_stacked, X_adv_auto, X_adv_ce, X_adv_rob, indices_test, pred_base, pred_stacked, pred_auto, pred_ce, pred_rob)
#comp_func_transfer(X_adv_stacked, indices_test, pred_base, pred_stacked, model_stacked, model)
#comp_func_transfer(X_adv_auto, indices_test, pred_base, pred_auto, model_auto, model)
#comp_func_transfer(X_adv_ce, indices_test, pred_base, pred_ce, model_ce, model)
#comp_func_transfer(X_adv_rob, indices_test, pred_base, pred_rob, model_rob, model)



####################################
    #MIML2
print("\n\n")        
print("MIMl2")    
mim_stacked = MomentumIterativeMethod(wrap_stacked, sess=sess)
mim_auto = MomentumIterativeMethod(wrap_auto, sess=sess)
mim_ce = MomentumIterativeMethod(wrap_ce, sess=sess)
mim_rob = MomentumIterativeMethod(wrap_rob, sess=sess)

X_adv_stacked = mim_stacked.generate_np(X_test[indices_test], **miml2_params)
X_adv_auto = mim_auto.generate_np(X_test[indices_test], **miml2_params)
X_adv_ce = mim_ce.generate_np(X_test[indices_test], **miml2_params)
X_adv_rob = mim_rob.generate_np(X_test[indices_test], **miml2_params)

X_adv_stacked = clip_adv(X_adv_stacked, X_test, indices_test, float(sys.argv[1]))
X_adv_auto = clip_adv(X_adv_auto, X_test, indices_test, float(sys.argv[1]))
X_adv_ce = clip_adv(X_adv_ce, X_test, indices_test, float(sys.argv[1]))
X_adv_rob = clip_adv(X_adv_rob, X_test, indices_test, float(sys.argv[1]))

comp_func(X_adv_stacked, X_adv_auto, X_adv_ce, X_adv_rob, indices_test, pred_base, pred_stacked, pred_auto, pred_ce, pred_rob)
#comp_func_transfer(X_adv_stacked, indices_test, pred_base, pred_stacked, model_stacked, model)
#comp_func_transfer(X_adv_auto, indices_test, pred_base, pred_auto, model_auto, model)
#comp_func_transfer(X_adv_ce, indices_test, pred_base, pred_ce, model_ce, model)
#comp_func_transfer(X_adv_rob, indices_test, pred_base, pred_rob, model_rob, model)



