import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"

from keras.models import Model, load_model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Activation, BatchNormalization, Dropout, Reshape, UpSampling2D
from keras.initializers import Constant
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras import backend 
import tensorflow as tf
import numpy as np
from datetime import datetime

from utils_func import metrics, clip_adv, clip_adv_l2
from cleverhans.utils_keras import KerasModelWrapper
from cleverhans.attacks import FastGradientMethod
from cleverhans.attacks import ProjectedGradientDescent 
from cleverhans.attacks import MomentumIterativeMethod


#Params FGSM
fgsm_params = {'eps': float(sys.argv[1]),
               'clip_min': 0.,
               'clip_max': 1.,
               }
#PARAMS PGD
pgd_params = {'eps': float(sys.argv[1]),
              'eps_iter': 0.01,
               'nb_iter': 1000,
              'ord': np.inf,
               'clip_min': 0.,
               'clip_max': 1.,
               'rand_init': True,
               }
#PARAMS MIM
mim_params = {'eps': float(sys.argv[1]),  
              'eps_iter': 0.01,
              'nb_iter': 100,
              'ord': np.inf,
               'clip_min': 0.,
               'clip_max': 1.         
               }
#PARAMS MIML2
miml2_params = {'eps': 1.0,  
              'eps_iter': 0.01,
              'nb_iter': 1000,
              'ord': 2,
               'clip_min': 0.,
               'clip_max': 1.         
               }    


def agree_func(indices_test, pred_adv, pred_adv_tot, pred, pred_tot):    
    
    c_1 = 0.0
    for i in range(len(indices_test)):
        if (pred_adv[i] != pred_adv_tot[i]):
            if (pred_adv[i] == pred[indices_test[i]]):
                c_1 = c_1+1 
    print("Detected and well-classified by base: " + str(c_1))
    
    c_2 = 0.0
    for i in range(len(indices_test)):
        if (pred_adv[i] != pred_adv_tot[i]):
            if (pred_adv[i] != pred[indices_test[i]]):
                c_2 = c_2+1 
    print("Detected and badly-classified by base: " + str(c_2))
    
    c_3 = 0.0
    for i in range(len(indices_test)):
        if (pred_adv[i] == pred_adv_tot[i]):
            if (pred_adv[i] == pred[indices_test[i]]):
                c_3 = c_3+1                     
    print("Undetected and well-classified by base: " + str(c_3))
    
    c_4 = 0.0
    for i in range(len(indices_test)):
        if (pred_adv[i] == pred_adv_tot[i]):
            if (pred_adv[i] != pred[indices_test[i]]):
                c_4 = c_4+1 
    print("Undetected and badly-classified by base: " + str(c_4))
    
    print((c_1 + c_3) / len(indices_test))
    print((c_1 + c_2 + c_3) / len(indices_test))
    

def comp_func(X_adv_stacked, X_adv_rob, X_adv_ce, indices_test, pred_base, pred_stacked, pred_ce, pred_rob):
    
    pred_stacked_adv = np.argmax(model_stacked.predict(X_adv_stacked), axis = 1)
    pred_ce_adv = np.argmax(model_ce.predict(X_adv_ce), axis = 1)
    pred_rob_adv = np.argmax(model_rob.predict(X_adv_rob), axis = 1)
    
    success_indices_stacked_adv = np.not_equal(pred_stacked_adv, y_test[indices_test])
    success_indices_ce_adv = np.not_equal(pred_ce_adv, y_test[indices_test])
    success_indices_rob_adv = np.not_equal(pred_rob_adv, y_test[indices_test])
    
    print(np.sum(success_indices_stacked_adv))
    print(np.sum(success_indices_ce_adv))
    print(np.sum(success_indices_rob_adv))
    
    cond = (success_indices_stacked_adv == success_indices_rob_adv) & (success_indices_rob_adv == success_indices_ce_adv) & (success_indices_rob_adv == True) 
    success_indices_adv = indices_test[cond] 
    
    print("metrics source models")
    print(metrics(model_stacked, X_adv_stacked[cond], X_test, pred_stacked, success_indices_adv))
    print(metrics(model_ce, X_adv_ce[cond], X_test, pred_ce, success_indices_adv))
    print(metrics(model_rob, X_adv_rob[cond], X_test, pred_rob, success_indices_adv))
    
    print("metrics base model")
    print(metrics(model, X_adv_stacked[cond], X_test, pred_base, success_indices_adv))
    print(metrics(model, X_adv_ce[cond], X_test, pred_base, success_indices_adv))
    print(metrics(model, X_adv_rob[cond], X_test, pred_base, success_indices_adv))


    pred_adv_basefromstacked = np.argmax(model.predict(X_adv_stacked[cond]), axis=1)
    pred_adv_basefromce = np.argmax(model.predict(X_adv_ce[cond]), axis=1)
    pred_adv_basefromrob = np.argmax(model.predict(X_adv_rob[cond]), axis=1)


    agree_func(success_indices_adv, pred_adv_basefromstacked, pred_stacked_adv[cond], pred_base, pred_stacked)
    agree_func(success_indices_adv, pred_adv_basefromce, pred_ce_adv[cond], pred_base, pred_ce)  
    agree_func(success_indices_adv, pred_adv_basefromrob, pred_rob_adv[cond], pred_base, pred_rob)  



def comp_func_transfer(X_adv_source, indices_test, pred_base, pred_source, model_source, model_base):
    
    print("metrics source model")
    print(metrics(model_source, X_adv_source, X_test, pred_source, indices_test))
    print("metrics base model")
    print(metrics(model_base, X_adv_source, X_test, pred_base, indices_test))

    pred_source_adv = np.argmax(model_source.predict(X_adv_source), axis = 1)
    pred_adv_basefromsource = np.argmax(model_base.predict(X_adv_source), axis=1)  
    agree_func(indices_test, pred_adv_basefromsource, pred_source_adv, pred_base, pred_source)
    
    
#Load data set
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
 
X_train = X_train.reshape(X_train.shape[0], 32, 32, 3)
X_test = X_test.reshape(X_test.shape[0], 32, 32, 3)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
 
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)

y_train = y_train[:,0]
y_test = y_test[:,0]



sess = tf.Session()
backend.set_session(sess)
backend._LEARNING_PHASE = tf.constant(0)
backend.set_learning_phase(0)




##############################################
##############################################
#######################
#Build models
model = load_model("models/CIFAR10_float.h5")
model_stacked = load_model("models/CIFAR10_stacked.h5")
model_ce = load_model("models/CIFAR10_ce.h5")
model_rob = load_model("models/CIFAR10_luring.h5")


print("Performances of models on test set:")
print("model_base: "+ str(model.evaluate(X_test, Y_test, verbose=0)))
print("model_stacked: " + str(model_stacked.evaluate(X_test, Y_test, verbose=0)))
print("model_ce: " + str(model_ce.evaluate(X_test, Y_test, verbose=0)))
print("model_luring: " + str(model_rob.evaluate(X_test, Y_test, verbose=0)))


#Get common success indices
pred_base = np.argmax(model.predict(X_test), axis = 1)
pred_stacked = np.argmax(model_stacked.predict(X_test), axis = 1)
pred_ce = np.argmax(model_ce.predict(X_test), axis = 1)
pred_rob = np.argmax(model_rob.predict(X_test), axis = 1)

success_indices_base = np.equal(pred_base, y_test)
success_indices_stacked = np.equal(pred_stacked, y_test)                 
success_indices_ce = np.equal(pred_ce, y_test)
success_indices_rob = np.equal(pred_rob, y_test)

success_indices = np.arange(0,len(X_test))[(success_indices_stacked == success_indices_rob) & 
                           (success_indices_rob == True) & (success_indices_base == success_indices_stacked)
                           & (success_indices_stacked == success_indices_ce)]

indices_test = np.random.choice(success_indices, 1000, replace=False)

c = 0
for i in range(len(X_test)):
    if (pred_base[i] == pred_stacked[i]) & (pred_stacked[i] == y_test[i]):
        c = c+1
print("Agreement test set stacked:" + str(c))

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


#Perform attacks
wrap_stacked = KerasModelWrapper(model_stacked)
wrap_ce = KerasModelWrapper(model_ce)
wrap_rob = KerasModelWrapper(model_rob)


####################################
    #FGSM
print("\n\n")    
print("FGSM")    
fgsm_stacked = FastGradientMethod(wrap_stacked, sess=sess)
fgsm_ce = FastGradientMethod(wrap_ce, sess=sess)
fgsm_rob = FastGradientMethod(wrap_rob, sess=sess)

X_adv_stacked = np.zeros((len(indices_test),32,32,3))
for i in np.arange(0,len(indices_test),100):
    X_adv_stacked[i:(i+100)] = fgsm_stacked.generate_np(X_test[indices_test[i:(i+100)]], **fgsm_params)
    
X_adv_ce = np.zeros((len(indices_test),32,32,3))
for i in np.arange(0,len(indices_test),100):
    X_adv_ce[i:(i+100)] = fgsm_ce.generate_np(X_test[indices_test[i:(i+100)]], **fgsm_params)
    
X_adv_rob = np.zeros((len(indices_test),32,32,3))
for i in np.arange(0,len(indices_test),100):
    X_adv_rob[i:(i+100)] = fgsm_rob.generate_np(X_test[indices_test[i:(i+100)]], **fgsm_params)

comp_func(X_adv_stacked, X_adv_rob, X_adv_ce, indices_test, pred_base, pred_stacked, pred_ce, pred_rob)
#comp_func_transfer(X_adv_stacked, indices_test, pred_base, pred_stacked, model_stacked, model)
#comp_func_transfer(X_adv_ce, indices_test, pred_base, pred_ce, model_ce, model)
#comp_func_transfer(X_adv_rob, indices_test, pred_base, pred_rob, model_rob, model)


####################################
    #PGD
print("\n\n")    
print("PGD")    
pgd_stacked = ProjectedGradientDescent(wrap_stacked, sess=sess)
pgd_ce = ProjectedGradientDescent(wrap_ce, sess=sess)
pgd_rob = ProjectedGradientDescent(wrap_rob, sess=sess)

X_adv_stacked = np.zeros((len(indices_test),32,32,3))
for i in np.arange(0,len(indices_test),100):
    X_adv_stacked[i:(i+100)] = pgd_stacked.generate_np(X_test[indices_test[i:(i+100)]], **pgd_params)

X_adv_ce = np.zeros((len(indices_test),32,32,3))
for i in np.arange(0,len(indices_test),100):
    X_adv_ce[i:(i+100)] = pgd_ce.generate_np(X_test[indices_test[i:(i+100)]], **pgd_params)

X_adv_rob = np.zeros((len(indices_test),32,32,3))
for i in np.arange(0,len(indices_test),100):
    X_adv_rob[i:(i+100)] = pgd_rob.generate_np(X_test[indices_test[i:(i+100)]], **pgd_params)
      
comp_func(X_adv_stacked, X_adv_rob, X_adv_ce, indices_test, pred_base, pred_stacked, pred_ce, pred_rob)
#comp_func_transfer(X_adv_stacked, indices_test, pred_base, pred_stacked, model_stacked, model)
#comp_func_transfer(X_adv_ce, indices_test, pred_base, pred_ce, model_ce, model)
#comp_func_transfer(X_adv_rob, indices_test, pred_base, pred_rob, model_rob, model)


###################################
    #MIM
print("\n\n")    
print("MIM")    
mim_stacked = MomentumIterativeMethod(wrap_stacked, sess=sess)
mim_ce = MomentumIterativeMethod(wrap_ce, sess=sess)
mim_rob = MomentumIterativeMethod(wrap_rob, sess=sess)

X_adv_stacked = np.zeros((len(indices_test),32,32,3))
for i in np.arange(0,len(indices_test),100):
    X_adv_stacked[i:(i+100)] = mim_stacked.generate_np(X_test[indices_test[i:(i+100)]], **mim_params)

X_adv_ce = np.zeros((len(indices_test),32,32,3))
for i in np.arange(0,len(indices_test),100):
    X_adv_ce[i:(i+100)] = mim_ce.generate_np(X_test[indices_test[i:(i+100)]], **mim_params)

X_adv_rob = np.zeros((len(indices_test),32,32,3))
for i in np.arange(0,len(indices_test),100):
    X_adv_rob[i:(i+100)] = mim_rob.generate_np(X_test[indices_test[i:(i+100)]], **mim_params)

comp_func(X_adv_stacked, X_adv_rob, X_adv_ce, indices_test, pred_base, pred_stacked, pred_ce, pred_rob)
#comp_func_transfer(X_adv_stacked, indices_test, pred_base, pred_stacked, model_stacked, model)
#comp_func_transfer(X_adv_ce, indices_test, pred_base, pred_ce, model_ce, model)
#comp_func_transfer(X_adv_rob, indices_test, pred_base, pred_rob, model_rob, model)


###################################
    #MIML2
print("\n\n")    
print("MIMl2")    
mim_stacked = MomentumIterativeMethod(wrap_stacked, sess=sess)
mim_ce = MomentumIterativeMethod(wrap_ce, sess=sess)
mim_rob = MomentumIterativeMethod(wrap_rob, sess=sess)

X_adv_stacked = np.zeros((len(indices_test),32,32,3))
for i in np.arange(0,len(indices_test),100):
    X_adv_stacked[i:(i+100)] = mim_stacked.generate_np(X_test[indices_test[i:(i+100)]], **miml2_params)
    
X_adv_ce = np.zeros((len(indices_test),32,32,3))
for i in np.arange(0,len(indices_test),100):
    X_adv_ce[i:(i+100)] = mim_ce.generate_np(X_test[indices_test[i:(i+100)]], **miml2_params)

X_adv_rob = np.zeros((len(indices_test),32,32,3))
for i in np.arange(0,len(indices_test),100):
    X_adv_rob[i:(i+100)] = mim_rob.generate_np(X_test[indices_test[i:(i+100)]], **miml2_params)

X_adv_stacked = clip_adv(X_adv_stacked, X_test, indices_test, float(sys.argv[1]))
X_adv_ce = clip_adv(X_adv_ce, X_test, indices_test, float(sys.argv[1]))
X_adv_rob = clip_adv(X_adv_rob, X_test, indices_test, float(sys.argv[1]))

comp_func(X_adv_stacked, X_adv_rob, X_adv_ce, indices_test, pred_base, pred_stacked, pred_ce, pred_rob)
#comp_func_transfer(X_adv_stacked, indices_test, pred_base, pred_stacked, model_stacked, model)
#comp_func_transfer(X_adv_ce, indices_test, pred_base, pred_ce, model_ce, model)
#comp_func_transfer(X_adv_rob, indices_test, pred_base, pred_rob, model_rob, model)


