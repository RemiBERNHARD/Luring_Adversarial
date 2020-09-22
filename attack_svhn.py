import sys
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]= "0"

from keras.models import load_model
from keras.utils import np_utils
from keras.datasets import mnist
from keras import backend 
import tensorflow as tf
import numpy as np

from utils_func import metrics, agree_func
from cleverhans.utils_keras import KerasModelWrapper
from cleverhans.attacks import FastGradientMethod, MomentumIterativeMethod
from MIM_InputDiverse import MomentumIterativeMethod_Diverse
from MIM_TI import MomentumIterativeMethod_TI
from MIM_TI_DIM import MomentumIterativeMethod_TI_DIM
from cleverhans.attacks import SPSA
from parsimonious_attack import ParsimoniousAttack


    

#################################    
####Load data set####
#################################
X_train = np.load("SVHN_data/X_train.npy")
X_test = np.load("SVHN_data/X_test.npy")
y_train = np.load("SVHN_data/y_train.npy")
y_test = np.load("SVHN_data/y_test.npy")
y_train[y_train==10]=0
y_test[y_test==10]=0
X_train = X_train.reshape(X_train.shape[0], 32, 32, 3)
X_test = X_test.reshape(X_test.shape[0], 32, 32, 3)
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
model = load_model("models/SVHN_float.h5")

model_type = sys.argv[2]
model_source = load_model("models/SVHN_" + sys.argv[2] + ".h5")
print("Crafting adversarial examples on model: " + model_type)
print("Size of the perturbation: " + sys.argv[1])
print("Performances of models on test set:")
print("model_base: "+ str(model.evaluate(X_test, Y_test, verbose=0)))
print("model_source: " + str(model_source.evaluate(X_test, Y_test, verbose=0)))
print("\n")

pred_base = np.argmax(model.predict(X_test), axis = 1)
pred_source = np.argmax(model_source.predict(X_test), axis = 1)   

success_indices_base = np.equal(pred_base, y_test)         
success_indices_source = np.equal(pred_source, y_test)

success_indices = np.arange(0,len(X_test))[(success_indices_base == success_indices_source) & (success_indices_source == True)] 

indices_test = np.random.choice(success_indices, 1000, replace=False)



#################################    
####Perform attacks####
#################################
wrap_source = KerasModelWrapper(model_source)

####################################
#FGSM
print("\n\n")        
print("FGSM")  

fgsm_params = {'eps': float(sys.argv[1]),
               'clip_min': 0.,
               'clip_max': 1.
               }

fgsm_source = FastGradientMethod(wrap_source, sess=sess)

X_adv_source = np.zeros((len(indices_test),32,32,3))
for i in np.arange(0,len(indices_test),500):
    X_adv_source[i:(i+500)] = fgsm_source.generate_np(X_test[indices_test[i:(i+500)]], **fgsm_params)


print("metrics source model")
print(metrics(model_source, X_adv_source, X_test, pred_source, indices_test))
print("metrics base model")
print(metrics(model, X_adv_source, X_test, pred_base, indices_test))

pred_source_adv = np.argmax(model_source.predict(X_adv_source), axis = 1)
pred_adv_basefromsource = np.argmax(model.predict(X_adv_source), axis=1)  
agree_func(indices_test, pred_adv_basefromsource, pred_source_adv, pred_base, pred_source)
print(" ")

####################################
#MIM
print("\n\n")        
print("MIM")  

mim_params = {'eps': float(sys.argv[1]),
              'eps_iter': 0.01,
              'nb_iter': 500,
              'ord': np.inf,
               'clip_min': 0.,
               'clip_max': 1.
               }

mim_source = MomentumIterativeMethod(wrap_source, sess=sess)

X_adv_source = np.zeros((len(indices_test),32,32,3))
for i in np.arange(0,len(indices_test),500):
    X_adv_source[i:(i+500)] = mim_source.generate_np(X_test[indices_test[i:(i+500)]], **mim_params)

print("metrics source model")
print(metrics(model_source, X_adv_source, X_test, pred_source, indices_test))
print("metrics base model")
print(metrics(model, X_adv_source, X_test, pred_base, indices_test))

pred_source_adv = np.argmax(model_source.predict(X_adv_source), axis = 1)
pred_adv_basefromsource = np.argmax(model.predict(X_adv_source), axis=1)  
agree_func(indices_test, pred_adv_basefromsource, pred_source_adv, pred_base, pred_source)
print(" ")

####################################
#MIM Diverse
print("\n\n")        
print("MIMDiverse")  

mim_params = {'eps': float(sys.argv[1]),
              'eps_iter': 0.01,
              'nb_iter': 500,
              'ord': np.inf,
               'clip_min': 0.,
               'clip_max': 1.
               }

for i in  np.arange(0.1,1.1,0.1):
    i = np.float32(i)
    mim_params['prob'] = i
    print(i)
    mim_source = MomentumIterativeMethod_Diverse(wrap_source, sess=sess)
    
    X_adv_source = np.zeros((len(indices_test),32,32,3))
    for i in np.arange(0,len(indices_test),500):
        X_adv_source[i:(i+500)] = mim_source.generate_np(X_test[indices_test[i:(i+500)]], **mim_params)

    print("metrics source model")
    print(metrics(model_source, X_adv_source, X_test, pred_source, indices_test))
    print("metrics base model")
    print(metrics(model, X_adv_source, X_test, pred_base, indices_test))
    
    pred_source_adv = np.argmax(model_source.predict(X_adv_source), axis = 1)
    pred_adv_basefromsource = np.argmax(model.predict(X_adv_source), axis=1)  
    agree_func(indices_test, pred_adv_basefromsource, pred_source_adv, pred_base, pred_source)
    print(" ")
    
####################################
#MIM TI
print("\n\n")        
print("MIM-TI")  

def gkern(kernlen=21, nsig=3):
  """Returns a 2D Gaussian kernel array."""
  import scipy.stats as st
  x = np.linspace(-nsig, nsig, kernlen)
  kern1d = st.norm.pdf(x)
  kernel_raw = np.outer(kern1d, kern1d)
  kernel = kernel_raw / kernel_raw.sum()
  return kernel    

for (i,j) in [(1,1), (3,3), (5,3), (10,3), (15,3)]:
    print((i,j))
    kernel = gkern(i, j).astype(np.float32)
    kernel = np.stack([kernel, kernel, kernel]).swapaxes(2, 0)
    kernel = np.expand_dims(kernel,3)

    #PARAMS MIM-TI
    mim_ti_params = {'eps': float(sys.argv[1]),
                  'eps_iter': 0.01,
                  'nb_iter': 500,
                  'ord': np.inf,
                   'clip_min': 0.,
                   'clip_max': 1.,
                   'kernel': kernel
                   }
    
    mim_ti_source = MomentumIterativeMethod_TI(wrap_source, sess=sess)
    
    X_adv_source = np.zeros((len(indices_test),32,32,3))
    for i in np.arange(0,len(indices_test),500):
        X_adv_source[i:(i+500)] = mim_ti_source.generate_np(X_test[indices_test[i:(i+500)]], **mim_ti_params)
    
    print("metrics source model")
    print(metrics(model_source, X_adv_source, X_test, pred_source, indices_test))
    print("metrics base model")
    print(metrics(model, X_adv_source, X_test, pred_base, indices_test))
    
    pred_source_adv = np.argmax(model_source.predict(X_adv_source), axis = 1)
    pred_adv_basefromsource = np.argmax(model.predict(X_adv_source), axis=1)  
    agree_func(indices_test, pred_adv_basefromsource, pred_source_adv, pred_base, pred_source)    

####################################
    #MIM-TI-DIM
print("\n\n")        
print("MIM-TI-DIM") 
    
def gkern(kernlen=21, nsig=3):
  """Returns a 2D Gaussian kernel array."""
  import scipy.stats as st
  x = np.linspace(-nsig, nsig, kernlen)
  kern1d = st.norm.pdf(x)
  kernel_raw = np.outer(kern1d, kern1d)
  kernel = kernel_raw / kernel_raw.sum()
  return kernel    

for (i,j) in [(1,1), (3,3), (5,3), (10,3), (15,3)]:
    print((i,j))
    kernel = gkern(i, j).astype(np.float32)
    kernel = np.stack([kernel, kernel, kernel]).swapaxes(2, 0)
    kernel = np.expand_dims(kernel,3)

    #PARAMS MIM-TI-DIM
    mim_ti_dim_params = {'eps': float(sys.argv[1]),
                  'eps_iter': 0.01,
                  'nb_iter': 500,
                  'ord': np.inf,
                   'clip_min': 0.,
                   'clip_max': 1.,
                   'kernel': kernel,
                   'prob': 1.0
                   }
    
    
    mim_ti_dim_source = MomentumIterativeMethod_TI_DIM(wrap_source, sess=sess)
    
    X_adv_source = np.zeros((len(indices_test),32,32,3))
    for i in np.arange(0,len(indices_test),500):
        X_adv_source[i:(i+500)] = mim_ti_dim_source.generate_np(X_test[indices_test[i:(i+500)]], **mim_ti_dim_params)
    
    print("metrics source model")
    print(metrics(model_source, X_adv_source, X_test, pred_source, indices_test))
    print("metrics base model")
    print(metrics(model, X_adv_source, X_test, pred_base, indices_test))
    
    pred_source_adv = np.argmax(model_source.predict(X_adv_source), axis = 1)
    pred_adv_basefromsource = np.argmax(model.predict(X_adv_source), axis=1)  
    agree_func(indices_test, pred_adv_basefromsource, pred_source_adv, pred_base, pred_source)    
    
####################################
#PARSI
print("\n\n")        
print("PARSI")  

parsi_attack = ParsimoniousAttack(model_source, max_queries=20000, epsilon=float(sys.argv[1]), block_size = 8, batch_size=256)

X_adv_source = np.zeros((len(indices_test),32,32,3))
for i in range(0, len(indices_test)): 
    X_adv_source[i] = parsi_attack.perturb(X_test[indices_test[i:(i+1)]], y_test[indices_test[i]], i, sess)[0]

print("metrics source model")
print(metrics(model_source, X_adv_source, X_test, pred_source, indices_test))
print("metrics base model")
print(metrics(model, X_adv_source, X_test, pred_base, indices_test))

pred_source_adv = np.argmax(model_source.predict(X_adv_source), axis = 1)
pred_adv_basefromsource = np.argmax(model.predict(X_adv_source), axis=1)  
agree_func(indices_test, pred_adv_basefromsource, pred_source_adv, pred_base, pred_source)

####################################
#SPSA
print("\n\n")        
print("SPSA")  

spsa_params = {'eps': float(sys.argv[1]),
               'learning_rate': 0.01,
               'delta': 0.01,
               'spsa_samples': 128,
               'spsa_iters': 1,
               'nb_iter': 100,
               'clip_min': 0.,
               'clip_max': 1.
               }

spsa_attack = SPSA(wrap_source, sess=sess)
x = tf.placeholder(dtype=tf.float32, shape=(None,32,32,3))
y = tf.placeholder(dtype=tf.float32, shape=(None,10))
x_adv = spsa_attack.generate(x, y, **spsa_params)
X_adv_source = np.zeros((len(indices_test),32,32,3))
for i in range(0, len(indices_test)):
    X_adv_source[i] = sess.run(x_adv, feed_dict={x: X_test[indices_test[i:(i+1)]], y: Y_test[indices_test[i:(i+1)]]})

print("metrics source model")
print(metrics(model_source, X_adv_source, X_test, pred_source, indices_test))
print("metrics base model")
print(metrics(model, X_adv_source, X_test, pred_base, indices_test))

pred_source_adv = np.argmax(model_source.predict(X_adv_source), axis = 1)
pred_adv_basefromsource = np.argmax(model.predict(X_adv_source), axis=1)  
agree_func(indices_test, pred_adv_basefromsource, pred_source_adv, pred_base, pred_source)
