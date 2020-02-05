import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"

from keras.models import Model, load_model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Activation, BatchNormalization, Dropout, Reshape, UpSampling2D
from keras.initializers import Constant
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
import keras.backend as K
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import sys


#Load data set
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.17, random_state=123)

Y_train = np_utils.to_categorical(y_train, 10)
Y_val = np_utils.to_categorical(y_val, 10)
Y_test = np_utils.to_categorical(y_test, 10)

generator = ImageDataGenerator()
generator.fit(X_train)

model_type = sys.argv[1]
print("Training model: " + model_type )



if (model_type == "float"):
    inputs = Input(shape=(28,28,1))
    l = Conv2D(32, kernel_size=(5, 5), strides=(1,1), activation='relu', padding="same", use_bias=False)(inputs)
    l = MaxPooling2D(pool_size=(2,2), strides=(2,2))(l)
    l = Conv2D(64, kernel_size=(5, 5), strides=(1,1), activation='relu', padding="same", use_bias=False)(l)
    l = MaxPooling2D(pool_size=(2,2), strides=(2,2))(l)
    l = Flatten()(l)
    l = Dense(1024, activation='relu')(l)
    predictions = Dense(10, activation='softmax')(l)

    model = Model(inputs=inputs, outputs=predictions)
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001, decay=0.00000000006), metrics=['accuracy'])
    filepath="mnist_weights_best_float.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]
    model.fit_generator(generator.flow(X_train, Y_train, batch_size=28), 
                        epochs=5, steps_per_epoch=len(X_train)//28, 
                        verbose=1, validation_data=(X_val, Y_val), callbacks=callbacks_list)
    model.load_weights("mnist_weights_best_float.hdf5")
    
    print(model.evaluate(X_train, Y_train, verbose=0))
    print(model.evaluate(X_test, Y_test, verbose=0))
    
    model.save("models/MNIST_float.h5")
    
    
    
if (model_type == "stacked"):
    inputs= Input(shape=(28,28,1))
    l = Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
    l = MaxPooling2D((2, 2), padding='same')(l)
    l = Conv2D(8, (3, 3), activation='relu', padding='same')(l)
    l = MaxPooling2D((2, 2), padding='same')(l)
    l = Conv2D(8, (3, 3), strides=(2,2), activation='relu', padding='same')(l)
    l = Flatten()(l)
    encoded = Reshape((4, 4, 8))(l)
    l = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
    l = UpSampling2D((2, 2))(l)
    l = Conv2D(8, (3, 3), activation='relu', padding='same')(l)
    l = UpSampling2D((2, 2))(l)
    l = Conv2D(16, (3, 3), activation='relu')(l)
    l = UpSampling2D((2, 2))(l)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(l)
    l = Conv2D(32, kernel_size=(5, 5), strides=(1,1), activation='relu', padding="same", use_bias=False)(decoded)
    l = MaxPooling2D(pool_size=(2,2), strides=(2,2))(l)
    l = Conv2D(64, kernel_size=(5, 5), strides=(1,1), activation='relu', padding="same", use_bias=False)(l)
    l = MaxPooling2D(pool_size=(2,2), strides=(2,2))(l)
    l = Flatten()(l)
    l = Dense(1024, activation='relu')(l)
    predictions = Dense(10, activation='softmax')(l)
    
    model = Model(inputs=inputs, outputs=predictions)
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001, decay=0.00000000006), metrics=['accuracy'])
    filepath="mnist_weights_best_stacked.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]
    model.fit_generator(generator.flow(X_train, Y_train, batch_size=28), 
                        epochs=5, steps_per_epoch=len(X_train)//28, 
                        verbose=1, validation_data=(X_val, Y_val), callbacks=callbacks_list)
    model.load_weights("mnist_weights_best_stacked.hdf5")
    
    print(model.evaluate(X_train, Y_train, verbose=0))
    print(model.evaluate(X_test, Y_test, verbose=0))
    
    model.save("models/MNIST_stacked.h5")
    
    
    
if (model_type == "auto"):
    inputs= Input(shape=(28,28,1))
    l = Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
    l = MaxPooling2D((2, 2), padding='same')(l)
    l = Conv2D(8, (3, 3), activation='relu', padding='same')(l)
    l = MaxPooling2D((2, 2), padding='same')(l)
    l = Conv2D(8, (3, 3), strides=(2,2), activation='relu', padding='same')(l)
    l = Flatten()(l)
    encoded = Reshape((4, 4, 8))(l)
    l = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
    l = UpSampling2D((2, 2))(l)
    l = Conv2D(8, (3, 3), activation='relu', padding='same')(l)
    l = UpSampling2D((2, 2))(l)
    l = Conv2D(16, (3, 3), activation='relu')(l)
    l = UpSampling2D((2, 2))(l)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(l)
        
    model_auto = Model(inputs, decoded)
    model_auto.compile(optimizer=Adam(lr=0.001, decay=0.00000000006), loss='binary_crossentropy')
    model_auto.fit(X_train, X_train, epochs=50, batch_size = 128, validation_data=(X_test, X_test))
    
    model_auto.save("models_p/MNIST_auto_p.h5")

    

if (model_type == "ce"):   
           
    sess = tf.Session()
    K.set_session(sess)
    
    model_base = load_model("models/MNIST_float.h5")    
    model_base.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
        
    inputs= Input(shape=(28,28,1))
    l = Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
    l = MaxPooling2D((2, 2), padding='same')(l)
    l = Conv2D(8, (3, 3), activation='relu', padding='same')(l)
    l = MaxPooling2D((2, 2), padding='same')(l)
    l = Conv2D(8, (3, 3), strides=(2,2), activation='relu', padding='same')(l)
    l = Flatten()(l)
    encoded = Reshape((4, 4, 8))(l)
    l = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
    l = UpSampling2D((2, 2))(l)
    l = Conv2D(8, (3, 3), activation='relu', padding='same')(l)
    l = UpSampling2D((2, 2))(l)
    l = Conv2D(16, (3, 3), activation='relu')(l)
    l = UpSampling2D((2, 2))(l)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(l)
        
    model_auto = Model(inputs, decoded)
    model_auto.compile(optimizer='adam', loss='binary_crossentropy')
    
    print(model_base.evaluate(X_train, Y_train, verbose=0))
    print(model_base.evaluate(X_test, Y_test, verbose=0))
    
    print(model_base.evaluate(model_auto.predict(X_train), Y_train, verbose=0))
    print(model_base.evaluate(model_auto.predict(X_test), Y_test, verbose=0))

    batch_size = 64
    generator = generator.flow(X_train, Y_train, batch_size=batch_size)
    #########################
    x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
    y = tf.placeholder(tf.float32, shape=(None, 10))
    
    preds = model_base(x)
    logits = model_base(model_auto(x))._op.inputs[0]
    ce_loss = tf.losses.softmax_cross_entropy(y, logits)
    tot_loss = ce_loss    
    #########################  

    step_size_schedule = [[0, 0.001], [35000, 0.0002], [45000, 0.0004]]
    global_step = tf.train.get_or_create_global_step()
    boundaries = [int(sss[0]) for sss in step_size_schedule]
    boundaries = boundaries[1:]
    values = [sss[1] for sss in step_size_schedule]
    learning_rate = tf.train.piecewise_constant(
        tf.cast(global_step, tf.int32),
        boundaries,
        values) 
    optimizer = tf.train.AdamOptimizer(learning_rate)  
    opt_op = optimizer.minimize(tot_loss, global_step=global_step, var_list=model_auto.weights)
    
    uninitialized_vars = []
    for var in tf.global_variables():
        try:
            sess.run(var)
        except tf.errors.FailedPreconditionError:
            uninitialized_vars.append(var)
    sess.run(tf.variables_initializer(uninitialized_vars))
    
    tamp_val_acc = model_base.evaluate(model_auto.predict(X_val), Y_val, verbose=0)[1]
    for step in np.arange(0, 50000):    
        x_batch, y_batch = next(generator)
        sess.run(opt_op, feed_dict={x: x_batch, y: y_batch})
        if (step % 1000 == 0):
            print("step number: " + str(step))          
            print(sess.run(tot_loss, feed_dict={x: X_train[0:1000], y: Y_train[0:1000]}))
            #Save best weights
            val_acc = model_base.evaluate(model_auto.predict(X_val), Y_val, verbose=0)[1]
            print("val_acc: " + str(val_acc))
            if (val_acc > tamp_val_acc):
                print("Best accuracy on validation set so far: " + str(val_acc))
                model_auto.save_weights("mnist_weights_best_ce.hdf5")
                tamp_val_acc = val_acc
            
    model_auto.load_weights("mnist_weights_best_ce.hdf5")   
    model_auto.save("models_p/MNIST_ce_p.h5")      
    
    
    
if (model_type == "luring_alt"):   
           
    sess = tf.Session()
    K.set_session(sess)
    
    model_base = load_model("models/MNIST_float.h5")    
    model_base.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
        
    inputs= Input(shape=(28,28,1))
    l = Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
    l = MaxPooling2D((2, 2), padding='same')(l)
    l = Conv2D(8, (3, 3), activation='relu', padding='same')(l)
    l = MaxPooling2D((2, 2), padding='same')(l)
    l = Conv2D(8, (3, 3), strides=(2,2), activation='relu', padding='same')(l)
    l = Flatten()(l)
    encoded = Reshape((4, 4, 8))(l)
    l = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
    l = UpSampling2D((2, 2))(l)
    l = Conv2D(8, (3, 3), activation='relu', padding='same')(l)
    l = UpSampling2D((2, 2))(l)
    l = Conv2D(16, (3, 3), activation='relu')(l)
    l = UpSampling2D((2, 2))(l)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(l)
        
    model_auto = Model(inputs, decoded)
    model_auto.compile(optimizer='adam', loss='binary_crossentropy')
    
    print(model_base.evaluate(X_train, Y_train, verbose=0))
    print(model_base.evaluate(X_test, Y_test, verbose=0))
    
    print(model_base.evaluate(model_auto.predict(X_train), Y_train, verbose=0))
    print(model_base.evaluate(model_auto.predict(X_test), Y_test, verbose=0))

    batch_size = 64
    generator = generator.flow(X_train, Y_train, batch_size=batch_size)
    #########################
    x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
    y = tf.placeholder(tf.float32, shape=(None, 10))
    
    preds = model_base(x)
    logits = model_base(model_auto(x))._op.inputs[0]
    preds_sort = tf.contrib.framework.sort(preds, axis=1, direction="DESCENDING")
    preds_max = tf.gather(preds_sort, axis=1, indices=[0])
    preds_secondmax = tf.gather(preds_sort, axis=1, indices=[1,2,3,4,5,6,7,8,9])
    c_max = tf.equal(preds, preds_max)
    d_max = tf.where(c_max, logits, tf.fill(tf.shape(logits), -float('Inf')))
    logits_max = tf.reduce_max(d_max, axis=1, keepdims=True)  
    c_secondmax = tf.not_equal(preds, preds_max)
    d_secondmax = tf.where(c_secondmax, logits, tf.fill(tf.shape(logits), -float('Inf')))
    logits_secondmax = tf.reduce_max(d_secondmax, axis=1, keepdims=True)     
    logits_diff = logits_max - logits_secondmax  
    logits_loss = - tf.reduce_mean(logits_diff) 
    tot_loss = logits_loss
    #########################  
        
    step_size_schedule = [[0, 0.001], [35000, 0.0002], [45000, 0.0004]]
    global_step = tf.train.get_or_create_global_step()
    boundaries = [int(sss[0]) for sss in step_size_schedule]
    boundaries = boundaries[1:]
    values = [sss[1] for sss in step_size_schedule]
    learning_rate = tf.train.piecewise_constant(
        tf.cast(global_step, tf.int32),
        boundaries,
        values) 
    optimizer = tf.train.AdamOptimizer(learning_rate)  
    opt_op = optimizer.minimize(tot_loss, global_step=global_step, var_list=model_auto.weights)
    
    uninitialized_vars = []
    for var in tf.global_variables():
        try:
            sess.run(var)
        except tf.errors.FailedPreconditionError:
            uninitialized_vars.append(var)
    sess.run(tf.variables_initializer(uninitialized_vars))
    
    tamp_val_acc = model_base.evaluate(model_auto.predict(X_val), Y_val, verbose=0)[1]
    for step in np.arange(0, 50000):    
        x_batch, y_batch = next(generator)
        sess.run(opt_op, feed_dict={x: x_batch, y: y_batch})
        if (step % 1000 == 0):
            print("step number: " + str(step))          
            print(sess.run(tot_loss, feed_dict={x: X_train[0:1000], y: Y_train[0:1000]}))
            #Save best weights
            val_acc = model_base.evaluate(model_auto.predict(X_val), Y_val, verbose=0)[1]
            print("val_acc: " + str(val_acc))
            if (val_acc > tamp_val_acc):
                print("Best accuracy on validation set so far: " + str(val_acc))
                model_auto.save_weights("mnist_weights_best_luring_alt.hdf5")
                tamp_val_acc = val_acc
            
    model_auto.load_weights("mnist_weights_best_luring_alt.hdf5")   
    model_auto.save("models_p/MNIST_luring_alt_p.h5")        
     


if (model_type == "luring"):
    
    sess = tf.Session()
    K.set_session(sess)
    
    model_base = load_model("models/MNIST_float.h5")    
    model_base.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
        
    inputs= Input(shape=(28,28,1))
    l = Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
    l = MaxPooling2D((2, 2), padding='same')(l)
    l = Conv2D(8, (3, 3), activation='relu', padding='same')(l)
    l = MaxPooling2D((2, 2), padding='same')(l)
    l = Conv2D(8, (3, 3), strides=(2,2), activation='relu', padding='same')(l)
    l = Flatten()(l)
    encoded = Reshape((4, 4, 8))(l)
    l = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
    l = UpSampling2D((2, 2))(l)
    l = Conv2D(8, (3, 3), activation='relu', padding='same')(l)
    l = UpSampling2D((2, 2))(l)
    l = Conv2D(16, (3, 3), activation='relu')(l)
    l = UpSampling2D((2, 2))(l)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(l)
        
    model_auto = Model(inputs, decoded)
    model_auto.compile(optimizer='adam', loss='binary_crossentropy')
    
    print(model_base.evaluate(X_train, Y_train, verbose=0))
    print(model_base.evaluate(X_test, Y_test, verbose=0))
    
    print(model_base.evaluate(model_auto.predict(X_train), Y_train, verbose=0))
    print(model_base.evaluate(model_auto.predict(X_test), Y_test, verbose=0))

    batch_size = 64
    generator = generator.flow(X_train, Y_train, batch_size=batch_size)
    #########################
    x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
    y = tf.placeholder(tf.float32, shape=(None, 10))
        
    def ZERO():
        return np.asarray(0., dtype=np.dtype('float32'))
    
    preds = model_base(x)
    logits = model_base(model_auto(x))._op.inputs[0]
    preds_sort = tf.contrib.framework.sort(preds, axis=1, direction="DESCENDING")
    preds_max = tf.gather(preds_sort, axis=1, indices=[0])
    preds_secondmax = tf.gather(preds_sort, axis=1, indices=[1,2,3,4,5,6,7,8,9])
    c_max = tf.equal(preds, preds_max)
    d_max = tf.where(c_max, logits, tf.fill(tf.shape(logits), -float('Inf')))
    logits_max = tf.reduce_max(d_max, axis=1, keepdims=True)  
    c_secondmax = tf.not_equal(preds, preds_max)
    d_secondmax = tf.where(c_secondmax, logits, tf.fill(tf.shape(logits), -float('Inf')))
    logits_secondmax = tf.reduce_max(d_secondmax, axis=1, keepdims=True)     

    tot_loss = tf.reduce_mean(tf.maximum(ZERO(), logits_secondmax - logits_max))

    preds_secondmax_2 = tf.gather(preds_sort, axis=1, indices=[1])
    c_secondmax_2 = tf.equal(preds, preds_secondmax_2)
    d_secondmax_2 = tf.where(c_secondmax_2, logits, tf.fill(tf.shape(logits), -float('Inf')))
    logits_secondmax_2 = tf.reduce_max(d_secondmax_2, axis=1, keepdims=True)
    logits_diff_2 = logits_max - logits_secondmax_2
    logits_loss_2 = - tf.reduce_mean(logits_diff_2)
    tot_loss_2 = logits_loss_2 
    #########################

    step_size_schedule = [[0, 0.001], [35000, 0.0002], [45000, 0.0004]]
    global_step = tf.train.get_or_create_global_step()
    boundaries = [int(sss[0]) for sss in step_size_schedule]
    boundaries = boundaries[1:]
    values = [sss[1] for sss in step_size_schedule]
    learning_rate = tf.train.piecewise_constant(
        tf.cast(global_step, tf.int32),
        boundaries,
        values) 
    optimizer = tf.train.AdamOptimizer(learning_rate)  
    opt_op = optimizer.minimize(tot_loss + tot_loss_2, global_step=global_step, var_list=model_auto.weights)
    
    uninitialized_vars = []
    for var in tf.global_variables():
        try:
            sess.run(var)
        except tf.errors.FailedPreconditionError:
            uninitialized_vars.append(var)
    sess.run(tf.variables_initializer(uninitialized_vars))
   
    tamp_val_acc = model_base.evaluate(model_auto.predict(X_val), Y_val, verbose=0)[1]
    for step in np.arange(0, 50000):    
        x_batch, y_batch = next(generator)
        sess.run(opt_op, feed_dict={x: x_batch, y: y_batch})
        if (step % 1000 == 0):
            print("step number: " + str(step))          
            print(sess.run(tot_loss, feed_dict={x: X_train[0:1000], y: Y_train[0:1000]}))
            print(sess.run(tot_loss_2, feed_dict={x: X_train[0:1000], y: Y_train[0:1000]}))
            #Save best weights
            val_acc = model_base.evaluate(model_auto.predict(X_val), Y_val, verbose=0)[1]
            print("val_acc: " + str(val_acc))
            if (val_acc > tamp_val_acc):
                print("Best accuracy on validation set so far: " + str(val_acc))
                model_auto.save_weights("mnist_weights_best_luring.hdf5")
                tamp_val_acc = val_acc
            
    model_auto.load_weights("mnist_weights_best_luring.hdf5")   
    model_auto.save("models_p/MNIST_luring_p.h5")        
     

##Save final model
if (model_type == "luring") | (model_type == "luring_alt") | (model_type == "ce"):

    model_base = load_model("models/MNIST_float.h5")    
        
    inputs= Input(shape=(28,28,1))
    l = Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
    l = MaxPooling2D((2, 2), padding='same')(l)
    l = Conv2D(8, (3, 3), activation='relu', padding='same')(l)
    l = MaxPooling2D((2, 2), padding='same')(l)
    l = Conv2D(8, (3, 3), strides=(2,2), activation='relu', padding='same')(l)
    l = Flatten()(l)
    encoded = Reshape((4, 4, 8))(l)
    l = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
    l = UpSampling2D((2, 2))(l)
    l = Conv2D(8, (3, 3), activation='relu', padding='same')(l)
    l = UpSampling2D((2, 2))(l)
    l = Conv2D(16, (3, 3), activation='relu')(l)
    l = UpSampling2D((2, 2))(l)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(l)
    
    l = Conv2D(32, kernel_size=(5, 5), strides=(1,1), activation='relu', padding="same", use_bias=False)(decoded)
    l = MaxPooling2D(pool_size=(2,2), strides=(2,2))(l)
    l = Conv2D(64, kernel_size=(5, 5), strides=(1,1), activation='relu', padding="same", use_bias=False)(l)
    l = MaxPooling2D(pool_size=(2,2), strides=(2,2))(l)
    l = Flatten()(l)
    l = Dense(1024, activation='relu')(l)
    predictions = Dense(10, activation='softmax')(l)
    
    model_built = Model(inputs, predictions)
    
    for i in range(len(model_auto.layers)):
        model_built.layers[i].set_weights(model_auto.layers[i].get_weights())
    j = 1
    for i in range(len(model_auto.layers), len(model_built.layers)):
        model_built.layers[i].set_weights(model_base.layers[j].get_weights())
        j = j + 1
    
    model_built.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])    
        
    print(model_built.evaluate(X_test, Y_test, verbose=0))
    
    model_built.save("models/MNIST_" + model_type + ".h5")




