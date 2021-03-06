import sys
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"

from keras.models import Model, load_model
from keras.layers import Input, Lambda, Conv2D, Conv2DTranspose, MaxPooling2D, Flatten, Dense, Activation, BatchNormalization, Dropout, Reshape, UpSampling2D
from keras.initializers import Constant
from keras.optimizers import Adam, SGD
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
import keras.backend as K
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split


#Load data set
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
 
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.17, random_state=123)

Y_train = np_utils.to_categorical(y_train, 10)
Y_val = np_utils.to_categorical(y_val, 10)
Y_test = np_utils.to_categorical(y_test, 10)

generator = ImageDataGenerator()
generator.fit(X_train)

model_type = sys.argv[1]
print("Training model: " + model_type )



if (model_type == "float"):
    inputs = Input(shape=(32,32,3))
    l = Conv2D(64, kernel_size=(3,3), strides=(1,1),padding="valid", bias=False)(inputs)
    l = BatchNormalization()(l)
    l = Activation("relu")(l)
    l = Conv2D(64, kernel_size=(3,3), strides=(1,1),padding="same", bias=False)(l)
    l = MaxPooling2D(pool_size=(2,2), strides=(2,2))(l)
    l = BatchNormalization()(l)
    l = Activation("relu")(l)
    l = Conv2D(128, kernel_size=(3,3), strides=(1,1),padding="same", bias=False)(l)
    l = BatchNormalization()(l)
    l = Activation("relu")(l) 
    l = Conv2D(128, kernel_size=(3,3), strides=(1,1),padding="same", bias=False)(l)
    l = MaxPooling2D(pool_size=(2,2), strides=(2,2))(l)
    l = BatchNormalization()(l)
    l = Activation("relu")(l)
    l = Conv2D(256, kernel_size=(3,3), strides=(1,1),padding="same", bias=False)(l)
    l = BatchNormalization()(l)
    l = Activation("relu")(l)
    l = Conv2D(256, kernel_size=(3,3), strides=(1,1),padding="same", bias=False)(l)
    l = MaxPooling2D(pool_size=(2,2), strides=(2,2))(l)
    l = BatchNormalization()(l)
    l = Activation("relu")(l)
    l = Dropout(0.25)(l)
    l = Flatten()(l)
    l = Dense(1024, bias=False)(l)
    l = BatchNormalization()(l)
    l = Activation("relu")(l)
    l = Dropout(0.25)(l)
    l = Dense(1024, bias=False)(l)
    l = BatchNormalization()(l)
    l = Activation("relu")(l)
    l = Dense(10, activation="softmax", bias=False)(l)
    predictions = l
    
    model = Model(inputs=inputs, outputs=predictions)
    model.compile(loss='categorical_crossentropy', optimizer=Adam(0.01), metrics=['accuracy'])
    filepath="svhn_weights_best_float.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]
    model.fit_generator(generator.flow(X_train, Y_train, batch_size=28), 
                        epochs=10, steps_per_epoch=len(X_train)//28, 
                        verbose=1, validation_data=(X_val, Y_val), callbacks=callbacks_list)
    model.load_weights("svhn_weights_best_float.hdf5")

    print(model.evaluate(X_train, Y_train, verbose=0))
    print(model.evaluate(X_test, Y_test, verbose=0))

    model.save("models/SVHN_float.h5")
    
    

if (model_type == "stacked"):
    inputs = Input(shape=(32, 32, 3))
    l = Conv2D(64, (3, 3), padding='same')(inputs)
    l = BatchNormalization()(l)
    l = Activation('relu')(l)
    l = Conv2D(128, (3, 3), padding='same')(inputs)
    l = Dropout(0.0)(l)
    l = BatchNormalization()(l)
    l = Activation('relu')(l)
    l = MaxPooling2D((2, 2), padding='same')(l)
    l = Conv2D(256, (3, 3), padding='same')(l)
    l = BatchNormalization()(l)
    l = Activation('relu')(l)
    l = MaxPooling2D((2, 2), padding='same')(l)
    l = Conv2D(512, (3, 3), padding='same')(l)
    l = BatchNormalization()(l)
    l = Activation('relu')(l)
    l = Conv2D(1024, (3, 3), padding='same')(l)
    l = BatchNormalization()(l)
    l = Activation('relu')(l)
    encoded = MaxPooling2D((2, 2), padding='same')(l)
    l = Conv2D(512, (3, 3), padding='same')(encoded)
    l = BatchNormalization()(l)
    l = Activation('relu')(l)
    l = Conv2D(512, (3, 3), padding='same')(l)
    l = BatchNormalization()(l)
    l = Activation('relu')(l)
    l = UpSampling2D((2, 2))(l)
    l = Conv2D(256, (3, 3), padding='same')(l)
    l = BatchNormalization()(l)
    l = Activation('relu')(l)
    l = UpSampling2D((2, 2))(l)
    l = Conv2D(128, (3, 3), padding='same')(l)
    l = BatchNormalization()(l)
    l = Activation('relu')(l)
    l = UpSampling2D((2, 2))(l)
    l = Conv2D(64, (3, 3), padding='same')(l)
    l = Dropout(0.0)(l)
    l = BatchNormalization()(l)
    l = Activation('relu')(l)
    l = Conv2D(3, (3, 3), padding='same')(l)
    l = BatchNormalization()(l)
    decoded = Activation('sigmoid')(l)
    l = Conv2D(64, kernel_size=(3,3), strides=(1,1),padding="valid", bias=False)(decoded)
    l = BatchNormalization()(l)
    l = Activation("relu")(l)
    l = Conv2D(64, kernel_size=(3,3), strides=(1,1),padding="same", bias=False)(l)
    l = MaxPooling2D(pool_size=(2,2), strides=(2,2))(l)
    l = BatchNormalization()(l)
    l = Activation("relu")(l)
    l = Conv2D(128, kernel_size=(3,3), strides=(1,1),padding="same", bias=False)(l)
    l = BatchNormalization()(l)
    l = Activation("relu")(l) 
    l = Conv2D(128, kernel_size=(3,3), strides=(1,1),padding="same", bias=False)(l)
    l = MaxPooling2D(pool_size=(2,2), strides=(2,2))(l)
    l = BatchNormalization()(l)
    l = Activation("relu")(l)
    l = Conv2D(256, kernel_size=(3,3), strides=(1,1),padding="same", bias=False)(l)
    l = BatchNormalization()(l)
    l = Activation("relu")(l)
    l = Conv2D(256, kernel_size=(3,3), strides=(1,1),padding="same", bias=False)(l)
    l = MaxPooling2D(pool_size=(2,2), strides=(2,2))(l)
    l = BatchNormalization()(l)
    l = Activation("relu")(l)
    l = Dropout(0.25)(l)
    l = Flatten()(l)
    l = Dense(1024, bias=False)(l)
    l = BatchNormalization()(l)
    l = Activation("relu")(l)
    l = Dropout(0.25)(l)
    l = Dense(1024, bias=False)(l)
    l = BatchNormalization()(l)
    l = Activation("relu")(l)
    predictions = Dense(10, activation="softmax", bias=False)(l)

    model = Model(inputs=inputs, outputs=predictions)
    filepath="svhn_weights_best_stacked.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]
    model.compile(loss='categorical_crossentropy', optimizer=Adam(0.01), metrics=['accuracy'])
    model.fit_generator(generator.flow(X_train, Y_train, batch_size=256), 
                        epochs=20, steps_per_epoch=len(X_train)//256, 
                        verbose=1, validation_data=(X_val, Y_val), callbacks=callbacks_list)
    model.load_weights("svhn_weights_best_stacked.hdf5")

    print(model.evaluate(X_train, Y_train, verbose=0))
    print(model.evaluate(X_test, Y_test, verbose=0))
    
    model.save("models/SVHN_stacked.h5")



if (model_type == "auto"):
    inputs = Input(shape=(32, 32, 3))
    l = Conv2D(128, (3, 3), padding='same')(inputs)
    l = Dropout(0.0)(l)
    l = BatchNormalization()(l)
    l = Activation('relu')(l)
    l = MaxPooling2D((2, 2), padding='same')(l)
    l = Conv2D(256, (3, 3), padding='same')(l)
    l = BatchNormalization()(l)
    l = Activation('relu')(l)
    l = MaxPooling2D((2, 2), padding='same')(l)
    l = Conv2D(512, (3, 3), padding='same')(l)
    l = BatchNormalization()(l)
    l = Activation('relu')(l)
    l = Conv2D(1024, (3, 3), padding='same')(l)
    l = BatchNormalization()(l)
    l = Activation('relu')(l)
    
    encoded = MaxPooling2D((2, 2), padding='same')(l)
    
    l = Conv2D(512, (3, 3), padding='same')(encoded)
    l = BatchNormalization()(l)
    l = Activation('relu')(l)
    l = Conv2D(512, (3, 3), padding='same')(l)
    l = BatchNormalization()(l)
    l = Activation('relu')(l)
    l = UpSampling2D((2, 2))(l)
    l = Conv2D(256, (3, 3), padding='same')(l)
    l = BatchNormalization()(l)
    l = Activation('relu')(l)
    l = UpSampling2D((2, 2))(l)
    l = Conv2D(128, (3, 3), padding='same')(l)
    l = BatchNormalization()(l)
    l = Activation('relu')(l)
    l = UpSampling2D((2, 2))(l)
    l = Conv2D(64, (3, 3), padding='same')(l)
    l = Dropout(0.0)(l)
    l = BatchNormalization()(l)
    l = Activation('relu')(l)
    l = Conv2D(3, (3, 3), padding='same')(l)
    l = BatchNormalization()(l)
    decoded = Activation('sigmoid')(l)

    model_auto = Model(inputs, decoded)   
    model_auto.compile(optimizer=Adam(lr=0.001), loss='binary_crossentropy') 
    model_auto.fit(X_train, X_train, epochs=5, batch_size = 128, validation_data=(X_test, X_test))
    
    model_auto.save("models_p/SVHN_auto_p.h5")
    
 
    
if (model_type == "ce"):
    
    sess = tf.Session()
    K.set_session(sess)
    
    model_base = load_model("models/SVHN_float.h5")    
    model_base.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])  
    
    inputs = Input(shape=(32, 32, 3))
    l = Conv2D(128, (3, 3), padding='same')(inputs)
    l = Dropout(0.0)(l)
    l = BatchNormalization()(l)
    l = Activation('relu')(l)
    l = MaxPooling2D((2, 2), padding='same')(l)
    l = Conv2D(256, (3, 3), padding='same')(l)
    l = BatchNormalization()(l)
    l = Activation('relu')(l)
    l = MaxPooling2D((2, 2), padding='same')(l)
    l = Conv2D(512, (3, 3), padding='same')(l)
    l = BatchNormalization()(l)
    l = Activation('relu')(l)
    l = Conv2D(1024, (3, 3), padding='same')(l)
    l = BatchNormalization()(l)
    l = Activation('relu')(l)
    
    encoded = MaxPooling2D((2, 2), padding='same')(l)
    
    l = Conv2D(512, (3, 3), padding='same')(encoded)
    l = BatchNormalization()(l)
    l = Activation('relu')(l)
    l = Conv2D(512, (3, 3), padding='same')(l)
    l = BatchNormalization()(l)
    l = Activation('relu')(l)
    l = UpSampling2D((2, 2))(l)
    l = Conv2D(256, (3, 3), padding='same')(l)
    l = BatchNormalization()(l)
    l = Activation('relu')(l)
    l = UpSampling2D((2, 2))(l)
    l = Conv2D(128, (3, 3), padding='same')(l)
    l = BatchNormalization()(l)
    l = Activation('relu')(l)
    l = UpSampling2D((2, 2))(l)
    l = Conv2D(64, (3, 3), padding='same')(l)
    l = Dropout(0.0)(l)
    l = BatchNormalization()(l)
    l = Activation('relu')(l)
    l = Conv2D(3, (3, 3), padding='same')(l)
    l = BatchNormalization()(l)
    decoded = Activation('sigmoid')(l)

    model_auto = Model(inputs, decoded)   
    model_auto.compile(optimizer='adam', loss='binary_crossentropy') 

    ######################
    ######################
    #Training    
    print(model_base.evaluate(X_train, Y_train, verbose=0)) 
    print(model_base.evaluate(X_test, Y_test, verbose=0))
    
    print(model_base.evaluate(model_auto.predict(X_train), Y_train, verbose=0))
    print(model_base.evaluate(model_auto.predict(X_test), Y_test, verbose=0))
    
    batch_size = 256     
    generator = generator.flow(X_train, Y_train, batch_size=batch_size) 
    #########################
    x = tf.placeholder(tf.float32, shape=(None, 32, 32, 3))
    y = tf.placeholder(tf.float32, shape=(None, 10))
       
    logits = model_base(model_auto(x))._op.inputs[0]
    ce_loss = tf.losses.softmax_cross_entropy(y, logits)
    tot_loss = ce_loss  
    #########################

    step_size_schedule = [[0, 0.0001], [30000, 0.00001], [40000, 0.000008]]
    global_step = tf.train.get_or_create_global_step()
    boundaries = [int(sss[0]) for sss in step_size_schedule]
    boundaries = boundaries[1:]
    values = [sss[1] for sss in step_size_schedule]
    learning_rate = tf.train.piecewise_constant(
        tf.cast(global_step, tf.int32),
        boundaries,
        values) 
    
    optimizer = tf.train.AdamOptimizer(learning_rate, epsilon=1e-8)  
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
        pred_base_batch = model_base.predict(x_batch)
        sess.run([opt_op, model_auto.updates], feed_dict={y: pred_base_batch, x: x_batch, model_auto.inputs[0]: x_batch, K.learning_phase(): 1 })
        if (step % 100 == 0):
            print("step number: " + str(step)) 
            print(sess.run(tot_loss, feed_dict={x: X_train[0:1000], y: Y_train[0:1000]}))
            #Save best weights
            val_acc = model_base.evaluate(model_auto.predict(X_val), Y_val, verbose=0)[1]
            print("val_acc: " + str(val_acc))
            if (val_acc > tamp_val_acc):
                print("Best accuracy on validation set so far: " + str(val_acc))
                model_auto.save_weights("svhn_weights_best_ce.hdf5")
                tamp_val_acc = val_acc
                
    model_auto.load_weights("svhn_weights_best_ce.hdf5")    
    model_auto.save("models_p/SVHN_ce_p.h5")            
    


if (model_type == "luring_alt"):
    
    sess = tf.Session()
    K.set_session(sess)
    
    model_base = load_model("models/SVHN_float.h5")    
    model_base.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])  
    
    inputs = Input(shape=(32, 32, 3))
    l = Conv2D(128, (3, 3), padding='same')(inputs)
    l = Dropout(0.0)(l)
    l = BatchNormalization()(l)
    l = Activation('relu')(l)
    l = MaxPooling2D((2, 2), padding='same')(l)
    l = Conv2D(256, (3, 3), padding='same')(l)
    l = BatchNormalization()(l)
    l = Activation('relu')(l)
    l = MaxPooling2D((2, 2), padding='same')(l)
    l = Conv2D(512, (3, 3), padding='same')(l)
    l = BatchNormalization()(l)
    l = Activation('relu')(l)
    l = Conv2D(1024, (3, 3), padding='same')(l)
    l = BatchNormalization()(l)
    l = Activation('relu')(l)
    
    encoded = MaxPooling2D((2, 2), padding='same')(l)
    
    l = Conv2D(512, (3, 3), padding='same')(encoded)
    l = BatchNormalization()(l)
    l = Activation('relu')(l)
    l = Conv2D(512, (3, 3), padding='same')(l)
    l = BatchNormalization()(l)
    l = Activation('relu')(l)
    l = UpSampling2D((2, 2))(l)
    l = Conv2D(256, (3, 3), padding='same')(l)
    l = BatchNormalization()(l)
    l = Activation('relu')(l)
    l = UpSampling2D((2, 2))(l)
    l = Conv2D(128, (3, 3), padding='same')(l)
    l = BatchNormalization()(l)
    l = Activation('relu')(l)
    l = UpSampling2D((2, 2))(l)
    l = Conv2D(64, (3, 3), padding='same')(l)
    l = Dropout(0.0)(l)
    l = BatchNormalization()(l)
    l = Activation('relu')(l)
    l = Conv2D(3, (3, 3), padding='same')(l)
    l = BatchNormalization()(l)
    decoded = Activation('sigmoid')(l)

    model_auto = Model(inputs, decoded)   
    model_auto.compile(optimizer='adam', loss='binary_crossentropy') 

    ######################
    ######################
    #Training    
    print(model_base.evaluate(X_train, Y_train, verbose=0)) 
    print(model_base.evaluate(X_test, Y_test, verbose=0))
    
    print(model_base.evaluate(model_auto.predict(X_train), Y_train, verbose=0))
    print(model_base.evaluate(model_auto.predict(X_test), Y_test, verbose=0))
    
    batch_size = 256
    generator = generator.flow(X_train, Y_train, batch_size=batch_size) 
    #########################
    x = tf.placeholder(tf.float32, shape=(None, 32, 32, 3))
    y = tf.placeholder(tf.float32, shape=(None, 10))
    
    preds = model_base(x)
    logits = model_base(model_auto(x))._op.inputs[0]
    preds_sort = tf.contrib.framework.sort(preds, axis=1, direction="DESCENDING")
    preds_max = tf.gather(preds_sort, axis=1, indices=[0])
    preds_secondmax = tf.gather(preds_sort, axis=1, indices=[1])
    c_max = tf.equal(preds, preds_max)
    d_max = tf.where(c_max, logits, tf.fill(tf.shape(logits), -float('Inf')))
    logits_max = tf.reduce_max(d_max, axis=1, keepdims=True)
    c_secondmax = tf.equal(preds, preds_secondmax)
    d_secondmax = tf.where(c_secondmax, logits, tf.fill(tf.shape(logits), -float('Inf')))
    logits_secondmax = tf.reduce_max(d_secondmax, axis=1, keepdims=True)
    logits_diff = logits_max - logits_secondmax
    logits_loss = - tf.reduce_mean(logits_diff)
    tot_loss = logits_loss 
    #########################
    
    step_size_schedule = [[0, 0.0001], [30000, 0.00001], [40000, 0.000008]]
    global_step = tf.train.get_or_create_global_step()
    boundaries = [int(sss[0]) for sss in step_size_schedule]
    boundaries = boundaries[1:]
    values = [sss[1] for sss in step_size_schedule]
    learning_rate = tf.train.piecewise_constant(
        tf.cast(global_step, tf.int32),
        boundaries,
        values) 
    
    optimizer = tf.train.AdamOptimizer(learning_rate, epsilon=1e-8)  
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
        sess.run([opt_op, model_auto.updates], feed_dict={y: y_batch, x: x_batch, model_auto.inputs[0]: x_batch, K.learning_phase(): 1 })
        if (step % 100 == 0):
            print("step number: " + str(step)) 
            print(sess.run(tot_loss, feed_dict={x: X_train[0:1000], y: Y_train[0:1000]}))
            #Save best weights
            val_acc = model_base.evaluate(model_auto.predict(X_val), Y_val, verbose=0)[1]
            print("val_acc: " + str(val_acc))
            if (val_acc > tamp_val_acc):
                print("Best accuracy on validation set so far: " + str(val_acc))
                model_auto.save_weights("weights_best_luring_alt.hdf5")
                tamp_val_acc = val_acc
                
    model_auto.load_weights("weights_best_luring_alt.hdf5")    
    model_auto.save("models_p/SVHN_luring_alt_p.h5")        
    
    
    
if (model_type == "luring"):
    
    sess = tf.Session()
    K.set_session(sess)
    
    model_base = load_model("models/SVHN_float.h5")    
    model_base.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])  
    
    inputs = Input(shape=(32, 32, 3))
    l = Conv2D(128, (3, 3), padding='same')(inputs)
    l = Dropout(0.15)(l)
    l = BatchNormalization()(l)
    l = Activation('relu')(l)
    l = MaxPooling2D((2, 2), padding='same')(l)
    l = Conv2D(256, (3, 3), padding='same')(l)
    l = BatchNormalization()(l)
    l = Activation('relu')(l)
    l = MaxPooling2D((2, 2), padding='same')(l)
    l = Conv2D(512, (3, 3), padding='same')(l)
    l = BatchNormalization()(l)
    l = Activation('relu')(l)
    l = Conv2D(1024, (3, 3), padding='same')(l)
    l = BatchNormalization()(l)
    l = Activation('relu')(l)
    
    encoded = MaxPooling2D((2, 2), padding='same')(l)
    
    l = Conv2D(512, (3, 3), padding='same')(encoded)
    l = BatchNormalization()(l)
    l = Activation('relu')(l)
    l = Conv2D(512, (3, 3), padding='same')(l)
    l = BatchNormalization()(l)
    l = Activation('relu')(l)
    l = UpSampling2D((2, 2))(l)
    l = Conv2D(256, (3, 3), padding='same')(l)
    l = BatchNormalization()(l)
    l = Activation('relu')(l)
    l = UpSampling2D((2, 2))(l)
    l = Conv2D(128, (3, 3), padding='same')(l)
    l = BatchNormalization()(l)
    l = Activation('relu')(l)
    l = UpSampling2D((2, 2))(l)
    l = Conv2D(64, (3, 3), padding='same')(l)
    l = Dropout(0.15)(l)
    l = BatchNormalization()(l)
    l = Activation('relu')(l)
    l = Conv2D(3, (3, 3), padding='same')(l)
    l = BatchNormalization()(l)
    decoded = Activation('sigmoid')(l)

    model_auto = Model(inputs, decoded)   
    model_auto.compile(optimizer='adam', loss='binary_crossentropy') 
    
    print(model_base.evaluate(X_train, Y_train, verbose=0))
    print(model_base.evaluate(X_test, Y_test, verbose=0))
    
    print(model_base.evaluate(model_auto.predict(X_train), Y_train, verbose=0))
    print(model_base.evaluate(model_auto.predict(X_test), Y_test, verbose=0))

    batch_size = 256

    generator = generator.flow(X_train, Y_train, batch_size=batch_size)
    #########################
    x = tf.placeholder(tf.float32, shape=(None, 32, 32, 3))
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

    step_size_schedule = [[0, 0.0001], [30000, 0.00001], [40000, 0.000008]]
    global_step = tf.train.get_or_create_global_step()
    boundaries = [int(sss[0]) for sss in step_size_schedule]
    boundaries = boundaries[1:]
    values = [sss[1] for sss in step_size_schedule]
    learning_rate = tf.train.piecewise_constant(
        tf.cast(global_step, tf.int32),
        boundaries,
        values) 
    optimizer = tf.train.AdamOptimizer(learning_rate, epsilon=1e-8)  
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
        sess.run([opt_op, model_auto.updates], feed_dict={y: y_batch, x: x_batch, model_auto.inputs[0]: x_batch, K.learning_phase(): 1 })
        if (step % 1000 == 0):
            print("step number: " + str(step))          
            print(sess.run(tot_loss, feed_dict={x: X_train[0:1000], y: Y_train[0:1000]}))
            print(sess.run(tot_loss_2, feed_dict={x: X_train[0:1000], y: Y_train[0:1000]}))
            #Save best weights
            val_acc = model_base.evaluate(model_auto.predict(X_val), Y_val, verbose=0)[1]
            print("val_acc: " + str(val_acc))
            if (val_acc > tamp_val_acc):
                print("Best accuracy on validation set so far: " + str(val_acc))
                model_auto.save_weights("svhn_weights_best_luring.hdf5") 
                tamp_val_acc = val_acc
            
    model_auto.load_weights("svhn_weights_best_luring.hdf5")   
    model_auto.save("models_p/SVHN_luring_p.h5")     


##Save final model
if (model_type == "luring") | (model_type == "luring_alt") | (model_type == "ce"):
 
    model_base = load_model("models/SVHN_float.h5")  

    inputs = Input(shape=(32, 32, 3))
    l = Conv2D(128, (3, 3), padding='same')(inputs)
    l = Dropout(0.0)(l)
    l = BatchNormalization()(l)
    l = Activation('relu')(l)
    l = MaxPooling2D((2, 2), padding='same')(l)
    l = Conv2D(256, (3, 3), padding='same')(l)
    l = BatchNormalization()(l)
    l = Activation('relu')(l)
    l = MaxPooling2D((2, 2), padding='same')(l)
    l = Conv2D(512, (3, 3), padding='same')(l)
    l = BatchNormalization()(l)
    l = Activation('relu')(l)
    l = Conv2D(1024, (3, 3), padding='same')(l)
    l = BatchNormalization()(l)
    l = Activation('relu')(l) 
    encoded = MaxPooling2D((2, 2), padding='same')(l) 
    l = Conv2D(512, (3, 3), padding='same')(encoded)
    l = BatchNormalization()(l)
    l = Activation('relu')(l)
    l = Conv2D(512, (3, 3), padding='same')(l)
    l = BatchNormalization()(l)
    l = Activation('relu')(l)
    l = UpSampling2D((2, 2))(l)
    l = Conv2D(256, (3, 3), padding='same')(l)
    l = BatchNormalization()(l)
    l = Activation('relu')(l)
    l = UpSampling2D((2, 2))(l)
    l = Conv2D(128, (3, 3), padding='same')(l)
    l = BatchNormalization()(l)
    l = Activation('relu')(l)
    l = UpSampling2D((2, 2))(l)
    l = Conv2D(64, (3, 3), padding='same')(l)
    l = Dropout(0.0)(l)
    l = BatchNormalization()(l)
    l = Activation('relu')(l)
    l = Conv2D(3, (3, 3), padding='same')(l)
    l = BatchNormalization()(l)
    decoded = Activation('sigmoid')(l)
    
    l = Conv2D(64, kernel_size=(3,3), strides=(1,1),padding="valid", use_bias=False)(decoded)
    l = BatchNormalization()(l)
    l = Activation("relu")(l)
    l = Conv2D(64, kernel_size=(3,3), strides=(1,1),padding="same", use_bias=False)(l)
    l = MaxPooling2D(pool_size=(2,2), strides=(2,2))(l)
    l = BatchNormalization()(l)
    l = Activation("relu")(l)
    l = Conv2D(128, kernel_size=(3,3), strides=(1,1),padding="same", use_bias=False)(l)
    l = BatchNormalization()(l)
    l = Activation("relu")(l) 
    l = Conv2D(128, kernel_size=(3,3), strides=(1,1),padding="same", use_bias=False)(l)
    l = MaxPooling2D(pool_size=(2,2), strides=(2,2))(l)
    l = BatchNormalization()(l)
    l = Activation("relu")(l)
    l = Conv2D(256, kernel_size=(3,3), strides=(1,1),padding="same", use_bias=False)(l)
    l = BatchNormalization()(l)
    l = Activation("relu")(l)
    l = Conv2D(256, kernel_size=(3,3), strides=(1,1),padding="same", use_bias=False)(l)
    l = MaxPooling2D(pool_size=(2,2), strides=(2,2))(l)
    l = BatchNormalization()(l)
    l = Activation("relu")(l)
    l = Dropout(0.25)(l)
    l = Flatten()(l)
    l = Dense(1024, use_bias=False)(l)
    l = BatchNormalization()(l)
    l = Activation("relu")(l)
    l = Dropout(0.25)(l)
    l = Dense(1024, use_bias=False)(l)
    l = BatchNormalization()(l)
    l = Activation("relu")(l)
    predictions = Dense(10, activation="softmax", use_bias=False)(l)
    
    model_built = Model(inputs, predictions)
    
    for i in range(len(model_auto.layers)):
        model_built.layers[i].set_weights(model_auto.layers[i].get_weights())
    j = 1
    for i in range(len(model_auto.layers), len(model_built.layers)):
        model_built.layers[i].set_weights(model_base.layers[j].get_weights())
        j = j + 1
    
    model_built.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
    
    model_built.save("models/SVHN_" + model_type + ".h5")





