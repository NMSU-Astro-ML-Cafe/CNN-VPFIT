# -*- coding: utf-8 -*-
"""
two cloud model           
"""

import numpy as np
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import random
from tensorflow.keras import layers, regularizers
from sklearn.model_selection import train_test_split
import sklearn
from sklearn.metrics import r2_score
#from keras import layers, regularizers
from tensorflow.keras.models import Model
from tensorflow import keras



#from google.colab import files
#uploaded = files.upload()

#d             = np.load('data.zip') # set up to work with CIV, but can change to MgII, Lya, OVI, whatever you like :)
spec1 = np.loadtxt('./data/MgII2796data.txt')
spec2 = np.loadtxt('./data/MgII2803data.txt')
labels = np.loadtxt('./data/labels.txt')


for i in range(6):
  print (labels[:,i].shape)
  labels[:,i] -=  np.mean(labels[:,i])
  labels[:,i] /=  np.std(labels[:,i])


spec = np.concatenate( (np.expand_dims(spec1, axis=2), np.expand_dims(spec2, axis=2)), axis=2)
(1000000, 450, 2)

#print(spec)

#labels: velocity1, logN1, doppler b1, velocity2, logN2, doppler b2
    
'''
#data augmentation to increase input data set
specs_to_add=30000
arr = np.zeros([specs_to_add,250])
arr_labels = np.zeros([specs_to_add,6])
for i in range(specs_to_add):
    shift = np.random.randint(specs_pre_aug.shape[1])
    row = np.random.randint(specs_pre_aug.shape[0])
    arr[i]=np.roll(specs_pre_aug[row], shift)
    arr_labels[i]=labels_pre_aug[row]
specs=np.concatenate((specs_pre_aug, arr), axis=0)    
labels=np.concatenate((labels_pre_aug, arr_labels), axis=0)
'''   
 
num_specs     = spec.shape[0]  
spec_pixels   = spec.shape[1]
rad           = np.random.randint(num_specs)
example_spec  = spec[rad]
example_label = labels[rad]


#fig, ax = plt.subplots(1)
#ax.plot(range(len(example_spec)), example_spec)
#fig, ax1 = plt.subplots(1)
#ax1.plot(range(len(labels[:,2])), labels[:,2])

bins     = np.linspace(0, len(labels), 32)
y_binned = np.digitize(labels, bins)

"random spec", rad, "total number", num_specs, "pixels", spec_pixels , "line properties", example_label

x_train, x_test, y_train, y_test = train_test_split(spec,labels,test_size=0.1,random_state=42)

x_train.shape, x_test.shape, y_train.shape, y_test.shape
#x_train = np.expand_dims(x_train,axis=2) # making 3D for keras model
#x_test = np.expand_dims(x_test,axis=2) # making 3D for keras model

num_input      = 450
num_classes    = 6
epochs   = 20
batch_size       = 32

#tf.reset_default_graph()

inputs = layers.Input(shape=(450,2))



x      = layers.Conv1D(filters=16, kernel_size=3, use_bias=False,padding='same') (inputs)
x      = layers.BatchNormalization() (x)
x      = layers.ReLU() (x)
x      = layers.Conv1D(filters=32, kernel_size=3, use_bias=False,padding='same') (x)
x      = layers.BatchNormalization() (x)
x      = layers.ReLU() (x)
x      = layers.Flatten() (x)
x      = layers.Dense(units=1024, use_bias=False) (x)
x      = layers.BatchNormalization() (x)
x      = layers.ReLU() (x)
x      = layers.Dense(units=1024, use_bias=False) (x)
x      = layers.BatchNormalization() (x)
x      = layers.ReLU() (x)
x      = layers.Dense(units=1024, use_bias=False) (x)
x      = layers.BatchNormalization() (x)
x      = layers.ReLU() (x)
out1   = layers.Dense(units=3, use_bias=False) (x)


y      = layers.Conv1D(filters=16, kernel_size=3, use_bias=False,padding='same') (inputs)
y      = layers.BatchNormalization() (y)
y      = layers.ReLU() (y)
y      = layers.Conv1D(filters=32, kernel_size=3, use_bias=False,padding='same') (y)
y      = layers.BatchNormalization() (y)
y      = layers.ReLU() (y)
y      = layers.Flatten() (y)
y      = layers.Dense(units=1024, use_bias=False) (y)
y      = layers.BatchNormalization() (y)
y      = layers.ReLU() (y)
y      = layers.Dense(units=1024, use_bias=False) (y)
y      = layers.BatchNormalization() (y)
y      = layers.ReLU() (y)
y      = layers.Dense(units=1024, use_bias=False) (y)
y      = layers.BatchNormalization() (y)
y      = layers.ReLU() (y)
out2   = layers.Dense(units=3, use_bias=False) (y)



model  = Model(inputs=[inputs], outputs=[out1, out2])

model.compile(loss=['mse','mse'],optimizer=keras.optimizers.RMSprop(lr=1e-6,rho=0.9))
#keras.optimizers.Adam(lr=1e-3))
#optimizer=tf.keras.optimizers.Adam(lr=1e-6))#, decay=1e-4))#,metrics=['mae'])

model.fit(x_train, [y_train[:,:3],y_train[:,3:]], batch_size=batch_size,epochs=epochs,verbose=2,validation_data=(x_test, [y_test[:,:3],y_test[:,3:]]))

model.summary() # print information about the model layer types, shapes, number of parameters

final_pred = model.predict(x_test, batch_size=batch_size, verbose=2) # use model.predict to make predictions for some subset of the data

model.save('./model_2cloud_cnn_epoch_20_lr_e-6_RMSprop_rho_0.9')


fig, ax = plt.subplots(1,num_classes, figsize=(20,4))
nn=0
ii=0 
nn_1=0     
titles = ["Velocity1", "logN1", "b1","Velocity2", "logN2", "b2"]
for ii in range(num_classes):
    print(ii)
    if(nn<=2):
        R2  = r2_score(y_test[:,nn],final_pred[0][:,nn])
        ax[ii].set_title(titles[ii])
        ax[ii].set_ylabel('the predicted values')
        ax[ii].set_xlabel('the true values')
        ax[ii].annotate('R2='+('%0.3f' % R2 )+'',xy=(0.05,0.9),xycoords=
          'axes fraction',bbox=dict(boxstyle="round",fc="w"), size=14)
        ax[ii].plot(y_test[:,nn], y_test[:,nn], 'k-') # where the trend should be
        ax[ii].plot(y_test[:,nn], final_pred[0][:,nn], 'bo', mfc='white', alpha=0.5)
        nn+=1
    elif(nn>2):
        R2  = r2_score(y_test[:,nn],final_pred[1][:,nn_1])
        ax[ii].set_title(titles[ii])
        ax[ii].set_ylabel('the predicted values')
        ax[ii].set_xlabel('the true values')
        ax[ii].annotate('R2='+('%0.3f' % R2 )+'',xy=(0.05,0.9),xycoords=
          'axes fraction',bbox=dict(boxstyle="round",fc="w"), size=14)
        ax[ii].plot(y_test[:,nn], y_test[:,nn], 'k-') # where the trend should be
        ax[ii].plot(y_test[:,nn], final_pred[1][:,nn_1], 'bo', mfc='white', alpha=0.5)
        nn_1+=1
        nn+=1
fig.tight_layout()
fig.savefig('2cloud_cnn_epoch_20_lr_e-6_RMSprop_rho_0point9.png')



