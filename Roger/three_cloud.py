# -*- coding: utf-8 -*-
"""
three cloud model data pipe

CNN uses the Euclidean metric to remove spectra that lead to high R2 scores.
Remaining spectra are written to a new file.

4/18/2020  Modified two cloud script to predict labels for three cloud spectra.
           R2 scores for second cloud label predictions were worse than others. 
4/19/2020  Used euclidian distance in 3D space (coordinates: velocity,logN,b) 
           between labels and predictions to filter out spectra that the CNN had 
           trouble predicting labels for. Said spectra had weird blended systems, 
           where one cloud was superimposed on another. In some spectra, it seems
           only two clouds were present. Other spectra had saturation.
4/22/2020  Created a data pipe (path) to strip weird blended spectra from testing
           data only. Predicted a second time and results were generally better.
           All R2 > 0.85, 2nd cloud logN and b R2 were the only ones < 90%
4/23/2020  Merged one cloud, two cloud, and three cloud spectra and labels and 
           filled missing labels with zeros where appropriately to force CNN to
           predict 0 for missing clouds. Testing CNN on discovery with 
           1M spectra and 40 epochs. Results worse. Weird line artifact. 
4/24/2020  Changing prediction plot to heat map to better observe scatter in 
           current plots. Look up Caitlin's proposed idea and implement. 
4/25/2020  
           
           
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
from sklearn.metrics import mean_absolute_error
from sklearn.metrics.pairwise import paired_distances
#from keras import layers, regularizers
from tensorflow.keras.models import Model
from tensorflow import keras
import matplotlib.colors as mcol
import matplotlib.cm as cm


#from google.colab import files
#uploaded = files.upload()

#d             = np.load('data.zip') # set up to work with CIV, but can change to MgII, Lya, OVI, whatever you like :)
spec1 = np.loadtxt('./data3/MgII2796data_10ksamples.txt')
spec2 = np.loadtxt('./data3/MgII2803data_10ksamples.txt')
labels = np.loadtxt('./data3/labels_10ksamples.txt')


for i in range(9):
  #print (labels[:,i].shape)
  labels[:,i] -=  np.mean(labels[:,i])
  labels[:,i] /=  np.std(labels[:,i])
  
  
spec = np.concatenate( (np.expand_dims(spec1, axis=2), np.expand_dims(spec2, axis=2)), axis=2)
(10000, 450, 2)

#print(spec)

#labels: velocity1, logN1, doppler b1, velocity2, logN2, doppler b2, velocity3, logN3, doppler b3
    
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

x_train, x_test, y_train, y_test = train_test_split(spec,labels,test_size=0.1, random_state=42)

x_train.shape, x_test.shape, y_train.shape, y_test.shape
#x_train = np.expand_dims(x_train,axis=2) # making 3D for keras model
#x_test = np.expand_dims(x_test,axis=2) # making 3D for keras model

num_input      = 450
num_classes    = 9
epochs   = 1
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

z      = layers.Conv1D(filters=16, kernel_size=3, use_bias=False,padding='same') (inputs)
z      = layers.BatchNormalization() (z)
z      = layers.ReLU() (z)
z      = layers.Conv1D(filters=32, kernel_size=3, use_bias=False,padding='same') (z)
z      = layers.BatchNormalization() (z)
z      = layers.ReLU() (z)
z      = layers.Flatten() (z)
z      = layers.Dense(units=1024, use_bias=False) (z)
z      = layers.BatchNormalization() (z)
z      = layers.ReLU() (z)
z      = layers.Dense(units=1024, use_bias=False) (z)
z      = layers.BatchNormalization() (z)
z      = layers.ReLU() (z)
z      = layers.Dense(units=1024, use_bias=False) (z)
z      = layers.BatchNormalization() (z)
z      = layers.ReLU() (z)
out3   = layers.Dense(units=3, use_bias=False) (z)

model  = Model(inputs=[inputs], outputs=[out1, out2, out3])

model.compile(loss=['mse','mse','mse'],optimizer=keras.optimizers.RMSprop(lr=1e-6,rho=0.9))
#keras.optimizers.Adam(lr=1e-3))
#optimizer=tf.keras.optimizers.Adam(lr=1e-5))#, decay=1e-4))#,metrics=['mae'])

model.fit(x_train, [y_train[:,:3],y_train[:,3:6],y_train[:,6:9]], batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(x_test, [y_test[:,:3],y_test[:,3:6],y_test[:,6:9]]))

#model.summary() # print information about the model layer types, shapes, number of parameters

final_pred = model.predict(x_test, batch_size=batch_size, verbose=0) # use model.predict to make predictions for some subset of the data

#model.save('./model_3cloud_cnn_epoch_40_lr_e-6_RMSprop_rho_0.9')


fig, ax = plt.subplots(3,3, figsize=(24,20))
#fig, ax_test = plt.subplots()
#fig, ax_test1 = plt.subplots()
#fig, ax_test2 = plt.subplots()
#fig, ax_test3 = plt.subplots()

nn=0
ii=0 
nn_1=0  
nn_2=0   
titles = ["Velocity1", "logN1", "b1","Velocity2", "logN2", "b2",
          "Velocity3", "logN3", "b3"]

#x_axis = np.linspace(0,450,450)
#ax_test.plot(x_axis,example_spec)
#ax_test1.plot(x_axis, x_test[2,:])
#ax_test1.plot(x_axis,spec1[rad,:])
#ax_test2.plot(x_axis,x_test[3,:,0])
#ax_test3.plot(x_axis,spec[3,:,0])

eucl_distance1 = paired_distances(y_test[:,:3],final_pred[0][:,:])
eucl_distance2 = paired_distances(y_test[:,3:6],final_pred[1][:,:])
eucl_distance3 = paired_distances(y_test[:,6:9],final_pred[2][:,:])

'''
count1=0
count2=0
count3=0

for index_test1, dist_test1 in enumerate(eucl_distance1):
    if(dist_test1>2.2):
        count1+=1
    #print('eucl distance1: ', i)
for index_test2, dist_test2 in enumerate(eucl_distance2):
    if(dist_test2>2.2):
        count2+=1
    #print('eucl distance2: ', i)
for index_test3, dist_test3 in enumerate(eucl_distance3):
    if(dist_test3>2.2):
        count3+=1
    #print('eucl distance3: ', i)
'''
'''
for index_test, dist_test in enumerate(eucl_distance1):
    if(dist_test<0.5):
        count4+=1
'''

x_test_modified1=[]
x_test_modified2=[]
x_test_modified3=[]
y_test_modified=[]
#x_test_modified=[[],[]]
#x_test_modified2.append([])

#for index1, dist1 in enumerate(eucl_distance1):
#    if(dist1<2.0):
#        x_test_modified1.append(x_test[index1,:])              
#num_spectra=0
for index2, dist2 in enumerate(eucl_distance2):
    if(dist2<2.0):
        x_test_modified2.append(np.expand_dims(x_test[index2], axis=0))
        y_test_modified.append(np.expand_dims(y_test[index2], axis=0))
        #num_spectra+=1
    
#for index3, dist3 in enumerate(eucl_distance3):
#    if(dist3<2.0):
#        x_test_modified3.append(x_test[index3,:])

x_test_modified2 = np.concatenate(x_test_modified2,axis=0)
y_test_modified = np.concatenate(y_test_modified,axis=0)
final_pred2 = model.predict(x_test_modified2, batch_size=batch_size, verbose=0)

#np.savetxt('./data3/MgII2796data_1ksamples_modified.txt', x_test_modified1 , delimiter=' ',newline='\n')
#np.savetxt('./data3/MgII2796data_1ksamples_modified.txt', x_test_modified1 , delimiter=' ',newline='\n')
#np.savetxt('./data3/MgII2796data_1ksamples_modified.txt', x_test_modified1 , delimiter=' ',newline='\n')




'''
fig4, ax4 = plt.subplots(count4,figsize=(10,40))
#fig.suptitle('Spectra of bad predictions')
fig4_number_plts1=0
fig4_number_plts2=0

for index, dist in enumerate(eucl_distance1):
    if(dist<0.5):
        ax4[fig4_number_plts1].plot(x_test[index,:,0])
        #ax[number_plts2].plot(x_test[index,:,1])
        fig4_number_plts1+=1
        fig4_number_plts2+=1
fig4.tight_layout()
fig4.savefig('test4.png')  
'''        

cm1 = mcol.LinearSegmentedColormap.from_list("MyCmapName",["r","y","g","b"])
cnorm1 = mcol.Normalize(vmin=min(y_test_modified[:,0]),vmax=max(y_test_modified[:,0]))
cnorm2 = mcol.Normalize(vmin=min(y_test_modified[:,1]),vmax=max(y_test_modified[:,1]))
cnorm3 = mcol.Normalize(vmin=min(y_test_modified[:,2]),vmax=max(y_test_modified[:,2]))
cpick1 = cm.ScalarMappable(norm=cnorm1,cmap=cm1)
cpick2 = cm.ScalarMappable(norm=cnorm2,cmap=cm1)
cpick3 = cm.ScalarMappable(norm=cnorm3,cmap=cm1)
cpick1.set_array([])
cpick2.set_array([])
cpick3.set_array([])


for ii in range(num_classes):
    if(nn<=2):
        R2  = r2_score(y_test_modified[:,nn],final_pred2[0][:,nn])            
        ax[0,ii].set_title(titles[ii])
        ax[0,ii].set_ylabel('the predicted values')
        ax[0,ii].set_xlabel('the true values')
        ax[0,ii].annotate('R2='+('%0.3f' % R2 )+'',xy=(0.05,0.9),xycoords=
          'axes fraction',bbox=dict(boxstyle="round",fc="w"), size=14)
        ax[0,ii].plot(y_test_modified[:,nn], y_test_modified[:,nn], 'k-') # where the trend should be
        ax[0,ii].scatter(y_test_modified[:,nn], final_pred2[0][:,nn], c=cpick1.to_rgba(len(y_test_modified[:,nn])), s=.1, alpha=0.5)
        nn+=1
        cb=plt.colorbar(cpick1,ax=ax[0,ii])
        #print('ax[ii]:', ax[ii])
    elif(2<nn<=5):
        R2  = r2_score(y_test_modified[:,nn],final_pred2[1][:,nn_1])
        ax[1,ii-3].set_title(titles[ii])
        ax[1,ii-3].set_ylabel('the predicted values')
        ax[1,ii-3].set_xlabel('the true values')
        ax[1,ii-3].annotate('R2='+('%0.3f' % R2 )+'',xy=(0.05,0.9),xycoords=
          'axes fraction',bbox=dict(boxstyle="round",fc="w"), size=14)
        ax[1,ii-3].plot(y_test_modified[:,nn], y_test_modified[:,nn], 'k-') # where the trend should be
        ax[1,ii-3].scatter(y_test_modified[:,nn], final_pred2[1][:,nn_1], c=cpick2.to_rgba(len(y_test_modified[:,nn])), s=.1, alpha=0.5)
        nn_1+=1
        nn+=1
        cb=plt.colorbar(cpick2,ax=ax[1,ii-3])
    elif(nn>5):
        R2  = r2_score(y_test_modified[:,nn],final_pred2[2][:,nn_2])
        ax[2,ii-6].set_title(titles[ii])
        ax[2,ii-6].set_ylabel('the predicted values')
        ax[2,ii-6].set_xlabel('the true values')
        ax[2,ii-6].annotate('R2='+('%0.3f' % R2 )+'',xy=(0.05,0.9),xycoords=
          'axes fraction',bbox=dict(boxstyle="round",fc="w"), size=14)
        ax[2,ii-6].plot(y_test_modified[:,nn], y_test_modified[:,nn], 'k-') # where the trend should be
        ax[2,ii-6].scatter(y_test_modified[:,nn], final_pred2[2][:,nn_2], c=cpick3.to_rgba(len(y_test_modified[:,nn])), s=.1, alpha=0.5)
        nn_2+=1
        nn+=1
        cb=plt.colorbar(cpick3,ax=ax[2,ii-6])
#cb=plt.colorbar(cpick1)#,label="Mass ($M_\odot$)")
#cb=plt.colorbar(cpick2)#,label="Mass ($M_\odot$)")
#cb=plt.colorbar(cpick3)#,label="Mass ($M_\odot$)")
#cb.set_label(label="Mass ($M_\odot$)",fontsize=20,labelpad=30)
fig.tight_layout()
fig.savefig('heatmap_test.png')

