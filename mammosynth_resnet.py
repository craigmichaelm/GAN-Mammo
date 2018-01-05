import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
import os
import readMammograms
import readsynthMammograms
import random
import cv2
import sklearn.model_selection
import helper
import pdb

# Runs a binary classification between cancer and noncancer patches with GAN data augmentation
# Densenet architecture

print("Loading in Mammogram data...")
mgrams = readMammograms.readData(0)
mgram_data = mgrams.data
mgram_labels = mgrams.labels
print("Mammogram data loaded.\n")

# Create training and test sets
X_tr, X_test, y_tr, y_test = sklearn.model_selection.train_test_split(mgram_data,mgram_labels,test_size=0.1,random_state=10)

# Importing synthetic mammograms for training
print("Loading synthetic mammograms")
synth = readsynthMammograms.readData(3)
synth_data = synth.data
synth_labels = synth.labels

X_train = X_tr + synth_data
y_train = y_tr + synth_labels
print("Real data augmented with synthetic data")


# Define number of layers
total_layers = 25 #Specify how deep we want our network
units_between_stride = total_layers / 5

# Define resnet
def resUnit(input_layer,i):
    with tf.variable_scope("res_unit"+str(i)):
        part1 = slim.batch_norm(input_layer,activation_fn=None)
        part2 = tf.nn.relu(part1)
        part3 = slim.conv2d(part2,64,[3,3],activation_fn=None)
        part4 = slim.batch_norm(part3,activation_fn=None)
        part5 = tf.nn.relu(part4)
        part6 = slim.conv2d(part5,64,[3,3],activation_fn=None)
        output = input_layer + part6
        return output

tf.reset_default_graph()

input_layer = tf.placeholder(shape=[None,32,32,1],dtype=tf.float32,name='input')
label_layer = tf.placeholder(shape=[None],dtype=tf.int32)
label_oh = slim.layers.one_hot_encoding(label_layer,10)
layer1 = slim.conv2d(input_layer,64,[3,3],normalizer_fn=slim.batch_norm,scope='conv_'+str(0))
for i in range(5):
    for j in range(int(units_between_stride)):
        layer1 = resUnit(layer1,j + (i*units_between_stride))
    layer1 = slim.conv2d(layer1,64,[3,3],stride=[2,2],normalizer_fn=slim.batch_norm,scope='conv_s_'+str(i))

top = slim.conv2d(layer1,10,[3,3],normalizer_fn=slim.batch_norm,activation_fn=None,scope='conv_top')

output = slim.layers.softmax(slim.layers.flatten(top))

loss = tf.reduce_mean(-tf.reduce_sum(label_oh * tf.log(output) + 1e-10, axis=[1]))
trainer = tf.train.AdamOptimizer(learning_rate=0.005)
update = trainer.minimize(loss)


# Training networks
init = tf.global_variables_initializer()
batch_size = 64
#currentCifar = 1
total_steps = 2001
l = []
a = []
aT = []
with tf.Session() as sess:
    sess.run(init)
    i = 1
    #draw = range(10000)
    while i < total_steps:

        x,y = helper.next_batch(batch_size, X_train, y_train)
        x = np.reshape(x,(batch_size,32,32,1),order='F')
        #x = (x/256.0)
        x = (x - np.mean(x,axis=0)) / np.std(x,axis=0)
        y = y[:] - 1
        #y = np.reshape(np.array(cifar['labels'])[batch_index],[batch_size,1])
        _,lossA,yP,LO = sess.run([update,loss,output,label_oh],feed_dict={input_layer:x,label_layer:np.hstack(y)})
        accuracy = np.sum(np.equal(np.hstack(y),np.argmax(yP,1)))/float(len(y))
        l.append(lossA)
        a.append(accuracy)


        if i % 1 == 0: print("Step: " + str(i) + " Loss: " + str(lossA) + " Accuracy: " + str(accuracy))
        if i % 2000 == 0:
            xT,yT = helper.next_batch(len(y_test), X_test, y_test)
            xT = np.reshape(xT,[-1,32,32,1],order='F')
            xT = (xT - np.mean(xT,axis=0)) / np.std(xT,axis=0)
            yT = yT[:] - 1
            lossT,yP = sess.run([loss,output],feed_dict={input_layer:xT,label_layer:yT})
            accuracy = np.sum(np.equal(yT,np.argmax(yP,1)))/float(len(yT))
            aT.append(accuracy)
            print("Test set accuracy: " + str(accuracy))
        i+= 1
