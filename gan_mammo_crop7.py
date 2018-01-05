import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.contrib.slim as slim
import os
import scipy.misc
import scipy
import readMammograms
import random
import cv2
import skimage.transform
import sklearn.preprocessing
import helper
import pdb

print("Loading in Mammogram data...")
mgrams = readMammograms.readData(0)
mgram_data = mgrams.data
mgram_labels = mgrams.labels
print("Mammogram data loaded.\n")


# Define generator network
def generator(z):

    zP = slim.fully_connected(z,4*4*256,normalizer_fn=slim.batch_norm,
        activation_fn=tf.nn.relu,scope='g_project',weights_initializer=initializer)

    zCon = tf.reshape(zP,[-1,4,4,256])

    gen1 = slim.convolution2d_transpose(zCon,num_outputs=64,kernel_size=[5,5],
        stride=[2,2],padding="SAME",normalizer_fn=slim.batch_norm,activation_fn=tf.nn.relu,
        scope='g_conv1', weights_initializer=initializer)

    gen2 = slim.convolution2d_transpose(gen1,num_outputs=32,kernel_size=[5,5],
        stride=[2,2],padding="SAME",normalizer_fn=slim.batch_norm,activation_fn=tf.nn.relu,
        scope='g_conv2', weights_initializer=initializer)

    gen3 = slim.convolution2d_transpose(gen2,num_outputs=16,kernel_size=[5,5],
        stride=[2,2],padding="SAME",normalizer_fn=slim.batch_norm,activation_fn=tf.nn.relu,
        scope='g_conv3', weights_initializer=initializer)

    #gen4 = slim.convolution2d_transpose(gen3,num_outputs=8,kernel_size=[5,5],
        #stride=[2,2],padding="SAME",normalizer_fn=slim.batch_norm,activation_fn=tf.nn.relu,
        #scope='g_conv4', weights_initializer=initializer)

    #gen5 = slim.convolution2d_transpose(gen4,num_outputs=4,kernel_size=[5,5],
#        stride=[1,1],padding="SAME",normalizer_fn=slim.batch_norm,activation_fn=tf.nn.relu,
#        scope='g_conv5', weights_initializer=initializer)

    g_out = slim.convolution2d_transpose(gen3,num_outputs=1,kernel_size=[32,32],
        padding="SAME",biases_initializer=None,activation_fn=tf.nn.tanh,scope='g_out',
        weights_initializer=initializer)

    return g_out


# Define descriminator network
def discriminator(bottom, reuse=False):

    dis1 = slim.convolution2d(bottom,16,[4,4],stride=[2,2],padding="SAME",
        biases_initializer=None,activation_fn=helper.lrelu,reuse=reuse,scope='d_conv1',
        weights_initializer=initializer)

    dis2 = slim.convolution2d(dis1,32,[4,4],stride=[2,2],padding="SAME",
        normalizer_fn=slim.batch_norm,activation_fn=helper.lrelu,reuse=reuse,scope='d_conv2',
        weights_initializer=initializer)

    dis3 = slim.convolution2d(dis2,64,[4,4],stride=[2,2],padding="SAME",
        normalizer_fn=slim.batch_norm,activation_fn=helper.lrelu,reuse=reuse,scope='d_conv3',
        weights_initializer=initializer)

    #dis4 = slim.convolution2d(dis2,64,[4,4],stride=[1,1],padding="SAME",
#        normalizer_fn=slim.batch_norm,activation_fn=helper.lrelu,reuse=reuse,scope='d_conv4',
#        weights_initializer=initializer)

    d_out = slim.fully_connected(slim.flatten(dis3),1,activation_fn=tf.nn.sigmoid,
    reuse=reuse,scope='d_out', weights_initializer=initializer)

    return d_out


# Connecting generator and discriminator networks
tf.reset_default_graph()

z_size = 100 #Size of z vector used for generator.

#This initializaer is used to initialize all the weights of the network.
initializer = tf.truncated_normal_initializer(stddev=0.02)

#These two placeholders are used for input into the generator and discriminator, respectively.
z_in = tf.placeholder(shape=[None,z_size],dtype=tf.float32) #Random vector
real_in = tf.placeholder(shape=[None,32,32,1],dtype=tf.float32) #Real images

Gz = generator(z_in) #Generates images from random z vectors
Dx = discriminator(real_in) #Produces probabilities for real images
Dg = discriminator(Gz,reuse=True) #Produces probabilities for generator images

#These functions together define the optimization objective of the GAN.
d_loss = -tf.reduce_mean(tf.log(Dx) + tf.log(1.-Dg)) #This optimizes the discriminator.
g_loss = -tf.reduce_mean(tf.log(Dg)) #This optimizes the generator.

tvars = tf.trainable_variables()

#The below code is responsible for applying gradient descent to update the GAN.
trainerD = tf.train.AdamOptimizer(learning_rate=0.0002,beta1=0.5)
trainerG = tf.train.AdamOptimizer(learning_rate=0.0002,beta1=0.5)
d_grads = trainerD.compute_gradients(d_loss,tvars[9:]) #Only update the weights for the discriminator network.
g_grads = trainerG.compute_gradients(g_loss,tvars[0:9]) #Only update the weights for the generator network.

update_D = trainerD.apply_gradients(d_grads)
update_G = trainerG.apply_gradients(g_grads)



# Start training loop
batch_size = 12 # Size of image batch to apply at each iteration.
iterations = 2000 # Total number of iterations to use.
cond =2 # Which condition to use: 0 = normal, 1 = benign, 2 = cancerous
sample_directory = './figs_cancer2' # Directory to save sample images from generator in.
model_directory = './models_cancer2' # Directory to save trained model to.

init = tf.global_variables_initializer()
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(init)
    print("Training start")

    for i in range(iterations):

        zs = np.random.uniform(-1.0,1.0,size=[batch_size,z_size]).astype(np.float32) #Generate a random z batch

        xs,_ = helper.next_batch_cond(batch_size, mgram_data, mgram_labels, cond)
        xs = (np.reshape(xs,[batch_size,32,32,1]) - 0.5) * 2.0 #Transform it to be between -1 and 1
        _,dLoss = sess.run([update_D,d_loss],feed_dict={z_in:zs,real_in:xs}) #Update the discriminator
        _,gLoss = sess.run([update_G,g_loss],feed_dict={z_in:zs}) #Update the generator, twice for good measure.
        _,gLoss = sess.run([update_G,g_loss],feed_dict={z_in:zs})
        if i % 1 == 0:
            print("Gen Loss " + str(gLoss) + " Disc Loss: " + str(dLoss))
            z2 = np.random.uniform(-1.0,1.0,size=[batch_size,z_size]).astype(np.float32) #Generate another z batch
            newZ = sess.run(Gz,feed_dict={z_in:z2}) #Use new z to get sample images from generator.
            newZ_filt = helper.filt(newZ[0])
            if not os.path.exists(sample_directory):
                os.makedirs(sample_directory)
            #Save sample generator images for viewing training progress.
            #newZint = interpolate(newZ[0])
            helper.save_images(np.reshape(newZ_filt,[1,52,52]),[1,1],sample_directory+'/can'+str(i)+'.png')
        if i % 100 == 0 and i != 0:
            if not os.path.exists(model_directory):
                os.makedirs(model_directory)
            saver.save(sess,model_directory+'/model-'+str(i)+'.cptk')
            print("Saved Model")

pdb.set_trace()
# Loading previous network to generate additional images
sample_directory = './figs_cancer_synth' #Directory to save sample images from generator in.
model_directory = './models_cancer' #Directory to load trained model from.
batch_size_sample = 20
path = '/Users/michaelcraig/software/Generative-Adversarial-Network-Tutorial/models_cancer/'
init = tf.global_variables_initializer()
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(init)
    #Reload the model.
    print('Loading Model...')
    ckpt = tf.train.get_checkpoint_state(path)
    saver.restore(sess,ckpt.model_checkpoint_path)

    for j in range(1000):
        pdb.set_trace()
        zs = np.random.uniform(-1.0,1.0,size=[batch_size_sample,z_size]).astype(np.float32) #Generate a random z batch
        newZ = sess.run(Gz,feed_dict={z_in:zs}) #Use new z to get sample images from generator.
        newZ_filt = helper.filt(newZ[0])
        if not os.path.exists(sample_directory):
            os.makedirs(sample_directory)
        helper.save_images(np.reshape(newZ_filt,[1,52,52]),[1,1],sample_directory+'/fig'+str(j)+'.png')
