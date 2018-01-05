import tensorflow as tf
import numpy as np
import scipy.misc
import cv2
import skimage.transform
import sklearn.preprocessing

#This function performns a leaky relu activation, which is needed for the discriminator network.
def lrelu(x, leak=0.2, name="lrelu"):
     with tf.variable_scope(name):
         f1 = 0.5 * (1 + leak)
         f2 = 0.5 * (1 - leak)
         return f1 * x + f2 * abs(x)

#The below functions are taken from carpdem20's implementation https://github.com/carpedm20/DCGAN-tensorflow
#They allow for saving sample images from the generator to follow progress
def save_images(images, size, image_path):
    return imsave(images, size, image_path)
    #return imsave(inverse_transform(images), size, image_path)

def filt(images):
    norm = sklearn.preprocessing.minmax_scale(np.reshape(images,(32,32)), (-0.999,0.999))
    rescale = skimage.transform.rescale(norm,1.625)
    kernel = np.ones((1,1),np.float32)/25
    return cv2.filter2D(rescale,-1,kernel)

def imsave(images, size, path):
    return scipy.misc.imsave(path, merge(images, size))

def inverse_transform(images):
    return (images+1.)/2.

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1]))

    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j*h:j*h+h, i*w:i*w+w] = image
    return img

# For getting batches
def next_batch(num, data, labels):
    '''
    Return a total of `num` random samples and labels.
    '''
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[ i] for i in idx]
    labels_shuffle = [labels[ i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)

# For getting batches with a specific diagnosis
def next_batch_cond(num, data, labels, cond):
    '''
    Return a total of `num` random samples and labels.
    '''
    data_con=[]
    label_con = []
    #idcon = np.arange(0,len(data))
    for j in np.arange(0,len(data)):
        if labels[j] == cond:
            data_con.append(data[j])
            label_con.append(labels[j])


    idx = np.arange(0,len(data_con))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data_con[i] for i in idx]
    labels_shuffle = [label_con[i] for i in idx]
    return np.asarray(data_shuffle), np.asarray(labels_shuffle)
