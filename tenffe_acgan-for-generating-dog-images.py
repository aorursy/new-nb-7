
from tqdm import tqdm_notebook



ComputeLB = True

DogsOnly = True



import numpy as np, pandas as pd, os

import xml.etree.ElementTree as ET 

import matplotlib.pyplot as plt, zipfile 

from PIL import Image 



ROOT = '../input/generative-dog-images/'

if not ComputeLB: ROOT = '../input/'

dogs_path = ROOT + 'all-dogs/all-dogs/'

IMAGES = os.listdir(dogs_path)

breeds = os.listdir(ROOT + 'annotation/Annotation/') 



idxIn = 0; namesIn = []

imagesIn = np.zeros((25000, 64, 64, 3))

# imagesIn = np.zeros((25000, 3, 64, 64))



# CROP WITH BOUNDING BOXES TO GET DOGS ONLY

# https://www.kaggle.com/paulorzp/show-annotations-and-breeds

if DogsOnly:

    for breed in tqdm_notebook(breeds):

        for dog in os.listdir(ROOT+'annotation/Annotation/'+breed):

            try: img = Image.open(ROOT+'all-dogs/all-dogs/'+dog+'.jpg') 

            except: continue           

            tree = ET.parse(ROOT+'annotation/Annotation/'+breed+'/'+dog)

            root = tree.getroot()

            objects = root.findall('object')

            for o in objects:

                bndbox = o.find('bndbox') 

                xmin = int(bndbox.find('xmin').text)

                ymin = int(bndbox.find('ymin').text)

                xmax = int(bndbox.find('xmax').text)

                ymax = int(bndbox.find('ymax').text)

                w = np.min((xmax - xmin, ymax - ymin))

                img2 = img.crop((xmin, ymin, xmin+w, ymin+w))

                img2 = img2.resize((64,64), Image.ANTIALIAS)

                imagesIn[idxIn,:,:,:] = np.asarray(img2)

                #if idxIn%1000==0: print(idxIn)

                namesIn.append(breed)

                idxIn += 1

    idx = np.arange(idxIn)

    np.random.shuffle(idx)

    imagesIn = imagesIn[idx,:,:,:]

    namesIn = np.array(namesIn)[idx]

    

# RANDOMLY CROP FULL IMAGES

else:

    IMAGES = np.sort(IMAGES)

    np.random.seed(810)

    x = np.random.choice(np.arange(20579),10000)

    np.random.seed(None)

    for k in tqdm_notebook(range(len(x))):

        img = Image.open(ROOT + 'all-dogs/all-dogs/' + IMAGES[x[k]])

        w = img.size[0]; h = img.size[1];

        if (k%2==0)|(k%3==0):

            w2 = 100; h2 = int(h/(w/100))

            a = 18; b = 0          

        else:

            a=0; b=0

            if w<h:

                w2 = 64; h2 = int((64/w)*h)

                b = (h2-64)//2

            else:

                h2 = 64; w2 = int((64/h)*w)

                a = (w2-64)//2

        img = img.resize((w2,h2), Image.ANTIALIAS)

        img = img.crop((0+a, 0+b, 64+a, 64+b))    

        imagesIn[idxIn,:,:,:] = np.asarray(img)

        namesIn.append(IMAGES[x[k]])

        #if idxIn%1000==0: print(idxIn)

        idxIn += 1

    

#DISPLAY CROPPED IMAGES

x = np.random.randint(0,idxIn,25)

for k in range(5):

    plt.figure(figsize=(15,3))

    for j in range(5):

        plt.subplot(1,5,j+1)

        img = Image.fromarray( imagesIn[x[k*5+j],:,:,:].astype('uint8') )

        plt.axis('off')

        if not DogsOnly: plt.title(namesIn[x[k*5+j]],fontsize=11)

        else: plt.title(namesIn[x[k*5+j]].split('-')[1],fontsize=11)

        plt.imshow(img)

    plt.show()
# the number of the dog breed

dognamesIn = [namesIn[i].split('-')[-1] for i in range(len(namesIn))]

print(len(dognamesIn))
from collections import defaultdict

dog_names = list(set(dognamesIn))

dog_names[:5]
dog_name2ID = defaultdict(int)

for i in range(len(dog_names)):

    dog_name2ID[dog_names[i]] = i

dog_name2ID
dogIDIn = [dog_name2ID[dognamesIn[i]] for i in range(len(dognamesIn))]

dogIDIn[:3]
import numpy as np

EYE_matrix = np.eye(120)

X_all = imagesIn

Y_all = np.array([EYE_matrix[i] for i in tqdm_notebook(dogIDIn)])

X_all.shape, Y_all.shape
# -*- coding: utf-8 -*-

import tensorflow as tf

import numpy as np

import os

import matplotlib.pyplot as plt


from imageio import imread, imsave, mimsave

import cv2

import glob

from tqdm import tqdm, tqdm_notebook
batch_size = 256

z_dim = 10000

WIDTH = 64

HEIGHT = 64

LABEL = 120

LAMBDA = 10

DIS_ITERS = 3 # 5



OUTPUT_DIR = 'samples'

if not os.path.exists(OUTPUT_DIR):

    os.mkdir(OUTPUT_DIR)



X = tf.placeholder(dtype=tf.float32, shape=[batch_size, HEIGHT, WIDTH, 3], name='X')

Y = tf.placeholder(dtype=tf.float32, shape=[batch_size, LABEL], name='Y')

noise = tf.placeholder(dtype=tf.float32, shape=[batch_size, z_dim], name='noise')

is_training = tf.placeholder(dtype=tf.bool, name='is_training')



def lrelu(x, leak=0.2):

    return tf.maximum(x, leak * x)
def discriminator(image, reuse=None, is_training=is_training):

    momentum = 0.9

    with tf.variable_scope('discriminator', reuse=reuse):

        h0 = lrelu(tf.layers.conv2d(image, kernel_size=5, filters=64, strides=2, padding='same'))

        

        h1 = lrelu(tf.layers.conv2d(h0, kernel_size=5, filters=128, strides=2, padding='same'))

        

        h2 = lrelu(tf.layers.conv2d(h1, kernel_size=5, filters=256, strides=2, padding='same'))

        

        h3 = lrelu(tf.layers.conv2d(h2, kernel_size=5, filters=512, strides=2, padding='same'))

        

        h4 = tf.contrib.layers.flatten(h3)

        Y_ = tf.layers.dense(h4, units=LABEL)

        h4 = tf.layers.dense(h4, units=1)

        return h4, Y_
def generator(z, label, is_training=is_training):

    momentum = 0.9

    with tf.variable_scope('generator', reuse=None):

        d = 4

        z = tf.concat([z, label], axis=1)

        h0 = tf.layers.dense(z, units=d * d * 512)

        h0 = tf.reshape(h0, shape=[-1, d, d, 512])

        h0 = tf.nn.relu(tf.contrib.layers.batch_norm(h0, is_training=is_training, decay=momentum))

        

        h1 = tf.layers.conv2d_transpose(h0, kernel_size=5, filters=256, strides=2, padding='same')

        h1 = tf.nn.relu(tf.contrib.layers.batch_norm(h1, is_training=is_training, decay=momentum))

        

        h2 = tf.layers.conv2d_transpose(h1, kernel_size=5, filters=128, strides=2, padding='same')

        h2 = tf.nn.relu(tf.contrib.layers.batch_norm(h2, is_training=is_training, decay=momentum))

        

        h3 = tf.layers.conv2d_transpose(h2, kernel_size=5, filters=64, strides=2, padding='same')

        h3 = tf.nn.relu(tf.contrib.layers.batch_norm(h3, is_training=is_training, decay=momentum))

        

        h4 = tf.layers.conv2d_transpose(h3, kernel_size=5, filters=3, strides=2, padding='same', activation=tf.nn.tanh, name='g')

        return h4
g = generator(noise, Y)

d_real, y_real = discriminator(X)

d_fake, y_fake = discriminator(g, reuse=True)



loss_d_real = -tf.reduce_mean(d_real)

loss_d_fake = tf.reduce_mean(d_fake)



loss_cls_real = tf.losses.mean_squared_error(Y, y_real)

loss_cls_fake = tf.losses.mean_squared_error(Y, y_fake)



loss_d = loss_d_real + loss_d_fake + loss_cls_real

loss_g = -tf.reduce_mean(d_fake) + loss_cls_fake



alpha = tf.random_uniform(shape=[batch_size, 1, 1, 1], minval=0., maxval=1.)

interpolates = alpha * X + (1 - alpha) * g

grad = tf.gradients(discriminator(interpolates, reuse=True), [interpolates])[0]

slop = tf.sqrt(tf.reduce_sum(tf.square(grad), axis=[1]))

gp = tf.reduce_mean((slop - 1.) ** 2)

loss_d += LAMBDA * gp



vars_g = [var for var in tf.trainable_variables() if var.name.startswith('generator')]

vars_d = [var for var in tf.trainable_variables() if var.name.startswith('discriminator')]
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

with tf.control_dependencies(update_ops):

    optimizer_d = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5).minimize(loss_d, var_list=vars_d)

    optimizer_g = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5).minimize(loss_g, var_list=vars_g)
def montage(images):    

    if isinstance(images, list):

        images = np.array(images)

    img_h = images.shape[1]

    img_w = images.shape[2]

    n_plots = int(np.ceil(np.sqrt(images.shape[0])))

    if len(images.shape) == 4 and images.shape[3] == 3:

        m = np.ones(

            (images.shape[1] * n_plots + n_plots + 1,

             images.shape[2] * n_plots + n_plots + 1, 3)) * 0.5

    elif len(images.shape) == 4 and images.shape[3] == 1:

        m = np.ones(

            (images.shape[1] * n_plots + n_plots + 1,

             images.shape[2] * n_plots + n_plots + 1, 1)) * 0.5

    elif len(images.shape) == 3:

        m = np.ones(

            (images.shape[1] * n_plots + n_plots + 1,

             images.shape[2] * n_plots + n_plots + 1)) * 0.5

    else:

        raise ValueError('Could not parse image shape of {}'.format(images.shape))

    for i in range(n_plots):

        for j in range(n_plots):

            this_filter = i * n_plots + j

            if this_filter < images.shape[0]:

                this_img = images[this_filter]

                m[1 + i + i * img_h:1 + i + (i + 1) * img_h,

                  1 + j + j * img_w:1 + j + (j + 1) * img_w] = this_img

    return m
for i in range(10):

    img = Image.fromarray( X_all[i,:,:,:].astype('uint8') )

    plt.imshow(img)

    plt.show()

    print(Y_all[i, :])
def get_random_batch():

    data_index = np.arange(X_all.shape[0])

    np.random.shuffle(data_index)

    data_index = data_index[:batch_size]

    X_batch = X_all[data_index, :, :, :]

    Y_batch = Y_all[data_index, :]

    

    return X_batch, Y_batch
sess = tf.Session()

sess.run(tf.global_variables_initializer())

zs = np.random.uniform(-1.0, 1.0, [batch_size // 2, z_dim]).astype(np.float32)

z_samples = []

y_samples = []

for i in range(batch_size // 2):

    LABEL_i = i % LABEL

    z_samples.append(zs[i, :])

    y_samples.append(Y_all[LABEL_i])

    z_samples.append(zs[i, :])

    y_samples.append(Y_all[-LABEL_i])

loss = {'d': [], 'g': []}



for i in tqdm_notebook(range(30000)):

    for j in range(DIS_ITERS):

        n = np.random.uniform(-1.0, 1.0, [batch_size, z_dim]).astype(np.float32)

        X_batch, Y_batch = get_random_batch()

        _, d_ls = sess.run([optimizer_d, loss_d], feed_dict={X: X_batch, Y: Y_batch, noise: n, is_training: True})

    

    _, g_ls = sess.run([optimizer_g, loss_g], feed_dict={X: X_batch, Y: Y_batch, noise: n, is_training: True})

    

    loss['d'].append(d_ls)

    loss['g'].append(g_ls)

    

    if i % 500 == 0:

        print(i, d_ls, g_ls)

        gen_imgs = sess.run(g, feed_dict={noise: z_samples, Y: y_samples, is_training: False})

        gen_imgs = (gen_imgs + 1) / 2

        imgs = [img[:, :, :] for img in gen_imgs]

        gen_imgs = montage(imgs)

        plt.axis('off')

        plt.imshow(gen_imgs)

        plt.show()



plt.plot(loss['d'], label='Discriminator')

plt.plot(loss['g'], label='Generator')

plt.legend(loc='upper right')

plt.show()
# Generate data

n_batches = 10000 // batch_size

last_batch_size = 10000 % batch_size



for i in tqdm_notebook(range(n_batches)):

    n = np.random.uniform(-1.0, 1.0, [batch_size, z_dim]).astype(np.float32)

    y_samples = Y_all[i * batch_size : (i+1) * batch_size]

    gen_imgs = sess.run(g, feed_dict={noise: z_samples, Y: y_samples, is_training: False})

    gen_imgs = (gen_imgs + 1) / 2

    for j in range(batch_size):

        imsave(os.path.join(GEN_DIR, f'sample_{i}_{j}.png'), gen_imgs[j])



for i in range(last_batch_size):

    n = np.random.uniform(-1.0, 1.0, [batch_size, z_dim]).astype(np.float32)

    gen_imgs = sess.run(g, feed_dict={noise: z_samples, Y: y_samples, is_training: False})

    gen_imgs = (gen_imgs + 1) / 2

    imsave(os.path.join(GEN_DIR, f'sample_{n_batches}_{i}.png'), gen_imgs[i])
from __future__ import absolute_import, division, print_function

import numpy as np

import os

import gzip, pickle

import tensorflow as tf

from scipy import linalg

import pathlib

import urllib

import warnings

from tqdm import tqdm

from PIL import Image



class KernelEvalException(Exception):

    pass



model_params = {

    'Inception': {

        'name': 'Inception', 

        'imsize': 64,

        'output_layer': 'Pretrained_Net/pool_3:0', 

        'input_layer': 'Pretrained_Net/ExpandDims:0',

        'output_shape': 2048,

        'cosine_distance_eps': 0.1

        }

}



def create_model_graph(pth):

    """Creates a graph from saved GraphDef file."""

    # Creates graph from saved graph_def.pb.

    with tf.gfile.FastGFile( pth, 'rb') as f:

        graph_def = tf.GraphDef()

        graph_def.ParseFromString( f.read())

        _ = tf.import_graph_def( graph_def, name='Pretrained_Net')



def _get_model_layer(sess, model_name):

    # layername = 'Pretrained_Net/final_layer/Mean:0'

    layername = model_params[model_name]['output_layer']

    layer = sess.graph.get_tensor_by_name(layername)

    ops = layer.graph.get_operations()

    for op_idx, op in enumerate(ops):

        for o in op.outputs:

            shape = o.get_shape()

            if shape._dims != []:

              shape = [s.value for s in shape]

              new_shape = []

              for j, s in enumerate(shape):

                if s == 1 and j == 0:

                  new_shape.append(None)

                else:

                  new_shape.append(s)

              o.__dict__['_shape_val'] = tf.TensorShape(new_shape)

    return layer



def get_activations(images, sess, model_name, batch_size=50, verbose=False):

    """Calculates the activations of the pool_3 layer for all images.



    Params:

    -- images      : Numpy array of dimension (n_images, hi, wi, 3). The values

                     must lie between 0 and 256.

    -- sess        : current session

    -- batch_size  : the images numpy array is split into batches with batch size

                     batch_size. A reasonable batch size depends on the disposable hardware.

    -- verbose    : If set to True and parameter out_step is given, the number of calculated

                     batches is reported.

    Returns:

    -- A numpy array of dimension (num images, 2048) that contains the

       activations of the given tensor when feeding inception with the query tensor.

    """

    inception_layer = _get_model_layer(sess, model_name)

    n_images = images.shape[0]

    if batch_size > n_images:

        print("warning: batch size is bigger than the data size. setting batch size to data size")

        batch_size = n_images

    n_batches = n_images//batch_size + 1

    pred_arr = np.empty((n_images,model_params[model_name]['output_shape']))

    for i in tqdm(range(n_batches)):

        if verbose:

            print("\rPropagating batch %d/%d" % (i+1, n_batches), end="", flush=True)

        start = i*batch_size

        if start+batch_size < n_images:

            end = start+batch_size

        else:

            end = n_images

                    

        batch = images[start:end]

        pred = sess.run(inception_layer, {model_params[model_name]['input_layer']: batch})

        pred_arr[start:end] = pred.reshape(-1,model_params[model_name]['output_shape'])

    if verbose:

        print(" done")

    return pred_arr





# def calculate_memorization_distance(features1, features2):

#     neigh = NearestNeighbors(n_neighbors=1, algorithm='kd_tree', metric='euclidean')

#     neigh.fit(features2) 

#     d, _ = neigh.kneighbors(features1, return_distance=True)

#     print('d.shape=',d.shape)

#     return np.mean(d)



def normalize_rows(x: np.ndarray):

    """

    function that normalizes each row of the matrix x to have unit length.



    Args:

     ``x``: A numpy matrix of shape (n, m)



    Returns:

     ``x``: The normalized (by row) numpy matrix.

    """

    return np.nan_to_num(x/np.linalg.norm(x, ord=2, axis=1, keepdims=True))





def cosine_distance(features1, features2):

    # print('rows of zeros in features1 = ',sum(np.sum(features1, axis=1) == 0))

    # print('rows of zeros in features2 = ',sum(np.sum(features2, axis=1) == 0))

    features1_nozero = features1[np.sum(features1, axis=1) != 0]

    features2_nozero = features2[np.sum(features2, axis=1) != 0]

    norm_f1 = normalize_rows(features1_nozero)

    norm_f2 = normalize_rows(features2_nozero)



    d = 1.0-np.abs(np.matmul(norm_f1, norm_f2.T))

    print('d.shape=',d.shape)

    print('np.min(d, axis=1).shape=',np.min(d, axis=1).shape)

    mean_min_d = np.mean(np.min(d, axis=1))

    print('distance=',mean_min_d)

    return mean_min_d





def distance_thresholding(d, eps):

    if d < eps:

        return d

    else:

        return 1



def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):

    """Numpy implementation of the Frechet Distance.

    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)

    and X_2 ~ N(mu_2, C_2) is

            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

            

    Stable version by Dougal J. Sutherland.



    Params:

    -- mu1 : Numpy array containing the activations of the pool_3 layer of the

             inception net ( like returned by the function 'get_predictions')

             for generated samples.

    -- mu2   : The sample mean over activations of the pool_3 layer, precalcualted

               on an representive data set.

    -- sigma1: The covariance matrix over activations of the pool_3 layer for

               generated samples.

    -- sigma2: The covariance matrix over activations of the pool_3 layer,

               precalcualted on an representive data set.



    Returns:

    --   : The Frechet Distance.

    """



    mu1 = np.atleast_1d(mu1)

    mu2 = np.atleast_1d(mu2)



    sigma1 = np.atleast_2d(sigma1)

    sigma2 = np.atleast_2d(sigma2)



    assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"

    assert sigma1.shape == sigma2.shape, "Training and test covariances have different dimensions"



    diff = mu1 - mu2



    # product might be almost singular

    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)

    if not np.isfinite(covmean).all():

        msg = "fid calculation produces singular product; adding %s to diagonal of cov estimates" % eps

        warnings.warn(msg)

        offset = np.eye(sigma1.shape[0]) * eps

        # covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    

    # numerical error might give slight imaginary component

    if np.iscomplexobj(covmean):

        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):

            m = np.max(np.abs(covmean.imag))

            raise ValueError("Imaginary component {}".format(m))

        covmean = covmean.real



    # covmean = tf.linalg.sqrtm(tf.linalg.matmul(sigma1,sigma2))



    print('covmean.shape=',covmean.shape)

    # tr_covmean = tf.linalg.trace(covmean)



    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

    # return diff.dot(diff) + tf.linalg.trace(sigma1) + tf.linalg.trace(sigma2) - 2 * tr_covmean

#-------------------------------------------------------------------------------





def calculate_activation_statistics(images, sess, model_name, batch_size=50, verbose=False):

    """Calculation of the statistics used by the FID.

    Params:

    -- images      : Numpy array of dimension (n_images, hi, wi, 3). The values

                     must lie between 0 and 255.

    -- sess        : current session

    -- batch_size  : the images numpy array is split into batches with batch size

                     batch_size. A reasonable batch size depends on the available hardware.

    -- verbose     : If set to True and parameter out_step is given, the number of calculated

                     batches is reported.

    Returns:

    -- mu    : The mean over samples of the activations of the pool_3 layer of

               the incption model.

    -- sigma : The covariance matrix of the activations of the pool_3 layer of

               the incption model.

    """

    act = get_activations(images, sess, model_name, batch_size, verbose)

    mu = np.mean(act, axis=0)

    sigma = np.cov(act, rowvar=False)

    return mu, sigma, act

    

def _handle_path_memorization(path, sess, model_name, is_checksize, is_check_png):

    path = pathlib.Path(path)

    files = list(path.glob('*.jpg')) + list(path.glob('*.png'))

    imsize = model_params[model_name]['imsize']



    # In production we don't resize input images. This is just for demo purpose. 

    x = np.array([np.array(img_read_checks(fn, imsize, is_checksize, imsize, is_check_png)) for fn in files])

    m, s, features = calculate_activation_statistics(x, sess, model_name)

    del x #clean up memory

    return m, s, features



# check for image size

def img_read_checks(filename, resize_to, is_checksize=False, check_imsize = 64, is_check_png = False):

    im = Image.open(str(filename))

    if is_checksize and im.size != (check_imsize,check_imsize):

        raise KernelEvalException('The images are not of size '+str(check_imsize))

    

    if is_check_png and im.format != 'PNG':

        raise KernelEvalException('Only PNG images should be submitted.')



    if resize_to is None:

        return im

    else:

        return im.resize((resize_to,resize_to),Image.ANTIALIAS)



def calculate_kid_given_paths(paths, model_name, model_path, feature_path=None, mm=[], ss=[], ff=[]):

    ''' Calculates the KID of two paths. '''

    tf.reset_default_graph()

    create_model_graph(str(model_path))

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        m1, s1, features1 = _handle_path_memorization(paths[0], sess, model_name, is_checksize = True, is_check_png = True)

        if len(mm) != 0:

            m2 = mm

            s2 = ss

            features2 = ff

        elif feature_path is None:

            m2, s2, features2 = _handle_path_memorization(paths[1], sess, model_name, is_checksize = False, is_check_png = False)

        else:

            with np.load(feature_path) as f:

                m2, s2, features2 = f['m'], f['s'], f['features']



        print('m1,m2 shape=',(m1.shape,m2.shape),'s1,s2=',(s1.shape,s2.shape))

        print('starting calculating FID')

        fid_value = calculate_frechet_distance(m1, s1, m2, s2)

        print('done with FID, starting distance calculation')

        distance = cosine_distance(features1, features2)        

        return fid_value, distance, m2, s2, features2
if ComputeLB:

  

    # UNCOMPRESS OUR IMGAES

    with zipfile.ZipFile("../working/images.zip","r") as z:

        z.extractall("../tmp/images2/")



    # COMPUTE LB SCORE

    m2 = []; s2 =[]; f2 = []

    user_images_unzipped_path = '../tmp/images2/'

    images_path = [user_images_unzipped_path,'../input/generative-dog-images/all-dogs/all-dogs/']

    public_path = '../input/dog-face-generation-competition-kid-metric-input/classify_image_graph_def.pb'



    fid_epsilon = 10e-15



    fid_value_public, distance_public, m2, s2, f2 = calculate_kid_given_paths(images_path, 'Inception', public_path, mm=m2, ss=s2, ff=f2)

    distance_public = distance_thresholding(distance_public, model_params['Inception']['cosine_distance_eps'])

    print("FID_public: ", fid_value_public, "distance_public: ", distance_public, "multiplied_public: ",

            fid_value_public /(distance_public + fid_epsilon))

    

    # REMOVE FILES TO PREVENT KERNEL ERROR OF TOO MANY FILES

    ! rm -r ../tmp