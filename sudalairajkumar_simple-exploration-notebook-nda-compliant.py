# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from scipy.misc import imread

import matplotlib.pyplot as plt

import seaborn as sns




from subprocess import check_output

print(check_output(["ls", "../input/train/"]).decode("utf8"))
sub_folders = check_output(["ls", "../input/train/"]).decode("utf8").strip().split('\n')

count_dict = {}

for sub_folder in sub_folders:

    num_of_files = len(check_output(["ls", "../input/train/"+sub_folder]).decode("utf8").strip().split('\n'))

    print("Number of files for the species",sub_folder,":",num_of_files)

    count_dict[sub_folder] = num_of_files

    

plt.figure(figsize=(12,4))

sns.barplot(list(count_dict.keys()), list(count_dict.values()), alpha=0.8)

plt.xlabel('Fish Species', fontsize=12)

plt.ylabel('Number of Images', fontsize=12)

plt.show()

    
num_test_files = len(check_output(["ls", "../input/test_stg1/"]).decode("utf8").strip().split('\n'))

print("Number of test files present :", num_test_files)
train_path = "../input/train/"

sub_folders = check_output(["ls", train_path]).decode("utf8").strip().split('\n')

different_file_sizes = {}

for sub_folder in sub_folders:

    file_names = check_output(["ls", train_path+sub_folder]).decode("utf8").strip().split('\n')

    for file_name in file_names:

        im_array = imread(train_path+sub_folder+"/"+file_name)

        size = "_".join(map(str,list(im_array.shape)))

        different_file_sizes[size] = different_file_sizes.get(size,0) + 1



plt.figure(figsize=(12,4))

sns.barplot(list(different_file_sizes.keys()), list(different_file_sizes.values()), alpha=0.8)

plt.xlabel('Image size', fontsize=12)

plt.ylabel('Number of Images', fontsize=12)

plt.title("Image size present in train dataset")

plt.xticks(rotation='vertical')

plt.show()
test_path = "../input/test_stg1/"

file_names = check_output(["ls", test_path]).decode("utf8").strip().split('\n')

different_file_sizes = {}

for file_name in file_names:

        size = "_".join(map(str,list(imread(test_path+file_name).shape)))

        different_file_sizes[size] = different_file_sizes.get(size,0) + 1



plt.figure(figsize=(12,4))

sns.barplot(list(different_file_sizes.keys()), list(different_file_sizes.values()), alpha=0.8)

plt.xlabel('File size', fontsize=12)

plt.ylabel('Number of Images', fontsize=12)

plt.xticks(rotation='vertical')

plt.title("Image size present in test dataset")

plt.show()
"""

import random

import matplotlib.animation as animation

from matplotlib import animation, rc

from IPython.display import HTML



random.seed(12345)

train_path = "../input/train/"

sub_folders = check_output(["ls", train_path]).decode("utf8").strip().split('\n')

different_file_sizes = {}

all_files = []

for sub_folder in sub_folders:

    file_names = check_output(["ls", train_path+sub_folder]).decode("utf8").strip().split('\n')

    selected_files = random.sample(file_names, 10)

    for file_name in selected_files:

        all_files.append([sub_folder,file_name])



fig = plt.figure()

sns.set_style("whitegrid", {'axes.grid' : False})

img_file = "".join([train_path, sub_folder, "/", file_name])

im = plt.imshow(imread(img_file), vmin=0, vmax=255)



def updatefig(ind):

    sub_folder = all_files[ind][0]

    file_name = all_files[ind][1]

    img_file = "".join([train_path, sub_folder, "/", file_name])

    im.set_array(imread(img_file))

    plt.title("Species : "+sub_folder, fontsize=15)

    return im,



ani = animation.FuncAnimation(fig, updatefig, frames=len(all_files))

ani.save('lb.gif', fps=1, writer='imagemagick')

#rc('animation', html='html5')

#HTML(ani.to_html5_video())

plt.show()

"""
import random

from subprocess import check_output

from scipy.misc import imread

import numpy as np

np.random.seed(2016)

from keras.datasets import mnist

from keras.models import Sequential

from keras.layers import Dense, Dropout, Activation, Flatten

from keras.layers import Convolution2D, MaxPooling2D

from keras.utils import np_utils

from keras import backend as K



batch_size = 1

nb_classes = 8

nb_epoch = 1



img_rows, img_cols, img_rgb = 500, 500, 3

nb_filters = 4

pool_size = (2, 2)

kernel_size = (3, 3)

input_shape = (img_rows, img_cols, 3)



species_map_dict = {

'ALB':0,

'BET':1,

'DOL':2,

'LAG':3,

'NoF':4,

'OTHER':5,

'SHARK':6,

'YFT':7

}



def batch_generator_train(sample_size):

	train_path = "../input/train/"

	all_files = []

	y_values = []

	sub_folders = check_output(["ls", train_path]).decode("utf8").strip().split('\n')

	for sub_folder in sub_folders:

		file_names = check_output(["ls", train_path+sub_folder]).decode("utf8").strip().split('\n')

		for file_name in file_names:

			all_files.append([sub_folder, '/', file_name])

			y_values.append(species_map_dict[sub_folder])

	number_of_images = range(len(all_files))



	counter = 0

	while True:

		image_index = random.choice(number_of_images)

		file_name = "".join([train_path] + all_files[image_index])

		print(file_name)

		y = [0]*8

		y[y_values[image_index]] = 1

		y = np.array(y).reshape(1,8)

		

		im_array = imread(file_name)

		X = np.zeros([1, img_rows, img_cols, img_rgb])

		#X[:im_array.shape[0], :im_array.shape[1], 3] = im_array.copy().astype('float32')

		X[0, :, :, :] = im_array[:500,:500,:].astype('float32')

		X /= 255.

        

		print(X.shape)

		yield X,y

		

		counter += 1

		#if counter == sample_size:

		#	break



def batch_generator_test(all_files):

	for file_name in all_files:

		file_name = test_path + file_name

		

		im_array = imread(file_name)

		X = np.zeros([1, img_rows, img_cols, img_rgb])

		X[0,:, :, :] = im_array[:500,:500,:].astype('float32')

		X /= 255.



		yield X





def keras_cnn_model():

	model = Sequential()

	model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],

                        border_mode='valid',

                        input_shape=input_shape))

	model.add(Activation('relu'))

	model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))

	model.add(Activation('relu'))

	model.add(MaxPooling2D(pool_size=pool_size))

	model.add(Dropout(0.25))	

	model.add(Flatten())

	model.add(Dense(128))

	model.add(Activation('relu'))

	model.add(Dropout(0.5))

	model.add(Dense(nb_classes))

	model.add(Activation('softmax'))

	model.compile(loss='categorical_crossentropy', optimizer='adadelta')

	return model



model = keras_cnn_model()

fit= model.fit_generator(

	generator = batch_generator_train(100),

	nb_epoch = 1,

	samples_per_epoch = 100

)



test_path = "../input/test_stg1/"

all_files = []

file_names = check_output(["ls", test_path]).decode("utf8").strip().split('\n')

for file_name in file_names:

	all_files.append(file_name)

#preds = model.predict_generator(generator=batch_generator_test(all_files), val_samples=len(all_files))



#out_df = pd.DataFrame(preds)

#out_df.columns = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']

#out_df['image'] = all_files

#out_df.to_csv("sample_sub_keras.csv", index=False)