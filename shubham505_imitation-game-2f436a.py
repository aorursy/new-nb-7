from PIL import Image, ImageStat, ImageEnhance

from multiprocessing import Pool, cpu_count

import glob, zipfile, os, itertools

import matplotlib.pyplot as plt


from sklearn import *

import pandas as pd

import numpy as np

import scipy, cv2

import imagehash



# Get statistical data

def get_features(path):

    try:

        st = []

        # Get pixel data of image

        img = Image.open(path)

        img = img.resize((100,int(img.size[1]/(img.size[0]/100))), Image.ANTIALIAS)

        img = img.crop((0, 0, 64, 64))

    

        # Start statistics by RGB of pixcel data of image

        im_stats_ = ImageStat.Stat(img)

        # total

        st += im_stats_.sum

        # Average value

        st += im_stats_.mean

        # Root mean square

        st += im_stats_.rms

        # dispersion

        st += im_stats_.var

        # standard deviation

        st += im_stats_.stddev

        img = np.array(img)

        m, s = cv2.meanStdDev(img)

        st += list(m)

        st += list(s)

        st += [cv2.Laplacian(img, cv2.CV_64F).var()]

        st += [cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5).var()]

        st += [cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5).var()]

        img = img[:,:,:3]

        st += [scipy.stats.kurtosis(img[:,:,0].ravel())]

        st += [scipy.stats.kurtosis(img[:,:,1].ravel())]

        st += [scipy.stats.kurtosis(img[:,:,2].ravel())]

        st += [scipy.stats.skew(img[:,:,0].ravel())]

        st += [scipy.stats.skew(img[:,:,1].ravel())]

        st += [scipy.stats.skew(img[:,:,2].ravel())]

    except:

        print(path)

    return [path, st]



# Parallel processing

def normalize_img(paths):

    imf_d = {}

    p = Pool(cpu_count())

    # get_features

    # Parallelize get_features function

    ret = p.map(get_features, paths)

    # Arrange the result of parallel processing

    for i in range(len(ret)):

        imf_d[ret[i][0]] = ret[i][1]

    ret = []

    fdata = [imf_d[f] for f in paths]

    return pd.DataFrame(fdata)



# Load path of image data

dog_bytes = pd.DataFrame(glob.glob('../input/all-dogs/all-dogs/**'), columns=['Path'])

# Get statistical data of pixcel data for each image

dog_bytes = pd.concat((dog_bytes, normalize_img(dog_bytes.Path.values)), axis=1)
labels = pd.DataFrame(glob.glob('../input/annotation/Annotation/**/**'), columns=['Path'])

labels['Labels'] = labels['Path'].map(lambda x: x.split('/')[4].split('-')[1])

labels['FileName'] = labels['Path'].map(lambda x: x.split('/')[-1] + '.jpg')

labels = {f:l for f,l in labels[['FileName', 'Labels']].values}



dog_bytes['FileName'] = dog_bytes['Path'].map(lambda x: x.split('/')[-1])

dog_bytes['Labels'] = dog_bytes['FileName'].map(labels)

dog_bytes.head()
# KMeans

# Divide image data into 100 classifications by KMeans method

dog_bytes['Group'] = cluster.KMeans(n_clusters=400, random_state=4).fit_predict(dog_bytes[list(range(30))])

#  Get 5 classifications with many from 400 classifications (display)

dog_bytes['Group'].value_counts()[:5]
# Generate a window to display the image

# Unit is in inches

fig = plt.figure(figsize=(8, 80))

samples = []

# Get 5 samples from image data of specific classification

for i in range(400):

    # Get image data of a specific classification

    g = dog_bytes[dog_bytes['Group'] == i]

    if len(g) > 5:

        # Get 5 samples from image data of specific classification

        samples += list(g['Path'].values[:5])



# Display images by classification

for i in range(len(samples))[:50]:

    # Get one of the 5 rows and 5 columns of windows

    ax = fig.add_subplot(len(samples)/5, 5, i+1, xticks=[], yticks=[])

    # Get image data

    img = Image.open(samples[i])

    # Resize image data

    # Unit is pixel

    # Resolution (dpi) = pixel / inch

    img = img.resize((100,int(img.size[1]/(img.size[0]/100))), Image.ANTIALIAS)

    img = img.crop((0, 0, 64, 64))

    plt.imshow(img)
def sim_img(path):

    img = Image.open(path).convert('RGB')

    img = img.resize((100,int(img.size[1]/(img.size[0]/100))), Image.ANTIALIAS)

    img = img.crop((0, 0, 64, 64))

    return img



samples = []

for i in range(400):

    g = dog_bytes[dog_bytes['Group'] == i]

    p = g['Path'].values

    for i in range(0,len(p)-2, 2):

        samples.append([p[i],p[i+1]])

    if len(samples) > 11_000: break

for i in range(0,len(samples)-1):

    samples.append([samples[i][0],samples[i+1][0]])

    if len(samples) > 11_000: break

print(len(samples))
z = zipfile.PyZipFile('images.zip', mode='w')

for i in range(10_000):

    p1, p2 = samples[i]

    # Mix two images in the same category and create a new image

    # out = p1 * (1 - 0.01) + p2 * 0.01

    im = Image.blend(sim_img(p1), sim_img(p2), alpha=0.01)

    f = str(i)+'.png'

    im.save(f,'PNG'); z.write(f); os.remove(f)

    if i % 1000==0:

        print(i)

print (len(z.namelist()))

z.close()
d = ensemble.RandomForestClassifier(n_jobs=-1, n_estimators=400, random_state=3)



groups = dog_bytes['Group'].value_counts().index[:100]

dog_bytes['Original'] = dog_bytes['Group'].map(lambda x: 0 if x in groups else 1) #target label



#Lets create test set

g = dog_bytes[dog_bytes['Group'].isin(groups)]

s = list([p for p,_ in itertools.groupby(sorted([sorted(p) for p in list(itertools.permutations(g['Path'].values[:60], 2))]))])

test = pd.DataFrame(s, columns=['Path1', 'Path2'])

test['Image'] = test.index.map(lambda x: 'test/' + str(x) + '.png')

os.mkdir('test/')

for i in range(len(test)):

    im = Image.blend(sim_img(test.Path1[i]), sim_img(test.Path2[i]), alpha=0.5)

    im.save(test['Image'][i],'PNG')

test = pd.concat((test, normalize_img(test.Image.values)), axis=1)



d.fit(dog_bytes[list(range(30))], dog_bytes['Original'])

test['Original'] = d.predict_proba(test[list(range(30))])[:,1]

test = test.sort_values(by=['Original'], ascending=False).reset_index(drop=True)

dog_bytes.Original.value_counts()
fig = plt.figure(figsize=(4, 20))

for i in range(10):

    ax = fig.add_subplot(10, 3, (3*i)+1, xticks=[], yticks=[])

    plt.imshow(Image.open(test.Path1[i]).resize((100,int(img.size[1]/(img.size[0]/100))), Image.ANTIALIAS))

    ax = fig.add_subplot(10, 3, (3*i)+2, xticks=[], yticks=[])

    plt.imshow(Image.open(test.Path2[i]).resize((100,int(img.size[1]/(img.size[0]/100))), Image.ANTIALIAS))

    ax = fig.add_subplot(10, 3, (3*i)+3, xticks=[], yticks=[])

    plt.imshow(Image.open(test.Image[i]))

import shutil; shutil.rmtree('test/')