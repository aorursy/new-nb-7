# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# basic modules

import seaborn as sns

import matplotlib.pyplot as plt

from tqdm import tqdm

import cv2

import matplotlib.animation as animation

from IPython.display import HTML

import imgaug

import random

from imgaug import augmenters as iaa

from imgaug import parameters as iap

from PIL import Image



#keras modules

from keras.preprocessing.image import img_to_array, load_img



#Pytorch modules

import torch

import torch.nn as nn

import torch.nn.parallel

import torch.backends.cudnn as cudnn

import torch.optim as optim

import torch.utils.data

import torchvision.datasets as dset

import torchvision.transforms as transforms

import torchvision.utils as vutils

from torchvision.utils import save_image



#Crop images using bounding box

import xml.etree.ElementTree as ET

import glob

img_list =os.listdir("../input/all-dogs/all-dogs/")
len(img_list)
root_images="../input/all-dogs/all-dogs/"

root_annots="../input/annotation/Annotation/"

croped_images="../croped_images/"
all_images=os.listdir("../input/all-dogs/all-dogs/")

print(f"Total images : {len(all_images)}")



breeds = glob.glob('../input/annotation/Annotation/*')

annotation=[]

for b in breeds:

    annotation+=glob.glob(b+"/*")

print(f"Total annotation : {len(annotation)}")



breed_map={}

for annot in annotation:

    breed=annot.split("/")[-2]

    index=breed.split("-")[0]

    breed_map.setdefault(index,breed)

    

print(f"Total Breeds : {len(breed_map)}")
def bounding_box(image):

    bpath=root_annots+str(breed_map[image.split("_")[0]])+"/"+str(image.split(".")[0])

    tree = ET.parse(bpath)

    root = tree.getroot()

    objects = root.findall('object')

    for o in objects:

        bndbox = o.find('bndbox') # reading bound box

        xmin = int(bndbox.find('xmin').text)

        ymin = int(bndbox.find('ymin').text)

        xmax = int(bndbox.find('xmax').text)

        ymax = int(bndbox.find('ymax').text)

        

    return (xmin,ymin,xmax,ymax)
if not os.path.exists(croped_images):

    os.mkdir(croped_images)
for i,image in enumerate(all_images):

    #print(image)

    bbox=bounding_box(image)

    im=Image.open(os.path.join(root_images,image))

    im=im.crop(bbox)

    im.save(croped_images + image, quality=95)
len(os.listdir("../croped_images/"))
temp_img = load_img('../croped_images/n02085620_3423.jpg')

temp_img_array  = img_to_array(temp_img)
temp_img
sns.set_style("white")

count = 1

plt.figure(figsize=[20, 20])

for img_name in img_list[:15]:

    #print("../input/all-dogs/all-dogs/%s.jpg" % img_name)

    img = cv2.imread("../croped_images/%s" % img_name)[...,[2, 1, 0]]

    plt.subplot(5, 5, count)

    plt.imshow(img)

    count += 1

    

plt.show()
manualSeed = random.randint(1000, 10000)

print("Random Seed: ", manualSeed)

random.seed(manualSeed)

torch.manual_seed(manualSeed)
# Root directory for dataset

dataroot = "../"

workers = 2

batch_size = 32

image_size = 64

nc = 3

nz = 128

ngf = 64

ndf = 64

num_epochs = 200

lr = 0.0001

beta1 = 0.5

# Number of GPUs available. Use 0 for CPU mode.

ngpu = 1
# Dataset Creator

rand_aff = random.uniform(3.0, 15.0)

rand_flip = random.uniform(0.3, 1.0)

rand_trans = random.uniform(0.3, 0.7)

rand_contr = random.uniform(0.2, 0.9)

random_transforms = [transforms.ColorJitter(contrast=rand_contr), transforms.RandomAffine(degrees=rand_aff)]

dataset = dset.ImageFolder(root=dataroot,

                           transform=transforms.Compose([

                               transforms.Resize(image_size),

                               transforms.CenterCrop(image_size),

                               transforms.RandomHorizontalFlip(p=rand_flip),

                               transforms.RandomApply(random_transforms, p=rand_trans),

                               transforms.ToTensor(),

                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),

                           ]))

# Create the dataloader

dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,

                                         shuffle=True, num_workers=workers)



# Decide which device we want to run on

device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
real_batch = next(iter(dataloader))

plt.figure(figsize=(8,8))

plt.axis("off")

plt.title("Training Images")

plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
def weights_init(m):

    classname = m.__class__.__name__

    if classname.find('Conv') != -1:

        nn.init.normal_(m.weight.data, 0.0, 0.02)

    elif classname.find('BatchNorm') != -1:

        nn.init.normal_(m.weight.data, 1.0, 0.02)

        nn.init.constant_(m.bias.data, 0)
# Generator

class Generator(nn.Module):

    def __init__(self, ngpu):

        super(Generator, self).__init__()

        self.ngpu = ngpu

        self.main = nn.Sequential(

            nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),

            nn.BatchNorm2d(ngf * 8),

            nn.ReLU(True),

            

            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),

            nn.BatchNorm2d(ngf * 4),

            nn.ReLU(True),



            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),

            nn.BatchNorm2d(ngf * 2),

            nn.ReLU(True),



            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),

            nn.BatchNorm2d(ngf),

            nn.ReLU(True),



            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),

            nn.Tanh()



        )



    def forward(self, input):

        return self.main(input)
# Create the generator

netG = Generator(ngpu).to(device)



# Handle multi-gpu if desired

if (device.type == 'cuda') and (ngpu > 1):

    netG = nn.DataParallel(netG, list(range(ngpu)))



# Apply the weights_init function to randomly initialize all weights

#  to mean=0, stdev=0.2.

netG.apply(weights_init)



# Print the model

print(netG)
class Discriminator(nn.Module):

    def __init__(self, ngpu):

        super(Discriminator, self).__init__()

        self.ngpu = ngpu

        self.main = nn.Sequential(



            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),

            nn.LeakyReLU(0.2, inplace=True),



            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),

            nn.BatchNorm2d(ndf * 2),

            nn.LeakyReLU(0.2, inplace=True),

            nn.Dropout2d(0.2),



            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),

            nn.BatchNorm2d(ndf * 4),

            nn.LeakyReLU(0.2, inplace=True),

            nn.Dropout2d(0.2),



            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),

            nn.BatchNorm2d(ndf * 8),

            nn.LeakyReLU(0.2, inplace=True),

            nn.Dropout2d(0.2),

            

            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),

            nn.Sigmoid()

        )



    def forward(self, input):

        return self.main(input)
# Create the Discriminator

netD = Discriminator(ngpu).to(device)



# Handle multi-gpu if desired

if (device.type == 'cuda') and (ngpu > 1):

    netD = nn.DataParallel(netD, list(range(ngpu)))



# Apply the weights_init function to randomly initialize all weights

#  to mean=0, stdev=0.2.

netD.apply(weights_init)



# Print the model

print(netD)
# Initialize BCELoss function

criterion = nn.BCELoss()



# Create batch of latent vectors that we will use to visualize

#  the progression of the generator

fixed_noise = torch.randn(64, nz, 1, 1, device=device)



# Establish convention for real and fake labels during training

real_label = 1

fake_label = 0



# Setup Adam optimizers for both G and D

optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))

optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
img_list = []

G_losses = []

D_losses = []

iters = 0



print("Training Loop...")

# For each epoch

for epoch in tqdm(range(num_epochs)):

    # For each batch in the dataloader

    for i, data in enumerate(dataloader, 0):



        ## Train with all-real batch

        netD.zero_grad()

        # Format batch

        real_cpu = data[0].to(device)

        b_size = real_cpu.size(0)

        label = torch.full((b_size,), real_label, device=device)

        # Forward pass real batch through D

        output = netD(real_cpu).view(-1)

        # Calculate loss on all-real batch

        errD_real = criterion(output, label)

        # Calculate gradients for D in backward pass

        errD_real.backward()

        D_x = output.mean().item()



        ## Train with all-fake batch

        # Generate batch of latent vectors

        noise = torch.randn(b_size, nz, 1, 1, device=device)

        # Generate fake image batch with G

        fake = netG(noise)

        label.fill_(fake_label)

        # Classify all fake batch with D

        output = netD(fake.detach()).view(-1)

        # Calculate D's loss on the all-fake batch

        errD_fake = criterion(output, label)

        # Calculate the gradients for this batch

        errD_fake.backward()

        D_G_z1 = output.mean().item()

        # Add the gradients from the all-real and all-fake batches

        errD = errD_real + errD_fake

        # Update D

        optimizerD.step()



        ############################

        # (2) Update G network: maximize log(D(G(z)))

        ###########################

        netG.zero_grad()

        label.fill_(real_label)  # fake labels are real for generator cost

        # Since we just updated D, perform another forward pass of all-fake batch through D

        output = netD(fake).view(-1)

        # Calculate G's loss based on this output

        errG = criterion(output, label)

        # Calculate gradients for G

        errG.backward()

        D_G_z2 = output.mean().item()

        # Update G

        optimizerG.step()



        # Save Losses for plotting later

        G_losses.append(errG.item())

        D_losses.append(errD.item())



        # Check how the generator is doing by saving G's output on fixed_noise

        if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):

            with torch.no_grad():

                fake = netG(fixed_noise).detach().cpu()

            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))



        iters += 1
plt.figure(figsize=(10,5))

plt.title("Generator and Discriminator Loss During Training")

plt.plot(G_losses,label="G")

plt.plot(D_losses,label="D")

plt.xlabel("iterations")

plt.ylabel("Loss")

plt.legend()

plt.show()
if not os.path.exists('../output_images'):

    os.mkdir('../output_images')

    

im_batch_size = 50

n_images=10000



for i_batch in tqdm(range(0, n_images, im_batch_size)):

    gen_z = torch.randn(im_batch_size, nz, 1, 1, device=device)

    gen_images = netG(gen_z)

    images = gen_images.to("cpu").clone().detach()

    images = images.numpy().transpose(0, 2, 3, 1)

    for i_image in range(gen_images.size(0)):

        save_image(gen_images[i_image, :, :, :], os.path.join('../output_images', f'image_{i_batch+i_image:05d}.png'))
import shutil

shutil.make_archive('images', 'zip', '../output_images')