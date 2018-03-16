import numpy as np
import os
import struct
from collections import defaultdict
import pickle
import matplotlib
import matplotlib.pyplot as plt

#this file contains the Digitgenerator class and it's methods and properties

#before using, be sure to download the mnist data from
#http://yann.lecun.com/exdb/mnist/
#you can use to load_mnist.sh script found in the github repository

class Digitgenerator(object):        
        
    #Initialization*****************************************************
    #Loads the mnist data from disk, transfers it from idx-format to numpy arrays and 
    #prepares a dictionary for fast retrieval of a digit index. The preprocessed files
    #are saved to disk. If the preprocessed files are found from the disk they are loaded.
    
    def __init__(self,mnist_folder='mnist_data',preprocessed_folder='.'):       
 
        #get the filenames for mnist labels, mnist images, saved labels, images and label dict
        mnistlbls,mnistimgs,lblsf,imgsf,lblsdictf = self.get_filenames(mnist_folder,preprocessed_folder)
   
        #read preprocessed data
        try:
            
            self._labels=np.load(lblsf, allow_pickle=False)
            self._images=np.load(imgsf, allow_pickle=False)
            print('Loaded presaved numpy arrays from disk')
            
            with open(lblsdictf, 'rb') as handle:
                self._labelsdict = pickle.load(handle)                                               

        #no data at disk, make from mnist
        except IOError as err:

            print('Loading MNIST data from file')
            if os.path.exists(mnistimgs) and os.path.exists(mnistlbls): 
                
                #load mnist data and make the labels dictionary
                self._labels, self._images = self.load_mnist(mnistlbls,mnistimgs)
                self._labelsdict = self.make_labels_dict(self._labels)
            
                #save
                np.save(lblsf, self._labels)
                np.save(imgsf, self._images)
                                               
                with open(lblsdictf, 'wb') as handle:
                    pickle.dump(self._labelsdict, handle, protocol=pickle.HIGHEST_PROTOCOL)

            else:
                raise Exception('MNIST data not found in ' + mnistlbls + '. Please run the load_minst.sh script')
           
        #problem with pickle (likely python version change from dumping), make the labelsdictionary again
        except ValueError as err:
                
            self._labelsdict = self.make_labels_dict(self._labels)
            with open(lblsdictf, 'wb') as handle:
                pickle.dump(self._labelsdict, handle, protocol=pickle.HIGHEST_PROTOCOL)
                
                   
                
    #properties**************************************************************
    #return the mnist images and labels as numpy arrays
    
    @property # read only because set only once, via constructor
    def images(self):
        return self._images
    

    @property # read only because set only once, via constructor
    def labels(self):
        return self._labels
                
                
    #internal class methods**************************************************
    #get_filenames: greate the filenames where to look for the data files
    #load_mnist: loads the idx format from file and returns numpy arrays
    #make_labels_dict: makes a dictionary where keys are the digit labels and values are the indexes
    #stack images: stacks horizontally the given images (arrays) into a one image
    
    #make the filenames
    @classmethod
    def get_filenames(cls,mnist_folder,preprocessed_folder):
    
        #file locations in IDX format
        mnistlabels = os.path.join(mnist_folder, "train-labels.idx1-ubyte")
        mnistimages = os.path.join(mnist_folder, "train-images.idx3-ubyte")

        #processed to numpy arrays
        labelsfile = os.path.join(preprocessed_folder, "labelsarray.npy")
        imagesfile = os.path.join(preprocessed_folder, "imagessarray.npy")
        labelsdictfile = os.path.join(preprocessed_folder, "labelsdict.pickle")
    
        return mnistlabels,mnistimages,labelsfile,imagesfile,labelsdictfile
        
    #load mnist data from the file, the original files are in IDX format and we want numpy arrays
    @staticmethod
    def load_mnist(mnistlabels,mnistimages):
        
    # Load everything to numpy arrays
        with open(mnistlabels, 'rb') as lblfile:
            magic, num = struct.unpack(">II", lblfile.read(8))
            labels = np.fromfile(lblfile, dtype=np.int8)

        with open(mnistimages, 'rb') as imgfile:
            magic, num, rows, cols = struct.unpack(">IIII", imgfile.read(16))
            images = np.fromfile(imgfile, dtype=np.uint8).reshape(len(labels), rows, cols)
            
        #in mnist the pixel values are from 0 (white) to 255 (black)
        #we want pixel values from 0 to 1 as float32
        images=images.astype(np.float32)/255.0
        
        return labels,images
    
    
    
    #make a dictionary for fast retrieval of indexes of labels (the handwritten digit)
    #keys are the labels and values are the indexes which represent the labels
    @staticmethod
    def make_labels_dict(labels):
        
        labelsdict = defaultdict(list)
    
        for label, ind in enumerate(labels.tolist()):
            labelsdict[ind].append(label)
        
        return labelsdict
    
    
    #add white space between the images and concatenate into a one image
    @staticmethod
    def stack_images(imglist,spacing_range):
        
        spaced_image_list=[]
        spaced_image_list.append(imglist[0])
        
        for ind in range(len(imglist)-1):
            
            #make the whitespace (value 1) of Uniform[spacing range] between images and stack
            x=np.random.randint(spacing_range[0],spacing_range[1]+1)          
            spaced_image_list.extend([np.zeros((28,x),dtype=np.float32),imglist[ind+1]])
            
        stacked_image=np.concatenate(spaced_image_list,axis=1)    
        
        #set the noisy pale pixels to 0
        pale_pixels = stacked_image < 0.0
        stacked_image[pale_pixels] = 0

        return stacked_image
        
        
    #Methods*************************************************************
    
    #given list of digits and spacing between them, return a single image (array) of the digits
    def draw_from_mnist(self,digits, spacing_range=(1,3), image_width=None):
        
        imglist=[]
        for digit in digits:
            
            #pick an image by random and append the image to imglists
            ind=np.random.choice(self._labelsdict[digit])
            imglist.append(self._images[ind,:,:])

        #stack images together and save to self._stacked_image
        self._stacked_image = self.stack_images(imglist,spacing_range)

        return self._stacked_image
    
    #train the variational autoencoder and print the trained variables into a file
    def train_vaemodel(self):
            
        #create the autoencoder instance and train
        vae=Vae()
        vae.train_vae(self.images,self.labels)
        
        return None
    
    
    #sample from conditional variational autoencoder
    def sample_from_vae(self,digits, spacing_range=(1,3), image_width=None,noise_prob=0):
            
        #create the autoencoder instance
        vae=Vae()
        
        #sample the list of digits from autoencoder
        imglist=vae.vae_generate(digits)
        
        #stack images together and save to self._stacked_image
        self._stacked_image = self.stack_images(imglist,spacing_range)
        
        #print
        plt.figure(figsize=(15, 5))
        plt.imshow(self._stacked_image, vmin=0, vmax=1, cmap="gray")
        plt.title("Autoencoded image")
        plt.show()
        
        return None
    
        #insert some noise to the sample and denoise using variational autoencoder
        #also prints the figures before and after autoencoding noise reduction
    def denoise(self,digits, spacing_range=(1,3), image_width=None,noise_prob=0.5):
        
        #create the autoencoder instance
        vae=Vae()
        
        imglist=[]
        for digit in digits:
            
            #pick an image by random and append the image to imglists
            ind=np.random.choice(self._labelsdict[digit])
            imglist.append(self._images[ind,:,:])
            
            
        #set some noise to the figure, with prob noise_prob
        for image in imglist:
                
            #make a replacement mask where True is with probability noise_prob
            mask=np.random.binomial(1, noise_prob,size=imglist[0].shape).astype(np.bool)

            #random array
            unknowns=np.random.randint(2, size=imglist[0].shape)

            #apply the mask and do assigment
            image[mask] = unknowns[mask]
                
        #stack and print
        stacked_image=self.stack_images(imglist,spacing_range)
        
        noisestr='noise: ' + str(noise_prob*100)+'%'
        
        plt.figure(figsize=(15, 5))
        plt.imshow(stacked_image, vmin=0, vmax=1, cmap="gray")
        plt.title("Original image, "+noisestr)
        plt.show()
            
        #run the images through autoencoder
        imglist=vae.ff_autoencode(imglist,digits)
        
        #stack images together and save to self._stacked_image
        self._stacked_image = self.stack_images(imglist,spacing_range)
        
        #print
        plt.figure(figsize=(15, 5))
        plt.imshow(self._stacked_image, vmin=0, vmax=1, cmap="gray")
        plt.title("Autoencoded image, "+noisestr)
        plt.show()
        
        return None
    
    
    #save the latest generate image as png
    def save_latest_as_png(self,filename='output_image.png'):
    
        try:
                matplotlib.image.imsave(filename, self._stacked_image,cmap='gray')
        except AttributeError:
                plt.imsave(filename, self._stacked_image,cmap='gray')
                
    #print the last generated image
    def print_latest(self,size=(15,5)):
    
        plt.figure(figsize=(15, 5))
        plt.imshow(self._stacked_image, vmin=0, vmax=1, cmap="gray")
        plt.title("Last generated image")
        plt.show()  
