# Handwritten digit generator

Handwritten digit generator and .png saver written in python based on MNIST database of 28x28 handwritten images (http://yann.lecun.com/exdb/mnist/). The digits can be sampled from MNIST database directly or generated using conditional variational autoencoder CVAE based on convolutional "deep" neural network. Autoencoders ability to denoise can be also tested.

Code is partly more polished, partly less polished.

Sample output from MNIST:  
![Sample output from MNIST](sample_output.png?raw=true "Sample output from MNIST")

Sample output from CVAE:  
![Sample output from MNIST](output_image_vae.png?raw=true "Sample output from VAE")

Input to CVAE with 50% noise and the denoised output:  
![noisy input to CVAE](noise_prevae.png?raw=true "Noisy input to CVAE")
 
![output from CVAE](noise_postvae.png?raw=true "Output from CVAE")


## Requirements:
2.7 or newer, numpy 1.10 or newer, matplotlib 1.4.0 or newer. For CVAE, tensorflow 1.2 or newer.

## Installation:
You can also download the package files by hand from https://github.com/kutvonenaki/python-handwritten-digit-gen
and the mnist data from http://yann.lecun.com/exdb/mnist/. Running the scripts as described below require bash environment. Windows users may consider downloading the files by hand. The default folder for mnist data is mnist_data/.

#### Installation requirements
Depending on the OS build, running the scripts as described below may requires installation of wget, git. Older python distributions (pre2.7.9, pre3.4) may require installation of pip or pip3 (python2 / python3).  

to install git: apt-get update && apt-get install -y git  
to install wget: apt-get update && apt-get install -y wget  
to install pip/pip3: apt-get update && apt-get install -y pip/pi3  

#### To clone the repository
git clone https://github.com/kutvonenaki/python-handwritten-digit-gen

#### Enter the directory
cd python-handwritten-digit-gen

#### Get MNIST data with a script:
./load_mnist.sh

#### Install the package with pip 
for python2: pip install .   
for python3: pip3 install .  

## Testing:
./test_output   
-runs extremely minimal tests and prints test_output.png
-requires nosetests and coverage, can be also run without those by python tests/test_sample.py

## Command line usage  

Installing the package sets up a command line command generate_png. Use $generate_png -h for showing the command help.

#### Example use:  

To generate a random sequence of 5 digits with the CVAE with default white space 4pix between the digits and default output filename image_out.png:  
$generate_png --random 5 --cvae 1

To generate sequence 1,3,5 with spacing ~ Uniform[2,5] and output filename sequence.png by direct sampling from MNIST:
$generate_png --digits 1 3 5 --spacing 2 5 --outfile sequence.png


# *class* digitgenerator.Digitgenerator
## *__init__*(mnist_folder='mnist_data',preprocessed_folder='.')

**Returns:**	*-type: self*  
Returns an instance of self.  

#### Parameters

**mnist_folder**            *-type:str, default='mnist_data'*  
-The default folder for the mnist_data  

**preprocessed_folder**     *-type:str, default='.'*  
-The default folder for the preprocessed data files (images,labels, dictionary file for fast retrieving of images)

#### Attributes

**labels**    *-type int8, shape (60000,)*
-The handwritten digit labels. A single integer from 0 to 9

**images**    *-type float32, shape(60000,28,28)*
-The digits as 28x28 images, the pixel values are from 0 (black) to 1 (white)

## Methods

## **draw_from_mnist**(self,digits, spacing_range, image_width=None)            
-Generate a sample from mnist and return a numpy arrays where the digits are horizontally stacked into a one array

**Returns:**  *2 dimensional numpy array of type float32, where the len of the first dimension is 28*

#### Parameters

**digits**            *-type: list of integers'*  
-A list-like containing the numerical values of the digits from which the sequence will be generated (for example [3, 5, 0]).

**spacing_range**     *-type: tuple of 2 integers, default=(4,4)*  
-a (minimum, maximum) pair (tuple), representing the min and max spacing between digits. The whitespace between the horizontally stacked images is sampled from Uniform[minimum,maximum]

**image_width**     *-type: integer, default=None*  
-The width of the total image. Not in use in the current version since the width of the image is determined by the number of digits and the sapling of spacing between them. Could be modified if image overlapping is wanted

## **sample_from_vae**(self,digits, spacing_range, image_width=None)            
-Generate a sample from the CVAE by sampling the latent space by random
-For explanation of parameters, see above
-If the variational autoencoder is not trained (no pretrained parameter files found), error is raised

**Returns:**  *2 dimensional numpy array of type float32, where the len of the first dimension is 28*

## **denoise**(self,digits, spacing_range, image_width=None,noise_prob=0.5)            
-Draw a sample from MNIST and add noise_prob of noise and denoise the images using CVAE.
-Prints the images before and after denoising.
-For other parameters, see above.

**Returns:**  *2 dimensional numpy array of type float32, where the len of the first dimension is 28*

## **train_vaemodel**(self,digits, spacing_range, image_width=None,noise_prob=0.5)            
-Trains the CVAE and saves the trained parameters to file in tf_vae_files/

## **save_latest_as_png** (self,filename='output_image.png')           
-Saves the last generated sample as png.

## **print_latest** (self,size=(15,5))           
-Prints the lates generated image.

## Code examples:

```python
>>> import numpy as np
>>> from digitgenerator import Digitgenerator
>>>
>>> #initialize an instance
>>> digitgen=Digitgenerator()
Loaded presaved numpy arrays from disk
>>>
>>> #draw sample from mnist
>>> outputarray=digitgen.draw_from_mnist([1,2,3],(5,10))
>>> print('The shape of the output array is', np.shape(outputarray))
The shape of the output array is (28, 100)
>>>
>>>#sample numbers from CVAE
>>>digitgen.sample_from_vae([0,1,2,3,4,5,6,7,8,9,0])
found pretrained variables which can be used
Restoring saved parameters
INFO:tensorflow:Restoring parameters from tf_vae_files\vae-0
>>>
>>>#print a sample from MNIST with 30% noise and denoise with CVAE
>>>digitgen.denoise([0,1,2,3,4,5,6,7,8,9,0],noise_prob=0.30)
>>>
>>> #save as the latest drawn sample as sequence_out.png
>>> digitgen.save_latest_as_png(filename='sequence_out.png')
```

## Notes
Multiple things could be improved:
-The Digitgenerator class uses Vae class under the hood and the current way is not optimized
-stack_images could be modified to allow 
-variational autoencoder could be used to generate coherent styles, now all digits are just randomized from the latent space
-parameter loading could be done better, now the parameters are loaded every time
-warping could be added
-would be interesting to try to combine variational aproach with capsule networks

