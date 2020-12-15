"""
This is a modified version of create_data_for_assignment.py script provided, which does the following:
* Read the resized coloured image.
* Split the image into the constituent Red, Green and Blue channels.
* For each image of the constituent channel, corrupt it and output the files required for performing optimisation.
"""
""" changes to the original code are indicated by ----> """
import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack as spfft
import scipy.ndimage as spimg
import imageio
#%% Discrete Cosine Transform 
def dct2(x):
    return spfft.dct(spfft.dct(x.T, norm='ortho', axis=0).T, norm='ortho', axis=0)

def idct2(x):
    return spfft.idct(spfft.idct(x.T, norm='ortho', axis=0).T, norm='ortho', axis=0)

#%% VARIABLES FOR YOU TO CHANGE
path_to_your_image="./rsz_profile.png"
zoom_out=0.9999999 #Fraction of the image you want to keep.
corruption=0.9#Fraction of the pixels that you want to discard
#%% Get image and create y
# read original image and downsize for speed
orig =imageio.imread(path_to_your_image) # ----> read in color
np.random.seed(0) # ----> fix the random seed so that it'll be easier to check
# extract small sample of signal
X = spimg.zoom(orig, zoom_out)
ny,nx,ncolor = X.shape[0],X.shape[1],X.shape[2]
corruption=1-corruption
k = round(nx * ny * corruption)
ri = np.random.choice(nx * ny, k, replace=False) # random sample of indices

for i in range(0,3): # ----> create output files for each of the three constituent channels
	Xorig = orig.copy() # ----> make a copy of the original image each time
	# ----> mult_factor decides the offset
	""" ---->(note that since the image is flattened, each time we sample indices with respect to a channel, 
	we need to choose the pixel value belonging to 'that' channel, hence the offset is required)"""
	
	if i==0: # red ----> indexing red with 0 and setting the other channel pixel values to 0 (to extract red only)
		mult_factor = 1
		blank_1 = 1
		blank_2 = 2
	elif i==1: # green ----> indexing green with 1 and setting the other channel pixel values to 0 (to extract green only)
		mult_factor = 2
		blank_1 = 0
		blank_2 = 2
	else: # blue ----> indexing blue with 2 and setting the other channel pixel values to 0 (to extract blue only)
		mult_factor = 3
		blank_1 = 0
		blank_2 = 1
	# ----> set the other colors to
	Xorig[:,:,blank_1]=0
	Xorig[:,:,blank_2]=0
	
	#Downsize image 
	X = spimg.zoom(Xorig, zoom_out)

	# extract small sample of signal
	b = X.T.flat[ri+(10000*(mult_factor-1))] # ----> note the offset here!
	b = np.expand_dims(b, axis=1)

	#%% CREATE A inverse and C
	# *******************************************************************************************
	"""This part consumes a lot of memory. Your PC might crash if the images you load are larger than 100 x 100 pixels """
	# create dct matrix operator using kron (memory errors for large ny*nx)
	Aa = np.kron(
	    np.float16(spfft.idct(np.identity(nx), norm='ortho', axis=0)),
	    np.float16(spfft.idct(np.identity(ny), norm='ortho', axis=0))
	    )
	A = Aa[ri,:] # same as B times A
	# *******************************************************************************************
	# create images of mask (for visualization)
	Xm = 255 * np.ones(X.shape)
	Xm.T.flat[ri+(10000*(mult_factor-1))] = X.T.flat[ri+(10000*(mult_factor-1))] # ----> note the offset here!
	Xm = Xm.astype(np.uint8)

	plt.imshow(Xorig)
	plt.title("Original")
	plt.show()

	plt.imshow(Xm)
	plt.title("Incomplete")
	plt.show()
	#%% SAVE MATRICES TO DRIVE

	import os
	dir_name="Try12" # ----> you could name this directory as you like!
	try:
	    os.mkdir(dir_name)
	except Exception as e:
	    pass

	np.save(dir_name+'/C'+str(i),A)
	np.save(dir_name+'/A_inv'+str(i),Aa)
	np.save(dir_name+'/y'+str(i),b)
	plt.imsave(dir_name+'/incomplete'+str(i)+'.png',Xm)
	plt.imsave(dir_name+'/original_with_crop'+str(i)+'.png',X)
	
	# ----> need to clear the arrays or else artefacts will end up in the reconstructed images!
	X = []
	b = []
	A = []
	Aa = []
	Xm = []
