# import libraries
import numpy as np
import matplotlib.pyplot as plt
import imageio

# load the image arrays corresponding to each of three constituent channels
r = np.load("./recon_img_array0.npy")
g = np.load("./recon_img_array1.npy")
b = np.load("./recon_img_array2.npy")

# reshape the flat arrays into a 100 x 100 matrix
r = r.reshape(100,100).T
g = g.reshape(100,100).T
b = b.reshape(100,100).T

# stack different channels into a single image
mix = np.dstack((r,g,b))

# perform normalisation of pixel values
mix = abs((mix-mix.min())/(mix.max()-mix.min()))

# plot the reconstructed image
plt.imshow(mix)
plt.axis('off')
plt.savefig("recon_color.png")
plt.close()