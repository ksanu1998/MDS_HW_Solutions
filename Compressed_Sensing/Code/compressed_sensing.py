"""
This code recovers an image x, given y, an incomplete measurement of x; C, a transformation matrix; and A_inv, inverse of the sensing matrix A
For achieving this, it solves an optimisation problem:
minimise(1-norm of s) such that 2-norm-squared(y-Cs)=0
Since the norm-squared constraint can be recast as an affine constraint, we provide the affine equality as our constraint instead.
The optimisation is run for the three constituent channels of Red, Green and Blue. So the corresponding input files are expected by this code.
"""

# import libraries
import cvxpy as cp # this is convex optimisation library
import numpy as np
import matplotlib.pyplot as plt

# run the optimisation for each constituent channel: Red, Green, Blue
for i in range(0,3):

	# loading the required matrices and vectors
	C = np.load("./C"+str(i)+".npy") # load the transformation matrix
	A_inv = np.load("./A_inv"+str(i)+".npy") # load the inverse of the sensing matrix A
	y = np.load("./y"+str(i)+".npy") # load the incomplete measurement of x
	s = cp.Variable(10000) # declaring the optimising variable

	# optimisation
	objective = cp.Minimize(cp.norm(s, 1)) # declaring the objective as minimisation of l1-norm of s
	constraints = [C@s == y.reshape(len(y),)] # declaring the constraint as an affine equality
	prob = cp.Problem(objective, constraints) # declaring the optimisation problem
	
	# solving the optimisation problem using ECOS solver
	obj = prob.solve(verbose=True, solver=cp.ECOS, max_iters=30, abstol=1e-6, reltol=1e-6, feastol=1e-6)
	
	# reconstruction
	recon_img = A_inv@s.value # reconstruct the original image by multiplying the s value with the sensing matrix

	# save numpy arrays so that they can be worked upon in future
	np.save('./s_value'+str(i),s.value)
	np.save('./recon_img_array'+str(i),recon_img)
	
	# plot the reconstructed image
	plt.imshow(recon_img.reshape(100,100).T)
	plt.axis('off') # not printing axis to focus only on the image
	plt.savefig('./recon'+str(i)+'.png')
	plt.close()