import numpy as np  
import matplotlib.pyplot as plt  
from mpl_toolkits import mplot3d
import math
from scipy.spatial import ConvexHull, convex_hull_plot_2d
from matplotlib import collections  as mc

# function to plot the contour of a saddle shaped curve
def x2y2_saddle_contour():
    x = np.arange(-5,6)
    y = np.arange(-5,6)
    X, Y = np.meshgrid(x, y)
    Z = eval('X**2-Y**2')
    fig,ax=plt.subplots(1,1)
    cp = ax.contourf(X,Y,Z)
    fig.colorbar(cp)
    plt.xlabel("x-axis")
    plt.ylabel("y-axis")
    plt.xticks(np.arange(-5,6,1))
    plt.yticks(np.arange(-5,6,1))
    ax.set_title(r'Contour plot of $f(x,y) = x^{2}-y^{2}\ \forall\ x,y \in\ [-5,5]$')
    plt.savefig('x2y2_saddle_contour.png')
    plt.close()

# function to plot the surface of a saddle shaped curve
def x2y2_saddle_surface():
    x = np.arange(-5,6)
    y = np.arange(-5,6)
    X, Y = np.meshgrid(x, y)
    Z = eval('X**2-Y**2')
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax.plot_surface(X,Y,Z,color="green")
    plt.xlabel("x-axis")
    plt.ylabel("y-axis")
    plt.xticks(np.arange(-5,6,1))
    plt.yticks(np.arange(-5,6,1))
    ax.set_title(r'Plot of $f(x,y) = x^{2}-y^{2}\ \forall\ x,y \in\ [-5,5]$')
    plt.savefig('x2y2_saddle_surface.png')
    plt.close()

# function to generate random points and plot the convex hull of those points
def conv_hull():
	np.random.seed(1)
	points = np.random.uniform(low=-5, high=5, size=(24,2))
	hull = ConvexHull(points)
	plt.scatter(points[:,0], points[:,1], c ="blue", label=r"$24$ points belonging to $[-5,5]$") 
	for simplex in hull.simplices:
		plt.plot(points[simplex, 0], points[simplex, 1], 'k-',c="red", label="Convex Hull")
	plt.xlabel("x-axis")
	plt.ylabel("y-axis")
	plt.title(r"Convex hull of a set of $24$ randomly generated points belonging to $[-5,5]\times[-5,5]$",fontsize=10)
	plt.grid(b=True,linewidth=0.5,which='both')
	plt.xticks(np.arange(-5,6,1))
	plt.yticks(np.arange(-5,6,1))
	plt.savefig('conv_hull.png')
	plt.close()

# function to implement the quadratic form xTAx where A is defined by [[cos sin],[sin cos]]
def my_func(i,x,y):
	return x*(x*math.cos(i)+y*math.sin(i))+y*(x*math.sin(i)+y*math.cos(i))

# function to find counter examples where xTAx is not convex
def psd():
	np.random.seed(1)
	v = np.random.randint(low=-10, high=10, size=(1,2))
	theta = np.arange(0,2*math.pi,0.1)
	counter_example_count = 0
	counter_rot_matrices = []
	rot_vec = []
	rot_vec_nonpsd = []
	theta_nonpsd = []
	theta_psd = []
	for i in theta:
		func_val = v[0][0]*(v[0][0]*math.cos(i)+v[0][1]*math.sin(i))+v[0][1]*(v[0][0]*math.sin(i)+v[0][1]*math.cos(i))
		if func_val < 0:
			counter_rot_matrices.append([[math.cos(i),math.sin(i)],[math.sin(i),math.cos(i)]])
			counter_example_count+=1
			rot_vec_nonpsd.append([(0,0),(v[0][0]*math.cos(i)+v[0][1]*math.sin(i),v[0][0]*math.sin(i)+v[0][1]*math.cos(i))])
			theta_nonpsd.append(i)
		else:
			rot_vec.append([(0,0),(v[0][0]*math.cos(i)+v[0][1]*math.sin(i),v[0][0]*math.sin(i)+v[0][1]*math.cos(i))])
			theta_psd.append(i)
		if counter_example_count==11:
			break
	f = open("counter_examples.txt", "w")
	for i in counter_rot_matrices:
		f.write(str(round(i[0][0],2))+' '+str(round(i[0][1],2))+'\n'+str(round(i[1][0],2))+' '+str(round(i[1][1],2))+'\n\n')
	f.write(str(v[0][0])+' '+str(v[0][1]))
	f.close()
	lc = mc.LineCollection(rot_vec,colors=["blue"],label=r"$x^{T}Ax\geq0$")
	lc_nonpsd = mc.LineCollection(rot_vec_nonpsd,colors=["red"],label=r"$x^{T}Ax<0$")
	fig, ax = plt.subplots()
	plt.xticks(np.arange(-10,10,math.pi))
	plt.yticks(np.arange(-10,10,math.pi))
	ax.add_collection(lc)
	ax.add_collection(lc_nonpsd)
	leg = plt.legend()
	plt.legend(fontsize=10)
	plt.xticks(np.arange(-10,10,math.pi))
	plt.yticks(np.arange(-10,10,math.pi))
	plt.xlabel("x-axis")
	plt.ylabel("y-axis")
	plt.title(r"A random vector in $[-10,10]$ rotated and scaled using the matrix $A$ which is defined as function of $\theta$",fontsize=8)
	plt.savefig('psd.png')
	plt.close()

	for i in theta_nonpsd:
		x = np.arange(-10,10)
		y = np.arange(-10,10)
		X, Y = np.meshgrid(x, y)
		zs = np.array([my_func(i,x,y) for x,y in zip(np.ravel(X), np.ravel(Y))])
		Z = zs.reshape(X.shape)
		fig = plt.figure()
		ax = plt.axes(projection="3d")
		ax.plot_surface(X,Y,Z,color="red")
		plt.xlabel("x-axis")
		plt.ylabel("y-axis")
		plt.title(r"Surface of $x^{T}Ax$ where $A$ is defined as a function of $\theta=$"+str(round(i/math.pi,2))+r"$\pi$",fontsize=10)
		plt.xticks(np.arange(-10,10,2))
		plt.yticks(np.arange(-10,10,2))
		plt.savefig(str(int(100*round(i/math.pi,2)))+'pi.png')
		plt.close()

	for i in theta_psd:
		x = np.arange(-10,10)
		y = np.arange(-10,10)
		X, Y = np.meshgrid(x, y)
		zs = np.array([my_func(i,x,y) for x,y in zip(np.ravel(X), np.ravel(Y))])
		Z = zs.reshape(X.shape)
		fig = plt.figure()
		ax = plt.axes(projection="3d")
		ax.plot_surface(X,Y,Z,color="blue")
		plt.xlabel("x-axis")
		plt.ylabel("y-axis")
		plt.title(r"Surface of $x^{T}Ax$ where $A$ is defined as a function of $\theta=$"+str(round(i/math.pi,2))+r"$\pi$",fontsize=10)
		plt.xticks(np.arange(-10,10,2))
		plt.yticks(np.arange(-10,10,2))
		plt.savefig(str(int(100*round(i/math.pi,2)))+'pi.png')
		plt.close()

x2y2_saddle_surface()
x2y2_saddle_contour()
conv_hull()
psd()