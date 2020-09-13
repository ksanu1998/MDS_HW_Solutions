import numpy as np  
import matplotlib.pyplot as plt  
from mpl_toolkits import mplot3d

def x2():
    x0 = np.array(range(-10, 11))
    y0 = eval('x0**2')
    plt.fill_between(x0, y0, 100, color='linen')
    plt.plot(x0, y0)
    plt.title(r'Plot of $f(x) = x^{2}\ \forall\ x \in\ [-10,10]$')
    plt.savefig('x2.png')
    plt.close()

def sinx():
    x1 = np.array(np.arange(-2*np.pi, 2*np.pi, 0.01))
    y1 = np.sin(x1)
    plt.fill_between(x1, y1, 1, color='linen')
    plt.plot(x1, y1)
    plt.title(r'Plot of $f(x) = \sin(x)\ \forall\ x \in\ [-2\pi,2\pi]$')
    plt.savefig('sinx.png')
    plt.close()

def xy2():
    x2 = np.array(range(-10, 11))
    y2 = np.array(range(-10, 11))
    X2, Y2 = np.meshgrid(x2, y2)
    Z2 = eval('(X2*Y2)**2')
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax.plot_surface(X2,Y2,Z2)
    ax.set_title(r'Plot of $f(x,y) = (xy)^{2}\ \forall\ x,y \in\ [-10,10]$')
    plt.savefig('xy2.png')
    plt.close()

def xy2contour():
    x3 = np.array(range(-10, 11))
    y3 = np.array(range(-10, 11))
    X3, Y3 = np.meshgrid(x3, y3)
    Z3 = eval('(X3*Y3)**2')
    fig3,ax3=plt.subplots(1,1)
    cp = ax3.contourf(X3,Y3,Z3)
    fig3.colorbar(cp)
    ax3.set_title(r'Contour plot of $f(x,y) = (xy)^{2}\ \forall\ x,y \in\ [-10,10]$')
    plt.savefig('xy2contour.png')
    plt.close()

x2()
sinx()
xy2()
xy2contour()