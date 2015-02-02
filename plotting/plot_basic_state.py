import matplotlib.pyplot as plt
import numpy as np
L = 20.e6
H = 10.e3
U = 30.
nsq = 1.e-4
f = 1.e-4
umax = 30.
N = 256
K = 101
theta0 = 300; g = 9.8066
scalefact = f*theta0/g
symmetric = False
# setup basic state wind, pv, pv gradient
y = np.arange(0,L,L/N,dtype=np.float32)
z = np.linspace(0,H,K)
y,z = np.meshgrid(y,z)
u = np.zeros((K,N),np.float32)
theta = np.zeros((K,N),np.float32)
pi = np.array(np.pi,np.float32)
l = 2.*pi/L
mu = l*np.sqrt(nsq)*H/f
# symmetric version
# no difference between upper and lower boundary
if symmetric:
    u = -0.5*U*np.sin(l*y)*np.sinh(mu*(z-0.5*H)/H)*np.sin(l*y)/np.sinh(0.5*mu)
    theta = scalefact*(0.5*U*mu/(l*H))*np.cosh(mu*(z-0.5*H)/H)*np.cos(l*y)/np.sinh(0.5*mu) + \
    theta0 + (theta0*nsq*z/g)
else:
    # if surface ekman damping on, basic state has no flow at surface.
    u = U*np.sin(l*y)*np.sinh(mu*z/H)*np.sin(l*y)/np.sinh(mu)
    theta = scalefact*(U*mu/(l*H))*np.cosh(mu*z/H)*np.cos(l*y)/np.sinh(mu) +\
    theta0 + (theta0*nsq*z/g)
print u.min(), u.max()
print theta.min(), theta.max()
cs1 = plt.contour(y,z,u,np.arange(-30,31,3),colors='k')
cs2 = plt.contour(y,z,theta,np.arange(250,361,5),colors='k',linestyles='dotted')
plt.show()
