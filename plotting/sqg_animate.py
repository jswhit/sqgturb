import matplotlib.pyplot as plt
import matplotlib.animation as animation
from netCDF4 import Dataset

filename = 'data/sqg_N256_aliased.nc'
vmin = -25; vmax = 25; levplot = 1

nc = Dataset(filename)
pv_var = nc['pv']
t_var = nc['t']
nsteps = len(t_var)-1
scalefact = nc.f*nc.theta0/nc.g

fig = plt.figure(figsize=(8,8))
fig.subplots_adjust(left=0, bottom=0.0, top=1., right=1.)
nout = 1

def initfig():
    global im
    ax = fig.add_subplot(111)
    ax.axis('off')
    pv = scalefact*pv_var[0,levplot,...]
    im = ax.imshow(pv,cmap=plt.cm.jet,interpolation='nearest',origin='lower',vmin=vmin,vmax=vmax)
    return im,

def updatefig(*args):
    global nout
    t = t_var[nout]
    pv = scalefact*pv_var[nout]
    hr = t/3600.
    print hr,pv.min(),pv.max()
    im.set_data(pv[levplot])
    nout += 1
    return im,

# interval=0 means draw as fast as possible
ani = animation.FuncAnimation(fig, updatefig, frames=nsteps, repeat=False,\
      init_func=initfig,interval=0,blit=True)
plt.show()
