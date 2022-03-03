import numpy as np
import math

def get_path(r, w, t, theta):
    x=0+r*np.cos(w*t+theta)
    y=0+r*np.sin(w*t+theta)
    z=1.5*np.ones_like(x)
    path=np.stack([x,y,z]).T
    return path


t=np.linspace(0,16,int(16/0.01))
p1=get_path(0.5, math.pi/4, t, 0)
p2=get_path(0.5, math.pi/4, t, math.pi/2)
p3=get_path(0.5, math.pi/4, t, math.pi)


np.save("p1.npy",p1)
np.save("p2.npy",p2)
np.save("p3.npy",p3)