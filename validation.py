import scipy.io as scio
import matplotlib.pyplot as plt
import numpy as np

def shown_x_y_z():
    fig = plt.figure(figsize=(10, 5))
    plt.tight_layout()
    # P_r,P_g,P_b
    ax1 = fig.add_subplot(311)
    ax1.plot(P_r[:,0],'r--')
    ax1.plot(P_g[:,0],'g')
    ax1.plot(P_b[:,0],'b--')
    ax1.set_xlabel('t')
    ax1.set_ylabel('x')
    ax2 = fig.add_subplot(312)
    ax2.plot(P_r[:,1],'r--')
    ax2.plot(P_g[:,1],'g')
    ax2.plot(P_b[:,1],'b--')
    ax2.set_xlabel('t')
    ax2.set_ylabel('y')
    ax3 = fig.add_subplot(313)
    ax3.plot(P_r[:,-1],'r--')
    ax3.plot(P_g[:,-1],'g')
    ax3.plot(P_b[:,-1],'b--')
    ax3.set_xlabel('t')
    ax3.set_ylabel('z')
    plt.show()

def convert(data):
    x=np.array(data['pos_x_traj']).reshape(-1)
    y=np.array(data['pos_y_traj']).reshape(-1)
    z=np.array(data['pos_z_traj']).reshape(-1)
    return np.stack([x,y,z]).T

path = 'resource/centralize/'
dataFiles=['p1.mat','p2.mat','p3.mat']
trajs=[]

for dataFile in dataFiles:
    data = scio.loadmat(path+dataFile)
    trajs.append(convert(data))

P_r,P_g,P_b = trajs[0],trajs[1],trajs[2]
shown_x_y_z()
