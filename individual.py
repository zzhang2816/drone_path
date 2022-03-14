import matplotlib.pyplot as plt
import numpy as np
import math
import matplotlib.animation
from scipy.io import savemat
import os

def get_dist(a,b):
    sz=min(a.shape[0],b.shape[0])
    dist = np.sum(np.power(a[:sz]-b[:sz],2),axis=1)
    # print(np.argmin(dist))
    return np.sqrt(np.min(dist))
    
class Circle:
    def __init__(self,startPoint, center):
        self.radius = math.sqrt((startPoint[0] - center[0])**2+(startPoint[1] - center[1])**2)
        self.center = center
        if startPoint[0] == center[0]:
            if center[1]>startPoint[1]:
                self.theta0 = math.pi/2
            else:
                self.theta0 = -math.pi/2
        else:
            x = (startPoint[1] - center[1])/(startPoint[0] - center[0])
            self.theta0 = math.atan(x)
            if self.theta0>0 and startPoint[1] - center[1]<0:
                self.theta0 = self.theta0+math.pi
            if self.theta0<0 and startPoint[1] - center[1]>0:
                self.theta0 = self.theta0+math.pi
            if startPoint[1] == center[1] and startPoint[0] < center[0]:
                self.theta0 = self.theta0+math.pi

    def getLinePoints(self, theta,time):
        w = theta/time
        print(f"w: {w}, v: {w*self.radius/scale}")
        t_move=np.linspace(0,time,time2intp(time))
        t_stop=np.ones(time2intp(stop_time))*time
        t=np.concatenate([t_move,t_stop])
        x = self.center[0] + np.cos(self.theta0+w*t)*self.radius
        y = self.center[1] + np.sin(self.theta0+w*t)*self.radius
        z = np.ones_like(t)*self.center[-1]
        return np.stack((x,y,z)).T

def time2intp(t):
    return int(10*t)

def line_move(startPos,endPos,time):
    speed = (np.array(endPos)-np.array(startPos))/time
    print("v: ",np.sqrt(np.sum(np.square(speed)))/scale)
    t_move=np.linspace(0,time,time2intp(time))
    t_stop=np.ones(time2intp(stop_time))*time
    t=np.concatenate([t_move,t_stop])
    path=startPos+np.dot(t.reshape(-1,1),speed.reshape(1,-1))
    return path

def Stop(startPos, time):
    sp = np.array([startPos])
    return np.repeat(sp,time2intp(time),axis=0)

def red():
    path=[]
    # wait 6s
    path.append(Stop([1,2,0.5],6*t))
    # take off
    path.append(line_move(startPos=[1,2,0.5], endPos=[1,2,1.7],time=3*t))
    path.append(line_move(startPos=[1,2,1.7], endPos=[2.8,2.8,1.7],time=6*t))
    # circle
    circle=Circle(startPoint=[2.8,2.8,1.7],center=[2,2,1.7]).getLinePoints(0.7*math.pi,time=0.7*16*t)
    path.append(circle)
    # fall_down
    endpos=path[-1][-1];path.append(line_move(startPos=endpos, endPos=[endpos[0],endpos[1],0.8],time=3*t))
    # stop
    path.append(Stop(path[-1][-1],2*t))
    # take off
    endpos=path[-1][-1];path.append(line_move(startPos=path[-1][-1], endPos=[endpos[0],endpos[1],1.3],time=2*t))
    # circle
    circle=Circle(startPoint=path[-1][-1],center=[2,2,1.3]).getLinePoints(0.5*math.pi,time=0.5*16*t)
    path.append(circle)
    # fall_down
    path.append(line_move(startPos=path[-1][-1], endPos=[1.7,1.5,1.3],time=3*t))
    endpos=path[-1][-1];path.append(line_move(startPos=endpos, endPos=[endpos[0],endpos[1],0.8],time=2*t))
    # wait 2s
    path.append(Stop(path[-1][-1],2*t))
    # take off
    path.append(line_move(startPos=path[-1][-1], endPos=[1,2,1.3],time=3*t))
    # move
    path.append(line_move(startPos=path[-1][-1], endPos=[1,3.5,1.3],time=5*t))
    # fall_down
    endpos=path[-1][-1];path.append(line_move(startPos=endpos, endPos=[endpos[0],endpos[1],0.8],time=2*t))
    return np.concatenate(path,axis=0)

def green():
    path=[]
    # wait 3s
    path.append(Stop([2,2,0.5],2*t))
    # take off
    path.append(line_move(startPos=[2,2,0.5], endPos=[2,2,1.5],time=3*t))
    path.append(line_move(startPos=[2,2,1.5], endPos=[3.2,3.2,1.7],time=5*t))
    # circle
    circle=Circle(startPoint=[3.2,3.2,1.7],center=[2,2,1.7]).getLinePoints(0.5*math.pi,time=0.5*16*t)
    path.append(circle)
    # fall_down
    endpos=path[-1][-1];path.append(line_move(startPos=endpos, endPos=[endpos[0],endpos[1],0.8],time=3*t))
    # stop
    path.append(Stop(path[-1][-1],2*t))
    # take off
    endpos=path[-1][-1];path.append(line_move(startPos=path[-1][-1], endPos=[endpos[0],endpos[1],1.5],time=3*t))
    # circle
    circle=Circle(startPoint=path[-1][-1],center=[2,2,1.5]).getLinePoints(0.95*math.pi,time=0.95*16*t)
    path.append(circle)
    # fall_down
    path.append(line_move(startPos=path[-1][-1], endPos=[2.5,1.5,1.4],time=3*t))
    endpos=path[-1][-1];path.append(line_move(startPos=endpos, endPos=[endpos[0],endpos[1],0.8],time=2*t))
    # wait 2s
    path.append(Stop(path[-1][-1],2*t))
    # take off
    endpos=path[-1][-1];path.append(line_move(startPos=path[-1][-1], endPos=[2.5,1.5,1.3],time=3*t))
    # move
    path.append(line_move(startPos=path[-1][-1], endPos=[2,3.5,1.3],time=5*t))
    # fall_down
    endpos=path[-1][-1];path.append(line_move(startPos=endpos, endPos=[endpos[0],endpos[1],0.8],time=2*t))
    return np.concatenate(path,axis=0)

def blue():
    path=[]
    # take off
    path.append(line_move(startPos=[3,2,0.5], endPos=[3.2,3.2,1.7],time=5*t))
    # circle
    circle=Circle(startPoint=[3.2,3.2,1.7],center=[2,2,1.7]).getLinePoints(0.7*math.pi,time=0.7*16*t)
    path.append(circle)
    # fall_down
    endpos=path[-1][-1];path.append(line_move(startPos=endpos, endPos=[endpos[0],endpos[1],0.8],time=3*t))
    # stop
    path.append(Stop(path[-1][-1],2*t))
    # take off
    endpos=path[-1][-1];path.append(line_move(startPos=path[-1][-1], endPos=[endpos[0],endpos[1],1.2],time=2*t))
    # circle
    circle=Circle(startPoint=path[-1][-1],center=[2,2,1.2]).getLinePoints(1*math.pi,time=(1*16+3)*t)
    path.append(circle)
    # fall_down
    path.append(line_move(startPos=path[-1][-1], endPos=[3.5,1.5,1.2],time=3*t))
    endpos=path[-1][-1];path.append(line_move(startPos=endpos, endPos=[endpos[0],endpos[1],0.8],time=2*t))
    # wait 2s
    path.append(Stop(path[-1][-1],2*t))
    # take off
    endpos=path[-1][-1];path.append(line_move(startPos=path[-1][-1], endPos=[3.2,1.5,1.3],time=3*t))
    # move
    path.append(line_move(startPos=path[-1][-1], endPos=[3,3.5,1.3],time=5*t))
    # fall_down
    endpos=path[-1][-1];path.append(line_move(startPos=endpos, endPos=[endpos[0],endpos[1],0.8],time=2*t))
    return np.concatenate(path,axis=0)

def show_path():
    fig=plt.figure()
    ax = fig.gca(projection='3d')
    ax.set(xlim3d=(-1.3, 1.3), xlabel='X')
    ax.set(ylim3d=(-1.3, 1.3), ylabel='Y')
    ax.set(zlim3d=(0, 2), zlabel='Z')
    ax.plot3D(P_r[:,0],P_r[:,1],P_r[:,2],color='r')
    ax.plot3D(P_g[:,0],P_g[:,1],P_g[:,2],color='g')
    ax.plot3D(P_b[:,0],P_b[:,1],P_b[:,2],color='b')
    plt.show() 

def show_ani():
    def update_graph(num):
        graph.set_data (data[:,num,0], data[:,num,1])
        graph.set_3d_properties(data[:,num,2])
        return graph

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    title = ax.set_title('3D Test')

    data=np.stack([P_r,P_g,P_b])

    ax.set(xlim3d=(-1.3, 1.3), xlabel='X')
    ax.set(ylim3d=(-1.3, 1.3), ylabel='Y')
    ax.set(zlim3d=(0, 2), zlabel='Z')
    graph, = ax.plot(data[:,0,0], data[:,0,1],data[:,0,2], linestyle="", marker="o")

    num = data.shape[1]
    ani = matplotlib.animation.FuncAnimation(fig, update_graph, num, 
                                interval=3)
    ani.save('video/decentralize.mp4',  
            writer = 'ffmpeg', fps = 30)

def shown_x_y_z():
    fig = plt.figure(figsize=(10, 5))
    plt.tight_layout()
    # P_r,P_g,P_b
    ax1 = fig.add_subplot(311)
    ax1.plot(P_r[:,0],'r--')
    ax1.plot(P_g[:,0],'g')
    ax1.plot(P_b[:,0],'b--')
    ax1.set_xlabel('t')
    ax1.set_ylabel('y')
    ax2 = fig.add_subplot(312)
    ax2.plot(P_r[:,1],'r--')
    ax2.plot(P_g[:,1],'g')
    ax2.plot(P_b[:,1],'b--')
    ax2.set_xlabel('t')
    ax2.set_ylabel('x')
    ax3 = fig.add_subplot(313)
    ax3.plot(P_r[:,-1],'r--')
    ax3.plot(P_g[:,-1],'g')
    ax3.plot(P_b[:,-1],'b--')
    ax3.set_xlabel('t')
    ax3.set_ylabel('z')
    plt.show()

if __name__ == '__main__':
    t=1 # decrease the time to speed up 50 interpolation per second
    stop_time=1
    scale=1.9

    print("red")
    P_r = red()
    print("green")
    P_g = green()
    print("blue")
    P_b = blue()

    bd_min=[]
    bd_max=[]
    for p in [P_r,P_g,P_b]:
        print(len(p))
        p[:,0:2]=(p[:,0:2]-2)/scale 
        p[:,1]=p[:,1]+0.2
    #     bd_min.append(np.min(p,axis=0))
    #     bd_max.append(np.max(p,axis=0))
    # print(np.min(np.array(bd_min),axis=0))
    # print(np.max(np.array(bd_max),axis=0))

    print(get_dist(P_r,P_g))
    print(get_dist(P_r,P_b))
    print(get_dist(P_g,P_b))

    # show_ani()
    show_path()
    # shown_x_y_z()

    # dir="./resource/decentralize/"
    # if not os.path.isdir(dir):
    #     os.mkdir(dir)

    # m1dic = {'pos_x_traj':P_r[:,1],'pos_y_traj':P_r[:,0],'pos_z_traj':P_r[:,2]}
    # m2dic = {'pos_x_traj':P_g[:,1],'pos_y_traj':P_g[:,0],'pos_z_traj':P_g[:,2]}
    # m3dic = {'pos_x_traj':P_b[:,1],'pos_y_traj':P_b[:,0],'pos_z_traj':P_b[:,2]}

    # savemat(dir+"p1.mat",m1dic)
    # savemat(dir+"p2.mat",m2dic)
    # savemat(dir+"p3.mat",m3dic)


