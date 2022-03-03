from json.tool import main
import matplotlib.pyplot as plt
import numpy as np
import math

def get_dist(a,b):
    sz=min(a.shape[0],b.shape[0])
    return np.sqrt(np.min(np.sum(np.power(a[:sz]-b[:sz],2),axis=1)))
    
class Bezier:
    # 输入控制点，Points是一个array,num是控制点间的插补个数
    def __init__(self,Points,InterpolationNum):
        self.demension=Points.shape[1]   # 点的维度
        self.order=Points.shape[0]-1     # 贝塞尔阶数=控制点个数-1
        self.num=InterpolationNum        # 相邻控制点的插补个数
        self.pointsNum=Points.shape[0]   # 控制点的个数
        self.Points=Points
        
    # 获取Bezeir所有插补点
    def getBezierPoints(self,method):
        if method==0:
            return self.DigitalAlgo()
        if method==1:
            return self.DeCasteljauAlgo()
    
    # 数值解法
    def DigitalAlgo(self):
        PB=np.zeros((self.pointsNum,self.demension)) # 求和前各项
        pis =[]                                      # 插补点
        for u in np.arange(0,1+1/self.num,1/self.num):
            for i in range(0,self.pointsNum):
                PB[i]=(math.factorial(self.order)/(math.factorial(i)*math.factorial(self.order-i)))*(u**i)*(1-u)**(self.order-i)*self.Points[i]
            pi=sum(PB).tolist()                      #求和得到一个插补点
            pis.append(pi)            
        return np.array(pis)

    # 德卡斯特里奥解法
    def DeCasteljauAlgo(self):
        pis =[]                          # 插补点
        for u in np.arange(0,1+1/self.num,1/self.num):
            Att=self.Points
            for i in np.arange(0,self.order):
                for j in np.arange(0,self.order-i):
                    Att[j]=(1.0-u)*Att[j]+u*Att[j+1]
            pis.append(Att[0].tolist())

        return np.array(pis)

class Circle:
    def __init__(self,startPoint, center, time):
        self.interp=time2intp(time)
        self.radius = math.sqrt((startPoint[0] - center[0])**2+(startPoint[1] - center[1])**2)
        self.center = center
        self.startHeight = startPoint[2]
        self.endHeight = 2*center[2]-startPoint[2]
        if startPoint[0] == center[0]:
            if center[1]>startPoint[1]:
                self.angle = math.pi/2
            else:
                self.angle = -math.pi/2
        else:
            x = (startPoint[1] - center[1])/(startPoint[0] - center[0])
            self.angle = math.atan(x)
            if startPoint[1] - center[1]<0:
                self.angle = self.angle+math.pi
            if startPoint[1] == center[1] and startPoint[0] < center[0]:
                self.angle = self.angle+math.pi

            
    
    def getLinePoints(self, theta):
        range = np.linspace(self.angle,self.angle-theta,self.interp+1)
        a = self.center[0] + np.cos(range)*self.radius
        b = self.center[1] + np.sin(range)*self.radius
        c = np.linspace(self.startHeight,self.endHeight,self.interp+1)
        return np.stack((a,b,c)).T

def time2intp(t):
    return int(50*t)

def Bezier_up(startPos,endPos,time):
    midPos=[startPos[0],endPos[1],startPos[2]]
    BezierPoints=np.array([
        startPos,
        midPos,
        endPos
        ])
    return Bezier(BezierPoints,time2intp(time)).getBezierPoints(0)

def Stop(startPos, time):
    sp = np.array([startPos])
    return np.repeat(sp,time2intp(time),axis=0)

t=1 # decrease the time to speed up 50 interpolation per second
def take_off(p1,p2,p3):
    startHeight=0.5
    startY=2
    # for the rightmost
    p3.append(Bezier_up(startPos=[3,startY,startHeight], endPos=[3.4,3.4,2],time=5*t))
    # for the middle
    p2_start = [2,startY,startHeight]
    p2.append(Stop(p2_start,t))
    p2.append(Bezier_up(startPos=p2_start, endPos=[3.4,3.4,2],time=5*t))
    # for the leftmost
    p1_start = [1,startY,startHeight]
    p1.append(Stop(p1_start,2*t))
    p1.append(Bezier_up(startPos=p1_start, endPos=[3.2,3.2,2],time=5*t))
    return p1, p2, p3

def circle(p1,p2,p3):
    # for the first
    p3_start=p3[-1][-1]
    c3=Circle(startPoint=p3_start,center=[2,2,p3_start[2]],time=22*t)
    p3.append(c3.getLinePoints(7*math.pi))
    # for the second
    p2_start=p2[-1][-1]
    c2=Circle(startPoint=p2_start,center=[2,2,p2_start[2]-0.1],time=8*t)
    p2.append(c2.getLinePoints(2*math.pi))
    p2_start=p2[-1][-1]
    c2=Circle(startPoint=p2_start,center=[2,2,p2_start[2]+0.1],time=8*t)
    p2.append(c2.getLinePoints(2*math.pi)[1:])
    p2_start=p2[-1][-1]
    c2=Circle(startPoint=p2_start,center=[2,2,p2_start[2]],time=4*t)
    p2.append(c2.getLinePoints(math.pi)[1:])
    # for the third
    p1_start=p1[-1][-1]
    c1=Circle(startPoint=p1_start,center=[2,2,p1_start[2]],time=16*t)
    p1.append(c1.getLinePoints(4*math.pi))
    return p1, p2, p3

def fall_down(p1,p2,p3):
    # for the third
    p3_start = p3[-1][-1]
    p3_end=[0.45,2,1]
    p3.append(Bezier_up(startPos=p3_start, endPos=p3_end,time=3*t))
    # for the second
    p2_start = p2[-1][-1]
    p2_end = [0,2,1]
    p2.append(Bezier_up(startPos=p2_start, endPos=p2_end,time=3*t))
    p2.append(Stop(p2_end,0.5*t))
    # for the first one
    p1_start = p1[-1][-1]
    p1_end = [0.9,2,1]
    p1.append(Bezier_up(startPos=p1_start, endPos= p1_end,time=3*t))
    p1.append(Stop(p1_end,3*t))
    return p1, p2, p3

def roundabout(p1,p2,p3):
    # for the first
    p3_start=p3[-1][-1]
    c3=Circle(startPoint=p3_start,center=[2,2,p3_start[2]],time=16*t)
    p3.append(c3.getLinePoints(3.5*math.pi))
    # for the second
    p2_start=p2[-1][-1]
    c2=Circle(startPoint=p2_start,center=[2,2,p2_start[2]+0.1],time=4*t)
    p2.append(c2.getLinePoints(math.pi))
    p2_start=p2[-1][-1]
    c2=Circle(startPoint=p2_start,center=[2,2,p2_start[2]-0.1],time=4*t)
    p2.append(c2.getLinePoints(math.pi)[1:])
    p2_start=p2[-1][-1]
    c2=Circle(startPoint=p2_start,center=[2,2,p2_start[2]],time=8*t)
    p2.append(c2.getLinePoints(2*math.pi)[1:])
    p2.append(Stop(p2_start,0.5*t))
    # for the third
    p1_start=p1[-1][-1]
    c1=Circle(startPoint=p1_start,center=[2,2,p1_start[2]],time=16*t)
    p1.append(c1.getLinePoints(4*math.pi))
    p1.append(Stop(p1_start,1*t))
    return p1, p2, p3 
      
if __name__ == '__main__':
    fig=plt.figure()
    ax = fig.gca(projection='3d')
    p1=[];p2=[];p3=[]
    p1,p2,p3=take_off(p1,p2,p3)
    p1,p2,p3=circle(p1,p2,p3)
    p1,p2,p3=fall_down(p1,p2,p3)
    p1,p2,p3=roundabout(p1,p2,p3)
    p1 = np.concatenate(p1,axis=0)
    p2 = np.concatenate(p2,axis=0)
    p3 = np.concatenate(p3,axis=0)
    for p in [p1,p2,p3]:
        p[:,0:2]=(p[:,0:2]-2)/1.5
        print(np.max(p1,axis=0))
        
    ax.set(xlim3d=(-1.3, 1.3), xlabel='X')
    ax.set(ylim3d=(-1.3, 1.3), ylabel='Y')
    ax.set(zlim3d=(0, 2), zlabel='Z')
    ax.plot3D(p1[:,0],p1[:,1],p1[:,2],color='r')
    ax.plot3D(p2[:,0],p2[:,1],p2[:,2],color='k')
    ax.plot3D(p3[:,0],p3[:,1],p3[:,2],color='b')

    np.save("p1.npy",p1)
    np.save("p2.npy",p2)
    np.save("p3.npy",p3)
    

    print(get_dist(p1,p2))
    print(get_dist(p1,p3))
    print(get_dist(p3,p2))


    plt.show() 

