from json.tool import main
import matplotlib.pyplot as plt
import numpy as np
import math

def get_dist(a,b):
    return np.sqrt(np.min(np.sum(np.power(a-b,2),axis=1)))

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

class Line:
    def __init__(self,Points,InterpolationNum):
        self.demension=Points.shape[1]    # 点的维数
        self.segmentNum=InterpolationNum-1 # 段数
        self.num=InterpolationNum         # 单段插补(点)数
        self.pointsNum=Points.shape[0]   # 点的个数
        self.Points=Points                # 所有点信息
        
    def getLinePoints(self):
        # 每一段的插补点
        pis=np.array(self.Points[0])
        # i是当前段
        for i in range(0,self.pointsNum-1):
            sp=self.Points[i]
            ep=self.Points[i+1]
            dp=(ep-sp)/(self.segmentNum)# 当前段每个维度最小位移
            for i in range(1,self.num):
                pi=sp+i*dp
                pis=np.vstack((pis,pi))         
        return pis

class Circle:
    def __init__(self,startPoint, center, InterpolationNum):
        self.interp=InterpolationNum
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
    
    def getLinePoints(self, theta):
        range = np.linspace(self.angle,self.angle-theta,self.interp+1)
        a = self.center[0] + np.cos(range)*self.radius
        b = self.center[1] + np.sin(range)*self.radius
        c = np.linspace(self.startHeight,self.endHeight,self.interp+1)
        return np.stack((a,b,c)).T

def generate_path(obj):
        pc = obj.c1.getLinePoints(-math.pi)
        pc2 = obj.c2.getLinePoints(math.pi)
        pc3 = obj.c3.getLinePoints(2*math.pi)
        pc4 = obj.c4.getLinePoints(1.5*math.pi)
        
        res = np.concatenate((pc,pc2,pc3,pc4))
        return res

class p1_confg():
    def __init__(self):
        self.c1 = Circle([1,3,1],[1.7,3.7,1.2],1000)
        self.c2 = Circle([2.4,4.4,1.4],[2.6,4.6,1.6],1000)
        self.c3 = Circle([2.8,4.8,1.8],[2.2,4.2,1.4],1000)
        self.c4 = Circle([2.8,4.8,1],[2.2,4.2,1.4],1000)
       
class p2_confg():
    def __init__(self):
        self.c1 = Circle([3,3,1],[3.5,3.5,1.5],1000)
        self.c2 = Circle([4,4,2],[4.5,4.5,2.5],1000)
        self.c3 = Circle([5,5,3],[3.5,3.5,2.7],1000)
        self.c4 = Circle([5,5,2.4],[3.5,3.5,2.7],1000)
    
class p3_confg():
    def __init__(self):
        self.c1 = Circle([5,3,1],[5.2,3.2,1.3],1000)
        self.c2 = Circle([5.4,3.4,1.6],[5.6,3.6,2],1000)
        self.c3 = Circle([5.8,3.8,2.4],[5,3,1.8],1000)
        self.c4 = Circle([5.8,3.8,1.2],[5,3,1.8],1000)

      
if __name__ == '__main__':
    fig=plt.figure()
    ax = fig.gca(projection='3d')

    c1 = p1_confg()
    p1 = generate_path(c1)
    # np.save("p1.npy",p1)

    c2 = p2_confg()
    p2 =  generate_path(c2)
    # np.save("p2.npy",p2)

    c3 = p3_confg()
    p3 = generate_path(c3)
    # np.save("p3.npy",p3)

    for p in [p1,p2,p3]:
        p[:,0:2]=(p[:,0:2]-3)/4*1.3
        p[:,-1]=p[:,-1]/1.5

    ax.set(xlim3d=(-1.3, 1.3), xlabel='X')
    ax.set(ylim3d=(-1.3, 1.3), ylabel='Y')
    ax.set(zlim3d=(0, 2), zlabel='Z')
    ax.plot3D(p1[:,0],p1[:,1],p1[:,2],color='r')
    ax.plot3D(p2[:,0],p2[:,1],p2[:,2],color='k')
    ax.plot3D(p3[:,0],p3[:,1],p3[:,2],color='b')

    print(get_dist(p1,p2))
    print(get_dist(p1,p3))
    print(get_dist(p3,p2))


    plt.show()

