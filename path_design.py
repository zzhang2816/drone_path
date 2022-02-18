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
        assert startPoint[0] != center[0]
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
    
    def getLinePoints(self):
        range = np.linspace(self.angle,self.angle-1.5*math.pi,self.interp+1)
        a = self.center[0] + np.cos(range)*self.radius
        b = self.center[1] + np.sin(range)*self.radius
        c = np.linspace(self.startHeight,self.endHeight,self.interp+1)
        return np.stack((a,b,c)).T

def generate_path(BezierPoints,LinePoints,start_point,center):
    # 垂直起飞
    groud_pos = [BezierPoints[0][0],BezierPoints[0][1],0]
    vt = np.linspace(groud_pos, BezierPoints[0],100)
    # 贝塞尔曲线连接控制点
    bz=Bezier(BezierPoints,1000)
    matpi=bz.getBezierPoints(0)
    # 直线连接控制点
    l=Line(LinePoints,1000)
    pl=l.getLinePoints()
    # 圆周
    c = Circle(start_point,center,1000)
    pc =c.getLinePoints()
    # 直线
    end_point=(pc[-1][0],0,pc[-1][2]-1.5)
    ht = np.linspace(pc[-1], end_point,500)
    res = np.concatenate((vt,matpi,pl,pc,ht))
    return res

class p1_confg():
    def __init__(self):
        st=[4,5,4]
        ed=[5,4,3.3]
        self.BezierPoints=np.array([
            [1,3,1],
            [1.5,1,1],
            [4,2,1],
            [4,3,2],
            [2,3,4],
            st,
            ])

        self.LinePoints=np.array([
            st,
            [5,4,3.3],
            [0,2.5,2],
            [2.5,5,3.3],
            ed
            ])
        self.start_point = ed
        self.center = [2.5,3,3.4]

class p2_confg():
    def __init__(self):
        st=[6,6,3.8]
        ed=[6,4,3.6]
        self.BezierPoints=np.array([
            [3,3,1],
            [4,2,1],
            [4,3,2],
            [2,3,4],
            st
            ])

        self.LinePoints=np.array([
            st,
            [6,4,3.6],
            [1,3.5,2.5],
            [4,4,3.6],
            ed])
        self.start_point = ed
        self.center = [4,3,3.8]

class p3_confg():
    def __init__(self):
        st=[4,6,3.3]
        ed=[4,1.5,2.5]
        self.BezierPoints=np.array([
            [5,3,0.5],
            [3,3.5,1.5],
            [2,3.5,3.3],
            st,
            ])
        
        self.LinePoints=np.array([
            st,
            [4,1.5,2.5],
            [1,3.5,1.5],
            [4,4,2.5],
            ed,
            ])
        self.start_point = ed
        self.center = [2,2,2.7]

if __name__ == '__main__':
    fig=plt.figure()
    ax = fig.gca(projection='3d')

    c1 = p1_confg()
    p1 = generate_path(c1.BezierPoints,c1.LinePoints,c1.start_point,c1.center)
    np.save("p1.npy",p1)
    ax.plot3D(p1[:,0],p1[:,1],p1[:,2],color='r')

    c2 = p2_confg()
    p2 = generate_path(c2.BezierPoints,c2.LinePoints,c2.start_point,c2.center)
    np.save("p2.npy",p2)
    ax.plot3D(p2[:,0],p2[:,1],p2[:,2],color='k')

    c3 = p3_confg()
    p3 = generate_path(c3.BezierPoints,c3.LinePoints,c3.start_point,c3.center)
    np.save("p3.npy",p3)
    ax.plot3D(p3[:,0],p3[:,1],p3[:,2],color='b')

    print(get_dist(p1,p2))
    print(get_dist(p1,p3))
    print(get_dist(p3,p2))

    plt.show()

