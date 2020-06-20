import numpy as np
from PIL import Image

class camera:
    """
    メモ:ベクトルは平行移動に対して不変なので、camposによるray方向の変化はない(起点はかわる)
    が、回転に関しては不変で無いので、回転が起きた場合は回転を考慮しなくてはならない。
    """
    def __init__(self):
        self.pos = np.array([0,0,0])
        self.lookAt = np.array([0,0,1])
        self.up = np.array([0,1,0])
        self.vAngle = 60*np.pi/180
        self.vAHalf = self.vAngle/2
        self.vScreenZ=0.5/np.tan(self.vAHalf)

    def ray(self,x,y):
        base = np.array([x,y,self.vScreenZ])
        return base/np.linalg.norm(base)

class object:
    def dist(self,x):
        pass

    def norm(self,x):
        eps = 0.0001
        buf = []
        arr = np.array([[eps,0,0],[0,eps,0],[0,0,eps]])
        for i in range(3):
            buf.append(self.dist(x+arr[i])-self.dist(x-arr[i]))
        buf = np.array(buf)
        return buf/np.linalg.norm(buf)
    

class sphere(object):
    def __init__(self,r,center):
        self.r = r
        self.cent = center
    def dist(self,x):
        return np.sqrt(np.dot(x-self.cent,x-self.cent))-self.r
    def norm(self,x):
        return super().norm(x)


pix = np.array([[[0,0,0] for i in range(128)] for j in range(128)]).astype(np.uint8)
step = 1/128
cam = camera()
sp = sphere(1,np.array([0,0,6]))
light = np.array([1,1,1])/np.linalg.norm([1,1,1])
for i in range(128):
    for j in range(128):
        tx,ty = step*i-0.5,step*j-0.5
        ray = cam.ray(tx,ty)

        totdis = 0
        maxdis = 10
        mindis = 0.01
        hitflag = False
        while totdis < maxdis:
            rayhead = cam.pos+totdis*ray
            d = sp.dist(rayhead)
            if abs(d) < mindis:
                hitflag = True
                break
            else:
                totdis += d
        if hitflag:
            bright = np.clip(np.dot(light,-sp.norm(cam.pos+totdis*ray)),0.1,0.9)
            #法線は面に垂直外に向いてるので、マイナスかけてやらないといい感じにならない。
            pix[i][j][0] = 255*bright
            pix[i][j][1] = 255*bright
            pix[i][j][2] = 255*bright

img = Image.fromarray(pix)
img.save("./res.png")