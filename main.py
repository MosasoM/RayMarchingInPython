import numpy as np
from PIL import Image
from multiprocessing import Pool
import itertools

"""
Lambert
Lr=ρd×(L⋅N)×id
Lambert反射はどの方向から見ても明るさが同じ(乱反射っぽい)
Lr:求めたい表面から発される日価値 pd:Lambert反射率 L:ライトベクトル N:法線ベクトル id:光の強さ

Phing
Lr=kaia+ρd(L⋅N)id+ρs(R⋅V)^α is
kaia:環境光、

Disney Diffuse
fd=σ/π(1+(FD90−1)(1−cosθl)^5)(1+(FD90−1)(1−cosθv)^5)
ここでFD90=0.5+2 roughness cos^2(θd)
θl:ライトと法線の角度? θv:視線と法線の角度？ θd:ハーフベクトルとライトベクトルの角度
ハーフベクトルはライトと視線を足して正規化したもの

fr,s=DGF/(4(n⋅l)(n⋅v))
D,G,Fはなんかややこしい項。nは法線、lはライトベクトル、vは視線ベクトル

D(h)=α^2/(π((n⋅h)2(α^2−1)+1)^2)
hはハーフベクトル。アルファはroughnessの二乗

G = 4(n・l)(n・v)V
V = 0.5/(Al+Av)

Al = n・v sqrt((n・l)^2・(1-α^2)+α^2)
Av = n・l sqrt((n・v)^2・(1-α^2)+α^2)

F(h)=F0+(1−F0)(1−(l⋅h))^5
F0は鏡面反射率

fr=fr,d(1−F)+fr,s
でdiffuseとspecularを合成する。Fはフレネル項

point rightとかは1/r^2での減衰がかかりうる


"""

class Camera:
    """
    メモ:ベクトルは平行移動に対して不変なので、camposによるray方向の変化はない(起点はかわる)
    が、回転に関しては不変で無いので、回転が起きた場合は回転を考慮しなくてはならない。
    """
    def __init__(self):
        self.pos = np.array([0,0,0])
        self.lookAt = np.array([0,0,1])
        self.up = np.array([0,1,0])
        self.vAngle = 30*np.pi/180
        self.vAHalf = self.vAngle/2
        self.vScreenZ=1/np.tan(self.vAHalf)

    def ray(self,x,y):
        base = np.array([x,y,self.vScreenZ])
        return base/np.linalg.norm(base)

class BRDF:
    def __init__(self,basecolor,f0,roughness):
        self.basecolor = basecolor/np.linalg.norm(basecolor)
        self.f0 = f0
        self.alpha = roughness ** 2
    def diffuse(self):
        return self.basecolor/np.pi
    def D(self,n,h):
        return np.clip(self.alpha**2/(np.pi* (np.dot(n,h)**2 * (self.alpha**2-1) + 1)**2),0,1000)
    def V(self,n,v,l):
        al = np.dot(n,v)*np.sqrt(np.dot(n,l)**2 * (1-self.alpha**2) +  self.alpha**2)
        av = np.dot(n,l)*np.sqrt(np.dot(n,v)**2 * (1-self.alpha**2) +  self.alpha**2)
        return 0.5/(al+av)
    def F(self,l,h):
        return self.f0 + (1-self.f0)*((1-np.dot(l,h))**5)
    
    def brdf(self,n,v,l):
        h = (v+l)/np.linalg.norm(v+l)
        f = self.F(l,h)
        diff = self.diffuse()
        d = self.D(n,h)
        v = self.V(n,v,l)
        return (1-f)*diff + d*v*f

class DiffuseMaterial:
    def __init__(self,basecolor):
        self.basecolor = basecolor/np.linalg.norm(basecolor)
    def brdf(self,n,v,l):
        return self.basecolor/np.pi

class Object:
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

    def mat_vecs(self,x,cam_pos,lay_vec):
        n = self.norm(x)
        v = (cam_pos-x)/np.linalg.norm(cam_pos-x)
        l = lay_vec
        return n,v,l

    def set_material(self,mat):
        self.mat = mat

class DirectionalLight:
    def __init__(self,direction,power):
        self.direction = direction/np.linalg.norm(direction)
        self.power = power
    def lighting(self,x):
        return -self.direction,self.power #光の方向として指定するが、light vectorとしてはマイナスになる。

class Sphere(Object):
    def __init__(self,r,center):
        self.r = r
        self.cent = center
    def dist(self,x):
        return np.sqrt(np.dot(x-self.cent,x-self.cent))-self.r
    def norm(self,x):
        return super().norm(x)

class Scene:
    def __init__(self,resolution=256):
        obj1 = Sphere(2,[0,0,20])
        # obj1.set_material(DiffuseMaterial(np.array([255,0,0])))
        obj1.set_material(BRDF(np.array([255,0,0]),np.array([0.6,0.6,0.6]),0.1))
        self.objs = [obj1]
        self.lights = [DirectionalLight(np.array([1,-1,1]),np.array([10,10,10]))]
        self.img = np.array([[[0,0,0] for i in range(resolution)] for j in range(resolution)]).astype(np.uint8)
        self.step = 2/resolution
        self.maxdis = 10
        self.mindis = 0.01
        self.resolutuon = resolution
        self.cam = None
        self.arg_ij = itertools.product(range(self.resolutuon),range(self.resolutuon))
    
    def render(self,camera):
        self.cam = camera
        with Pool(8) as p:
            ret = p.starmap(self.marching, self.arg_ij)
        with Pool(8) as p:
            ret2 = p.starmap(self.color,ret)
        for r in ret2:
            self.img[r[0]][r[1]][0] = r[2][0]*255
            self.img[r[0]][r[1]][1] = r[2][1]*255
            self.img[r[0]][r[1]][2] = r[2][2]*255
        tmp = Image.fromarray(self.img)
        tmp.save("./res.png")

    def color(self,i,j,hitflag,p,x):
        if hitflag == -1:
            return [i,j,np.array([0,0,0])]
        else:
            obj = self.objs[hitflag]
            buf = np.array([0,0,0])
            for light in self.lights:
                lv,lpow = light.lighting(x)
                n,v,l = obj.mat_vecs(x,p,lv)
                brdf = obj.mat.brdf(n,v,l)
                buf = buf + np.clip(np.clip(np.dot(l,n)+0.1,0,1)*lpow*brdf,0,1)
            return [i,j,buf]


    def marching(self,i,j):
        tx,ty = self.step*j-1,-self.step*i+1
        ray = self.cam.ray(tx,ty)
        totdis = 0
        maxdis = 30
        mindis = 0.0001
        hitflag = -1
        p = self.cam.pos
        while totdis < maxdis:
            rayhead = p+totdis*ray
            ds = [o.dist(rayhead) for o in self.objs]
            near_ind = np.argmin(ds)
            d = ds[near_ind]
            if abs(d) < mindis:
                hitflag = near_ind
                break
            else:
                totdis += d
        return [i,j,hitflag,p,p+totdis*ray]

sc = Scene()
cam = Camera()
sc.render(cam)
