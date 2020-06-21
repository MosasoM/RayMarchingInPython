import numpy as np
from PIL import Image
from multiprocessing import Pool
import itertools

"""
フォーカルブラー
ぐろー
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
        self.basecolor = basecolor
        self.f0 = f0
        self.alpha = roughness ** 2
        self.glow = False
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

class GlowMaterial:
    def __init__(self,glowcolor,alpha):
        self.glowcolor = glowcolor
        self.alpha = alpha
        self.glow = True

class DiffuseMaterial:
    def __init__(self,basecolor):
        self.basecolor = basecolor/np.linalg.norm(basecolor)
    def brdf(self,n,v,l):
        return self.basecolor/np.pi

class BasicMaterial:
    def __init__(self,metallic,basecolor,reflection,alpha):
        self.basecolor = basecolor
        self.reflection = reflection
        self.alpha = alpha
        self.metal = metallic
    def brdf(self,n,v,l):
        diff = self.basecolor*(1-self.metal)/np.pi
        spec = self.metal*self.reflection*(np.dot(n,l)**self.alpha)*(self.alpha+1)/(2*np.pi)
        return diff+spec
    def diff(self):
        return self.basecolor*(1-self.metal)/np.pi

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
    def __init__(self,r,pos):
        self.r = r
        self.pos = pos
    def dist(self,x):
        return np.sqrt(np.dot(x-self.pos,x-self.pos))-self.r
    def norm(self,x):
        return super().norm(x)

class Box(Object):
    def __init__(self,halfsize,pos):
        self.size = halfsize
        self.pos = pos
    def dist(self,x):
        p = np.abs(x-self.pos)
        hoge = np.clip(p-self.size,0,100000000)
        return np.sqrt(np.dot(hoge,hoge))


class Scene:
    def __init__(self,resolution=256):
        obj1 = Sphere(1.5,[0,0,20])
        obj2 = Box(np.array([5,5,5]),np.array([0,0,40]))
        # obj1.set_material(DiffuseMaterial(np.array([255,0,0])))
        obj2.set_material(BRDF(np.array([1,0,0]),np.array([0.6,0.6,0.6]),0.1))
        # obj2.set_material(BRDF(np.array([1,1,0]),np.array([0.02,0.02,0.02]),0.1))
        obj1.set_material(GlowMaterial(np.array([0,1,1]),2))
        # obj1.set_material(BasicMaterial(1,np.array([1,0,0]),np.array([1,1,1]),30))
        self.objs = [obj1,obj2]
        self.lights = [DirectionalLight(np.array([1,-1,1]),np.array([1,1,1]))]
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
            ret2 = p.starmap(self.hitcolor,ret)
        for r in ret2:
            self.img[r[0]][r[1]][0] += np.clip(r[2][0],0,1)*255
            self.img[r[0]][r[1]][1] += np.clip(r[2][1],0,1)*255
            self.img[r[0]][r[1]][2] += np.clip(r[2][2],0,1)*255

        with Pool(8) as p:
            ret3 = p.starmap(self.glowcolor,ret)
        for r in ret3:
            self.img[r[0]][r[1]][0] += np.clip(r[2][0],0,1)*255
            self.img[r[0]][r[1]][1] += np.clip(r[2][1],0,1)*255
            self.img[r[0]][r[1]][2] += np.clip(r[2][2],0,1)*255

        np.clip(self.img,0,255)
        tmp = Image.fromarray(self.img)
        tmp.save("./res.png")

    def hitcolor(self,i,j,hitflag,p,x,shortest2objs):
        if hitflag == -1:
            return [i,j,np.array([0,0,0])]
        else:
            obj = self.objs[hitflag]
            if not obj.mat.glow:
                buf = np.array([0,0,0])
                for light in self.lights:
                    lv,lpow = light.lighting(x)
                    n,v,l = obj.mat_vecs(x,p,lv)
                    brdf = obj.mat.brdf(n,v,l)
                    lcos = np.clip(np.dot(l,n),0,1)
                    buf = buf + np.clip(lcos*lpow*brdf,0,1)
                buf = buf+obj.mat.diffuse()*np.array([0.4,0.4,0.4])
                return [i,j,buf]
            else:
                return [i,j,obj.mat.glowcolor]

    def glowcolor(self,i,j,hitflag,p,x,shortest2objs):
        if hitflag != -1:
            hoge = self.cam.pos-self.objs[hitflag].pos
            obstdist = np.sqrt(np.dot(hoge,hoge))
        else:
            obstdist = 1e9

        buf = np.array([0,0,0])
        for ind in range(len(self.objs)):
            if self.objs[ind].mat.glow:
                hoge = self.cam.pos-self.objs[ind].pos
                gdist = np.sqrt(np.dot(hoge,hoge))
                if gdist < obstdist:
                    buf = buf + self.objs[ind].mat.glowcolor*((shortest2objs[ind]+1.5)**(-self.objs[ind].mat.alpha))
        return [i,j,buf]


            


    def marching(self,i,j):
        tx,ty = self.step*j-1,-self.step*i+1
        ray = self.cam.ray(tx,ty)
        totdis = 0
        maxdis = 100
        mindis = 0.001
        hitflag = -1
        p = self.cam.pos
        numobj = len(self.objs)
        shortest2objs = [1e9 for hoge in range(numobj)]
        while totdis < maxdis:
            rayhead = p+totdis*ray
            ds = [o.dist(rayhead) for o in self.objs]
            for hoge in range(numobj):
                shortest2objs[hoge] = min(shortest2objs[hoge],ds[hoge])
            near_ind = np.argmin(ds)
            d = ds[near_ind]
            if abs(d) < mindis:
                hitflag = near_ind
                break
            else:
                totdis += d
        return [i,j,hitflag,p,p+totdis*ray,shortest2objs]

sc = Scene()
cam = Camera()
sc.render(cam)
