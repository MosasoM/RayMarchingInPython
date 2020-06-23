import numpy as np
from PIL import Image
from multiprocessing import Pool
import itertools

"""
フォーカルブラー
ぐろー
右手系になりました！！！！！
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
        self.vScreenY=1/np.tan(self.vAHalf)

    def ray(self,x,z):
        base = np.array([x,self.vScreenY,z])
        return base/np.linalg.norm(base)

class BRDF:
    def __init__(self,basecolor,f0,roughness):
        self.basecolor = basecolor
        self.f0 = f0
        self.alpha = roughness ** 2
        self.glow = False
    def diffuse(self):
        return np.clip(self.basecolor/np.pi,0,1)
    def D(self,n,h):
        return np.clip(self.alpha**2/(np.pi* (np.dot(n,h)**2 * (self.alpha**2-1) + 1)**2),0,1)
    def V(self,n,v,l):
        al = np.dot(n,v)*np.sqrt(np.dot(n,l)**2 * (1-self.alpha**2) +  self.alpha**2)
        av = np.dot(n,l)*np.sqrt(np.dot(n,v)**2 * (1-self.alpha**2) +  self.alpha**2)
        return np.clip(0.5/(al+av),0,1)
    def F(self,l,h):
        return np.clip(self.f0 + (1-self.f0)*((1-np.dot(l,h))**5),0,1)
    
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

    def mat_vecs(self,x,cam_pos):
        n = self.norm(x)
        v = (cam_pos-x)/np.linalg.norm(cam_pos-x)
        return n,v

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
        # obj1 = Sphere(1.5,[0,0,20])
        obj1 = Box(np.array([2,2,2]),np.array([0,20,0]))
        obj2 = Box(np.array([5,20,1]),np.array([0,20,-3]))
        # obj1.set_material(DiffuseMaterial(np.array([255,0,0])))
        # obj2.set_material(BRDF(np.array([1,0,0]),np.array([0.6,0.6,0.6]),0.1))
        obj2.set_material(BRDF(np.array([0.9,0.9,0.9]),np.array([0.7,0.7,0.7]),0.1))
        # obj1.set_material(BRDF(np.array([0,1,0]),np.array([0.7,0.7,0.7]),0.1))
        obj1.set_material(BRDF(np.array([230/255,180/255,34/255]),np.array([0.6,0.6,0.6]),0.1))
        # obj2.set_material(BRDF(np.array([1,1,0]),np.array([0.02,0.02,0.02]),0.1))
        # obj1.set_material(GlowMaterial(np.array([0,1,1]),2))
        # obj1.set_material(BasicMaterial(1,np.array([1,0,0]),np.array([1,1,1]),30))
        self.objs = [obj1,obj2]
        self.lights = [DirectionalLight(np.array([1,1,-1]),np.array([5,5,5]))]
        self.img = np.array([[[0,0,0] for i in range(resolution)] for j in range(resolution)]).astype(np.uint8)
        self.step = 2/resolution
        self.maxdis = 100
        self.mindis = 0.001
        self.resolutuon = resolution
        self.reflection_max = 1
        self.cam = None
        self.arg_ij = itertools.product(range(self.resolutuon),range(self.resolutuon))
    
    def render(self,camera):
        self.cam = camera

        with Pool(8) as p:
            ret3 = p.starmap(self.cast_ray,self.arg_ij)
        # print(ret3)
        for r in ret3:
            self.img[r[0]][r[1]][0] += np.clip(r[2][0],0,1)*255
            self.img[r[0]][r[1]][1] += np.clip(r[2][1],0,1)*255
            self.img[r[0]][r[1]][2] += np.clip(r[2][2],0,1)*255

        np.clip(self.img,0,255)
        tmp = Image.fromarray(self.img)
        tmp.save("./res.png")

    def ray_marching(self,origin,ray):
        totdis = 0
        hitflag = -1
        numobj = len(self.objs)
        shortest2objs = [1e9 for hoge in range(numobj)]

        while totdis < self.maxdis:
            rayhead = origin + ray*totdis
            ds = [o.dist(rayhead) for o in self.objs]
            for hoge in range(numobj):
                shortest2objs[hoge] = min(shortest2objs[hoge],ds[hoge])
            near_ind = np.argmin(ds)
            d = ds[near_ind]
            if abs(d) < self.mindis:
                hitflag = near_ind
                break
            else:
                totdis += d
        return hitflag,totdis,shortest2objs
    
    def is_hit(self,origin,ray):
        totdis = 0
        hitflag = -1

        while totdis < self.maxdis:
            rayhead = origin + ray*totdis
            ds = [o.dist(rayhead) for o in self.objs]
            near_ind = np.argmin(ds)
            d = ds[near_ind]
            if abs(d) < self.mindis:
                hitflag = near_ind
                break
            else:
                totdis += d
        return hitflag

    def cast_ray(self,i,j):
        
        tx,tz = self.step*j-1,-self.step*i+1
        ray = self.cam.ray(tx,tz)
        origin = self.cam.pos
        totcol = self.color(1,self.reflection_max,origin,ray)

        return [i,j,totcol]

    def color(self,rnum,rmax,origin,ray):
        hitflag,totdis,shortest2objs = self.ray_marching(origin,ray)
        totcol = np.array([0,0,0])
        if hitflag == -1:
            return totcol
        else:
            hitpos = origin + ray*totdis
            n,v = self.objs[hitflag].mat_vecs(hitpos,origin)
            hitpos = hitpos + n*0.02
            for light in self.lights:
                lv,lpow = light.lighting(hitpos)
                totcol = totcol + self.hcolor(hitflag,hitpos,n,v,lv,lpow) #BRDF
            totcol = totcol + self.gcolor(hitflag,origin,shortest2objs) #GLOW

            if rnum < rmax:
                refv = self.reflect(v,n)
                refcol = self.color(rnum+1,rmax,hitpos,refv)
                totcol = totcol + refcol
            
            return totcol
            
        
    
    def reflect(self,v,n):
        # vは視線ベクトルなど物体から視点方向に生える向き
        return -v+2*np.dot(v,n)*n/np.linalg.norm(-v+2*np.dot(v,n)*n)

    def hcolor(self,hitflag,hitpos,n,v,l,lpow):
        is_shadow = self.is_shadow(hitpos,l)
        if is_shadow:
            lp = lpow*0.2
        else:
            lp = lpow*np.clip(np.dot(l,n),0.2,1)
        tmp = np.array([0,0,0])
        if hitflag != -1:
            obj = self.objs[hitflag]
            if not obj.mat.glow:
                brdf = np.clip(obj.mat.brdf(n,v,l),0,1)
                tmp = tmp + np.clip(lp*brdf,0,1)
            else:
                tmp = tmp + obj.mat.glowcolor
        return tmp

    def gcolor(self,hitflag,origin,shortest2objs):
        tmp = np.array([0,0,0])
        if hitflag != -1:
            hoge = origin-self.objs[hitflag].pos
            obstdist = np.sqrt(np.dot(hoge,hoge))
        else:
            obstdist = 1e9

        for ind in range(len(self.objs)):
            if self.objs[ind].mat.glow:
                hoge = origin-self.objs[ind].pos
                gdist = np.sqrt(np.dot(hoge,hoge))
                if gdist < obstdist:
                    tmp = tmp + self.objs[ind].mat.glowcolor*((shortest2objs[ind]+1.5)**(-self.objs[ind].mat.alpha))

        return tmp
    
    def is_shadow(self,origin,l):
        hitflag = self.is_hit(origin,l)
        if hitflag == -1:
            return False
        else:
            return True



sc = Scene()
cam = Camera()
sc.render(cam)
