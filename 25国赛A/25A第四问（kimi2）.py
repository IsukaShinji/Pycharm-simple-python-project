# -*- coding: utf-8 -*-
"""
2025 A-prob4 极速控制台版
FY1固定（反向静止即刻爆炸），只优化FY2/FY3
运行：python this.py
"""
import numpy as np, math, time
from random import uniform
from concurrent.futures import ThreadPoolExecutor, as_completed

# ===================== 用户可调参数 =====================
N_THREAD   = 32          # 线程
PARTICLE   = 320         # PSO粒子数
ITER       = 2000       # PSO迭代数
DT_CHECK   = 0.01      # 时间步长（s）
PRINT_STEP = 20         # 进度打印间隔
# =======================================================
# -------------------- 固定常量（赛题） --------------------
CYL_C  = np.array([0., 200., 0.])
CYL_R, CYL_H = 7.0, 10.0
M1_POS0 = np.array([20000., 0., 2000.])
M1_VEL  = 300. * (-M1_POS0) / np.linalg.norm(M1_POS0)
M1_T_MAX = np.linalg.norm(M1_POS0) / 300.
# 将 UAV_POS 转换为 NumPy 数组
UAV_POS = np.array([
    [17800., 0., 1800.],    # FY1
    [12000., 1400., 1400.], # FY2
    [6000., -3000., 700.]   # FY3
])
V_BOUNDS = (70., 140.)
TH_BOUNDS = (0., 360.)
SMOKE_R, SMK_LIFE, SMK_SINK, G = 10.0, 20.0, 3.0, 9.8
# FY1 固定（反向静止即刻爆炸）
FY1_TH, FY1_V, FY1_T1, FY1_DT = 180.0, 0.0, 0.0, 0.0
# -------------------- Numba 核心 --------------------
try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False; njit = lambda f: f

@njit(fastmath=True, cache=True)
def cyl_samples(c, r, h, n_z, n_a):
    pts = []
    for z in np.linspace(c[2], c[2]+h, n_z):
        for k in range(n_a):
            ang = 2.*math.pi*k/n_a
            pts.append([c[0]+r*math.cos(ang), c[1]+r*math.sin(ang), z])
    return np.array(pts, dtype=np.float64)

CYL_PTS = cyl_samples(CYL_C, CYL_R, CYL_H, 3, 72)
N_CYL   = len(CYL_PTS)

@njit(fastmath=True, cache=True)
def missile_pos(t):
    if t >= M1_T_MAX-1e-8: return math.nan, math.nan, math.nan
    x = M1_POS0[0] + M1_VEL[0]*t
    y = M1_POS0[1] + M1_VEL[1]*t
    z = M1_POS0[2] + M1_VEL[2]*t
    return x, y, z

@njit(fastmath=True, cache=True)
def smoke_c(t, t_det, xe, ye, ze):
    if t < t_det-1e-8 or t > t_det+SMK_LIFE+1e-8: return math.nan, math.nan, math.nan
    return xe, ye, ze - SMK_SINK*(t-t_det)

@njit(fastmath=True, cache=True)
def hit(ax, ay, az, bx, by, bz, ox, oy, oz):
    abx, aby, abz = bx-ax, by-ay, bz-az
    aox, aoy, aoz = ax-ox, ay-oy, az-oz
    seg2 = abx*abx+aby*aby+abz*abz
    if seg2<1e-16: return (aox*aox+aoy*aoy+aoz*aoz)<=SMOKE_R*SMOKE_R+1e-8
    a,b,c = seg2, 2*(aox*abx+aoy*aby+aoz*abz), (aox*aox+aoy*aoy+aoz*aoz)-SMOKE_R*SMOKE_R
    d = b*b-4*a*c
    if d<-1e-8: return False
    sq = math.sqrt(max(d,0.))
    s1,s2 = (-b-sq)/(2*a), (-b+sq)/(2*a)
    return s1<=1.+1e-8 and s2>=-1e-8

@njit(fastmath=True, cache=True)
def fitness(x8, fy1_xyz):
    th2,v2,t12,dt2, th3,v3,t13,dt3 = x8
    for th,v,tr,dt in [(th2,v2,t12,dt2),(th3,v3,t13,dt3)]:
        if not (TH_BOUNDS[0]<=th<=TH_BOUNDS[1] and V_BOUNDS[0]<=v<=V_BOUNDS[1]
                and tr>=-1e-8 and dt>=-1e-8 and tr+dt<=M1_T_MAX-1e-8): return 0.
    tdet2, tdet3 = t12+dt2, t13+dt3
    # FY2/FY3 爆炸中心
    th2,th3 = math.radians(th2), math.radians(th3)
    xe2 = UAV_POS[1, 0] + v2*tdet2*math.cos(th2)
    ye2 = UAV_POS[1, 1] + v2*tdet2*math.sin(th2)
    ze2 = UAV_POS[1, 2] - 0.5*G*dt2*dt2
    xe3 = UAV_POS[2, 0] + v3*tdet3*math.cos(th3)
    ye3 = UAV_POS[2, 1] + v3*tdet3*math.sin(th3)
    ze3 = UAV_POS[2, 2] - 0.5*G*dt3*dt3
    if ze2<-1e-8 or ze3<-1e-8: return 0.
    # 并集扫描
    t_start, t_end = 0.0, min(max(tdet2, tdet3)+SMK_LIFE, M1_T_MAX)
    steps = int((t_end-t_start)/DT_CHECK)+1
    total = 0.0
    for k in range(steps):
        t = t_start + k*DT_CHECK
        mx,my,mz = missile_pos(t)
        if math.isnan(mx): continue
        all_ok = True
        for j in range(N_CYL):
            cx,cy,cz = CYL_PTS[j,0], CYL_PTS[j,1], CYL_PTS[j,2]
            ok = False
            # FY1 始终存在
            sx,sy,sz = fy1_xyz[0], fy1_xyz[1], fy1_xyz[2] - SMK_SINK*t
            if hit(mx,my,mz, cx,cy,cz, sx,sy,sz): ok=True
            # FY2
            if not ok:
                sx,sy,sz = smoke_c(t, tdet2, xe2, ye2, ze2)
                if not math.isnan(sx) and hit(mx,my,mz, cx,cy,cz, sx,sy,sz): ok=True
            # FY3
            if not ok:
                sx,sy,sz = smoke_c(t, tdet3, xe3, ye3, ze3)
                if not math.isnan(sx) and hit(mx,my,mz, cx,cy,cz, sx,sy,sz): ok=True
            if not ok:
                all_ok=False; break
        if all_ok: total += DT_CHECK
    # 晚高峰奖励
    if t_end>=55.: total*=1.05
    return total

# -------------------- PSO --------------------
def init_pop(size):
    pop=np.empty((size,8))
    for i in range(size):
        pop[i,0]=uniform(*TH_BOUNDS); pop[i,1]=uniform(*V_BOUNDS)
        pop[i,2]=uniform(0,M1_T_MAX); max_dt=min(SMK_LIFE,math.sqrt(2*UAV_POS[1, 2]/G),M1_T_MAX-pop[i,2])
        pop[i,3]=uniform(0,max_dt)
        pop[i,4]=uniform(*TH_BOUNDS); pop[i,5]=uniform(*V_BOUNDS)
        pop[i,6]=uniform(0,M1_T_MAX); max_dt=min(SMK_LIFE,math.sqrt(2*UAV_POS[2, 2]/G),M1_T_MAX-pop[i,6])
        pop[i,7]=uniform(0,max_dt)
    return pop

def eval_pop(pop):
    fit=np.empty(len(pop))
    fy1_xyz=(UAV_POS[0, 0], UAV_POS[0, 1], UAV_POS[0, 2]-0.0)  # 固定爆炸中心
    with ThreadPoolExecutor(max_workers=N_THREAD) as ex:
        fut={ex.submit(fitness, p, fy1_xyz): i for i,p in enumerate(pop)}
        for f in as_completed(fut): fit[fut[f]]=f.result()
    return fit

def pso():
    particles=init_pop(PARTICLE)
    velocities=np.zeros_like(particles)
    pbest=particles.copy(); pbest_fit=eval_pop(pbest)
    gbest=pbest[np.argmax(pbest_fit)]; gbest_fit=pbest_fit.max()
    for it in range(1,ITER+1):
        # PSO更新
        r1,r2=np.random.rand(PARTICLE,8),np.random.rand(PARTICLE,8)
        velocities=0.7*velocities+1.5*r1*(pbest-particles)+1.5*r2*(gbest-particles)
        particles+=velocities
        # 边界修正
        for i in range(PARTICLE):
            particles[i,0]=np.clip(particles[i,0],*TH_BOUNDS)
            particles[i,1]=np.clip(particles[i,1],*V_BOUNDS)
            particles[i,2]=np.clip(particles[i,2],0,M1_T_MAX); max_dt=min(SMK_LIFE,math.sqrt(2*UAV_POS[1, 2]/G),M1_T_MAX-particles[i,2])
            particles[i,3]=np.clip(particles[i,3],0,max_dt)
            particles[i,4]=np.clip(particles[i,4],*TH_BOUNDS)
            particles[i,5]=np.clip(particles[i,5],*V_BOUNDS)
            particles[i,6]=np.clip(particles[i,6],0,M1_T_MAX); max_dt=min(SMK_LIFE,math.sqrt(2*UAV_POS[2, 2]/G),M1_T_MAX-particles[i,6])
            particles[i,7]=np.clip(particles[i,7],0,max_dt)
        fit=eval_pop(particles)
        # 更新最优
        mask=fit>pbest_fit; pbest[mask]=particles[mask]; pbest_fit[mask]=fit[mask]
        if fit.max()>gbest_fit: gbest=particles[np.argmax(fit)]; gbest_fit=fit.max()
        if it%PRINT_STEP==0 or it==ITER: print(f'[PSO] iter {it:3d}/{ITER}  best={gbest_fit:.6f}s')
    return gbest, gbest_fit

# -------------------- 控制台打印 result2 表头 --------------------
def print_result2(x, best_fit):
    fy1_xyz=(UAV_POS[0, 0], UAV_POS[0, 1], UAV_POS[0, 2]-0.0)
    th2,v2,t12,dt2,th3,v3,t13,dt3=x
    # 计算投放/起爆点
    th2,th3=math.radians(th2),math.radians(th3)
    # FY2
    drop2=[UAV_POS[1, 0]+v2*t12*math.cos(th2), UAV_POS[1, 1]+v2*t12*math.sin(th2), UAV_POS[1, 2]]
    det2 =[drop2[0]+v2*dt2*math.cos(th2), drop2[1]+v2*dt2*math.sin(th2), drop2[2]-0.5*G*dt2*dt2]
    # FY3
    drop3=[UAV_POS[2, 0]+v3*t13*math.cos(th3), UAV_POS[2, 1]+v3*t13*math.sin(th3), UAV_POS[2, 2]]
    det3 =[drop3[0]+v3*dt3*math.cos(th3), drop3[1]+v3*dt3*math.sin(th3), drop3[2]-0.5*G*dt3*dt3]
    # 表头
    header=['无人机编号','无人机运动方向(°)','无人机运动速度(m/s)',
            '投放点x(m)','投放点y(m)','投放点z(m)',
            '起爆点x(m)','起爆点y(m)','起爆点z(m)','有效干扰时长(s)']
    print('\t'.join(header))
    # 三行数据
    line=lambda u,th,v,dx,dy,dz,dex,dey,dez,t: f'{u}\t{th:.6f}\t{v:.6f}\t{dx:.6f}\t{dy:.6f}\t{dz:.6f}\t{dex:.6f}\t{dey:.6f}\t{dez:.6f}\t{t:.6f}'
    print(line('FY1',FY1_TH,FY1_V, UAV_POS[0, 0], UAV_POS[0, 1], UAV_POS[0, 2], fy1_xyz[0], fy1_xyz[1], fy1_xyz[2], best_fit))
    print(line('FY2',x[0],x[1], drop2[0], drop2[1], drop2[2], det2[0], det2[1], det2[2], best_fit))
    print(line('FY3',x[4],x[5], drop3[0], drop3[1], drop3[2], det3[0], det3[1], det3[2], best_fit))

# -------------------- main --------------------
def main():
    print('==== 2025 A-prob4 极速控制台版 ====')
    print('FY1 固定：反向静止即刻爆炸，只优化 FY2/FY3')
    t0=time.time()
    best_x, best_fit = pso()
    print('\n==== result2 表格（控制台输出）====')
    print_result2(best_x, best_fit)
    print(f'\n总耗时 {time.time()-t0:.2f}s')

if __name__=='__main__':
    main()