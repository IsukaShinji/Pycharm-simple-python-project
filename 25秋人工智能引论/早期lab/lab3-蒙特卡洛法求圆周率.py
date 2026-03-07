import random
n=10000000
def mont():
    inside=0
    for i in range(n):
        x,y=random.random(),random.random()
        if x*x+y*y<=1:
            inside+=1
    return 4*inside/n

if __name__=='__main__':
    pi=mont()
    print(f'采样点数 = {n}')
    print(f'π 估算值 ≈ {pi:.6f}')
    print(f'绝对误差 ≈ {abs(pi - 3.141592653589793):.6f}')