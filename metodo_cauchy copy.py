import math
import numpy as np

def regla_eliminacion(x1,x2,fx1,fx2,a,b) -> tuple[float,float]:
    if fx1 > fx2:
        return x1, b
    
    if fx1 < fx2:
        return a, x2
    
    return x1, x2

def w_to_x(w:float,a,b) -> float:
    return w * (b-a) + a

def busquedaDorada(funcion, epsilon:float, a:float=None, b:float=None) -> float:
    PHI = (1 + math.sqrt(5)) / 2 - 1
    aw, bw = 0, 1
    Lw = 1
    k = 1

    while Lw > epsilon:
        w2 = aw + PHI*Lw
        w1 = bw + PHI*Lw
        aw, bw = regla_eliminacion(w1, w2, funcion(w_to_x(w1,a,b)),
                                   funcion(w_to_x(w2,a,b)),aw,bw)
        k+=1
        Lw = bw - aw
    return (w_to_x(aw,a,b)+w_to_x(bw,a,b))/2




def gradiente(f,x,deltaX=0.001):
    grad=[]
    for i in range(0,len(x)):
        xp = x.copy()
        xn = x.copy()
        xp[i] = xp[i]+deltaX
        xn[i] = xn[i]-deltaX
        grad.append((f(xp)-f(xn))/(2*deltaX))
    return grad

def cauchy(funcion,x0,epsilon1,epsilon2,M):

    terminar=False
    xk=x0
    k=0
    while not terminar:
        
        grad = np.array(gradiente(funcion,xk))

        if np.linalg.norm(grad) < epsilon1 or k >= M:
            terminar=True
        else:

            def alpha_funcion(alpha):
                return funcion(xk-alpha*grad)
            
            alpha = busquedaDorada(alpha_funcion,epsilon=epsilon2,a=0.0,b=1.0)
            x_k1 = xk - alpha*grad
            # print(xk,alpha,grad,x_k1)

            if np.linalg.norm(x_k1-xk)/(np.linalg.norm(xk)+0.00001) <= epsilon2:
                terminar = True
            else:
                k = k+1
                xk = x_k1
    return xk

def exhaustive_search(x):#limit_inf:x[0],limit_sup:x[1]
    accuracy=0.5
    f=lambda x: x**2 + 3
    npoints=2*(x[1] - (x[0]))/accuracy
    # npoints=int(npoints)
    # limites=np.linspace(x[0],x[1],npoints)

    # Delta=(b-a)/n
    Delta=(x[1]-(x[0]))/npoints


    x1=x[0]
    x2=x[0]+Delta
    x3=x2+Delta
    while True:
        fx1=f(x1)
        fx2=f(x2)
        fx3=f(x3)
        if(fx1 > fx2) and (fx2 < fx3):
            return np.sum([x1,x3])/2
        else: 
            x1=x2
            x2=x3
            x3=x2+Delta

        if x3 > x[1]:
            return np.sum([x1,x3])/2
            break
        # print(fx1,fx2,fx3)


himmenblau = lambda x: (((x[0]**2)+x[1]-11)**2) + ((x[0]+(x[1]**2)-7)**2)
print('prueba1: ',cauchy(himmenblau,np.array([0.0,0.0]),0.001,0.001,100))

# Prueba con busqueda exhaustiva
print('prueba2: ',cauchy(exhaustive_search,np.array([1.0,2.0]),0.001,0.001,100))