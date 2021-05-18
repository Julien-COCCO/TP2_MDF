## TP2 Equation d'advection

import numpy as np
import matplotlib.pyplot as plt

nu = 2
tmax = 2

def mat(n,d,vh,vb):
    M = d*np.eye(n)
    for i in range(n-1):
        M[i,i+1] = vh
        M[i+1,i] = vb
    M[0,n-1] = vb
    M[n-1,0] = vh
    return(M)

def solEC(h,tau):
    N = round((1/h))
    nmax = int(tmax/tau)
    c = nu*tau/h
    # Construction du vecteur u0 pour t=0
    u0 = np.zeros(N+1)
    x = np.linspace(0,1,N+1)
    for i in range(N+1):
        u0[i] = (np.sin(np.pi*x[i]))**10
    #
    U = np.zeros((nmax,N+1))
    U[0,:] = u0
    M = mat(N+1,1,-c/2,c/2)
    invM = np.linalg.inv(M)
    for i in range(1,nmax):
        U[i,:] = np.linalg.solve(invM,U[i-1,:])
    return(U,x)

def solIC(h,tau):
    N = round((1/h))
    nmax = int(tmax/tau)
    c = nu*tau/h
    # Construction du vecteur u0 pour t=0
    u0 = np.zeros(N+1)
    x = np.linspace(0,1,N+1)
    for i in range(N+1):
        u0[i] = (np.sin(np.pi*x[i]))**10
    #
    U = np.zeros((nmax,N+1))
    U[0,:] = u0
    M = mat(N+1,1,c/2,-c/2)
    for i in range(1,nmax):
        U[i,:] = np.linalg.solve(M,U[i-1,:])
    return(U,x)

def solDA(h,tau):
    N = round((1/h))
    nmax = int(tmax/tau)
    c = nu*tau/h
    # Construction du vecteur u0 pour t=0
    u0 = np.zeros(N+1)
    x = np.linspace(0,1,N+1)
    for i in range(N+1):
        u0[i] = (np.sin(np.pi*x[i]))**10
    #
    U = np.zeros((nmax,N+1))
    U[0,:] = u0
    M = mat(N+1,1-c,0,c)
    invM = np.linalg.inv(M)
    for i in range(1,nmax):
        U[i,:] = np.linalg.solve(invM,U[i-1,:])
    return(U,x)

def solLF(h,tau):
    N = round((1/h))
    nmax = int(tmax/tau)
    c = nu*tau/h
    # Construction du vecteur u0 pour t=0
    u0 = np.zeros(N+1)
    x = np.linspace(0,1,N+1)
    for i in range(N+1):
        u0[i] = (np.sin(np.pi*x[i]))**10
    #
    U = np.zeros((nmax,N+1))
    U[0,:] = u0
    M = mat(N+1,0,(1-c)/2,(1+c)/2)
    invM = np.linalg.inv(M)
    for i in range(1,nmax):
        U[i,:] = np.linalg.solve(invM,U[i-1,:])
    return(U,x)

def solLW(h,tau):
    N = round((1/h))
    nmax = int(tmax/tau)
    c = nu*tau/h
    # Construction du vecteur u0 pour t=0
    u0 = np.zeros(N+1)
    x = np.linspace(0,1,N+1)
    for i in range(N+1):
        u0[i] = (np.sin(np.pi*x[i]))**10
    #
    U = np.zeros((nmax,N+1))
    U[0,:] = u0
    M = mat(N+1,1-c**2,(c**2-c)/2,(c**2+c)/2)
    invM = np.linalg.inv(M)
    for i in range(1,nmax):
        U[i,:] = np.linalg.solve(invM,U[i-1,:])
    return(U,x)


UEC1,x1 = solEC(0.02,0.01)
UIC1,x1 = solIC(0.02,0.01)
UDA1,x1 = solDA(0.02,0.01)
ULF1,x1 = solLF(0.02,0.01)
ULW1,x1 = solLW(0.02,0.01)

UEC2,x2 = solEC(0.002,0.005)
UIC2,x2 = solIC(0.002,0.005)
UDA2,x2 = solDA(0.002,0.005)
ULF2,x2 = solLF(0.002,0.005)
ULW2,x2 = solLW(0.002,0.005)

UEC3,x3 = solEC(0.002,0.002)
UIC3,x3 = solIC(0.002,0.002)
UDA3,x3 = solDA(0.002,0.002)
ULF3,x3 = solLF(0.002,0.002)
ULW3,x3 = solLW(0.002,0.002)

UEC4,x4 = solEC(0.005,0.0002)
UIC4,x4 = solIC(0.005,0.0002)
UDA4,x4 = solDA(0.005,0.0002)
ULF4,x4 = solLF(0.005,0.0002)
ULW4,x4 = solLW(0.005,0.0002)


plt.figure("Explicite centré")
plt.subplot(4,1,1)
plt.plot(x1,UEC1[0,:],label="t=0")
plt.plot(x1,UEC1[int(0.1/0.01),:],label="t=0.1")
plt.plot(x1,UEC1[int(1/0.01),:],label="t=1")
plt.plot(x1,UEC1[int(2/0.01)-1,:],label="t=2")
plt.subplot(4,1,2)
plt.plot(x2,UEC2[0,:],label="t=0")
plt.plot(x2,UEC2[int(0.1/0.005),:],label="t=0.1")
plt.plot(x2,UEC2[int(1/0.005),:],label="t=1")
plt.plot(x2,UEC2[int(2/0.005)-1,:],label="t=2")
plt.subplot(4,1,3)
plt.plot(x3,UEC3[0,:],label="t=0")
plt.plot(x3,UEC3[int(0.1/0.002),:],label="t=0.1")
plt.plot(x3,UEC3[int(1/0.002),:],label="t=1")
plt.plot(x3,UEC3[int(2/0.002)-1,:],label="t=2")
plt.subplot(4,1,4)
plt.plot(x4,UEC4[0,:],label="t=0")
plt.plot(x4,UEC4[int(0.1/0.0002),:],label="t=0.1")
plt.plot(x4,UEC4[int(1/0.0002),:],label="t=1")
plt.plot(x4,UEC4[int(2/0.0002)-1,:],label="t=2")
plt.legend()
plt.show()

plt.figure("Implicite centré")
plt.subplot(4,1,1)
plt.plot(x1,UIC1[0,:],label="t=0")
plt.plot(x1,UIC1[int(0.1/0.01),:],label="t=0.1")
plt.plot(x1,UIC1[int(1/0.01),:],label="t=1")
plt.plot(x1,UIC1[int(2/0.01)-1,:],label="t=2")
plt.subplot(4,1,2)
plt.plot(x2,UIC2[0,:],label="t=0")
plt.plot(x2,UIC2[int(0.1/0.005),:],label="t=0.1")
plt.plot(x2,UIC2[int(1/0.005),:],label="t=1")
plt.plot(x2,UIC2[int(2/0.005)-1,:],label="t=2")
plt.subplot(4,1,3)
plt.plot(x3,UIC3[0,:],label="t=0")
plt.plot(x3,UIC3[int(0.1/0.002),:],label="t=0.1")
plt.plot(x3,UIC3[int(1/0.002),:],label="t=1")
plt.plot(x3,UIC3[int(2/0.002)-1,:],label="t=2")
plt.subplot(4,1,4)
plt.plot(x4,UIC4[0,:],label="t=0")
plt.plot(x4,UIC4[int(0.1/0.0002),:],label="t=0.1")
plt.plot(x4,UIC4[int(1/0.0002),:],label="t=1")
plt.plot(x4,UIC4[int(2/0.0002)-1,:],label="t=2")
plt.legend()
plt.show()

plt.figure("Décentré amont")
plt.subplot(4,1,1)
plt.plot(x1,UDA1[0,:],label="t=0")
plt.plot(x1,UDA1[int(0.1/0.01),:],label="t=0.1")
plt.plot(x1,UDA1[int(1/0.01),:],label="t=1")
plt.plot(x1,UDA1[int(2/0.01)-1,:],label="t=2")
plt.subplot(4,1,2)
plt.plot(x2,UDA2[0,:],label="t=0")
plt.plot(x2,UDA2[int(0.1/0.005),:],label="t=0.1")
plt.plot(x2,UDA2[int(1/0.005),:],label="t=1")
plt.plot(x2,UDA2[int(2/0.005)-1,:],label="t=2")
plt.subplot(4,1,3)
plt.plot(x3,UDA3[0,:],label="t=0")
plt.plot(x3,UDA3[int(0.1/0.002),:],label="t=0.1")
plt.plot(x3,UDA3[int(1/0.002),:],label="t=1")
plt.plot(x3,UDA3[int(2/0.002)-1,:],label="t=2")
plt.subplot(4,1,4)
plt.plot(x4,UDA4[0,:],label="t=0")
plt.plot(x4,UDA4[int(0.1/0.0002),:],label="t=0.1")
plt.plot(x4,UDA4[int(1/0.0002),:],label="t=1")
plt.plot(x4,UDA4[int(2/0.0002)-1,:],label="t=2")
plt.legend()
plt.show()

plt.figure("Lax-Friedrichs")
plt.subplot(4,1,1)
plt.plot(x1,ULF1[0,:],label="t=0")
plt.plot(x1,ULF1[int(0.1/0.01),:],label="t=0.1")
plt.plot(x1,ULF1[int(1/0.01),:],label="t=1")
plt.plot(x1,ULF1[int(2/0.01)-1,:],label="t=2")
plt.subplot(4,1,2)
plt.plot(x2,ULF2[0,:],label="t=0")
plt.plot(x2,ULF2[int(0.1/0.005),:],label="t=0.1")
plt.plot(x2,ULF2[int(1/0.005),:],label="t=1")
plt.plot(x2,ULF2[int(2/0.005)-1,:],label="t=2")
plt.subplot(4,1,3)
plt.plot(x3,ULF3[0,:],label="t=0")
plt.plot(x3,ULF3[int(0.1/0.002),:],label="t=0.1")
plt.plot(x3,ULF3[int(1/0.002),:],label="t=1")
plt.plot(x3,ULF3[int(2/0.002)-1,:],label="t=2")
plt.subplot(4,1,4)
plt.plot(x4,ULF4[0,:],label="t=0")
plt.plot(x4,ULF4[int(0.1/0.0002),:],label="t=0.1")
plt.plot(x4,ULF4[int(1/0.0002),:],label="t=1")
plt.plot(x4,ULF4[int(2/0.0002)-1,:],label="t=2")
plt.legend()
plt.show()

plt.figure("Lax-Wendroff")
plt.subplot(4,1,1)
plt.plot(x1,ULW1[0,:],label="t=0")
plt.plot(x1,ULW1[int(0.1/0.01),:],label="t=0.1")
plt.plot(x1,ULW1[int(1/0.01),:],label="t=1")
plt.plot(x1,ULW1[int(2/0.01)-1,:],label="t=2")
plt.subplot(4,1,2)
plt.plot(x2,ULW2[0,:],label="t=0")
plt.plot(x2,ULW2[int(0.1/0.005),:],label="t=0.1")
plt.plot(x2,ULW2[int(1/0.005),:],label="t=1")
plt.plot(x2,ULW2[int(2/0.005)-1,:],label="t=2")
plt.subplot(4,1,3)
plt.plot(x3,ULW3[0,:],label="t=0")
plt.plot(x3,ULW3[int(0.1/0.002),:],label="t=0.1")
plt.plot(x3,ULW3[int(1/0.002),:],label="t=1")
plt.plot(x3,ULW3[int(2/0.002)-1,:],label="t=2")
plt.subplot(4,1,4)
plt.plot(x4,ULW4[0,:],label="t=0")
plt.plot(x4,ULW4[int(0.1/0.0002),:],label="t=0.1")
plt.plot(x4,ULW4[int(1/0.0002),:],label="t=1")
plt.plot(x4,ULW4[int(2/0.0002)-1,:],label="t=2")
plt.legend()
plt.show()

