import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

def dichtomy(Q, eps, A, B):
  a = A
  b = B
  while (b - a >= eps):
    c = (a + b) / 2.
    Qc = Q(c)
    x = (a + c) / 2.
    Qx = Q(x)
    y = (c + b) / 2.
    Qy = Q(y)
    
    if (Qx <= Qc) and (Qc < Qy):
      b = c
      c = x
      Qc = Qx
    elif (Qc < Qx) and (Qc <= Qy):
      a = x
      b = y
    else:
      a = c
      c = y
      Qc = Qy
    
  return (a, b)

def Fib(n):
  if (n == 0 or n == 1):
    return 1
  else:
    return Fib(n - 1) + Fib(n - 2)
  
def Fibonacci(Q, eps, delta, A , B):
  a = A
  b = B
  n = int(np.log((b - a) / (delta - eps) * 5**0.5 + 1) * 1 / np.log(0.5 + 5**0.5 / 2.))
  
  F = [1, 1]
  for k in range(2, n):
    F.append(F[k - 2] + F[k - 1])
  
  for k in range(2, n - 1):
    lam = F[n - k] / float(F[n - k + 1])
    
    x = lam * a + (1 - lam) * b
    y = lam * b + (1 - lam) * a
    
    if (Q(x) > Q(y)):
      a = x
    else:
      b = y
      
  lam = 0.5 + eps
  x = lam * a + (1 - lam) * b
  y = lam * b + (1 - lam) * a
  if (Q(x) > Q(y)):
    a = x
  else:
    b = y
    
  return (a + b) / 2
  
def crt1(h, Q, dQ, x, d, t):
  if t < 0:
    return False

  if (Q(x + t * d) <= (Q(x) + h * np.dot(dQ(x), d) * t)):
    return True
  else:
    return False
  
def crt2(h, Q, dQ, x, d, t):
  if t < 0:
    return False

  if (abs(np.dot(dQ(x + t * d), d)) <= h * abs(np.dot(dQ(x), d))):
    return True
  else:
    return False
  
def ArmihoRule(a, h, y, Q, dQ, x, d):
  t = a
  while (not crt1(h, Q, dQ, x, d, t)):
    t = t * y
  
  return t

def OneDimOpt(etha, mu, delta, Q, dQ, x, d):
  phi = lambda t : Q(x + t * d)
  dphi = lambda t : np.dot(dQ(x + t * d), d)
  
  T = 0
  T1 = delta
  dlt = delta
  while(phi(T) > phi(T1) and dphi(T1) <= 1e-3):
    T = T1
    dlt = 2 * dlt
    T1 = T1 + dlt
  T = T1
  
  t = 0.
  lam = (5**0.5 - 1) / 2
  a = 0
  b = T
  while (not (crt1(mu, Q, dQ, x, d, t) and crt2(etha, Q, dQ, x, d, t))):
    z = lam * a + (1 - lam) * b
    y = lam * b + (1 - lam) * a
   
    if (phi(z) > phi(y)):
      a = z
    else:
      b = y
    t = (a + b) / 2
  return t

def GradDesc(Q, dQ, x0, eps):
  x = x0
  d = -dQ(x)
  
  k = 0
  while(np.linalg.norm(dQ(x)) >= eps):
    t = OneDimOpt(0.001, 0.0005, 0.15, Q, dQ, x, d)
    x = x + t * d
    d = -dQ(x)
    k += 1
    
  return (x, k)
  
def Newton(Q, dQ, invddQ, x0, eps):
  x = x0
  d = np.matmul(-invddQ(x), dQ(x))
  
  k = 0
  while(np.linalg.norm(dQ(x)) >= eps):
    x = x + d
    d = np.matmul(-invddQ(x), dQ(x))
    k += 1
    
  return (x, k)

def NewtonRafson(Q, dQ, invddQ, x0, eps):
  x = x0
  d = np.matmul(-invddQ(x), dQ(x))
  
  k = 0
  while(np.linalg.norm(dQ(x)) >= eps):
    t = OneDimOpt(0.001, 0.0005, 0.15, Q, dQ, x, d)
    x = x + t * d
    d = np.matmul(-invddQ(x), dQ(x))
    k += 1
    
  return (x, k)

def ConjGrads(Q, dQ, x0, eps):
  x = x0
  N = len(x0)
  p = [None] * N
  p[0] = -dQ(x)
  k = 0
  
  count = 0
  
  while (np.linalg.norm(dQ(x)) >= eps):
    count += 1
    if (np.dot(dQ(x), p[k]) >= 0):
      k = 0
      p[0] = -dQ(x)
      continue
    
    t = OneDimOpt(0.001, 0.0005, 0.15, Q, dQ, x, p[k])
    xprev = x
    x = x + t * p[k]
    k += 1
    if (k == N):
      k = 0
      p[0] = -dQ(x)
      continue

    beta = np.dot(dQ(x), dQ(x) - dQ(xprev)) / np.linalg.norm(dQ(xprev))**2
    p[k] = -dQ(x) + beta * p[k - 1]
    
  return (x, count)

def BFGS(Q, dQ, x0, eps):
  N = len(x0)
  x = sp.Matrix([[x0[0]], [x0[1]]])
  k = 0
  G = sp.eye(N)
  dq = sp.Matrix([[dQ(x)[0]], [dQ(x)[1]]])
  d = G * (-dq)
  xarray = np.array([float(x[0]), float(x[1])])
  while(np.linalg.norm(dQ(xarray)) >= eps):
    xarray = np.array([float(x[0]), float(x[1])])
    darray = np.array([float(d[0]), float(d[1])])
    t = OneDimOpt(0.001, 0.0005, 0.15, Q, dQ, xarray, darray)
    dx = sp.Matrix([[-x[0]], [-x[1]]])
    dy = sp.Matrix([[-dQ(x)[0]], [-dQ(x)[1]]])
    x = x + t * d
    dx += sp.Matrix([[x[0]], [x[1]]])
    dy += sp.Matrix([[dQ(x)[0]], [dQ(x)[1]]])
    
    G = G - G * dx * dx.T * G / (dx.T * G * dx)[0] + dy * dy.T / (dy.T * dy)[0]
    d = -1 * G.inv() * sp.Matrix([[dQ(x)[0]], [dQ(x)[1]]])
    k += 1
    
  return (x, k)

def HJ_Config(Q, orts, h, x):
  N = len(orts)
  z = x
  for i in range(N):
    if (Q(x + h * orts[i]) < Q(x)):
      z = x + h * orts[i]
    elif (Q(x - h * orts[i]) < Q(x)):
      z = x - h * orts[i]
  return z

def HookeJeevs(Q, x0, orts, eps, H):
  h = H
  x = x0
  y = x
  k = 0
  
  count = 0
  f = True
  while (f):
    count += 1
    xt = HJ_Config(Q, orts, h, y)
    
    if (Q(xt) < Q(x)):
      y = x + 2 * (xt - x)
      x = xt
      k += 1
      continue
    else:
      if (k > 0):
        k = 0
        y = x
        continue
      if (h < eps):
        f = False
      else:
        h = h / 2.
        continue
  
  return (x, count)