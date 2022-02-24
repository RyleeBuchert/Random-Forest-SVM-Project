import numpy as np
import random
import math
import pandas as pd

"""
class svm():
    def __init__(self):
        
"""
#sum(aiy^i <x^i , x> + b)
def linearKernel(a, x, xi, yi, b):
    return a * yi * np.inner(xi, x).sum() + b

def kKernel(a, x, xi, yi, c):
    return (np.dot(np.transpose(x), yi) + c)**2

def gaussianKernel(a, x, xi, yi, c):
    sigma = 0.5
    k = np.exp(-(np.linalg.norm(xi - yi) ** 2 / (2 * sigma ** 2)))
    return k
    

def computeL(initA, aj, yi, yj, c):
    if yi == yj:
        return max(0, initA + aj - c)
    else:
        return max(0, initA - aj)
    
def computeH(initA, aj, yi, yj, c):
    if yi == yj:
        return min(c, initA + aj)
    else:
        return min(c, c + initA + aj)

def eta(xi, xj):
    ij = np.inner(xi, xj)
    ii = np.inner(xi, xi)
    jj = np.inner(xj, xj)
    return 2 * (ij - ii - jj)

def error(xi, y, a, x, b):
    
    f = linearKernel(a,x, xi, y, b)
    return f - y
    """
    error = 0
    for j in range(y.size):
        error += a[j] * y[j] * np.dot(x[i], x[j])
    error += b
    error += y[i]
    return error
    """
    

def clipAlpha(aj, h, l):
    if aj < l:
        return l
    elif aj > h:
        return h
    else:
        return aj
    
def calculateI(ai, yi, yj, aj, aj2):
    return ai + (yi * yj) * (aj - aj2)

def calculateJ(aj, yj, e, ej, eta):
    return aj - ((yj * (e - ej)) / eta)

def calc(b, xi, yi, a, aj, e, ej, xj, yj, aOld, ajOld):
    aD = a- aOld
    bs = b - e - yi
    br = b - ej - yi
    pr = yj * (aj - ajOld)
    xij = np.inner(xi, xj)
    xii = np.inner(xi, xi)
    inners = aD * xii - pr * xij
    b1 = bs * inners
    b2 = br * inners
    return b1, b2

def compute(b1, b2, a, aj, c):
    if aj > 0 and aj < c:
        return b2
    elif a > 0 and a < c:
        return b1
    else:
        return (b1+b2)/2
    
def rand(m, i):
    while True:
        j = random.ranint(0, m)
        if j == i:
            break
    return j

def SMO(c, tol, passes, x, y):
    m = x.shape[0]
    b = 0
    a = np.zeros(m)
    aOld = np.zeros(m)
    passing = 0
    while passing < passes:
        changes = 0
        for i in range(1, m):
            e = error(x[i], y, a, x, b)
            ye = y[i] * e
            if (ye < (-1 * tol) and a[i] < c) or (ye > tol and a[i] > 0):
                j = rand(m, i)
                ej = error(x[j], y[j], a[j], x, b)
                aOld[i] = a[i]
                aOld[j] = a[j]
                l = computeL(a[i], a[j], y[i], y[j], c)
                h = computeH(a[i], a[j], y[i], y[j], c)
                if l == h:
                    continue
                eta = eta(x[i], x[j])
                if eta > 0:
                    continue
                a[j] = clipAlpha(calculateJ(a[j], e, ej, eta),h, l)
                temp = math.abs(a[j] - aOld[j])
                if temp < 0.0001:
                    continue
                a[i] = calculateI(a[i], y[i], y[j], aOld[j], a[j])
                b1, b2 = calc(b, x[i], y[i], a[i], a[j], e, ej, x[j], y[j], aOld[i], aOld[j])
                b = compute(b1, b2, a[i], a[j], c)
                changes += 1
        if changes == 0:
            passing += 1
        else:
            passing = 0
    return a, b


if __name__ == "__main__":  
    
    # load blobs data
    blobs_data = pd.read_csv('data\\blobs2.csv')
    blobs_Y = blobs_data['class']
    blobs_X = blobs_data.drop(columns='class', axis=1)

    # hyperparameters
    c = 1000
    tol = 0.001
    passes = 100
    a, b = SMO(c, tol, passes, blobs_X, blobs_Y)