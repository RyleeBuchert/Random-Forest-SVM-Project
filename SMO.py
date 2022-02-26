import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd

"""
class svm():
    def __init__(self):
        
"""
#sum(aiy^i <x^i , x> + b)
def linearKernel(a, x, xi, y, b):
    #linK = a * yi * np.inner(xi, x).sum()
    #linK += b
    linK = 0
    for j in range(y.size):
        linK += a[j] * y[j] * np.dot(x[xi], np.transpose(x[j]))
    linK += b
    return linK

def kKernel(a, x, xi, y, c):
    linK = 0
    for j in range(y.size):
        linK += np.dot(x[xi], np.transpose(x[j]))
    linK += c
    return linK

def gaussianKernel(a, x, xi, y, c):
    sigma = 0.5
    k = 0
    for j in range(y.size):
        k = np.exp(-(np.linalg.norm(x[j] - y[j]) ** 2) / (2 * sigma ** 2))
    return k
    

def computeL(ai, aj, yi, yj, c):
    if yi == yj:
        return max(0, ai + aj - c)
    else:
        return max(0, aj - ai)
    
def computeH(ai, aj, yi, yj, c):
    if yi == yj:
        return min(c, ai + aj)
    else:
        return min(c, c + aj - ai)

def cEta(xi, xj):
    ij = np.inner(xi, xj)
    ii = np.inner(xi, xi)
    jj = np.inner(xj, xj)
    return 2 * (ij - ii - jj)

def error(xi, y, a, x, b):
    #use this line to change the kernel type
    f = gaussianKernel(a,x, xi, y, b)
    return f - y[xi]
    """
    error = 0
    for j in range(y.size):
        error += a[j] * y[j] * np.dot(x[xi], np.transpose(x[j]))
    error += b
    error += y[xi]
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
    aD = yi * (a - aOld)
    bs = b - e 
    br = b - ej
    pr = yj * (aj - ajOld)
    xij = np.inner(xi, xj)
    xii = np.inner(xi, xi)
    xjj = np.inner(xj, xj)
    inners = aD * xii - pr * xij
    innerz = aD * xij - pr * xjj
    b1 = bs - inners
    b2 = br - innerz
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
        j = random.randint(1, m)
        if j != i:
            break
    return j

def SMO(c, tol, passes, x, y):
    m = len(y)
    b = 0
    a = np.zeros(m)
    aOld = np.zeros(m)
    passing = 0
    while passing < passes:
        print(passing)
        changes = 0
        for i in range(0, m):
            #test with new error method change
            #e = error(x[i], y[i], a[i], x, b)
            e = error(i, y, a, x, b)
            ye = e
            #ye = y[i] * e
            if (ye < (-1 * tol) and a[i] < c) or (ye > tol and a[i] > 0):
                j = rand(m, i)
                if j == m:
                    j -= 1
                #test with new error method
                #ej = error(x[j], y[j], a[j], x, b)
                ej = error(j, y, a, x, b)
                aOld[i] = a[i]
                aOld[j] = a[j]
                l = computeL(a[i], a[j], y[i], y[j], c)
                h = computeH(a[i], a[j], y[i], y[j], c)
                if l == h:
                    continue
                eta = cEta(x[i], x[j])
                if eta >= 0:
                    continue
                aj = calculateJ(a[j], y[j], e, ej, eta)
                a[j] = clipAlpha(aj, h, l)
                temp = abs(a[j] - aOld[j])
                if temp < 0.0001:
                    continue
                a[i] = calculateI(a[i], y[i], y[j], aOld[j], a[j])
                b1, b2 = calc(b, x[i], y[i], a[i], a[j], e, ej, x[j], y[j], aOld[i], aOld[j])
                bnew = compute(b1, b2, a[i], a[j], c)
                if bnew != b:
                    b = bnew
                    changes += 1
        if changes == 0:
            passing += 1
        else:
            passing = 0
    return a, b

def errorBound(x, y, w, i):
    count = 0
    for i in range(y.size):
        pred = np.dot(x[i, :], np.transpose(w)) + i
        if pred > 0:
            if y[i] < 0:
                count += 1
        else:
            if y[i] > 0:
                count += 1
    return count/y.size
        

if __name__ == "__main__":  
    #change kernerl type with error method
    # load blobs data
    '''blobs_data = pd.read_csv('dataset\\blobs.csv')
    blobs_Y = blobs_data['class']
    blobs_X = blobs_data.drop(columns='class', axis=1)
    """
    print(blobs_X.to_numpy())
    print(blobs_Y.to_numpy())
    input()
    """
    # hyperparameters
    blobs_X = blobs_X.to_numpy()
    blobs_Y = blobs_Y.to_numpy()
    
    for i in range(len(blobs_Y)):
        if blobs_Y[i] == 0:
            blobs_Y[i] = -1
    
    c = 100
    tol = 0.001
    passes = 70
    a, b = SMO(c, tol, passes, blobs_X, blobs_Y)
    print(a)
    print(b)
    w = np.zeros((1, blobs_X.shape[1]))
    for i in range(blobs_Y.size):
        w += a[i] * blobs_Y[i] * blobs_X[i, :]
    xPlt = np.zeros((blobs_X.shape[0], 1))
    for i in range(blobs_Y.size):
        xPlt[i][0] = -1 * (w[0][0] * blobs_X[i][0] + b) / w[0][1]
    for i in range(blobs_Y.size):
        if blobs_Y[i] == 1:
            plt.scatter([blobs_X[i][0]], [blobs_X[i][1]], c='red')
        else:
            plt.scatter([blobs_X[i][0]], [blobs_X[i][1]], c='blue')
    plt.plot(blobs_X[:, 0], xPlt[:, 0], color = 'black')
    plt.show()
    print(errorBound(blobs_X, blobs_Y, w, b))
    '''
    blobs_data = pd.read_csv('dataset\\adult_data.csv')
    blobs_Y = blobs_data['income']
    blobs_X = blobs_data.drop(columns='income', axis=1)
    """
    print(blobs_X.to_numpy())
    print(blobs_Y.to_numpy())
    input()
    """
    # hyperparameters
    blobs_X = blobs_X.to_numpy()
    blobs_Y = blobs_Y.to_numpy()
    
    for i in range(len(blobs_Y)):
        if blobs_Y[i] == '<=50K':
            blobs_Y[i] = -1
        else:
            blobs_Y[i] = 1
    #print(blobs_Y)
    
    
    #print(blobs_X[1][1])
    """from sklearn.feature_extraction.text import HashingVectorizer
    vectorizer = HashingVectorizer(n_features = 20)
    for i in range(len(blobs_X)):
        for j in range(len(blobs_X[i])):
            if isinstance(blobs_X[i][j], str):
                rawS = [fr"{blobs_X[i][j]}"]
                #print(rawS)
                blobs_X[i][j] = vectorizer.transform(rawS).toarray()
                #print(blobs_X[i][j][0][1])
                max = 0.0
                min = 0.0
                mxCount = 0
                mnCount = 0
                for k in range(len(blobs_X[i][j][0])):
                    if blobs_X[i][j][0][k] >= max:
                        max == blobs_X[i][j][0][k]
                        mxCount += 1
                    elif blobs_X[i][j][0][k] <= min:
                        min = blobs_X[i][j][0][k]
                        mnCount += 1
                        
                #print(max)
                #print(min)
                if mnCount > mxCount:
                    blobs_X[i][j] = min
                elif mxCount > mnCount:
                    blobs_X[i][j] = max
                else:
                    minMax = [min, max]
                    blobs_X[i][j] = random.choice(minMax)
                #print(blobs_X[i][j])
                #input()
    print(blobs_X[1][1])
    input()
    """
    X = [[0, 0, 0]]*669
    #print(X)
    #print(d)
    for i in range(len(blobs_X)):
        d = []
        #for j in range(len(blobs_X[i])):
            #if isinstance(blobs_X[i][j], int):
        d.append(blobs_X[i][0])
        d.append(blobs_X[i][3])
        d.append(blobs_X[i][9])
        #print(d)
        #print(blobs_X[i][9])
        X[i][0] = d[0]
        X[i][1] = d[1]
        X[i][2] = d[2]

                
    X =pd.DataFrame(X)
    X = X.to_numpy()
    Y = blobs_Y
    
    #print(X)
    c = 1000
    tol = 0.001
    passes = 70
    a, b = SMO(c, tol, passes, X, Y)
    print(a)
    print(b)
    w = np.zeros((1, X.shape[1]))
    for i in range(Y.size):
        w += a[i] * Y[i] * X[i, :]
    xPlt = np.zeros((X.shape[0], 1))
    for i in range(Y.size):
        xPlt[i][0] = -1 * (w[0][0] * X[i][0] + b) / w[0][1]
    for i in range(Y.size):
        if Y[i] == 1:
            plt.scatter([X[i][0]], [X[i][1]], c='red')
        else:
            plt.scatter([X[i][0]], [X[i][1]], c='blue')
    plt.plot(X[:, 0], xPlt[:, 0], color = 'black')
    plt.show()
    print(errorBound(X, Y, w, b))