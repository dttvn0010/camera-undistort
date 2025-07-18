import numpy as np
import math
from random import random

def find_transform(src_points, dst_points):
    print(src_points)
    print(dst_points)
    x = dst_points[:,0]
    y = dst_points[:,1]
    X = src_points[:,0]
    Y = src_points[:,1]
    M = np.stack([-y, x, Y,-X]).T
    N = x*Y - X * y
    print(M)
    print(N)
    print('MTM=', M.T @ M)
    print('MTN=', M.T @ N)
    X0,Y0,x0,y0 = np.linalg.solve(M.T @ M, M.T @ N)
    print('XYxy=',X0,Y0,x0,y0)
    Xm,Ym,xm,ym = X0,Y0,x0,y0
    a,b,c,d = np.linalg.solve(M.T@M, M.T @ np.ones(N.shape))
    print('abcd=',a,b,c,d)
    n = 0
    eps = 1e-2
    while n < 1000:
        D = xm * Ym - Xm * ym
        eX = Xm - X0 - a*D
        eY = Ym - Y0 - b*D
        ex = xm - x0 - c*D
        ey = ym - y0 - d*D
        #print(n, eX, eY, ex, ey)
        if abs(eX) + abs(eY) + abs(ex) + abs(ey) < eps:
            break
        H = np.array([
            [1+a*ym, -a*xm, -a*Ym, a*Xm],
            [b*ym, 1-b*xm, -b*Ym, b*Xm],
            [c*ym, -c*xm, 1-c*Ym, c*Xm],
            [d*ym, -d*xm, -d*Ym, 1+d*Xm],
        ])
        D = np.linalg.solve(H, np.array([eX,eY,ex,ey]))
        Xm -= D[0]
        Ym -= D[1]
        xm -= D[2]
        ym -= D[3]
        n += 1
    #
    X0,Y0,x0,y0 = Xm,Ym,xm,ym
    print(Xm,Ym,xm,ym)
    M = np.stack([
        X*x - X*x0,
        Y*x - Y *x0,
        x-x0
    ]).T
    N = X - X0
    print('M.T@M=', M.T@M)
    print('M.T@N=', M.T@N)
    A,B,C  = np.linalg.solve(M.T@M, M.T@N)
    print('ABC=', A, B, C)
    #
    δx = 1/(x-x0)
    δy = 1/(y-y0)
    Δx = X0 * δx
    Δy = Y0 * δy
    eps = 1e-2
    n = 0
    while n < 1000:
        Dn = (A*δy + B*δx - δx*δy)
        Nx = (B*Δx - B*Δy - C*δy - Δx*δy)
        Ny = (-A*Δx + A*Δy - C*δx - Δy*δx)
        Xm = Nx/Dn
        Ym = Ny/Dn
        Ex = Xm - X
        Ey = Ym - Y
        Dn2 = Dn ** 2
        dx2 = δx ** 2
        dy2 = δy ** 2
        Xm2 = Xm ** 2
        Ym2 = Ym ** 2
        D_A = np.sum(-2*δy*Ex*Nx/Dn2 + Ey*(-2*δy*Ny/Dn2 + 2*(-Δx + Δy)/Dn))
        D_B = np.sum(-2*δx*Ey*Ny/Dn2 + Ex*(-2*δx*Nx/Dn2 + 2*(Δx - Δy)/Dn))
        D_C = np.sum(-2*δx*Ey/Dn - 2*δy*Ex/Dn)
        D_AA = np.sum(2*(2*dy2*Ex*Xm + dy2*Xm2 - 2*δy*Ey*(-Δx + Δy - δy*Ym) + (-Δx + Δy - δy*Ym)**2)/Dn2)
        D_AB = np.sum(2*(2*δx*δy*Ex*Xm - δx*Ey*(-Δx + Δy - 2*δy*Ym) - δx*(-Δx + Δy - δy*Ym)*Ym - δy*Ex*(Δx - Δy) - δy*(Δx - Δy - δx*Xm)*Xm)/Dn2)
        D_AC = np.sum(2*(δx*δy*Ey - δx*(-Δx + Δy - δy*Ym) + dy2*Ex + dy2*Xm)/Dn2)
        D_BB = np.sum(2*(2*dx2*Ey*Ym + dx2*Ym2 - 2*δx*Ex*(Δx - Δy - δx*Xm) + (Δx - Δy - δx*Xm)**2)/Dn2)
        D_BC = np.sum(2*(dx2*Ey + dx2*Ym + δx*δy*Ex - δy*(Δx - Δy - δx*Xm))/Dn2)
        D_CC = np.sum(2*(dx2 + dy2)/Dn2)
        G = np.array([
            D_A, D_B, D_C
        ])
        
        if abs(D_A) + abs(D_B) + abs(D_C) < eps:
            break
        #
        H = np.array([
            [D_AA, D_AB, D_AC],
            [D_AB, D_BB, D_BC],
            [D_AC, D_BC, D_CC]
        ])
        D = np.linalg.solve(H, G)
        A -= D[0]
        B -= D[1]
        C -= D[2]
        n +=1
    print(A,B,C,X0,Y0,x0,y0)
    return A,B,C,X0,Y0,x0,y0

'''
from sympy import *
 

Δx, Δy, δx, δy = symbols('Δx Δy δx δy')
A, B, C = symbols('A B C')
X,Y = symbols('X Y')
D = A * δy + B * δx -δx*δy
Xm = ((B-δy)*(C+Δx)-B*(C+Δy))/D
Xm = simplify(expand(Xm))
Ym = (-A*(C+Δx)+(A-δx)*(C+Δy))/D
Ym = simplify(expand(Ym))
Ex = (Xm-X)
Ey = (Ym-Y)
E2 = Ex**2 + Ey**2
D_A = Derivative(E2, A)
D_B = Derivative(E2, B)
D_C = Derivative(E2, C)
D_AA = Derivative(D_A, A)
D_AB = Derivative(D_A, B)
D_AC = Derivative(D_A, C)
D_BB = Derivative(D_B, B)
D_BC = Derivative(D_B, C)
D_CC = Derivative(D_C, C)
'''

w1 = 216.2
w2 = 216.1
w3 = 101.5
w4 = 104.1
h1 = 114.1
h2 = 113.1
h3 = 50.4
h4 = 56.7

dst_points = np.array([
    [119, 67],
    [610, 63],
    [1173,50],
    [1203,321],
    [1236,621],
    [626,626],
    [64, 634],
    [99, 303],
], dtype='float32')

dst_points2 = np.array([
    [119, 67],
    [610, 63],
    [1172,50],
    [1200,321],
    [1236,621],
    [626,626],
    [64, 634],
    [99, 303],
], dtype='float32')
        
dst_points = np.array([
    [119, 67],
    [611, 63],
    [1171,49],
    [1201,321],
    [1236,621],
    [626,626],
    [64, 634],
    [100, 303],
], dtype='float32')       

tops = [[119, 67], [610, 63], [1173,50]]
bottoms =[[64, 634], [626,626], [1236,621]]

def lingress(X, y):
    Sxx = np.sum(X*X)
    Sxy = np.sum(X*y)
    Sx = np.sum(X)
    Sy = np.sum(y)
    n = len(X)
    a = (n*Sxy - Sx*Sy) / (n*Sxx - Sx*Sx)
    b = (Sy*Sxx - Sx*Sxy) / (n*Sxx - Sx*Sx)
    return a , b

def find_rotation(tops, bottoms):
    X1 = np.array([pt[0] for pt in tops], dtype='float')
    y1 = np.array([pt[1] for pt in tops], dtype='float')
    X2 = np.array([pt[0] for pt in bottoms], dtype='float')
    y2 = np.array([pt[1] for pt in bottoms], dtype='float')
    a1,_ = lingress(X1,y1)
    a2,_ = lingress(X2,y2)
    return math.atan((a1+a2)/2)

α = find_rotation(tops, bottoms)
cosα, sinα = math.cos(α), math.sin(α)
print(α)

dst_points = np.array([
    [x*cosα+y*sinα, y*cosα-x*sinα]
    for x,y in dst_points
])
src_points = np.array([
    [0,0],
    [w3,w3/w1*(h1-h2)/2],
    [w1,(h1-h2)/2],
    [(1-h4/h2)*w1+h4/h2*w2,(h1-h2)/2+h4],
    [(w1+w2)/2, (h1+h2)/2],
    [w4 + (w1-w2)/2, (((1-w4/w1)*h1 + w4/w1 *h2)+h1)/2],
    [(w1-w2)/2, h1],
    [(1-h3/h1)*(w1-w2)/2, h3]
], dtype='float32')

trans1 = find_transform(src_points[[0,1,5,6,7]], dst_points[[0,1,5,6,7]])
trans2 = find_transform(src_points[[1,2,3,4,5]], dst_points[[1,2,3,4,5]])

def find_inv(x,y):
    x, y = x*cosα+y*sinα, y*cosα-x*sinα
    if (x-dst_points[0,0])/(dst_points[2,0]-dst_points[0,0]) < w3/w1:
        A,B,C,X0,Y0,x0,y0 = trans1
    else:
        A,B,C,X0,Y0,x0,y0 = trans2
    δx = 1/(x-x0)
    δy = 1/(y-y0)
    Δx = X0 * δx
    Δy = Y0 * δy
    X = (B*Δx - B*Δy - C*δy - Δx*δy)/(A*δy + B*δx - δx*δy)
    Y = (-A*Δx + A*Δy - C*δx - Δy*δx)/(A*δy + B*δx - δx*δy)
    return X, Y

print(find_inv(1029, 145))
#print(find_inv(178, 475))
#print(find_inv(646, 189))

#print(find_inv(376,268))
#print(find_inv(905,216))
#print(find_inv(783,418))

#print(find_inv(1099,364))
#print(find_inv(460,266))
#print(find_inv(488,512))

import cv2
detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50))
img = cv2.imread('1.png', 0)
corners, ids, _ = detector.detectMarkers(img)
all_points = np.concatenate(corners).reshape((-1,2))
ymin = np.min(all_points[:,1])
ymax = np.max(all_points[:,1])
y25 = (ymin * 3 + ymax) / 4
y75 = (ymin  + 3 * ymax)/4
top_corners = [c[0] for c in corners if np.mean(c[0][:,1]) < y25]
mid_corners = [c[0] for c in corners if y25 < np.mean(c[0][:,1]) < y75]
bottom_corners = [c[0] for c in corners if np.mean(c[0][:,1]) > y75]

top_corners = sorted(top_corners, key=lambda c: np.mean(c[:,0]))
mid_corners = sorted(mid_corners, key=lambda c: np.mean(c[:,0]))
bottom_corners = sorted(bottom_corners, key=lambda c: np.mean(c[:,0]))

def get_top_left(corner):
    return corner[np.argmin(corner[:,0] + corner[:,1])]

def get_top_right(corner):
    return corner[np.argmax(corner[:,0] - corner[:,1])]

def get_bottom_left(corner):
    return corner[np.argmin(corner[:,0] - corner[:,1])]

def get_bottom_right(corner):
    return corner[np.argmax(corner[:,0] + corner[:,1])]

dst_points =[
    get_top_left(top_corners[0]),
    get_top_left(top_corners[1]),
    get_top_right(top_corners[2]),
    get_top_right(mid_corners[1]),
    get_bottom_right(bottom_corners[2]),
    get_bottom_left(bottom_corners[1]),
    get_bottom_left(bottom_corners[0]),
    get_top_left(mid_corners[0])
]