import numpy as np
import cv2

cap = cv2.VideoCapture('1.mp4')

n = 0
n_point = 0

while cap.isOpened():
    ret, im = cap.read()
    if ret != True:
        break
    
    n += 1
    kx = np.array([[-1,0,1]])
    ky = np.array([[-1],[0],[1]])
    im_e = (
        np.abs(cv2.filter2D(src=im[:,:,0], ddepth=-1, kernel=kx)) + 
        np.abs(cv2.filter2D(src=im[:,:,1], ddepth=-1, kernel=kx)) +
        np.abs(cv2.filter2D(src=im[:,:,2], ddepth=-1, kernel=kx)) +
        np.abs(cv2.filter2D(src=im[:,:,0], ddepth=-1, kernel=ky)) +
        np.abs(cv2.filter2D(src=im[:,:,1], ddepth=-1, kernel=ky)) +
        np.abs(cv2.filter2D(src=im[:,:,2], ddepth=-1, kernel=ky))
    )

    km = np.ones((21,21))/21/21
    im_d = cv2.filter2D(src=im_e, ddepth=-1, kernel=km)
    Y,X = np.where(im_d>56)
    points = list(zip(X,Y))
        
    def d2(x1,y1,x2,y2):
        return (x2-x1)*(x2-x1) + (y2-y1)*(y2-y1)
        
    def group_points(points):
        groups = []
        for x,y in points:
            found = False
            for g in groups:
                if d2(x,y, g['x'], g['y']) < 30 * 30 :
                    g['points'].append((x,y))
                    g['x'] = np.mean([p[0] for p in g['points']])
                    g['y'] = np.mean([p[1] for p in g['points']])
                    found = True
            if not found:
                groups.append({'x': x, 'y': y, 'points': [(x,y)]})
        return groups

    groups = group_points(points)
    bounds = []
    for g in groups:
        x = int(g['x'] + 0.5)
        y = int(g['y'] + 0.5)
        bounds.append([x,y])
        _ = cv2.circle(im, (x,y), 5, (0,0,255), -1)

    if len(bounds) != 4: continue
    cx = np.mean([p[0] for p in bounds])
    cy = np.mean([p[1] for p in bounds])
    point1 = next(p for p in bounds if p[0] < cx and p[1] < cy)
    point2 = next(p for p in bounds if p[0] > cx and p[1] < cy)
    point3 = next(p for p in bounds if p[0] > cx and p[1] > cy)
    point4 = next(p for p in bounds if p[0] < cx and p[1] > cy)

    mask = np.zeros(im.shape[:2], dtype='uint8')
    cv2.fillPoly(mask, pts=[np.array([point1, point2, point3, point4])], color=1)
    im_e = im_e * mask
    im_e[point1[1]-30:point1[1]+30, point1[0]-30:point1[0]+30] = 0
    im_e[point2[1]-30:point2[1]+30, point2[0]-30:point2[0]+30] = 0
    im_e[point3[1]-30:point3[1]+30, point3[0]-30:point3[0]+30] = 0
    im_e[point4[1]-30:point3[1]+30, point4[0]-30:point4[0]+30] = 0

    Y,X = np.where(im_e>132)
    points = list(zip(X,Y))
    groups = group_points(points)

    if len(groups) > n_point:
        n_point = len(groups)
        print(n_point, n)

    for g in groups:
        x = int(g['x'] + 0.5)
        y = int(g['y'] + 0.5)
        if any(d2(x,y,p[0],p[1]) < 50 * 50 for p in bounds):
            continue
        _ = cv2.circle(im, (x,y), 3, (0,255,0), -1)

    
    if im.shape[1] > 1024:
        im = cv2.resize(im, (1024, 1024*im.shape[0]//im.shape[1]))

    cv2.imshow('Frame',im)
    
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()    