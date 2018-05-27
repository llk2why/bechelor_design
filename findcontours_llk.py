import cv2
import numpy as np
# from queue import Queue #python3
from Queue import Queue #python2.7

def dfs(l, x, y, newl, id, nid):
    # print([x, y, id])
    row,col = l.shape
    q = Queue()
    d =[
        [-1, -1], [-1, 0], [-1, 1],
        [0, -1], [0, 1],
        [1, -1], [1, 0], [1, 1],
    ] 
    q.put((x,y))
    l[x][y] = -1
    newl[x][y] = nid
    while(not q.empty()):
        x,y = q.get()
        for di in range(8):
            nx = x+d[di][0]
            ny = y+d[di][1]
            if (nx>=0 and nx<row and ny>=0 and ny<col and l[nx][ny]==id):
                l[nx][ny] = -1
                newl[nx][ny] = nid
                q.put((nx,ny))


def findContours(label,k):
    row,col = label.shape
    l = [[] for i in range(k)]
    r = [[] for i in range(k)]
    for i in range(row):
        dl = {}
        dr = {}
        for j in range(col):
            if label[i][j] not in dl:
                dl[label[i][j]] = j
            dr[label[i][j]] = j
        for k,v in dl.items():
            l[k].append([[v, i]])
        for k,v in dr.items():
            r[k].append([[v, i]])
    res = [
        np.array(l[i]+r[i][::-1])
        for i in range(len(l))
    ]
    return res

def ddfs(l, cc):
    p = 0
    center_c = []
    newl = np.zeros(l.shape)
    row,col = l.shape
    for i in range(row):
        for j in range(col):
            if (l[i][j]>=0):
                center_c.append(cc[l[i][j]])
                dfs(l, i, j, newl, l[i][j], p)
                p += 1
    return (newl, p, center_c)


def findContoursAnddfs(t, center_c):
    tt,ttt, tttt = ddfs(t, center_c)
    cnt = findContours(tt.astype(np.int16), ttt)
    return (cnt, ttt, tttt)
 
def main():
    t = np.array([[4,4,4,6,7],[4,4,4,6,7],[4,4,4,6,7]])
    findContoursAnddfs(t)

if __name__ == '__main__':
    main()
