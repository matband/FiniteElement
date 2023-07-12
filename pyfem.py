import numpy as np
import scipy.sparse
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.patches as mpatches
import matplotlib.tri as mtri
import matplotlib.transforms as transforms
from matplotlib.collections import LineCollection
import subprocess
from matplotlib import cm
import pandas as pd
import math
import scipy.sparse.linalg as linalg
import warnings
import configFEM 
import vtk

warnings.filterwarnings("error")

def readNode(file):
    nd = np.array([])
    ndFile = open(file+'.1.node')
    lines = ndFile.readlines()
    l = np.fromstring(lines[0], sep = ' ')
    nNodes = int(l[0])
    l = np.fromstring(lines[1], sep = ' ')
    nd = np.append(nd,l)
    for l in lines[2:]:
        if (l[0] != "#"):
            l = np.fromstring(l, sep = ' ')
            nd = np.vstack((nd,l))
    ndFile.close()
    return (nd,nNodes)

def readElemBar(file):
    el = np.array([],dtype = int)
    elFile = open(file+'.1.ele')
    lines = elFile.readlines()
    l = np.fromstring(lines[0], sep = ' ')
    nNodesElem = 2
    l = np.fromstring(lines[1], dtype = int, sep = ' ')
    el = np.append(el,l[1:3]-1)
    el = np.vstack((el,l[2:4]-1))
    el = np.vstack((el,l[1:4:2]-1))
    for l in lines[2:]:
        if (l[0] != "#"):
            l = np.fromstring(l,dtype = int, sep = ' ')
            el = np.vstack((el,l[1:3]-1))
            el = np.vstack((el,l[2:4]-1))
            el = np.vstack((el,l[1:4:2]-1))
    elFile.close()
    return (el,el.shape[0],2)

def readElem(file):
    el = np.array([],dtype = int)
    elFile = open(file+'.1.ele')
    lines = elFile.readlines()
    l = np.fromstring(lines[1],dtype = int, sep = ' ')
    el = np.append(el,l[1:]-1)
    for l in lines[2:]:
        if (l[0] != "#"):
            l = np.fromstring(l,dtype = int, sep = ' ')
            el = np.vstack((el,l[1:]-1))
    elFile.close()
    return (el,el.shape[0],el.shape[1])

def fixDofBycoords(coord,dof):
        a


#######################################
# PLOT FUNCTIONS
def plotElem(elem,node, lbName = False,alp = 1):
    xlim,ylim, _, _, _, _ = setlim(node)
    for cont,e in enumerate(elem):
        x1,x2 = 0,0
        path = mpath.Path
        path_data = [(path.MOVETO, [node[e[0]][0], node[e[0]][1]])]
        if model == 3: e = e[:3]
        for j in range(e.shape[0]):
            path_data.append((path.LINETO,[node[e[j]][0], node[e[j]][1]]))
            x1 += node[e[j]][0]
            x2 += node[e[j]][1]
        x1/=3
        x2/=3
        path_data.append((path.CLOSEPOLY, [node[e[0]][0], node[e[0]][1]]))
        codes, verts = zip(*path_data)
        path = mpath.Path(verts, codes)
        patch = mpatches.PathPatch(path,color='grey', ec = 'black', alpha = alp)
        ax.add_patch(patch)
        if lbName:
            plt.annotate(str(cont), xy=(x1,x2), ha="center",bbox=dict(facecolor='white', edgecolor='black', alpha= 0.5, boxstyle='round,pad=0.1'))
        # plt.annotate(str(elemAngle[cont-1]), xy=(x1,x2), ha="center",bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.1'))
    
def plotElemDisp(elem,nodeNew,totalDisp):
    xlim,ylim, _, _, _, _ = setlim(nodeNew)
    norm = plt.Normalize(totalDisp.min(), totalDisp.max())
    colors = cm.rainbow(norm(totalDisp))
    for cont,e in enumerate(elem):
        x1,x2 = 0,0
        path = mpath.Path
        path_data = [(path.MOVETO, [nodeNew[e[0]][0], nodeNew[e[0]][1]])]
        for j in range(e.size):
            path_data.append((path.LINETO,[nodeNew[e[j]][0], nodeNew[e[j]][1]]))
            x1 += nodeNew[e[j]][0]
            x2 += nodeNew[e[j]][1]
        x1/=e.size
        x2/=e.size
        path_data.append((path.CLOSEPOLY, [nodeNew[e[0]][0], nodeNew[e[0]][1]]))
        codes, verts = zip(*path_data)
        path = mpath.Path(verts, codes)
        patch = mpatches.PathPatch(path,color=0.5*(colors[e[0]]+colors[e[1]]), ec =0.5*(colors[e[0]]+colors[e[1]]), alpha = 1)
        ax.add_patch(patch)

    fig.colorbar(cm.ScalarMappable(norm=norm,cmap='rainbow'), ax = ax, shrink = 0.7, location='left')


def plotElemDispTR3(elem,nodeNew,disp,smooth = False,edge=True):
    norm = plt.Normalize(disp.min(), disp.max())
    fig.colorbar(cm.ScalarMappable(norm=norm,cmap='rainbow'), ax = ax, shrink = 0.7, location='left')
    disp = (disp - disp.min())/(disp.max()-disp.min())
    x = nodeNew[:,0]
    y = nodeNew[:,1]
    e = elem[:,:3]
    setlim(nodeNew)

    for cont,e in enumerate(elem):
        x = np.array([nodeNew[e[0]][0],nodeNew[e[1]][0],nodeNew[e[2]][0]])
        y = np.array([nodeNew[e[0]][1],nodeNew[e[1]][1],nodeNew[e[2]][1]])
        if edge:
            plt.triplot(x,y,color='black',linewidth=0.5)

        if(smooth):
            tri = mtri.Triangulation(x,y,triangles=np.array([[0,1,2]]))
            plt.tripcolor(tri,disp[[e[0],e[1],e[2]]],vmin=0,vmax=1,cmap='rainbow',alpha=1,shading='gouraud')
        else:
            z = 1/3*disp[e[0]]+1/3*disp[e[1]]+1/3*disp[e[2]]
            x = np.append(x,x.sum()/3)
            y = np.append(y,y.sum()/3)
            v = np.append(disp[[e[0],e[1],e[2]]],z)
            plt.tricontourf(x,y,v,vmax=1,vmin=0,cmap='rainbow')

def plotElemDispTR6(elem,nodeNew,disp,smooth = False,edge=True):
    ax,
    norm = plt.Normalize(disp.min(), disp.max())
    fig.colorbar(cm.ScalarMappable(norm=norm,cmap='rainbow'), ax = ax, shrink = 0.7, location='left')
    disp = (disp - disp.min())/(disp.max()-disp.min())
    x = nodeNew[:,0]
    y = nodeNew[:,1]
    e = elem[:,:6]
    setlim(nodeNew)

    for cont,e in enumerate(elem):
        x = np.array([nodeNew[e[0]][0],nodeNew[e[1]][0],nodeNew[e[2]][0],nodeNew[e[3]][0],nodeNew[e[4]][0],nodeNew[e[5]][0]])
        y = np.array([nodeNew[e[0]][1],nodeNew[e[1]][1],nodeNew[e[2]][1],nodeNew[e[3]][1],nodeNew[e[4]][1],nodeNew[e[5]][1]])

        if edge:
            plt.triplot(x,y,color='black',linewidth=0.5)

        if(smooth):
            tri = mtri.Triangulation(x,y)
            plt.tripcolor(tri,disp[[e[0],e[1],e[2]]],vmin=0,vmax=1,cmap='rainbow',alpha=1,shading='gouraud')
        else:
            z = 1/3*disp[e[0]]+1/3*disp[e[1]]+1/3*disp[e[2]]
            x = np.append(x,x.sum()/6)
            y = np.append(y,y.sum()/6)
            v = np.append(disp[[e[0],e[1],e[2],e[3],e[4],e[5]]],z)
            plt.tricontourf(x,y,v,vmax=1,vmin=0,cmap='rainbow')


def plotEdges(node,elem):
    for i,e in enumerate(elem):
        v0 = node[e[0]]
        v1 = node[e[1]]
        v2 = node[e[2]]
        edge0 = (v0+v1)/(2) 
        edge1 = (v1+v2)/(2) 
        edge2 = (v2+v0)/(2) 

        plt.annotate(f'e{i}/0', xy = edge0, fontsize = 'x-small',bbox=dict(facecolor='white', edgecolor='black', alpha= 0.5, boxstyle='round,pad=0.1'))
        plt.annotate(f'e{i}/1', xy = edge1, fontsize = 'x-small',bbox=dict(facecolor='white', edgecolor='black', alpha= 0.5, boxstyle='round,pad=0.1'))
        plt.annotate(f'e{i}/2', xy = edge2, fontsize = 'x-small',bbox=dict(facecolor='white', edgecolor='black', alpha= 0.5, boxstyle='round,pad=0.1'))


def plotNode(node,disp = [],totalDisp = [], pos = False, plotId = False,plotCoord = False):
    xlim,ylim, _, _, _, _ = setlim(node)
    if (not pos):
        for c,i in enumerate(node):
            plt.scatter(i[0],i[1],color = 'grey')
            if plotId:
                plt.annotate(str(c),xy =(i-0.02*(xlim+ylim)),color = 'black')
            if plotCoord:
                plt.annotate(str(f'({i[0]:.2f},{i[1]:.2f})'),xy =(i),color = 'black', fontsize = 8)
    else:
        nodeNew = node.copy()
        
        for i in range(nNodes):
            for j in range(nDofNode):
                nodeNew[i,j] += disp[nDofNode*i+j]
        norm = plt.Normalize(totalDisp.min(), totalDisp.max())
        colors = cm.rainbow(norm(totalDisp))
        for c,i in enumerate(nodeNew):        
            sct = plt.scatter(i[0],i[1],color = 'black')
            plt.annotate(str(c),xy =(i+0.01*(xlim+ylim)),color = 'black')
    setlim(node)


def plotElemStBar(elem,nd,s,E):
    xlim,ylim, _, _, _, _ = setlim(nd)
    norm = plt.Normalize(s.min(), s.max())
    colors = cm.rainbow(norm(s))
    for cont,e in enumerate(elem):
        x1,x2 = 0,0
        path = mpath.Path
        path_data = [(path.MOVETO, [nd[e[0]][0], nd[e[0]][1]])]
        for j in range(e.size):
            path_data.append((path.LINETO,[nd[e[j]][0], nd[e[j]][1]]))
            x1 += nd[e[j]][0]
            x2 += nd[e[j]][1]
        x1/=e.size
        x2/=e.size
        path_data.append((path.CLOSEPOLY, [nd[e[0]][0], nd[e[0]][1]]))
        codes, verts = zip(*path_data)
        path = mpath.Path(verts, codes)
        patch = mpatches.PathPatch(path,color=colors[cont], ec =colors[cont])
        ax.add_patch(patch)
    fig.colorbar(cm.ScalarMappable(norm=norm,cmap='rainbow'), ax = ax, shrink = 0.7, location='left', label = 'strain')
    s = s * E
    norm = plt.Normalize(s.min(), s.max())
    fig.colorbar(cm.ScalarMappable(norm=norm,cmap='rainbow'), ax = ax, shrink = 0.7, location='left', label = 'stress')
    
def plotElemBeam(elem,nd,st):
    
    min_value = np.min(st)
    max_value = np.max(st)

    cmap = cm.get_cmap('rainbow') 
    norm = plt.Normalize(0,1)
    sm = plt.cm.ScalarMappable(cmap=cmap)
    sm.set_array([min_value,max_value]) 

    ns = 10
    for i, element in enumerate(elem):

        node1 = nd[element[0]]
        node2 = nd[element[1]]
        
        seg = []
        xpoints = np.linspace(node1[0],node2[0],ns)
        ypoints = np.linspace(node1[1],node2[1],ns)
        for k in range(ns-1):
            s = ((xpoints[k],ypoints[k]),(xpoints[k+1],ypoints[k+1]))
            seg.append(s)

        value = st[i]
        if (min_value == 0 and max_value == 0):
            normalized_value = 0
        else:
            normalized_value = (value - min_value) / (max_value - min_value)
        gradient = np.linspace(normalized_value[0], normalized_value[1], ns)

        lc = LineCollection(seg, cmap = cmap,norm = norm, array = gradient)

        # Criação do plot
        ax.add_collection(lc)
    ax.autoscale()

    fig.colorbar(sm)
    
def plotElemStTR3(elem,nd,stVec,direction,edge = True):
    xlim,ylim, _, _, _, _ = setlim(nd)
    st = stVec.T[direction]
    norm = plt.Normalize(st.min(), st.max())
    colors = cm.rainbow(norm(st))
    fig.colorbar(cm.ScalarMappable(norm=norm,cmap='rainbow'), ax = ax, shrink = 0.7, location='left', label = 'stress')
    st = (st - st.min())/(st.max()-st.min())
    for cont,e in enumerate(elem):
        x1,x2 = 0,0
        path = mpath.Path
        path_data = [(path.MOVETO, [nd[e[0]][0], nd[e[0]][1]])]
        for j in range(e.size):
            path_data.append((path.LINETO,[nd[e[j]][0], nd[e[j]][1]]))
            x1 += nd[e[j]][0]
            x2 += nd[e[j]][1]
        x1/=e.size
        x2/=e.size
        path_data.append((path.CLOSEPOLY, [nd[e[0]][0], nd[e[0]][1]]))
        codes, verts = zip(*path_data)
        path = mpath.Path(verts, codes)
        if edge:
            patch = mpatches.PathPatch(path,color=colors[cont], ec =colors[cont])
        else:
            patch = mpatches.PathPatch(path,color=colors[cont], ec =None)
        ax.add_patch(patch)
        
def plotElemStTR6(elem,nd,stVec,direction,smooth = False,edge = True):
    st = stVec.T[[0+direction,3+direction,6+direction]]
    norm = plt.Normalize(st.min(), st.max())
    fig.colorbar(cm.ScalarMappable(norm=norm,cmap='rainbow'), ax = ax, shrink = 0.7, location='left', label = 'stress')
    st = (st - st.min())/(st.max()-st.min())
    # st = (st - st.min())/(st.max()-st.min())
    xlim,ylim, _, _, _, _ = setlim(nd)
    for cont,e in enumerate(elem):
        x = np.array(nd[(e[:]),0])
        y = np.array(nd[(e[:]),1])
        v = np.array(st[:,cont])
        if edge:
            plt.triplot(x,y,color='black',linewidth=0.5)

        tri = mtri.Triangulation(x[:3],y[:3],triangles=np.array([[0,1,2]]))
        if(smooth):
            plt.tripcolor(tri,v,vmin=0,vmax=1,cmap='rainbow',alpha=1,shading='gouraud')
        else:
            plt.tricontourf(tri,v,vmax=1,vmin=0,cmap='rainbow')
    
def plotConstraint(center, direction,xlim,ylim, xmax, xmin, ymax, ymin):

    Path = mpath.Path
    if(direction == 0):
        a = -0.025*(xlim)
        b = 0.025*(ylim)
        if((xmax - center[0])<(center[0]-xmin)):
            a=-a
            b= -b
        path_data = [
            (Path.MOVETO, [center[0], center[1]]),
            (Path.LINETO, [center[0]+a, center[1]-b]),
            (Path.LINETO, [center[0]+a, center[1]+b]),
            (Path.CLOSEPOLY, [center[0], center[1]]),
            (Path.MOVETO, [center[0]+1.25*a, center[1]-b]),
            (Path.LINETO, [center[0]+1.25*a, center[1]+b])
            ]
    else:
        a = 0.025*(xlim)
        b = 0.025*(ylim)
        path_data = [
            (Path.MOVETO, [center[0], center[1]]),
            (Path.LINETO, [center[0]-a, center[1]-b]),
            (Path.LINETO, [center[0]+a, center[1]-b]),
            (Path.CLOSEPOLY, [center[0], center[1]]),
            (Path.MOVETO, [center[0]-a, center[1]-b*1.25]),
            (Path.LINETO, [center[0]+a, center[1]-b*1.25])
            ]
    codes, verts = zip(*path_data)
    path = mpath.Path(verts, codes)
    patch = mpatches.PathPatch(path,color='grey', ec = 'black')
    ax.add_patch(patch)

def plotFixedEnd(node,theta,n):

    s = 0.05*(xlim) *np.sin(theta*np.pi/180)
    c = 0.05*(ylim) * np.cos(theta*np.pi/180)

    Path = mpath.Path
    path_data =  [(Path.MOVETO, [node[n][0], node[n][1]]),
            (Path.LINETO, [node[n][0]-s, node[n][1]+c]),
            (Path.LINETO, [node[n][0]+s, node[n][1]-c]),
            (Path.CLOSEPOLY, [node[n][0], node[n][1]])
            ]

    codes, verts = zip(*path_data)
    path = mpath.Path(verts, codes)
    patch = mpatches.PathPatch(path,color='grey', ec = 'black', linewidth = 2)
    ax.add_patch(patch)


def plotLoad(center,direction,value,xlim,ylim):
    a=0.0
    b=1
    if(value < 0):
        a=-1
        b=0
    if(direction == 0):
        ar = mpatches.FancyArrow(center[0],center[1],dx=(a+b)*xlim*0.075 ,dy=0,length_includes_head=True,width = ylim*0.0125,head_length = xlim*0.025, color = 'orange', ec='black')
        plt.annotate(str(f'{value:.2f}'), xy=(center[0]+xlim*0.1*(a+b),center[1]+ylim*0.05*(-a+b)), ha="center")
    else:
        ar = mpatches.FancyArrow(center[0],center[1]-(a)*ylim*0.075,dx=0 ,dy=(a+b)*ylim*0.075,length_includes_head=True,width = 0.0125*xlim,head_length = 0.025*ylim, color = 'orange', ec='black')
        plt.annotate(str(f'{value:.2f}'), xy=(center[0],center[1]-ylim*0.1*(a-b)), ha="center")

    ax.add_patch(ar)
        

def plotDirichlet(nodeDofList, node, elemAngle = []):
    xlim,ylim, xmax, xmin, ymax, ymin = setlim(node)
    for i in range(nodeDofList.shape[0]):
        if(all(k == True for k in nodeDofList[i]) and model == 1):
            plotFixedEnd(node,elemAngle[i],i)
        else:
            for j in range(nodeDofList[i].size):
                
                if (nodeDofList[i][j]):
                    center = node[i]
                    plotConstraint(center,j,xlim,ylim, xmax, xmin, ymax, ymin)
            

def plotNewmann(nodeLoadList, node, docList = []):
    xlim,ylim, _, _, _, _ = setlim(node)
    docL = np.flip(np.append(docList,-1))
    cont = 0
    for i in range(nodeLoadList.size):
        while (cont+i == docL[-1]):
            docL = np.delete(docL,-1)
            cont +=1
        if(nodeLoadList[i]!=0):
            center = node[(i+cont)//nDofNode]
            plotLoad(center,(i+cont)%nDofNode,nodeLoadList[i],xlim,ylim)



############################
#CALCULATE ELEMENT ROTATIONS
def get_angle(x1,y1,x2,y2):
    return math.degrees(math.atan2(y2-y1, x2-x1))

def getElemAngle(elem,node,nElem):
    elemAngle = np.zeros((nElem),dtype=int)

    for e in range(nElem):
        x1 = node[elem[e][0]][0] 
        y1 = node[elem[e][0]][1]
        x2 = node[elem[e][1]][0] 
        y2 = node[elem[e][1]][1]

        elemAngle[e] = get_angle(x1,y1,x2,y2)
    return(elemAngle)

# nd = pd.read_csv('nodes.txt', sep='\s+', header = None)
# nNodes = node.shape[0]
# node = nd.to_numpy()
# print(node)
# elem = np.loadtxt('elem.txt', dtype = int, delimiter=' ')
# nElem = elem.shape[0]
#=
def getStiffnessBeam(E,I,A,L):
    L2 = L*L
    L3 = L*L*L
    return np.array(
    [[E*A/L  ,0,0,-E*A/L,0,0],
    [0, 12*E*I/L3 , 6*E*I/L2 ,0,-12*E*I/L3, 6*E*I/L2 ],
    [0, 6*E*I/L2 , 4*E*I/L  ,0, -6*E*I/L2, 2*E*I/L  ],
    [ -E*A/L  ,0,0,  E*A/L  ,0,0],
    [0,-12*E*I/L3,-6*E*I/L2 ,0, 12*E*I/L3,-6*E*I/L2 ],
    [0, 6*E*I/L2 , 2*E*I/L  ,0,-6*E*I/L2, 4*E*I/L  ], 
    ])

def getStiffnessTruss(E,A,L):
    return E*A/L * np.array(
    [[1,0,-1,0],
     [0,0,0,0],
     [-1,0,1,0],
     [0,0,0,0],
    ])

def rotateT(mat,theta):
    s = np.sin(theta* np.pi / 180)
    c = np.cos(theta* np.pi / 180)
    rot = np.array(
        [[c,s,0,0],
        [-s,c,0,0],
        [0,0,c,s],
        [0,0,-s,c]]
    )
    return rot.T @ mat @ rot

def rotateBeam(mat,theta):
    s = np.sin(theta* np.pi / 180)
    c = np.cos(theta* np.pi / 180)
    rot = np.array(
       [[ c,s,0,0,0,0],
        [-s,c,0,0,0,0],
        [0,0,1,0,0,0],
        [0,0,0, c,s,0],
        [0,0,0,-s,c,0],
        [0,0,0,0,0,1]]
    )
    return rot.T @ mat @ rot

def mean(x1,x2):
    return [(x1[0]+x2[0])*0.5,(x1[1]+x2[1])*0.5]

def calcNorm(x,y):
    return math.sqrt(x*x+y*y)


def findNear(node,nNeighs = 1):
    nBinds = 0
    nearList = [[]]
    for i in range(nNodes):
        nearList.append([])
        distls = np.zeros(nNodes)
        for j in range(nNodes):
            if(i==j):
                distls[j] = 0
            distls[j] = math.dist(node[i],node[j])
            
        min = distls[0]
        ix = 0
        if(i == 0):
            min = distls[1]
        for j in range(nNodes):
            if(distls[j]<min and distls[j]!=0):
                ix = j
                min = distls[j]
        # print(f"min distance from node {i} is {min} from node {ix}")
        for j in range(nNodes):
            if(i<j):
                if(distls[j]<(1+nNeighs)*min and distls[j]!=0):
                    nearList[i].append(j)
                    nBinds+=1
    return nearList, nBinds


def bindnear(nearList):
    elem2 = np.array([[]], dtype = int)
    item = [0,nearList[0][0]]
    elem2 = np.append(elem2,item)
    for j in nearList[0][1:]:
        elem2 = np.vstack((elem2,[0,j]))
    for i in range(1,nNodes):
        for j in nearList[i]:
            if(i<j):
                elem2 = np.vstack((elem2,[i,j]))
    return elem2

def getElemBin(bitv,nearList,intersecting = []):
    elem = np.array([[0,0]],dtype = int)
    cont = 0
    for i,nd in enumerate(nearList):
        for j in nd:
            if (bitv[cont]):
                elem = np.vstack((elem,[i,j]))
            cont+=1
    elem = elem[1:]
    for i in intersecting:
        if bitv[i[0]]:
            for gene in i[1:]:
                bitv[gene] = False
    return elem


def getL(nNodesElem,nodeL,node):
    x,y = np.array([]),np.array([])
    for i in range(nNodesElem):
        ix = nodeL[i]
        x = np.append(x,node[ix][0])
        y = np.append(y,node[ix][1])
    l = math.dist([x[0],y[0]],[x[1],y[1]])
    return l

def getObj(elem,node,nNodesElem):
    
    sumL = sum(list(map(lambda x: getL(nNodesElem,x,node),elem)))
    return sumL
    

def assignloadcoord(node,elem,loadcoord,surfaceForces):
    for load in loadcoord:
        edgelist = np.array([[0,0,0]])
        
        d = load[2]
        if load[3] ==1:
            mcoord = node[:,d].max()
        else: mcoord = node[:,d].min()
        for i,e in enumerate(elem):
            v0 = e[0]
            v1 = e[1]
            v2 = e[2]
            if(abs(node[v0,d]-mcoord)<0.1) and (abs(node[v1,d]-mcoord)<0.1):
                dis = math.dist(node[v0],node[v1])
                edgelist = np.vstack((edgelist,[i,0,dis]))
            if(abs(node[v1,d]-mcoord)<0.1) and (abs(node[v2,d]-mcoord)<0.1):
                dis = math.dist(node[v2],node[v1])
                edgelist = np.vstack((edgelist,[i,1,dis]))
            if(abs(node[v0,d]-mcoord)<0.1) and (abs(node[v2,d]-mcoord)<0.1):
                dis = math.dist(node[v0],node[v2])
                edgelist = np.vstack((edgelist,[i,2,dis]))
        edgelist = edgelist[1:]
        sumd = edgelist[:,2].sum()
        for ed in edgelist:
            surfaceForces.append([int(ed[0]),int(ed[1]),load[1],load[0]*ed[2]/sumd])
    return surfaceForces




#Stiffness matrix assemble
def globalstiffAssemble(node, elem,elemAngle,nElem,nNodes,nNodesElem,nDofNode,model,E,A,I):
    K = scipy.sparse.lil_matrix((nNodes*nDofNode,nNodes*nDofNode), dtype =np.double)
    lList = np.zeros(nElem)
    if (model==0):
        
        for i in range(nElem):
            nodeL = elem[i]
            theta = elemAngle[i]
            l = getL(nNodesElem,nodeL,node)
            lList[i] = l
            kEl = rotateT(getStiffnessTruss(E,A,l),theta)
            for j in range(nNodesElem):
                actualNode = nodeL[j]
                for k in range(nDofNode):
                    ix = nDofNode*actualNode+k
                    for icol in range(nNodesElem):
                        for jcol in range(nDofNode):
                            col = nDofNode*icol+jcol
                            K[ix,nDofNode*nodeL[icol]+jcol] += kEl[nDofNode*j+k,col] 
        return K, lList
    elif (model==1):
        for i in range(nElem):
            nodeL = elem[i]
            theta = elemAngle[i]
            l = getL(nNodesElem,nodeL,node)
            lList[i] = l
            kEl = rotateBeam(getStiffnessBeam(E,I,A,l),theta)
            for j in range(nNodesElem):
                actualNode = nodeL[j]
                for k in range(nDofNode):
                    ix = nDofNode*actualNode+k
                    for icol in range(nNodesElem):
                        for jcol in range(nDofNode):
                            col = nDofNode*icol+jcol
                            K[ix,nDofNode*nodeL[icol]+jcol] += kEl[nDofNode*j+k,col]
        return K, lList
    

def getBC(fixedDof,fixedNode,loadDof):
    doc = 0
    nodeDofList = np.zeros((nNodes,nDofNode),dtype =bool)
    nodeLoadList = np.zeros((nNodes*nDofNode), dtype = np.double)
    
    for i in loadDof:
        nodeLoadList[nDofNode*i[0]+i[1]] = i[2]

    for i in fixedNode:
        for j in range(nodeDofList[i].size):
            if (not nodeDofList[i,j]):
                nodeDofList[i,j] = 1
                doc +=1
            
    for i in fixedDof:
        if (not nodeDofList[i[0],i[1]]):
            nodeDofList[i[0],i[1]] = 1
            doc +=1

    return nodeDofList,nodeLoadList,doc
    
def removeFreeNodes(node,elem,nodeLoadList):
    freenodeList = np.empty(0,dtype = int)
    for i in range(node.shape[0]):
        if i not in elem and nodeLoadList[nDofNode*i:nDofNode*i+1] == 0:
            freenodeList = np.append(freenodeList,[nDofNode*i,nDofNode*i+1])
    return freenodeList 

def assignBC(K,doc,nodeLoadList,nodeDofList, nNodes, nDofNode, freenodeList = np.array([],dtype = int)):
    cont = 0
    docList= np.zeros(doc,dtype = int)
    for i in freenodeList:
        if not nodeDofList[int(i//nDofNode),int(i%nDofNode)]:
            docList = np.append(docList,int(i))

    for i in range(nNodes):
        for j in range(nDofNode):
            if (nodeDofList[i,j]):
                dof = nDofNode*i+j-cont
                docList[cont] = nDofNode*i+j
                K= scipy.sparse.hstack([ K[:,0:dof] , K[:,dof+1:] ])
                K = K.tolil()
                K= scipy.sparse.vstack([ K[0:dof,:] , K[dof+1:,:] ]) 
                K = K.tolil()
                cont+=1
    
    docList =np.sort(docList)
    for i in docList[::-1]:
        nodeLoadList = np.delete(nodeLoadList,i)


    K= K[K.getnnz(1)>0][:,K.getnnz(0)>0]

    return K,docList,nodeLoadList

def solve(K,nodeLoadList):
    return linalg.spsolve(K.tocsr(),nodeLoadList)


def calcStrainT(node, nodeNew,elem):
    strain = np.zeros(elem.shape[0])
    c=0
    for e in elem:
        v1n = nodeNew[e[0]]
        v2n = nodeNew[e[1]]
        v1o = node[e[0]]
        v2o = node[e[1]]
        l = math.dist(v1o,v2o)
        lnew = math.dist(v1n,v2n)
        strain[c] = (lnew-l)/l
        c+=1
    return strain

def calcStress(strain,E):
    return strain*E

def calcElementForces(elem,dispEx,lList,elemAngle,E,I,A):
    axial = np.zeros(elem.shape[0])
    shearForce = np.zeros((elem.shape[0],2))
    bendingMoment = np.zeros((elem.shape[0],2))
    for i,e in enumerate(elem):
        n0 = e[0]
        n1 = e[1]
        elemDisps= np.array(dispEx[[3*e[0],3*e[0]+1,3*e[0]+2,3*e[1],3*e[1]+1,3*e[1]+2]])
        elemDisps = rotateBeam(elemDisps,elemAngle[i])
        l = lList[i]
        sfbm = (1/l*l*l)*E*I*np.array([[12,6*l,-12,6*l],[6*l,4*l*l,-6*l,2*l*l],[-12,-6*l,12,-6*l],[6*l,2*l*l,-6*l,4*l*l]])
        sfbm = sfbm@np.array([elemDisps[[1,2,4,5]]]).T
        axial [i] = A*E/l * (elemDisps[3] - elemDisps[0])
        shearForce[i,0],shearForce[i,1] = -sfbm[0],sfbm[2]
        bendingMoment[i,0],bendingMoment[i,1] = -sfbm[1],sfbm[3]

    return axial,shearForce,bendingMoment

def getDisp(disp,docList,nDofNode,nNodes):
    dispEx = np.zeros(nDofNode*nNodes)
    cont = 0 
    docL = np.flip(np.append(docList,-1))

    for i in disp:
        while (cont == docL[-1]):
            docL = np.delete(docL,-1)
            cont+=1
        dispEx[cont] = i
        cont+=1
    return dispEx

def scale(dispEx,fs):
    dispExmin = min(dispEx)
    scale= (1/abs(dispExmin)/10 * fs) if (abs(dispExmin) != np.nan and dispExmin!= 0) else 1

    return  dispEx * scale

def nodeLoads(nodeLoadList,nodeDofList,elem):
    for i,nd in enumerate(nodeDofList):
        for j,dof in enumerate(nodeDofList[i]):
            c = i*nDofNode+j
            if nodeLoadList[c] == 0 and not dof:
                if(i in (elem)):
                    nodeLoadList[c]+=0.1
    return nodeLoadList


def getTotalDisp(dispEx,nNodes,nDofNode):

    totalDisp = np.zeros(nNodes)
    for x in range(nNodes):
        totalDisp[x] = calcNorm(dispEx[nDofNode*x],dispEx[nDofNode*x+1])
    return totalDisp


def updatePos(node,dispEx,nNodes,nDofNode,dim):
    nodeNew = node.copy()
    for i in range(nNodes):
        for j in range(dim):
            d = dispEx[nDofNode*i+j]
            nodeNew[i,j] += d
    return nodeNew

def scaleNew (node,dispExScaled,nNodes,nDofNode,dim):
    nodeNewScaled = node.copy()
    for i in range(nNodes):
        for j in range(dim):
            nodeNewScaled[i,j] += dispExScaled[nDofNode*i+j]
    return nodeNewScaled

def setlim(n):
    xmin = min(n[:,0])
    xmax = max(n[:,0])
    ymin = min(n[:,1])
    ymax = max(n[:,1])

    xlen = abs(xmax-xmin)
    ylen = abs(ymax-ymin)
    if(xlen>ylen):                
        plt.xlim([xmin-xlen*0.2,xmax+xlen*0.2])
        plt.ylim([ymin-xlen*0.2,ymax+xlen*0.2])
        xlim = -xmin-xlen*0.2+xmax+xlen*0.2
        ylim = -ymin-xlen*0.2+ymax+xlen*0.2
    else:                
        plt.xlim([xmin-ylen*0.2,xmax+ylen*0.2])
        plt.ylim([ymin-ylen*0.2,ymax+ylen*0.2])
        xlim = xmax+xlen*0.2-xmin-xlen*0.2
        if xlim <0.3:
            xlim = 0.3
        
        ylim = ymin-ylen*0.2+ymax+ylen*0.2
        if ylim < 0.3:
            ylim = 0.3
    return xlim,ylim, xmax, xmin, ymax, ymin 

def outputVTK(elem,node,disp,st,order,filename):
    if order == 2:
        stressxx = st.T[[0,3,6]].T
        stressyy = st.T[[1,4,7]].T
        stressxy = st.T[[2,5,8]].T
        
    elif order == 1:
        stressxx = st.T[[0,0,0]].T
        stressyy = st.T[[1,1,1]].T
        stressxy = st.T[[2,2,2]].T
    vonmises = np.sqrt(np.multiply(stressxx,stressxx) - np.multiply(stressxx,stressyy) + np.multiply(stressyy,stressyy) + 3*np.multiply(stressxy,stressxy))
    nstresses = 4
    nodecount = np.zeros((disp.shape[0],nstresses))
    nodalStress = np.zeros((disp.shape[0],nstresses))
    for i,e in enumerate(elem):
        for j,n in enumerate(e[:3]):
            nodecount[n,0]+=1
            nodalStress[n,0]+=stressxx[i,j]
            nodecount[n,1]+=1
            nodalStress[n,1]+=stressyy[i,j]
            nodecount[n,2]+=1
            nodalStress[n,2]+=stressxy[i,j]
            nodecount[n,3]+=1
            nodalStress[n,3]+=vonmises[i,j]

        for j,n in enumerate(e[5:2:-1]):
            nodecount[n,0]+=1
            nodalStress[n,0]+=stressxx[i,j]
            nodecount[n,1]+=1
            nodalStress[n,1]+=stressyy[i,j]
            nodecount[n,2]+=1
            nodalStress[n,2]+=stressxy[i,j]
            nodecount[n,3]+=1
            nodalStress[n,3]+=vonmises[i,j]
            

    for i,n in enumerate(nodalStress):
        try:
            nodalStress[i]/= nodecount[i]
        except:
            nodalStress[i] = 0

    # Criar o dataset VTK
    my_vtk_dataset = vtk.vtkUnstructuredGrid()

    # Adicionar os pontos
    vtk_points = vtk.vtkPoints()
    for id, point in enumerate(node):
        vtk_points.InsertPoint(id, np.append(point,0))
    my_vtk_dataset.SetPoints(vtk_points)

    # Adicionar as células
    if order == 1:
        for element in elem:
            vtk_cell = vtk.vtkTriangle()
            vtk_cell.GetPointIds().SetId(0, element[0])
            vtk_cell.GetPointIds().SetId(1, element[1])
            vtk_cell.GetPointIds().SetId(2, element[2])
            my_vtk_dataset.InsertNextCell(vtk_cell.GetCellType(), vtk_cell.GetPointIds())
    if order == 2:
        # Adicionar as células
        for element in elem:
            vtk_cell = vtk.vtkQuadraticTriangle()
            vtk_cell.GetPointIds().SetId(0, element[0])
            vtk_cell.GetPointIds().SetId(1, element[1])
            vtk_cell.GetPointIds().SetId(2, element[2])
            vtk_cell.GetPointIds().SetId(3, element[5])
            vtk_cell.GetPointIds().SetId(4, element[3])
            vtk_cell.GetPointIds().SetId(5, element[4])
            my_vtk_dataset.InsertNextCell(vtk_cell.GetCellType(), vtk_cell.GetPointIds())

    # Adicionar os deslocamentos nodais como dados de ponto
    disp_array = vtk.vtkDoubleArray()
    disp_array.SetNumberOfComponents(disp.shape[1])
    disp_array.SetNumberOfTuples(disp.shape[0])
    disp_array.SetName('Displacements')
    for id, value in enumerate(disp):
        disp_array.SetTuple(id, value)
    my_vtk_dataset.GetPointData().AddArray(disp_array)

    # Adicionar o campo de tensões como dados de célula
    stress_array = vtk.vtkDoubleArray()
    stress_array.SetNumberOfComponents(nstresses)
    stress_array.SetNumberOfTuples(elem.shape[0])
    stress_array.SetName('Element Stress')
    for id, value in enumerate(elem):
        stress_array.SetTuple(id, [stressxx[id].mean(),stressyy[id].mean(),stressxy[id].mean(),vonmises[id].mean()])
    stress_array.SetComponentName(0, 'Average XX') 
    stress_array.SetComponentName(1, 'Average YY')  
    stress_array.SetComponentName(2, 'Average XY')
    stress_array.SetComponentName(3, 'Von mises')  
    my_vtk_dataset.GetCellData().AddArray(stress_array)

    # Adicionar os valores interpolados como dados de célula
    st_array = vtk.vtkDoubleArray()
    st_array.SetNumberOfComponents(nstresses)
    st_array.SetNumberOfTuples(disp.shape[0])
    st_array.SetName("Nodal Stress")
    for id, value in enumerate(nodalStress):
        st_array.SetTuple(id, value)
    
    st_array.SetComponentName(0, 'XX') 
    st_array.SetComponentName(1, 'YY')  
    st_array.SetComponentName(2, 'XY')
    st_array.SetComponentName(3, 'Von mises')  
    my_vtk_dataset.GetPointData().AddArray(st_array)


    # Salvar o dataset como arquivo VTK
    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetFileName(f"{filename}.vtu")
    writer.SetInputData(my_vtk_dataset)
    writer.Write()


def optruss(config, elem,plot = False, plotFinal = False):

    E,A,mass,dim,node,nodeDofList,nodeLoadList_a,doc = config.unpack()
    global xlim,ylim, xmax, xmin, ymax, ymin 
    global nNodes, nElem, nDofNode, model
    global fig, ax
    nElem = elem.shape[0]
    model = 0
    nNodesElem = 2
    elemAngle = getElemAngle(elem,node,nElem)
    for i in range(node.shape[0]):
        if(i not in elem):
            if(nodeLoadList_a[i*nDofNode] != 0 or nodeLoadList_a[i*nDofNode+1]!= 0):
                # if plotFinal:
                #     plt.annotate('erro 1', xy = (1,1))
                #     plt.show()
                return False , False, False, False

    #Boundary Conditions

    K, lList = globalstiffAssemble(node,elem,elemAngle,nElem,nNodes,nNodesElem,nDofNode,model,E,A,-1)

    for i,e in enumerate(elem):
        gk = lList[i]*A*mass
        nodeLoadList_a[nDofNode*e[0]+1] -= gk*0.5
        nodeLoadList_a[nDofNode*e[1]+1] -= gk*0.5

    freeNodeList = removeFreeNodes(node,elem,nodeLoadList_a)
    K,docList,nodeLoadList = assignBC(K,doc,nodeLoadList_a,nodeDofList, nNodes, nDofNode,freeNodeList)

    try:
        nll = nodeLoadList.copy()
        nll[::2]+=0.0001
        solve(K,nll)
        nll2 = nodeLoadList.copy()
        nll2[1::2]-=0.0001
        disp = solve(K,nll2)
        disp = solve(K,nodeLoadList)
        dispEx = getDisp(disp,docList,nDofNode,nNodes)
    except:
        # if plotFinal:
        #     plt.annotate('erro 2', xy = (1,1))
        #     plt.show()
        return False , False, False, False

    totalDisp = getTotalDisp(dispEx,nNodes,nDofNode)
    nodeNew = updatePos(node,dispEx,nNodes,nDofNode,dim)
    strain = calcStrainT(node, nodeNew,elem)
    stress = calcStress(strain,E)
    if(plot):
        fig, ax = plt.subplots()
        plotNewmann(nodeLoadList,node,docList)
        plotDirichlet(nodeDofList,node,elemAngle)
        plotElem(elem,node)
        plt.show()
        if(plotFinal):
            dispExScaled = scale(dispEx,5) 
            nodeNewScaled = scaleNew(node,dispExScaled,nNodes,nDofNode,dim)
            fig, ax = plt.subplots()
            plotElemStBar(elem,nodeNewScaled,strain,E)
            setlim(nodeNewScaled)
            plt.title("Stresses")
            plt.show()
    return dispEx,totalDisp,stress,True

def readCppOut(filename):
    
    disp = pd.read_csv(filename+'displacements.csv', sep=';',header=None).to_numpy()
    stress = pd.read_csv(filename+'stresses.csv', sep=';',header=None).to_numpy()
    strain = pd.read_csv(filename+'strains.csv', sep=';',header=None).to_numpy()
    return disp[:,0],stress,strain


##############################
#FEM 
def fem(inputOpt,inputBC,outputOpt,plotOpt,E,A,I,materialClasses,t,fs,mod,dim,fileTriangle,argTriangle,node,elem,fixedDof,fixedNode,coordsFixedDof,coordsFixedNode,loadDof,loadcoord,surfaceForces,bodyForces,filename):
    global xlim,ylim, xmax, xmin, ymax, ymin 
    global fig, ax
    global nNodes, nElem, nDofNode, model
    
    model = mod
    
    fig, ax = plt.subplots()
    triangle = inputOpt['Input with Triangle lib']

    nNodes = node.shape[0]
    if(inputOpt['Element Binding by Proximity']):
        nearList,nBinds = findNear(node)
        elem = bindnear(nearList)

    nElem = elem.shape[0]

    #######
    #Pre Process


    #DoF per Node
    dofNodeList = [
        2, #Truss: 2 DoF per node
        3, #Bernoulli Euler: 3 DoF per node
        2, #2d Linear Triangles 
        2, #2d Quadratic Triangles
        2, #2d Cubic Triangles
    ]

    nDofNode = dofNodeList[model]

    #Nodes per Element
    nodeElemList = [
        2, #Truss: 2 nodes per Element
        2, #Bernoulli Euler: 2 nodes per Element
        3, #2d Linear Triangles 
        6, #2d Quadratic Triangles
        10 #2d Cubic Triangles
    ]

    nNodesElem = nodeElemList[model]

    if(triangle):
        if model == 3:
            argTriangle+= ' -o2 '
        command = './input/triangle '+ argTriangle +' ./input/'+fileTriangle+'.poly'
        proc = subprocess.Popen(command)
        proc.wait()
        file = './input/'+fileTriangle
        if model <=1:
            nod, nNodes = readNode(file)
            node = nod[:,1:3]
            elem, nElem, nNodesElem = readElemBar(file)
        if model >= 2:
            nod, nNodes = readNode(file)
            node = nod[:,1:3]
            elem, nElem, _ = readElem(file)

    
    xlim,ylim, xmax, xmin, ymax, ymin = setlim(node)

    nNodes = node.shape[0]
    

    #########
    #Boundary Conditions
        
    doc = 0
    nodeDofList = np.zeros((nNodes,nDofNode),dtype =bool)
    nodeLoadList = np.zeros((nNodes*nDofNode), dtype = np.double)

    for i in loadDof:
            nodeLoadList[nDofNode*i[0]+i[1]] = i[2]

    if inputBC['Distributed Load on Boundary']:
        surfaceForces = assignloadcoord(node,elem,loadcoord,surfaceForces)

    for i in fixedNode:
        for j in range(nodeDofList[i].size):
            if (not nodeDofList[i,j]):
                nodeDofList[i,j] = 1
                doc +=1
            
    for i in fixedDof:
        if (not nodeDofList[i[0],i[1]]):
            nodeDofList[i[0],i[1]] = 1
            doc +=1
        

    if (inputBC['fixDofCoord']):
        for c in coordsFixedDof:
            for i,n in enumerate(node):
                if abs(n[c[1]]-c[0]) < 0.001:
                    if(not nodeDofList[i,c[2]]):
                            nodeDofList[i,c[2]] = True
                            doc +=1 
    if (inputBC['fixNodeCoord']):    
        for c in coordsFixedNode:
            for i,n in enumerate(node):
                if abs(n[c[1]]-c[0]) < 0.001:
                    nodeDofList[i,0],nodeDofList[i,1] = True,True
                    doc +=2

    if outputOpt['plot Nodes']:
        xlim,ylim, xmax, xmin, ymax, ymin  = setlim(node)
        plId,plCd = 0,0
        if(plotOpt['Plot Nodes with node id']): plId = 1
        if(plotOpt['Plot Nodes with node coords']): plCd = 1            
        plotNode(node,plotId = plId, plotCoord=plCd)
        plt.show() 
        fig, ax = plt.subplots()
    
    elemAngle = getElemAngle(elem,node,nElem)
    if outputOpt['plot Elements']:
        xlim,ylim, xmax, xmin, ymax, ymin  = setlim(node)
        if plotOpt['Plot Elements with Constraints']:
            plotDirichlet(nodeDofList,node,elemAngle)
        if plotOpt['Plot Elements with Loads']:
            plotNewmann(nodeLoadList,node)
        plotElem(elem,node,lbName = plotOpt['Plot Elements with element id'])
        if(model == 2 or model == 3):
            if plotOpt['Plot Elements with Edges id (Triangles)']:
                plotEdges(node,elem)
        plt.show()
        fig, ax = plt.subplots()
    
    if(model == 2):
        #processing by c++ with triangular 2d elements
        elemProperties = np.zeros((nElem,3),dtype=int)
        elemProperties[:,0] = 0
        elemProperties[:,1] = 0
        elemProperties[:,2] = 3
        constraints = np.array([-1,-1],dtype = int)
        for i,a in enumerate(nodeDofList):
            if(i not in constraints):
                if a[0] == 1 and a[1] == 1:
                    constraints = np.vstack((constraints,[i,-1]))
                elif a[0] == 1 and a[1] == 0:
                    constraints = np.vstack((constraints,[i,0]))
                elif a[0] == 0 and a[1] == 1:
                    constraints = np.vstack((constraints,[i,1]))
        constraints = constraints[1:]
        configFEM.configFEM(node,elem,elemProperties,materialClasses,constraints,loadDof,surfaceForces,bodyForces,t,filename)
        command = './femexec '+filename
        if outputOpt['Solve']:
            proc = subprocess.Popen(command)
            proc.wait()

    if(model == 3):
        #processing by c++ with triangular 2d elements
        elemProperties = np.zeros((nElem,3),dtype=int)
        elemProperties[:,0] = 0
        elemProperties[:,1] = 1
        elemProperties[:,2] = 6
        constraints = np.array([-1,-1],dtype = int)
        for i,a in enumerate(nodeDofList):
            #MUDAR 
            if(i not in constraints):
                if a[0] == 1 and a[1] == 1:
                    constraints = np.vstack((constraints,[i,-1]))
                elif a[0] == 1 and a[1] == 0:
                    constraints = np.vstack((constraints,[i,0]))
                elif a[0] == 0 and a[1] == 1:
                    constraints = np.vstack((constraints,[i,1]))

        constraints = constraints[1:]
        configFEM.configFEM(node,elem,elemProperties,materialClasses,constraints,loadDof,surfaceForces,bodyForces,t,filename)
        command = './femexec '+filename
        if outputOpt['Solve']:
            proc = subprocess.Popen(command)
            proc.wait()


    if outputOpt['Solve']:
        if(model<2):
            K,lList = globalstiffAssemble(node,elem,elemAngle,nElem,nNodes,nNodesElem,nDofNode,model,E,A,I)
            freeNodeList = removeFreeNodes(node,elem,nodeLoadList)
            K,docList,nodeLoadList = assignBC(K,doc,nodeLoadList,nodeDofList, nNodes, nDofNode,freeNodeList)
            disp = solve(K,nodeLoadList)
            dispEx = getDisp(disp,docList,nDofNode,nNodes)
        else:
            dispEx,stress,strain = readCppOut(filename)
            docList = []

        dispExScaled = scale(dispEx,fs)
        totalDisp = getTotalDisp(dispEx,nNodes,nDofNode)        
        nodeNew = updatePos(node,dispEx,nNodes,nDofNode,dim)      
        nodeNewScaled = nodeNew
        if plotOpt['Plot Post Processing with scaled displacements']:
            nodeNewScaled = scaleNew(node,dispExScaled,nNodes,nDofNode,dim)  

        if model == 0:
            strain = calcStrainT(node, nodeNew,elem)
            stress = calcStress(strain,E)

        edBool = plotOpt['Plot Element Edges']
        
        ############
        #strain/stress

        if model == 0:
            if outputOpt['plot Stress/Strain']:
                if plotOpt['Plot Post Processing with Loads']:
                    plotNewmann(nodeLoadList,nodeNewScaled,docList)
                if plotOpt['Plot Post Processing with Constraints']:
                    plotDirichlet(nodeDofList,node,elemAngle)
                if plotOpt['Plot Post Processing with Undeformed configuration']:
                    plotElem(elem,node,False,0.05)
                plotElemStBar(elem,nodeNewScaled,strain,E)
                plt.show()
                fig, ax = plt.subplots()
        
        if model == 1:
            
            axial,shear,bmom = calcElementForces(elem,dispEx,lList,elemAngle,E,I,A)
            if outputOpt['plot Axial Force']:
                if plotOpt['Plot Post Processing with Loads']:
                    plotNewmann(nodeLoadList,nodeNewScaled,docList)
                if plotOpt['Plot Post Processing with Constraints']:
                    plotDirichlet(nodeDofList,node,elemAngle)
                if plotOpt['Plot Post Processing with Undeformed configuration']:
                    plotElem(elem,node,False,0.05)
                plotElemBeam(elem,nodeNewScaled,np.column_stack((axial,axial)))
                plt.title("Axial Forces")
                plt.show()
                fig, ax = plt.subplots()
            
            if outputOpt['plot Shear Force']:
                if plotOpt['Plot Post Processing with Loads']:
                    plotNewmann(nodeLoadList,nodeNewScaled,docList)
                if plotOpt['Plot Post Processing with Constraints']:
                    plotDirichlet(nodeDofList,node,elemAngle)
                if plotOpt['Plot Post Processing with Undeformed configuration']:
                    plotElem(elem,node,False,0.05)
                plotElemBeam(elem,nodeNewScaled,shear)
                plt.title("Shear Forces")
                plt.show()
                fig, ax = plt.subplots()
            
            if outputOpt['plot Bending Moment']:
                if plotOpt['Plot Post Processing with Loads']:
                    plotNewmann(nodeLoadList,nodeNewScaled,docList)
                if plotOpt['Plot Post Processing with Constraints']:
                    plotDirichlet(nodeDofList,node,elemAngle)
                if plotOpt['Plot Post Processing with Undeformed configuration']:
                    plotElem(elem,node,False,0.05)
                plotElemBeam(elem,nodeNewScaled,bmom)
                plt.title("Bending Moments")
                plt.show()
                fig, ax = plt.subplots()
        
        elif model == 2:
            if outputOpt['plot Stress XX']:
                if plotOpt['Plot Post Processing with Loads']:
                    plotNewmann(nodeLoadList,nodeNewScaled,docList)
                if plotOpt['Plot Post Processing with Constraints']:
                    plotDirichlet(nodeDofList,node,elemAngle)
                if plotOpt['Plot Post Processing with Undeformed configuration']:
                    plotElem(elem,node,False,0.01)

                plotElemStTR3(elem,nodeNewScaled,stress,0,edge = edBool)
                setlim(nodeNewScaled)
                plt.title("Strain/Stress xx")
                plt.show()
                fig, ax = plt.subplots()

            if outputOpt['plot Stress YY']:
                if plotOpt['Plot Post Processing with Loads']:
                    plotNewmann(nodeLoadList,nodeNewScaled,docList)
                if plotOpt['Plot Post Processing with Constraints']:
                    plotDirichlet(nodeDofList,node,elemAngle)
                if plotOpt['Plot Post Processing with Undeformed configuration']:
                    plotElem(elem,node,False,0.01)

                plotElemStTR3(elem,nodeNewScaled,stress,1,edge = edBool)
                setlim(nodeNewScaled)
                plt.title("Strain/Stress yy")
                plt.show()
                fig, ax = plt.subplots()

            if outputOpt['plot Stress XY']:
                if plotOpt['Plot Post Processing with Loads']:
                    plotNewmann(nodeLoadList,nodeNewScaled,docList)
                if plotOpt['Plot Post Processing with Constraints']:
                    plotDirichlet(nodeDofList,node,elemAngle)
                if plotOpt['Plot Post Processing with Undeformed configuration']:
                    plotElem(elem,node,False,0.01)

                plotElemStTR3(elem,nodeNewScaled,stress,2,edge = edBool)
                setlim(nodeNewScaled)
                plt.title("Strain/Stress xy")
                plt.show()
                fig, ax = plt.subplots()

        elif model == 3:
            if outputOpt['plot Stress XX']:
                if plotOpt['Plot Post Processing with Loads']:
                    plotNewmann(nodeLoadList,nodeNewScaled,docList)
                if plotOpt['Plot Post Processing with Constraints']:
                    plotDirichlet(nodeDofList,node,elemAngle)
                if plotOpt['Plot Post Processing with Undeformed configuration']:
                    plotElem(elem,node,False,0.01)

                plotElemStTR6(elem,nodeNewScaled,stress,0,edge = edBool)
                setlim(nodeNewScaled)
                plt.title("Stress XX")
                plt.show()
                fig, ax = plt.subplots()

            if outputOpt['plot Stress YY']:
                if plotOpt['Plot Post Processing with Loads']:
                    plotNewmann(nodeLoadList,nodeNewScaled,docList)
                if plotOpt['Plot Post Processing with Constraints']:
                    plotDirichlet(nodeDofList,node,elemAngle)
                if plotOpt['Plot Post Processing with Undeformed configuration']:
                    plotElem(elem,node,False,0.01)

                plotElemStTR6(elem,nodeNewScaled,stress,1,edge = edBool)
                setlim(nodeNewScaled)
                plt.title("Stress YY")
                plt.show()
                fig, ax = plt.subplots()

            if outputOpt['plot Stress XY']:
                if plotOpt['Plot Post Processing with Loads']:
                    plotNewmann(nodeLoadList,nodeNewScaled,docList)
                if plotOpt['Plot Post Processing with Constraints']:
                    plotDirichlet(nodeDofList,node,elemAngle)
                if plotOpt['Plot Post Processing with Undeformed configuration']:
                    plotElem(elem,node,False,0.01)

                plotElemStTR6(elem,nodeNewScaled,stress,2,edge = edBool)
                setlim(nodeNewScaled)
                plt.title("Stress XY")
                plt.show()
                fig, ax = plt.subplots()



        ##############
        #Displacements
        if outputOpt['plot Total Displacements']:
            if plotOpt['Plot Post Processing with Loads']:
                plotNewmann(nodeLoadList,nodeNewScaled,docList)
            if plotOpt['Plot Post Processing with Constraints']:
                plotDirichlet(nodeDofList,node,elemAngle)
            if plotOpt['Plot Post Processing with Undeformed configuration']:
                plotElem(elem,node,False,0.05)
            if model<2:
                plotElemDisp(elem,nodeNewScaled,totalDisp)
            if model == 2:
                plotElemDispTR3(elem,nodeNewScaled,totalDisp,edge = edBool)
            if model == 3:
                plotElemDispTR6(elem,nodeNewScaled,totalDisp,edge = edBool)
            setlim(nodeNewScaled)
            plt.title("Displacements Magnitude")
            plt.show()
            fig, ax = plt.subplots()
        if(model != 1):
            dirDisp = np.empty((2,nNodes),dtype = float)
            dirDisp[0] = dispEx[::2]
            dirDisp[1] = dispEx[1::2]
        else:
            dirDisp = np.empty((3,nNodes),dtype = float)
            dirDisp[0] = dispEx[::3]
            dirDisp[1] = dispEx[1::3]
        if outputOpt['plot X Displacement']:    
            if plotOpt['Plot Post Processing with Loads']:
                plotNewmann(nodeLoadList,nodeNewScaled,docList)
            if plotOpt['Plot Post Processing with Constraints']:
                plotDirichlet(nodeDofList,node,elemAngle)
            if plotOpt['Plot Post Processing with Undeformed configuration']:
                plotElem(elem,node,False,0.05)
            if model<2:
                plotElemDisp(elem,nodeNewScaled,totalDisp)
            if model == 2:
                plotElemDispTR3(elem,nodeNewScaled,dirDisp[0],edge = edBool)
            if model == 3:
                plotElemDispTR6(elem,nodeNewScaled,dirDisp[0],edge = edBool)
            setlim(nodeNewScaled)
            plt.title("Displacements X")
            plt.show()
            fig, ax = plt.subplots()

        if outputOpt['plot Y Displacement']:
            if plotOpt['Plot Post Processing with Loads']:
                plotNewmann(nodeLoadList,nodeNewScaled,docList)
            if plotOpt['Plot Post Processing with Constraints']:
                plotDirichlet(nodeDofList,node,elemAngle)
            if plotOpt['Plot Post Processing with Undeformed configuration']:
                plotElem(elem,node,False,0.05)
            if model<2:
                plotElemDisp(elem,nodeNewScaled,totalDisp)
            if model == 2:
                plotElemDispTR3(elem,nodeNewScaled,dirDisp[1],edge = edBool)
            if model == 3:
                plotElemDispTR6(elem,nodeNewScaled,dirDisp[1],edge = edBool)
            setlim(nodeNewScaled)
            plt.title("Displacements Y")
            plt.show()
        if model>1 and outputOpt['Output vtk file']:

            outputVTK(elem,node,dirDisp.T,stress,model-1,filename)

        if outputOpt['Output csv with Nodal Displacement']:
            df = pd.DataFrame([])
            for i in range(nDofNode):
                df[f'u[{i}]'] = (dispEx[i::nDofNode])
            df.to_csv("./output/TotalDisplacement.csv",sep=';')        
            maxdisplacement = abs(df).idxmax().to_numpy()
            # print("Max Displacements:")
            # for i in range(nDofNode):
            #     c = mean(node[maxdisplacement[i],0],node[maxdisplacement[i],1])
            #     print(f'{i}: node {maxdisplacement[i]} : ({c[0]:.2f},{c[1]:.2f}) = {df.iloc[maxdisplacement[i],i]}')
                
        if outputOpt['Output csv with Element Stress']:
            df = pd.DataFrame([])
            if (model == 0):
                df[f's[{i}]'] = stress
                df.to_csv("./output/Stress.csv",sep=';')
            if(model == 1):
                for i in range(nDofNode):
                    df[f'u[{i}]'] = (stress[i::nDofNode])
                df.to_csv("./output/Stress.csv",sep=';')
            if(model == 2):
                df = pd.DataFrame(stress)
            if(model == 3):
                df = pd.DataFrame(stress)
            maxstress = df.idxmax().to_numpy()
            minstress = df.idxmin().to_numpy()
            print("Max/Min Stresses:")
            if (model == 0):
                c = mean(node[elem[maxstress[0],0]],node[elem[maxstress[0],1]])
                cmin = mean(node[elem[minstress[0],0]],node[elem[minstress[0],1]])
                print(f'{i} - Max: el. {maxstress[0]} : ({c[0]:.2f},{c[1]:.2f}) = {df.iloc[maxstress[0],0]}')
                print(f'{i} - Min: el. {minstress[0]} : ({cmin[0]:.2f},{cmin[1]:.2f}) = {df.iloc[minstress[0],0]}')
            elif(model == 1):
                for i in range(nDofNode):
                    c = mean(node[elem[maxstress[i],0]],node[elem[maxstress[i],1]])
                    cmin = mean(node[elem[minstress[i],0]],node[elem[minstress[i],1]])
                    print(f'{i} - Max: el. {maxstress[i]} : ({c[0]:.2f},{c[1]:.2f}) = {df.iloc[maxstress[i],i]}')
                    print(f'{i} - Min: el. {minstress[i]} : ({cmin[0]:.2f},{cmin[1]:.2f}) = {df.iloc[minstress[i],i]}')
            elif(model == 2):
                label = ['XX','YY','XY']
                for i in range(3):
                    c = mean(node[elem[maxstress[i],0]],node[elem[maxstress[i],1]])
                    cmin = mean(node[elem[minstress[i],0]],node[elem[minstress[i],1]])
                    print(f'{i} - Max Stress {label[i]}: el. {maxstress[i]} : ({c[0]:.2f},{c[1]:.2f}) = {df.iloc[maxstress[i],i]}')
                    print(f'{i} - Min Stress {label[i]}: el. {minstress[i]} : ({cmin[0]:.2f},{cmin[1]:.2f}) = {df.iloc[minstress[i],i]}')
            elif(model == 3):
                label = ['XX','YY','XY']
                for i in range(3):
                    c = mean(node[elem[maxstress[i],0]],node[elem[maxstress[i],1]])
                    cmin = mean(node[elem[minstress[i],0]],node[elem[minstress[i],1]])
                    print(f'{i} - Max Stress {label[i]}: el. {max(maxstress[i],maxstress[i+3],maxstress[i+6])} : ({c[0]:.2f},{c[1]:.2f}) = {df.iloc[maxstress[i],i]}')
                    print(f'{i} - Min Stress {label[i]}: el. {min(minstress[i],minstress[i+3],minstress[i+6])} : ({cmin[0]:.2f},{cmin[1]:.2f}) = {df.iloc[minstress[i],i]}')
                
        if outputOpt['Output csv with Element Strain']:
            df = pd.DataFrame([])
            if (model == 0):
                df[f's[{i}]'] = strain
                df.to_csv("./output/Strain.csv",sep=';')
            if(model == 1):
                for i in range(nDofNode):
                    df[f'u[{i}]'] = (strain[i::nDofNode])
                df.to_csv("./output/Strain.csv",sep=';')
            if(model > 1):
                df = pd.DataFrame(strain)
            maxstrain = df.idxmax().to_numpy()
            minstrain = df.idxmin().to_numpy()
            print("Max/Min Strains:")
            if (model == 0):
                c = mean(node[elem[maxstrain[0],0]],node[elem[maxstrain[0],1]])
                cmin = mean(node[elem[minstrain[0],0]],node[elem[minstrain[0],1]])
                print(f'{i} - Max: el. {maxstrain[0]} : ({c[0]:.2f},{c[1]:.2f}) = {df.iloc[maxstrain[0],0]}')
                print(f'{i} - Min: el. {minstrain[0]} : ({cmin[0]:.2f},{cmin[1]:.2f}) = {df.iloc[minstrain[0],0]}')
            elif(model == 1):
                for i in range(nDofNode):
                    c = mean(node[elem[maxstrain[i],0]],node[elem[maxstrain[i],1]])
                    cmin = mean(node[elem[minstrain[i],0]],node[elem[minstrain[i],1]])
                    print(f'{i} - Max: el. {maxstrain[i]} : ({c[0]:.2f},{c[1]:.2f}) = {df.iloc[maxstrain[i],i]}')
                    print(f'{i} - Min: el. {minstrain[i]} : ({cmin[0]:.2f},{cmin[1]:.2f}) = {df.iloc[minstrain[i],i]}')
            
            elif(model == 2):
                label = ['XX','YY','XY']
                for i in range(3):
                    c = mean(node[elem[maxstrain[i],0]],node[elem[maxstrain[i],1]])
                    cmin = mean(node[elem[minstrain[i],0]],node[elem[minstrain[i],1]])
                    print(f'{i} - Max strain {label[i]}: el. {maxstrain[i]} : ({c[0]:.2f},{c[1]:.2f}) = {df.iloc[maxstrain[i],i]}')
                    print(f'{i} - Min strain {label[i]}: el. {minstrain[i]} : ({cmin[0]:.2f},{cmin[1]:.2f}) = {df.iloc[minstrain[i],i]}')
            elif(model == 3):
                label = ['XX','YY','XY']
                for i in range(3):
                    c = mean(node[elem[maxstrain[i],0]],node[elem[maxstrain[i],1]])
                    cmin = mean(node[elem[minstrain[i],0]],node[elem[minstrain[i],1]])
                    print(f'{i} - Max strain {label[i]}: el. {max(maxstrain[i],maxstrain[i+3],maxstrain[i+6])} : ({c[0]:.2f},{c[1]:.2f}) = {df.iloc[maxstrain[i],i]}')
                    print(f'{i} - Min strain {label[i]}: el. {min(minstrain[i],minstrain[i+3],minstrain[i+6])} : ({cmin[0]:.2f},{cmin[1]:.2f}) = {df.iloc[minstrain[i],i]}')

