import pandas as pd
import numpy as np

file = 'exemplo1.csv'



df = pd.read_csv('input/'+file)

df = df.reset_index()
dfBc = df.iloc[:,:-4:-1]
df = df.iloc[:,2:-3]

df.columns=['x2','y2','x1','y1']
dfBc = dfBc.sum(axis=1)

df['bc'] = dfBc.rank(method='dense',ascending = False).astype(int)

df = df.sort_values(by=['bc','x1','y1']).reset_index().iloc[:,1:]

node = np.array([[0,0,0]])
for ix,row in df.iterrows():
    a = np.array([row['x1'],row['y1'],row['bc']])
    node = np.vstack((node,a))
node = node[1:]
print(node)


elem = np.array([[0,0,0]],dtype=int)
for ix,row in df.iterrows():
    e = np.array([0,0,0],dtype=int)
    for j,nd in enumerate(node):
        if(abs(row['x1'] - nd[0]) < 10E-6 and abs(row['y1'] - nd[1])< 10E-6):
            e[0] = j
        if(abs(row['x2'] - nd[0]) < 10E-6 and abs(row['y2'] - nd[1])< 10E-6):
            e[1] = j
    e[2] = row['bc']
    elem = np.vstack((elem,e))
elem = elem[1:]
print(elem)

nHoles = max(df['bc']) -1 


holeList = np.zeros((nHoles,2))
for i in range(nHoles):
    h = i+1
    nodeHole = node[node[:,-1] == h+1]
    x = np.sum(nodeHole, axis=0)
    if(nodeHole.shape[0]!=0):
        x = x/nodeHole.shape[0]
    holeList[i] = x[:-1]

for h in holeList:
    print(h)

firstLine = [node.shape[0],2,0,1]
elemLine = [elem.shape[0],1]



f = open('input/'+file[:-4]+'.poly','w')
f.write(f'{node.shape[0]} 2 0 1\n')
for i,row in enumerate(node):
    f.write(f'{i+1} {row[0]:.4f} {row[1]:.4f} {int(row[2])}\n')
f.write(f'{elem.shape[0]} 1\n')
for i,row in enumerate(elem):
    f.write(f'{i+1} {int(row[0]+1)} {int(row[1]+1)} {int(row[2])}\n')
f.write(f'{nHoles}\n')
for i,row in enumerate(holeList):
    f.write(f'{i+1} {row[0]:.3f} {row[1]:.3f}\n')
f.close()