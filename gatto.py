import numpy as np
import pyfem
import pandas as pd
import random
import matplotlib.pyplot as plt
import time
#Input Data
#Material / Cross Section Geometry

E = 2*10E11  # Young Modulus (Pa)
A = 0.1   # Cross Section Area (M^2)
I = 10E-4  # Area Moment of Inertia
M = 0    # Mass of material kg/m3

###############
# Yield point in kN/m

# 1 kN/cm2 = 10 MPa = 10000 kN/m2
fyd = 250000

##########################
# Constrained Displacements(Dirichlet)
# fixed DoF: 
# To fix dof: [node,dof] on fixedDof list

fixedDof = [
    [0,0],
    [0,1],
]

# fixed Node 
# To fix node: node id on fixedNode list

fixedNode = [8]

# Prescribed Loads (Newmann)
# Load on DoF
# To prescribe load: [node,dof,load value]
# On Truss - Fx; 1 - Fy

loadDof = [
    [3,1,-100]
    ]



# Integer Number of Configurations per Generation
popsize = 2**8

# True or False - Plot Every Generations' best Config
plotEveryGeneration = False

# True of False - Show Execution Time on Every Generation
showExecTimeEveryGeneration = False

 # Integer - Quantity of Leaders to be assigned onto next generation without crossover or mutation
nLeaders = 2

# float [0,1] -  a or a,b  - random(a%,b%) percentage of mutation with every Config at the generation excluding Leaders // if input = a, nMutations = random(2,a%)
pcMutation =0.1,0.3  

# float [0,1] - Preference - preference of selecting the best configurations to generate others by crossing over 
preference = 0.5

# Integer Number of Generations, if empty nGenerations = quantity of possible Binds divided by 2
nGenerations = 25

# True or False, True if you want to refine the previous optimal config. Need to assign configToRefine
refine = 0

#Output of previous optimal config as an Integer to be converted to Binary string
configToRefine = 152258728

# Integer - Range of neighbors to bind. Neighboor = Adjacent node. Works well with 1!
nNeighs = 1 

# True or False - True if you want to input node configuration with Txt File
nodeInputWithTxt = False
txtfile = 'file'

# True or False - True if you want to input a initial element configuration 
inputElements = 0




node = np.array([[0,0],[0,1]],dtype = np.double)

for i in range(1,5): 
    node = np.vstack((node,[i,0]))
    node = np.vstack((node,[i,1]))
    # node = np.vstack((node,[i,2]))

elements = np.array([[0,3]])

for i in [0,1,3,4,6,7,10,12,13,16]:
    elements = np.vstack((elements,[i,i+3]))


for i in [9,12,15]:
    elements = np.vstack((elements,[i,i+4]))

for i in [1,4,7,10]:
    elements = np.vstack((elements,[i,i+2]))

for i in [3,6,9,12,15]:
    elements = np.vstack((elements,[i,i+1]))


class cfg:
    def __init__(self, E,A,M,dim,node,fixedDof,fixedNode,loadDof,nearList,nBinds,plotEveryGeneration,showExecTimeEveryGeneration,nLeaders,pcMutation,preference):
        
        nodeDofList,nodeLoadList,doc = pyfem.getBC(fixedDof,fixedNode,loadDof)
        self.E = E
        self.A = A
        self.M = M
        self.dim = dim
        self.node = node
        self.nodeDofList = nodeDofList
        self.nodeLoadList = nodeLoadList
        self.doc = doc
        self.nearList = nearList
        self.nBinds = nBinds
        self.plotEveryGeneration = plotEveryGeneration
        self.showExecTimeEveryGeneration = showExecTimeEveryGeneration
        self.nLeaders = nLeaders
        self.pcMutation = pcMutation
        self.preference = preference
    def unpack(self):
        return self.E,self.A,self.M,self.dim,self.node,self.nodeDofList,self.nodeLoadList,self.doc
    
def bits(n):
    while n:
        b = n & (~n+1)
        yield b
        n ^= b

def getIndByElem(elements,nearList,nBinds):
    ind = np.zeros((nBinds),dtype=bool)
    cont = 0
    for i,elist in enumerate(nearList):
        for e in elist:
            if([i,e] in elements.tolist()):
                ind[cont] = True
            cont+=1
    return ind


def mutation(ind,s):
    position = random.randint(0,s-1)
    ind[position] = not ind[position]
    return ind

def spCrossOver(ind1,ind2,s):
    index = random.randint(0,s-1)
    return np.append(ind1[:index] , ind2[index:]), np.append(ind2[:index] , ind1[index:])

def dpCrossOver(ind1,ind2,s):
    index = random.randint(0,(s-1)//2)
    index2 = random.randint((s-1)//2,s-1)
    return np.append(np.append(ind1[:index] , ind2[index:index2]), ind1[index2:]), np.append(np.append(ind2[:index] , ind1[index:index2]), ind2[index2:])


def popFilterViable(popProps):
    return  popProps[popProps[:,-1] == 1]

def getPosInList(n1,n2,nearList):
    c=0
    for k in range(len(nearList)):
        for w in range(len(nearList[k])):
            if(k == n1 and nearList[n1][w] == n2):
                return c
            c+=1

def removeZeroStress(elem,stress,nodeDofList,nearList,ind):
    refined = ind.copy()
    for j,e in enumerate(elem):
        if abs(stress[j]) < 2:
            n1 = elem[j,0]
            n2 = elem[j,1]
            # if(not all(nodeDofList[n1]) and not all(nodeDofList[n2])):
            posInList = getPosInList(n1,n2,nearList)
            refined[posInList] = False
    return refined

def parsePopProps(popProps):

    popProps = popFilterViable(popProps)
    popProps = popProps[popProps[:, 1].argsort()]

    return popProps

def iterateGeneration(pop,popsize,configOpt):

    acumTime = 0
    popProps = np.zeros((popsize,7))
    for i in range(popsize):
        popProps[i,0] = i
        startTime2 = time.time()
        elem = pyfem.getElemBin(pop[i],configOpt.nearList)

        dispEx,totalDisp,stress,success = pyfem.optruss(configOpt,elem)
        if (success == False):
            popProps[i,1:] = np.nan
            continue

        stopTime2 = time.time()
        acumTime +=stopTime2-startTime2
        obj = pyfem.getObj(elem,node,2)
        
        ###########
        # Stress

        maxstress = np.amax(stress)
        minstress = np.amin(stress)

        popProps[i,2] = minstress
        popProps[i,3] = maxstress

        minstressId = np.argmin(stress)
        lminStress = pyfem.getL(2,elem[minstressId],configOpt.node)
        
        pcrit = np.pi*np.pi*E*I/lminStress/lminStress
        if(minstress==0):
            ck = 0
        else:
            ck = (-minstress*A)/pcrit

        popProps[i,4] = ck
        ###########
        # Adding max stress contribution to be a decider for equal objective functions by volume
        obj += max(-popProps[i,1],popProps[i,2])*0.000000001
        
        ##########
        # Displacement

        popProps[i,5] = np.amax(totalDisp)


        #Constraint Verification
        if (popProps[i,5]<1 and max(maxstress,-minstress) < fyd and abs(max(maxstress,-minstress)>0.1)):
            popProps[i,6] = 1

        else: popProps[i,6] = 0
        popProps[i,1] = obj


    popProps = parsePopProps(popProps)
    if popProps.shape[0]==0:
        return popProps, pop[0], acumTime  
    elem = pyfem.getElemBin(pop[int(popProps[0,0])],configOpt.nearList)
    dispEx,totalDisp,stress,success = pyfem.optruss(configOpt,elem,configOpt.plotEveryGeneration)
    indRefined = removeZeroStress(elem,stress,configOpt.nodeDofList,configOpt.nearList,pop[int(popProps[0,0])])
    return popProps, indRefined, acumTime   


def getOdds(popProps,preference):
    ncands = popProps.shape[0]
    selectionOdd = np.zeros(ncands)
    sum=0
    for i in range(ncands):
        n = 2/(popProps[i,1])/(i+ncands*2*preference)
        sum+=n
        selectionOdd[i] = sum
    return selectionOdd,sum

def getCands(selectionOdd,sum):
    n1,n2 = random.random()*sum,random.random()*sum
    return np.argmax(selectionOdd>n1),np.argmax(selectionOdd>n2)

def plotOccurs(popProps,preference):
    selectionOdd,sum = getOdds(popProps,preference)
    ncands = popProps.shape[0]
    occur = np.zeros(ncands)
    for k in range(100000):
        n1,n2 = getCands(selectionOdd,sum)
        occur[n1]+=1
        occur[n2]+=1

    print('Selection Odds: ',selectionOdd)
    print('Occurencies: ',occur)
    plt.plot(np.arange(ncands),occur,label = f'Pref:{preference}')
    plt.legend()
def assignNewPopBegin(popProps,pop,indRefined,n):
    
    newPop = np.zeros((popsize,nBinds),dtype = bool)
    idLeader = int(popProps[0,0])
    newPop[0] = pop[idLeader]
    for i in range (1,n):
        if popProps.shape[0]<=i:
            newPop[i] = mutation(pop[idLeader],pop[idLeader].shape[0])
        else:
            id = int(popProps[i,0])    
            newPop[i] = pop[id]

    newPop[n],newPop[n+1] = spCrossOver(newPop[0],newPop[1],nBinds)
    newPop[-1] = indRefined
    return newPop,n+2

def fillNewPop(pop,popProps,indRefined,n,pcMutation,preference):

    newPop,cont = assignNewPopBegin(popProps,pop,indRefined,n)    
    selectionOdd,sum = getOdds(popProps,preference)
    while(cont < popsize-1):
        n1,n2 = getCands(selectionOdd,sum)
        if(random.random()<0.5):
            ind1, ind2 = spCrossOver(pop[n1],pop[n2],nBinds)        
        else:
            ind1, ind2 = dpCrossOver(pop[n1],pop[n2],nBinds)
        for i in ind1,ind2:
            if cont<popsize-1:
                newPop[cont]=i
                cont+=1

    if(type(pcMutation) == float):
        minrange = 2
        maxrange = int(newPop[0].shape[0]*pcMutation)
    else:
        minrange = int(newPop[0].shape[0]*pcMutation[0])
        maxrange = int(newPop[0].shape[0]*pcMutation[1])


    for i in range(n,newPop.shape[0]-1):
        for j in range(random.randint(minrange,maxrange)):
            newPop[i] = mutation(newPop[i],newPop[i].shape[0])
    return newPop

def refineOutput(a,popsize,nBinds):
    return np.array([ [c == '1' for c in format( a, f'0{nBinds}b')] for i in range(popsize)])


pyfem.model = 0
pyfem.nNodes = node.shape[0]
pyfem.nDofNode = 2

sf = 8 #SCALE FACTOR: 1-10
dim = 2

nearList,nBinds = pyfem.findNear(node,nNeighs)

if refine:
    pop = refineOutput(configToRefine,popsize,nBinds)

configOpt = cfg(E,A,M,dim,node,fixedDof,fixedNode,loadDof,nearList,nBinds,plotEveryGeneration,showExecTimeEveryGeneration,nLeaders,pcMutation,preference)


try:
    if(nGenerations == '' or nGenerations == 0):
        nGenerations = nBinds//2
    print(f'Number of Generations: {nGenerations}')
except:
    nGenerations = nBinds//2
    print(f'Number of Generations: {nGenerations}')


if(inputElements):
    pop[:] = getIndByElem(elements,nearList,nBinds)
    pyfem.optruss(configOpt,elements,True,True)
    pyfem.plotElem(elements,node)
    plt.show()

pyfem.fig, pyfem.ax = plt.subplots() 
pyfem.plotNode(node,plotId=True)
pyfem.plotNewmann(configOpt.nodeLoadList,node)
pyfem.plotDirichlet(configOpt.nodeDofList,node)
plt.show()
totalTime = time.time()

pop = np.array([ [c == '1' for c in format(random.randint(1,2**nBinds) , f'0{nBinds}b')] for i in range(popsize)])
pop[0] = [True for i in range(nBinds)]

bestconfigeverygen = []
configOpt = cfg(E,A,M,dim,node,fixedDof,fixedNode,loadDof,nearList,nBinds,plotEveryGeneration,showExecTimeEveryGeneration,nLeaders,pcMutation,preference)

for i in range(nGenerations):
    execGen = time.time()
    print(f'Generation {i}')
    popProps,indRefined, acumTime = iterateGeneration(pop,popsize,configOpt)
    popProps = parsePopProps(popProps)
    oldPop = pop
    oldpopProps = popProps
    # plotOccurs(popProps,configOpt.preference)
    # plotOccurs(popProps,0.5)
    # plotOccurs(popProps,1)
    # plt.show()
    c = 0
    while(popProps.shape[0] == 0):
        print('No feasible Configurations Found... Repeating Generation')
        if(np.array_equal(pop,oldPop)):
            pop = np.array([ [c == '1' for c in format(random.randint(1,2**nBinds) , f'0{nBinds}b')] for i in range(popsize)])
        popProps,indRefined, acumTime = iterateGeneration(pop,popsize,configOpt)
        popProps = parsePopProps(popProps)
        c+=1
        if(c >10):
            print('Error: Feasible Configuration not Found')
            exit()
    pop = fillNewPop(pop,popProps,indRefined,configOpt.nLeaders,configOpt.pcMutation,configOpt.preference)
    print(pd.DataFrame(popProps, columns = ['Index','Objective','Min Stress','Max Stress', 'Buckling Critical Coeff' , 'Displacement', 'Is Feasible']))
    execGenTime = time.time() - execGen
    bestconfigeverygen.append(popProps[0,1])
    if(configOpt.showExecTimeEveryGeneration):
        print(f'Execution time on FEM for generation {i} = {acumTime}')
        print(f'Total Execution time for generation {i} = {execGenTime}')
        if oldpopProps.shape[0]>0:
            if(np.all(np.all(popProps[:popProps.shape[0]//2,1:] == popProps[0,1:], axis = 1)) and np.array_equal(pop[int(popProps[0,0])],oldPop[int(oldpopProps[0,0])])):
                if(i>1):
                    break

finaltime = time.time() - totalTime

plt.plot(range(nGenerations),bestconfigeverygen,label = f'p {preference}')
plt.xlabel('Generations')
plt.ylabel('Objective')

print(f'Total Execution time = {finaltime//60}min {finaltime%60}s')
popProps,indRefined, acumTime = iterateGeneration(pop,popsize,configOpt)
pyfem.optruss(configOpt,pyfem.getElemBin(pop[0],configOpt.nearList),True,True)
print(f'Best combination: ',pop[0].astype(int))

a = pop[0].tolist()
print(f'Best combination as integer: {sum(v << i for i, v in enumerate(reversed(a)))}')

