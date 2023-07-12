import numpy as np
from pyfem import *

##########################################################################
# IO Options

inputOpt = {
    'Input with Triangle lib' : 1,
    'Element Binding by Proximity' : 0
}

inputBC = {
    'fixDofCoord' : 0,
    'fixNodeCoord' : 0,
    'Distributed Load on Boundary' : 0,
}
outputOpt = {
    'plot Nodes': 0,
    'plot Elements' : 0,
    'Solve' : 1,
    'plot X Displacement' : 0,
    'plot Y Displacement' : 0,
    'plot Total Displacements' : 0, 
    'Output csv with Nodal Displacement' : 0,
    #For Trusses
    'plot Stress/Strain' : 1,
    'Output csv with Truss Stress/Strain' : 1,
    #For Beams
    'plot Axial Force' : 1,
    'plot Shear Force' : 1,
    'plot Bending Moment' : 1,
    #For Plane State Elements
    'Output vtk file' :1,
    'plot Stress XX' : 0,
    'plot Stress YY' : 0,
    'plot Stress XY' : 0,
    'Output csv with Element Stress' : 1,
    'Output csv with Element Strain' : 1,
}

plotOpt = {
    'Plot Nodes with node id' : 0,
    'Plot Nodes with node coords' : 0,
    'Plot Elements with element id' : 0,
    'Plot Elements with Loads' : 1,
    'Plot Elements with Constraints' : 1,
    'Plot Post Processing with Loads' : 0,
    'Plot Post Processing with Constraints' : 1,
    'Plot Post Processing with Undeformed configuration' : 1,
    'Plot Post Processing with scaled displacements' : 0,
    ##For Plane State Elements
    'Plot Element Edges' : 1,
    'Plot Elements with Edges id (Triangles)' : 0,

}
##############################################################################

#Input Data

#FEM MODEL
# 0 : Truss
# 1 : Bernoulli Euler Beam
# 2 : 2d Linear Triangles 
# 3 : 2d Quadratic Triangles

model = 3

##############
# 1D ELEMENTS
#Material / Cross Section Geometry

E = 2*10E11  # Young Modulus (Pa)
A = 1    # Cross Section Area (cm^2)
I = 10**4 # Area Moment of Inertia (Model 1)


##############
# 2D TRIANGULAR ELEMENTS

# Set Material Class - E [Pa],nu,rho[kg/m3]
materialClasses = [
    [73119000,0.33,2794],
]

# Plane State
# 0 : Plane Stresss
# 1 : Plane Strain
planeProblem = 0

#Thickness 
t = 1

#Dimension (Currently only implemented for dim = 2)
dim = 2

#################
# FILE NAMES
#To Use with Triangle
fileTriangle = 'mesh01'
argTriangle = '-p'

#Filename
filename = 'mesh01'

#########################################################
#BOUNDARY CONDITIONS

sf = 1 #SCALE FACTOR: 1-10

# Constrained Displacements(Dirichlet)
# fixed DoF: 
# To fix dof: [node,dof] on fixedDof list

fixedDof = [
    
]

# fixed Node 
# To fix node: node id on fixedNode list

fixedNode = [0,2,4]


# fixed DoF by coords: 
# To fix dof: [coord,direction,dof] on fixedDof list
# i.e. [0,0,1] -> fix y on [0,0] node
#  
coordsFixedDof = [ 
 
]

# fixed Node by coords: 
# To fix all dof of node: [coord,direction] on fixedDof list
# i.e. [0,0] -> fix [0,0] node
#  
coordsFixedNode = [ 

]



# Prescribed Loads (Newmann)
# Values in kN 
# Load on DoF
# To prescribe load: [node,dof,load]
# On Truss/Beam: 0 - Fx; 1 - Fy
# On Beam: 2 - Moment

# Distributed Load on Boundary
# To
#[value, horizontal:0/vertical:1, direction of load, min:0/max:1]
distributedLoadCoord = [
    [335,1,1,1]    
]

loadDof = [
]

#Surface Forces List -  [Element Id , Edge ID , Direction ,value]

surfaceForces = [
    [1,0,1,-0.5]
]

#Body Forces List - [Direction , value]
bodyForces = [
    
]


#####
#TRUSS EXAMPLE FOR HARDCODE NODE AND ELEMENT

# #Node list 
# #Coordinates of each node 
# #Size: nNodes x dim
node = np.array([
    [0,0]
    ],dtype = np.double)

for i in range(5):
    node = np.vstack((node,[4.16*(i),0]))
    node = np.vstack((node,[4.16*(i),3.2+i*(6.5-3.2)/5]))
for i in range(5,11):
    node = np.vstack((node,[4.16*(i-5)+20.8,0]))
    node = np.vstack((node,[4.16*(i-5)+20.8,6.5-(i-5)*(6.5-3.2)/5]))
node = node[1:]
elem = np.array([[0,1]])

for i in range(1,13):
    elem = np.vstack((elem,[i,i+1]))

for i in range(8,19,2):
    elem = np.vstack((elem,[i,i+3]))

for i in range(0,20):
    elem = np.vstack((elem,[i,i+2]))
for i in [14,16,18,20]:
    elem = np.vstack((elem,[i,i+1]))

####
# Finite Element Procedure Call - Modifying may cause
fem(inputOpt,inputBC,outputOpt,plotOpt,E,A,I,materialClasses,t,sf,model,dim,fileTriangle,argTriangle,node,elem,fixedDof,fixedNode,coordsFixedDof,coordsFixedNode,loadDof,distributedLoadCoord,surfaceForces,bodyForces,filename)
