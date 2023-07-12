#include "fem.h"

Node2::Node2(): id(0),c({0,0}){}

Node2::Node2(unsigned int m_id, const std::array<double,2> m_coords): id(m_id), c(m_coords){}
Node2::Node2(unsigned int m_id, double m_coords_x, double m_coords_y): id(m_id){c = {m_coords_x,m_coords_y};}


void Node2::setID(unsigned int i) { id = i;}
void Node2::setC(const std::array<double,2> m_coords) {c = m_coords;}


double& Node2::operator()(const int i){return c[i];}
const double& Node2::operator()(const int i) const{return c[i];}
Node2& Node2::operator=(const double (&arr)[2]){c[0] = arr[0]; c[1] = arr[1]; return *this;}

Node3::Node3(): id(0),c({0,0,0}){}

Node3::Node3(unsigned int m_id, const std::array<double,3> m_coords): id(m_id), c(m_coords){}
Node3::Node3(unsigned int m_id, double m_coords_x, double m_coords_y, double m_coords_z): id(m_id){c = {m_coords_x,m_coords_y,m_coords_z};}


void Node3::setID(unsigned int i) { id = i;}
void Node3::setC(const std::array<double,3> m_coords) {c = m_coords;}


double& Node3::operator()(const int i){return c[i];}
const double& Node3::operator()(const int i) const{return c[i];}
Node3& Node3::operator=(const double (&arr)[3]){c[0] = arr[0]; c[1] = arr[1], c[2] = arr[2]; return *this;}

Material::Material(const unsigned int Id, const double E_m, const double Nu, const double Rho):id(Id),E(E_m),nu(Nu),rho(Rho){}

dMatrix Material::computeC(bool ept){
    if(ept){
        double ct = E/(1-nu*nu);
        dMatrix preC(3,3,{{1,nu,0},{nu,1,0},{0,0,0.5*(1-nu)}});
        return ct * preC;
    }
    else{
        double ct = E/((1+nu)*1-2*nu);
        dMatrix preC(3,3,{{1-nu,nu,0},{nu,1-nu,0},{0,0,0.5*(1-2*nu)}});
        return ct * preC;
    }
}

Element::Element(const unsigned int m_id, const unsigned int m_elemType, 
const unsigned int m_nNodesElem,const std::vector<unsigned int> m_elemBuffer, const unsigned int m_mat_id): id(m_id),elemType(m_elemType),nNodesElem(m_nNodesElem),elemBuffer(m_elemBuffer),mat_id(m_mat_id){}

Element::Element(const unsigned int m_id, const unsigned int m_elemType, 
const unsigned int m_nNodesElem,const unsigned int (&m_elemBuffer)[],const unsigned int m_mat_id):id(m_id),elemType(m_elemType),nNodesElem(m_nNodesElem),mat_id(m_mat_id){
    elemBuffer.resize(m_nNodesElem);
    for(int i = 0;i<m_nNodesElem;i++){
        elemBuffer[i] = m_elemBuffer[i];
    }
}
const unsigned int Element::operator()(const int i)const{return elemBuffer[i];}
TR3::TR3(const unsigned int (&m_elemBuffer)[3], unsigned int m_id, const unsigned int m_mat_id):Element(m_id,0,3,{m_elemBuffer[0],m_elemBuffer[1],m_elemBuffer[2]},m_mat_id){}
TR3::TR3(const std::vector<unsigned int> m_elemBuffer, const unsigned int m_id, const unsigned int m_mat_id) : Element(m_id,0,3,m_elemBuffer,m_mat_id){}

TR6::TR6(const unsigned int (&m_elemBuffer)[6], unsigned int m_id, const unsigned int m_mat_id):Element(m_id,1,6,{m_elemBuffer[0],m_elemBuffer[1],m_elemBuffer[2],m_elemBuffer[3],m_elemBuffer[4],m_elemBuffer[5]},m_mat_id){}
TR6::TR6(const std::vector<unsigned int> m_elemBuffer, const unsigned int m_id, const unsigned int m_mat_id) : Element(m_id,1,6,m_elemBuffer,m_mat_id){}


Mesh::Mesh():EPT(0),t(0),nElem(0),nNodes(0),nMats(0),elemType(0){}

void Mesh::initMesh(std::string path){

    std::ifstream cfgFile(path);
    std::string line,split;
    path = path;
    std::getline(cfgFile, line);
    std::getline(cfgFile, line);
    std::getline(cfgFile, line);
    std::getline(cfgFile, line);
    std::istringstream ss(line);

    ss>>split;
    // node list number
    nNodes = std::stoi(split);

    // elem list number
    ss>>split;
    nElem = std::stoi(split);

    // material list number
    ss>>split;
    nMats = std::stoi(split);

    // fixednode number
    ss>>split;
    nFixedNodes = std::stoi(split);

    // point loads number
    ss>>split;
    nPointLoads = std::stoi(split);
    
    // surface forces number
    ss>>split;
    nSurfForces = std::stoi(split);

    // body forces number
    ss>>split;
    nBodyForces = std::stoi(split);
    std::getline(cfgFile, line);
    // node list
    node = (class Node2*) malloc(nNodes*sizeof(class Node2));
    for (int i = 0; i<nNodes;i++){
        std::getline(cfgFile, line);
        std::istringstream ss(line);
        double x,y;
        ss >> split;
        x = std::stod(split);
        ss >> split;
        y = std::stod(split);
        node[i] = Node2(i,x,y);
    }
    // for (int i = 0; i < nNodes;i++){
    //     printf("node %d has x = %f, y= %f",node[i].getID(),node[i](0),node[i](1));
    // }
    std::getline(cfgFile, line);

    elem = new Element*[nElem];
    for (int i = 0; i<nElem;i++){
        std::getline(cfgFile, line);
        std::istringstream ss(line);
        int elemType,nNodesElem,matID;
        ss >> split;
        matID = std::stoi(split);
        ss >> split;
        elemType = std::stoi(split);
        ss >> split;
        nNodesElem = std::stoi(split);
        if (elemType == 0){
            unsigned int a,b,c;
            ss >> split;
            a = std::stoi(split);
            ss >> split;
            b = std::stoi(split);
            ss >> split;
            c = std::stoi(split);
            elem[i] = new TR3({a, b, c}, i, matID);
        }
        if (elemType == 1){
            unsigned int a,b,c,d,e,f;
            ss >> split;
            a = std::stoi(split);
            ss >> split;
            b = std::stoi(split);
            ss >> split;
            c = std::stoi(split);
            ss >> split;
            d = std::stoi(split);
            ss >> split;
            e = std::stoi(split);
            ss >> split;
            f = std::stoi(split);
            elem[i] = new TR6({a, b, c, d, e, f}, i, matID);
        }
    }
    if(elem[0]->getElemType() == 0){
        elemType = 0;
    }else if(elem[0]->getElemType() == 1){
        elemType = 1;
    }
    //wip
    nDofs = nNodes*node[0].getSize();
    // for (int i = 0; i<nElem;i++){
    //     printf("Elem %d contains nodes %d,%d,%d\n",elem[i]->getID(),(*elem[i])(0),(*elem[i])(1),(*elem[i])(2));
    // }

    std::getline(cfgFile, line);
    // material list
    mat = (class Material*) malloc(nMats*sizeof(class Material));
    for (int i = 0; i<nMats;i++){
        std::getline(cfgFile, line);
        std::istringstream ss(line);
        double E,nu,rho;
        ss >> split;
        E = std::stod(split);
        ss >> split;
        nu = std::stod(split);
        ss >> split;
        rho = std::stod(split);
        mat[i] = Material(i,E,nu,rho);
    }
    // for (int i = 0; i<nMats;i++){
    //     printf("Mat %d has E = %.2f, nu = %.2f, G=%.2f, rho = %.2f \n",mat[i].getID(),mat[i].getE(),mat[i].getNu(),mat[i].getG(),mat[i].getRho());
    // }

    std::getline(cfgFile, line);
    // fixednode list
    for (int i = 0; i<nFixedNodes;i++){
        std::getline(cfgFile, line);
        std::istringstream ss(line);
        int node_id,dof_id;
        ss >> split;
        node_id = std::stoi(split);
        ss >> split;
        dof_id = std::stoi(split);
        if (dof_id == -1){
            for(int i = 0; i< node[node_id].getSize();i++){
                fixed_dofs.push_back(node[node_id].getSize()*node_id+i);
            }
        }
        else{
            fixed_dofs.push_back(node[node_id].getSize()*node_id+dof_id);
        }

    }

    // for (auto it = begin (fixed_dofs); it != end (fixed_dofs); ++it) {
    //     int data = *it;
    //     std::cout<<data;
    // }

    std::getline(cfgFile, line);
    // point loads list
    pLoads.resize(nPointLoads);
    for (int i = 0; i<nPointLoads;i++){
        std::getline(cfgFile, line);
        std::istringstream ss(line);
        int node_id,dof_id;
        double value;
        ss >> split;
        node_id = std::stoi(split);
        ss >> split;
        dof_id = std::stoi(split);
        ss >> split;
        value = std::stod(split); 
        pLoads[i].dofID = dof_id;
        pLoads[i].nodeID = node_id;
        pLoads[i].v = value;
    }
    sForces.resize(nSurfForces);
    std::getline(cfgFile, line);

    // surf forces list
    for (int i = 0; i<nSurfForces;i++){
        std::getline(cfgFile, line);
        std::istringstream ss(line);
        unsigned int direction,elemId,edgeId;
        double value;
        ss >> split;
        elemId = std::stoi(split);
        ss >> split;
        edgeId = std::stoi(split);
        ss >> split;
        direction = std::stoi(split);
        ss >> split;
        value = std::stod(split); 
        sForces[i].elemID = elemId;
        sForces[i].edgeID = edgeId;
        sForces[i].direction = direction;
        sForces[i].v = value;
    }

    bForces.resize(nBodyForces);
    std::getline(cfgFile, line);
    // body forces list
    for (int i = 0; i<nBodyForces;i++){
        std::getline(cfgFile, line);
        std::istringstream ss(line);
        unsigned int direction;
        double value;
        ss >> split;
        direction = std::stoi(split);
        ss >> split;
        value = std::stod(split); 
        bForces[i].direction = direction;
        bForces[i].v = value;
    }

    // options
    
    std::getline(cfgFile, line);
    std::getline(cfgFile, line);
    std::istringstream sst(line);
    sst >> split;
    t = std::stod(split);

    EPT = (t != 0);
    if(!EPT) t = 1;

    while (std::getline(cfgFile, line)){
        std::istringstream iss(line);
        int a, b;
        if (!(iss >> a >> b)) { break; } 
    }
	

}

SparseMatrix Mesh::assemblyGlobalStiffnessMatrix(){

    SparseMatrix K(nDofs,nDofs);
    
    for (int i = 0; i< nElem;i++){
        dMatrix KEl = calcKEl(i);
        for (int j = 0; j<elem[i]->getNodesElem();j++){
            int actualNode = (*elem[i])(j);
            int nDofNode = node[actualNode].getSize();
            
            for (int k = 0; k<nDofNode;k++){
                int ix = nDofNode*actualNode+k;
                for(int icol = 0; icol<(*elem[i]).getNodesElem();icol++){
                    for(int jcol = 0;jcol < nDofNode;jcol++){
                        int localcol = nDofNode*icol+jcol;
                        int globalcol =nDofNode*(*elem[i])(icol)+jcol; 
                        int localrow = nDofNode*j+k;
                        K.addElem(ix,globalcol, K(ix,globalcol) + KEl(localrow,localcol));
                    }
                }
            }
        }
    }
    return K;
}
void Mesh::applyBodyForces(VectorXd &loads){
    if(elemType == 0){
        int nDofNode = node[0].getSize();
        double bodyToNodal;
        for (int force = 0; force<nBodyForces;force++){
            for (int i = 0; i < nElem; i++){
                bodyToNodal = elem[i]->getArea()*t*bForces[force].v/3;
                loads[(*elem[i])(0)* nDofNode +bForces[force].direction]+=bodyToNodal;
                loads[(*elem[i])(1)* nDofNode +bForces[force].direction]+=bodyToNodal;
                loads[(*elem[i])(2)* nDofNode +bForces[force].direction]+=bodyToNodal;
            }
        }
    }
    if(elemType ==1){
        int nDofNode = node[0].getSize();
        double bodyToNodal;
        for (int force = 0; force<nBodyForces;force++){
            for (int i = 0; i < nElem; i++){
                bodyToNodal = elem[i]->getArea()*t*bForces[force].v/6;
                loads[(*elem[i])(3)* nDofNode +bForces[force].direction]+=bodyToNodal;
                loads[(*elem[i])(4)* nDofNode +bForces[force].direction]+=bodyToNodal;
                loads[(*elem[i])(5)* nDofNode +bForces[force].direction]+=bodyToNodal;
            }
        }
    }
}
void Mesh::applySurfaceForces(VectorXd &loads){
    if(elemType == 0){
        int nDofNode = node[0].getSize();
        double edgeLength; // 0 -> nodes 0,1/ 1 -> nodes 1,2/ 2-> nodes 2,0
        double surfToNodal;
        Node2 v1,v2;
        int edgeVertex1, edgeVertex2;
        for (int f = 0; f<nSurfForces;f++){
            edgeVertex1 = sForces[f].edgeID;
            edgeVertex2 = (sForces[f].edgeID+1)%3;
            Element *elemF = elem[sForces[f].elemID];
            v1 = node[(*elemF)(edgeVertex1)];
            v2 = node[(*elemF)(edgeVertex2)];
            edgeLength =std::sqrt((v2(0)-v1(0))*(v2(0)-v1(0))+(v2(1)-v1(1))*(v2(1)-v1(1)));
            loads[(*elemF)(edgeVertex1)* nDofNode +sForces[f].direction]+=sForces[f].v*edgeLength*0.5;
            loads[(*elemF)(edgeVertex2)* nDofNode +sForces[f].direction]+=sForces[f].v*edgeLength*0.5;
        }
    }
    if(elemType == 1){
        int nDofNode = node[0].getSize();
        double edgeLength; // 0 -> nodes 0,5,1/ 1 -> nodes 1,3,2/ 2-> nodes 2,4,0
        double surfToNodal;
        Node2 v1,v2,v3;
        int edgeVertex1, edgeVertex2, edgeVertex3;
        for (int f = 0; f<nSurfForces;f++){
            edgeVertex1 = sForces[f].edgeID;
            edgeVertex2 = (5-2*sForces[f].edgeID);
            if (sForces[f].edgeID == 2) edgeVertex2=4;
            edgeVertex3 = (sForces[f].edgeID+1)%3;
            Element *elemF = elem[sForces[f].elemID];
            v1 = node[(*elemF)(edgeVertex1)];
            v2 = node[(*elemF)(edgeVertex2)];
            v3 = node[(*elemF)(edgeVertex3)];
            edgeLength =std::sqrt((v3(0)-v1(0))*(v3(0)-v1(0))+(v3(1)-v1(1))*(v3(1)-v1(1)));
            loads[(*elemF)(edgeVertex1)* nDofNode +sForces[f].direction]+=sForces[f].v*edgeLength/6;
            loads[(*elemF)(edgeVertex2)* nDofNode +sForces[f].direction]+=sForces[f].v*edgeLength*2/3;
            loads[(*elemF)(edgeVertex3)* nDofNode +sForces[f].direction]+=sForces[f].v*edgeLength/6;
        }
    }
}
//wip - mixed elements formulation
VectorXd Mesh::getNodalLoads() {
    VectorXd loads(nDofs,0);
    int nDofNode = node[0].getSize();
    for (int i = 0; i < nPointLoads; ++i) {
        loads(pLoads[i].nodeID * nDofNode + pLoads[i].dofID) = loads(pLoads[i].nodeID * nDofNode + pLoads[i].dofID)+pLoads[i].v;
    }
    applyBodyForces(loads);
    applySurfaceForces(loads);
    return loads;
}

dMatrix Mesh::calcKEl(const unsigned int i){

    elem[i]->getInterpolation(node);
    dMatrix KEl =  elem[i]->kEl(t,mat,EPT);

    return KEl;

}
void TR3::getInterpolation(Node2 *nodes){

    std::array<double, 2> n1, n2, n3;
    n1 = nodes[elemBuffer[0]].getC();
    n2 = nodes[elemBuffer[1]].getC();
    n3 = nodes[elemBuffer[2]].getC();

    a(0) = n2[0]*n3[1] - n3[0]*n2[1];
    a(1) = n3[0]*n1[1] - n1[0]*n3[1];
    a(2) = n1[0]*n2[1] - n2[0]*n1[1];

    b(0) = n2[1] - n3[1]; // y2-y3
    b(1) = n3[1] - n1[1]; // y3-y1
    b(2) = n1[1] - n2[1]; // y1-y2

    c(0) = n3[0] - n2[0]; // x3-x2
    c(1) = n1[0] - n3[0]; // x1-x3
    c(2) = n2[0] - n1[0]; // x2-x1
    area = 0.5*fabs(a(0) +n1[0]*b(0) + n1[1]*c(0));   
}
dMatrix TR3::H(double r,double s){ 

    std::vector<double> h;
    h[0] = 1-r-s;
    h[1] = r;
    h[2] = s;
    dMatrix H(2,6,
    {{h[0],0,h[1],0,h[2],0},
    {0,h[0],0,h[1],0,h[2]}});
    return H;
    }
dMatrix TR3::H_global(double x,double y){ 

    std::vector<double> h;
    h = g2l(x,y).getRowVector(0);
    dMatrix H(2,6,
    {{h[0],0,h[1],0,h[2],0},
    {0,h[0],0,h[1],0,h[2]}});
    return H;
    }

dMatrix TR3::Bl(double r,double s){ 

    dMatrix B(3,6, 
    {{b(0),0,b(1),0,b(2),0},
    {0,c(0),0,c(1),0,c(2)},
    {c(0),b(0),c(1),b(1),c(2),b(2)}});    
    
    return (1/(2*area))*B;
    }
dMatrix TR3::J(double r, double s){
    return dMatrix(2,2,{{c(1),-b(1)},{-c(0),b(0)}});
}
double TR3::detJ(double r, double s){
    return 2*area;
}
dMatrix TR3::invJ(double r = 0, double s = 0){
    dMatrix A(2,2,{{b(1),b(2)},{c(1),c(2)}});
    A = (1.0/2.0*area) * A;
    return A;
}
dMatrix TR3::g2l(double x, double y){
    std::array<double,3> h;
    for(int i = 0; i<nNodesElem;i++){
        h[i]= (a(i)+b(i)*x+c(i)*y)/(2*area);
    }
    return dMatrix(1,3,{{h[0],h[1],h[2]}}); // h[0] = 1-r-s h[1] = r; h[2] = s
}
dMatrix TR3::kEl(double t, Material *mat, bool ept){
    dMatrix kEl(6,6), B(6,3), C(3,3);
    C = mat[mat_id].computeC(ept);
    B = Bl(0,0);
    kEl = ~B * C * B;
    return kEl*area*t;
}

void TR3::GLPoint(double &xi, double &eta, double &alpha, int i = 0){
    eta = 1.0/3.0;
    xi = 1.0/3.0;
    alpha = 1.0;
}

void TR6::getInterpolation(Node2 *nodes){

    std::array<double, 2> n1, n2, n3;
    n1 = nodes[elemBuffer[0]].getC();
    n2 = nodes[elemBuffer[1]].getC();
    n3 = nodes[elemBuffer[2]].getC();

    a(0) = n2[0]*n3[1] - n3[0]*n2[1];
    a(1) = n3[0]*n1[1] - n1[0]*n3[1];
    a(2) = n1[0]*n2[1] - n2[0]*n1[1];

    b(0) = n2[1] - n3[1]; // y2-y3
    b(1) = n3[1] - n1[1]; // y3-y1
    b(2) = n1[1] - n2[1]; // y1-y2

    c(0) = n3[0] - n2[0]; // x3-x2
    c(1) = n1[0] - n3[0]; // x1-x3
    c(2) = n2[0] - n1[0]; // x2-x1 
    area = 0.5*fabs(a(0) +n1[0]*b(0) + n1[1]*c(0));   
}
dMatrix TR6::g2l(double x, double y){
    std::array<double,3> h;
    for(int i = 0; i<nNodesElem;i++){
        h[i]= (a(i)+b(i)*x+c(i)*y)/(2.0*area);
    }
    return dMatrix(1,3,{{h[0],h[1],h[2]}}); // h[0] = 1-r-s h[1] = r; h[2] = s
}
dMatrix TR6::H(double r,double s){ 

    std::vector<double> h;
    h[0] = 1-r-s;
    h[1] = r;
    h[2] = s;
    dMatrix H(2,12);
    H(0,0) = h[0]*(2*h[0]-1);
    H(1,1) = h[0]*(2*h[0]-1);
    H(0,2) = h[1]*(2*h[1]-1);
    H(1,3) = h[1]*(2*h[1]-1);
    H(0,4) = h[2]*(2*h[2]-1);
    H(1,5) = h[2]*(2*h[2]-1);
    H(0,6) = 4*h[1]*h[0];
    H(1,7) = 4*h[1]*h[0];
    H(0,8) = 4*h[1]*h[2];
    H(1,9) = 4*h[1]*h[2];
    H(0,10) = 4*h[2]*h[0];
    H(1,11) = 4*h[2]*h[0];
    
    return H;
    }
dMatrix TR6::H_global(double x,double y){ 

    std::vector<double> h;
    h = g2l(x,y).getRowVector(0);
    dMatrix H(2,12);
    H(0,0) = h[0]*(2*h[0]-1);
    H(1,1) = h[0]*(2*h[0]-1);
    H(0,2) = h[1]*(2*h[1]-1);
    H(1,3) = h[1]*(2*h[1]-1);
    H(0,4) = h[2]*(2*h[2]-1);
    H(1,5) = h[2]*(2*h[2]-1);
    H(0,6) = 4*h[2]*h[0];
    H(1,7) = 4*h[2]*h[0];
    H(0,8) = 4*h[1]*h[0];
    H(1,9) = 4*h[1]*h[0];
    H(0,10) = 4*h[2]*h[1];
    H(1,11) = 4*h[2]*h[1];
    return H;
    }
dMatrix TR6::Bl(double r,double s){ 
    dMatrix B(4,12);

    dMatrix delta(3,4);
    delta(0,0) = 1.0;
    delta(2,1) = 1.0;
    delta(2,2) = 1.0;
    delta(1,3) = 1.0;
    dMatrix inv = invJ(0,0);
    dMatrix jacInv (4,4,
    {{inv(0,0),inv(0,1),0,0},
    {inv(1,0),inv(1,1),0,0},
    {0,0,inv(0,0),inv(0,1)},
    {0,0,inv(1,0),inv(1,1)}
    });
    //u
    B(0,0) = 4.0*r+4.0*s-3; // dh1 dr
    B(1,0) = 4.0*r+4.0*s-3; // dh1 ds
    //v
    B(2,1) = 4.0*r+4.0*s-3; // dh1 dr
    B(3,1) = 4.0*r+4.0*s-3; // dh1 ds

    //u
    B(0,2) = 4.0*r-1; // dh2 dr
    B(1,2) = 0; // dh2 ds
    //v
    B(2,3) = 4.0*r-1; // dh2 dr
    B(3,3) = 0; // dh2 ds

    //u
    B(0,4) = 0; //dh3 dr
    B(1,4) = 4.0*s-1; //dh3 ds
    //v
    B(2,5) = 0; //dh3 dr
    B(3,5) = 4.0*s-1; //dh3 ds

    //u
    B(0,6) = 4.0*s; //dh4 dr
    B(1,6) = 4.0*r; // dh4 ds
    //v
    B(2,7) = 4.0*s; //dh4 dr
    B(3,7) = 4.0*r; // dh4 ds
    
    //u
    B(0,8) = -4.0*s; //dh5 dr
    B(1,8) = -4.0*(2.0*s+r-1); // dh5 ds
    //v
    B(2,9) = -4.0*s; //dh5 dr
    B(3,9) = -4.0*(2.0*s+r-1); // dh5 ds

    //u
    B(0,10) = -4.0*(2.0*r+s-1); // dh6 dr
    B(1,10) = -4.0*r; // dh6 ds
    //v
    B(2,11) = -4.0*(2.0*r+s-1); // dh6 dr
    B(3,11) = -4.0*r; // dh6 ds


    B = delta*jacInv*B;
    return B;
    }



dMatrix TR6::J(double r=0, double s=0){
    return dMatrix(2,2,{{c(1),-b(1)},{-c(0),b(0)}});
}
double TR6::detJ(double r=0, double s=0){
    return 2.0*area;
}
dMatrix TR6::invJ(double r = 0, double s = 0){
    dMatrix A(2,2,{{b(1),b(2)},{c(1),c(2)}});
    A = (1.0/(2.0*area)) * A;
    return A;
}
dMatrix TR6::kEl(double t, Material *mat, bool ept){
    dMatrix kEl(12,12), B(12,3), C(3,3);
    double r,s,alpha, scalar;

    C = mat[mat_id].computeC(ept);
    
    for(int i = 0;i<3;i++){
        GLPoint(r,s,alpha,i);
        B = Bl(r,s);        
        scalar = (alpha*detJ());
        dMatrix A = (~B*C*B)*scalar;
        dMatrix B = kEl + A;
        kEl = B;

    }

    return t*kEl;
}

void TR6::GLPoint(double &xi, double &eta, double &alpha, int i){
    alpha = 1.0 / 6.0;
    eta = 1.0 / 2.0;
    xi = 1.0 / 2.0;
    if (i == 1)
        eta = 0.0;
    else if (i == 2)
        xi = 0.0;
}




void FEM::initFEM(std::string &f){
    std::string path = f+".txt";
    mesh.initMesh(path);
    dofDisplacements.resize(mesh.getnDofs());
    file = f;
    // More to do
}

void Mesh::assignBoundaryConditions(SparseMatrix &K, VectorXd &nodalLoads){
    for(int i = fixed_dofs.size()-1; i >= 0 && i < fixed_dofs.size();i--){
        K.eraseCol(fixed_dofs[i]);
        K.eraseRow(fixed_dofs[i]);
        // for (int j = 0; j<K.getRows();j++){
        //     K(j,fixed_dofs[i]) = 0;
        // }
        // for (int k = 0; k< K.getCols();k++){
        //     K(fixed_dofs[i],k) = 0;
        // }
        K(fixed_dofs[i],fixed_dofs[i])=1;
        nodalLoads(fixed_dofs[i])=0;
    }
}

void FEM::solve(){

    Timer timer;
    timer.start();
    SparseMatrix K = mesh.assemblyGlobalStiffnessMatrix();
    std::cout << "Duration Assembly: " << timer.stop() << std::endl;
    
    timer.start();
    VectorXd nodalLoads = mesh.getNodalLoads();
    std::cout << "Duration Nodal: " << timer.stop() << std::endl;

    timer.start();
    mesh.assignBoundaryConditions(K,nodalLoads);
    std::cout << "Duration BC: " << timer.stop() << std::endl;

    
    timer.start();
    dofDisplacements = K.conjugateGradient(nodalLoads);
    std::cout << "Duration Conjugate: " << timer.stop() << std::endl;

    timer.start();
    strains = mesh.getElementStrains(dofDisplacements);
    stresses = mesh.getElementStresses(strains);
    std::cout << "Duration Strains " << timer.stop() << std::endl;
    outputResults(";");
}


void FEM::outputResults(const std::string& separator) {
    // Output file names
    std::string displacementsFile = file+"displacements.csv";
    std::string strainsFile = file+"strains.csv";
    std::string stressesFile = file+"stresses.csv";

    // Open output files
    std::ofstream displacementsOutput(displacementsFile);
    std::ofstream strainsOutput(strainsFile);
    std::ofstream stressesOutput(stressesFile);

    if (!displacementsOutput || !strainsOutput || !stressesOutput) {
        std::cout << "Error opening output files." << std::endl;
        return;
    }

    for (int i = 0; i < dofDisplacements.size(); i++) {
        displacementsOutput << dofDisplacements(i) << std::endl;
    }

    for (int i = 0; i < strains.rows(); i++) {
        for (int j = 0; j < strains.cols(); j++) {
            strainsOutput << strains(i, j);
            if (j != strains.cols() - 1) {
                strainsOutput << separator;
            }
        }
        strainsOutput << std::endl;
    }

    for (int i = 0; i < stresses.rows(); i++) {
        for (int j = 0; j < stresses.cols(); j++) {
            stressesOutput << stresses(i, j);
            if (j != stresses.cols() - 1) {
                stressesOutput << separator;
            }
        }
        stressesOutput << std::endl;
    }

    displacementsOutput.close();
    strainsOutput.close();
    stressesOutput.close();

    std::cout << "Results output to CSV files: "
              << displacementsFile << ", "
              << strainsFile << ", "
              << stressesFile << std::endl;
}

dMatrix Mesh::getElementStrains(VectorXd disp){
    //wip different elements coupling
    unsigned int elemType = elem[0]->getElemType();
    if(elemType == 0){
        dMatrix strain(nElem,3);
        for(int i = 0; i< nElem;i++){
            dMatrix elemDisplacements(elem[i]->getNodesElem()*node[(*elem[i])(0)].getSize(),1);
            for(int j = 0; j< 6;j++){
                elemDisplacements(j,0) = disp[2*(*elem[i])(int(j*0.5))+j%2];
            }
            dMatrix a= elem[i]->Bl(0,0)*elemDisplacements;
            strain.assignRow(i,(~a).getRowVector(0));
        }
        return strain;
    }
    
    if (elemType == 1){
        dMatrix strain(nElem,9);
        for(int i = 0; i< nElem;i++){
            dMatrix elemDisplacements(elem[i]->getNodesElem()*node[(*elem[i])(0)].getSize(),1);
            for(int j = 0; j< 12;j++){
                elemDisplacements(j,0) = disp[2*(*elem[i])(int(j*0.5))+j%2];
            }
            dMatrix a= elem[i]->Bl(0,0)*elemDisplacements;
            strain(i,0) = a(0,0);
            strain(i,1) = a(1,0);
            strain(i,2) = a(2,0);
            a= elem[i]->Bl(1,0)*elemDisplacements;
            strain(i,3) = a(0,0);
            strain(i,4) = a(1,0);
            strain(i,5) = a(2,0);
            a= elem[i]->Bl(0,1)*elemDisplacements;
            strain(i,6) = a(0,0);
            strain(i,7) = a(1,0);
            strain(i,8) = a(2,0);
        }
        return strain;
    }
}
 
dMatrix Mesh::getElementStresses(dMatrix strain){
    //wip different elements coupling
    dMatrix stress (strain.rows(),strain.cols());
    if (elemType==0){
        for (int i = 0; i< strain.rows();i++){
            //Calc stress of element i by multiplying strain to the material C matrix (hookes law)
            dMatrix strainRow = strain.getRowMatrix(i);
            stress.assignRow(i,(~(mat[elem[i]->getMatID()].computeC(EPT)*~strainRow)).getRowVector(0));
        }
    }
    else if (elemType==1){
        for (int i = 0; i< strain.rows();i++){
            //Calc stress of element i by multiplying strain to the material C matrix (hookes law)
            dMatrix strainRow(1,3,{{strain(i,0),strain(i,1),strain(i,2)}});
            strainRow=(mat[elem[i]->getMatID()].computeC(EPT)*~strainRow);
            stress(i,0) = strainRow(0,0);
            stress(i,1) = strainRow(1,0);
            stress(i,2) = strainRow(2,0);
            std::vector<double> a({strain(i,3),strain(i,4),strain(i,5)});
            strainRow = ~strainRow;
            strainRow.assignRow(0,a);
            strainRow=(mat[elem[i]->getMatID()].computeC(EPT)*~strainRow);
            stress(i,3) = strainRow(0,0);
            stress(i,4) = strainRow(1,0);
            stress(i,5) = strainRow(2,0);
            strainRow = ~strainRow;
            strainRow.assignRow(0,std::vector<double>({strain(i,6),strain(i,7),strain(i,8)}));
            strainRow=(mat[elem[i]->getMatID()].computeC(EPT)*~strainRow);
            stress(i,6) = strainRow(0,0);
            stress(i,7) = strainRow(1,0);
            stress(i,8) = strainRow(2,0);
        }
    }
    return stress;
}
