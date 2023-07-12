#pragma once

#include <sstream>
#include <fstream>
#include <vector>
#include <array>
#include <string>
#include "datastructs.h"
#include <algorithm>

struct pointLoad {
    unsigned int nodeID;
    unsigned int dofID;
    double v;
};

struct surfForce{
    unsigned int direction;
    unsigned int elemID;
    unsigned int edgeID;
    double v;
};
struct bodyForce{
    unsigned int direction;
    double v;
};

// Class for nodes
class Node2 {

    private:
        unsigned int id;
        std::array<double,2> c;
    public:
        Node2();
        Node2(unsigned int m_id, std::array<double,2> m_coords);
        Node2(unsigned int m_id, double m_coords_x, double m_coords_y);
        ~Node2(){};
        
        void setID(unsigned int i);
        void setC(std::array<double,2> m_coords);
        inline unsigned int getID() const {return id;}
        inline unsigned int getSize() const {return 2;}
        inline const std::array<double,2> &getC() const{return c;}


        double& operator()(const int i);
        const double& operator()(const int i) const;
        Node2& operator=(const double (&arr)[2]);
        
};
class Node3 {

    private:
        unsigned int id;
        std::array<double,3> c;
    public:
        Node3();
        Node3(unsigned int m_id, std::array<double,3> m_coords);
        Node3(unsigned int m_id, double m_coords_x, double m_coords_y, double m_coords_z);
        ~Node3(){};
        
        void setID(unsigned int i);
        void setC(std::array<double,3> m_coords);
        inline unsigned int getID() const {return id;}
        inline unsigned int getSize() const {return 3;}
        inline const std::array<double,3> &getC() const{return c;}


        double& operator()(const int i);
        const double& operator()(const int i) const;
        Node3& operator=(const double (&arr)[3]);
        
};


class Material {
    private:
        unsigned int id;
        double E;
        double nu;
        double G = E/(2*(1+nu));
        double rho;
    
    public:
    
        Material(const unsigned int Id, const double E_m, const double Nu, const double Rho);
        ~Material(){};

        inline unsigned int getID() const {return id;}
        inline double getE() const {return E;}
        inline double getNu() const {return nu;}
        inline double getG() const {return G;}
        inline double getRho() const {return rho;}

        dMatrix computeC(bool ept);

};

// Base class for elements
class Element {
    protected:
        
        unsigned int id;
        unsigned int mat_id;
        unsigned int elemType;
        unsigned int nNodesElem;
        std::vector<unsigned int> elemBuffer;
        double area;

    public:
        virtual ~Element() {} 
        Element(){}
        Element(const unsigned int m_id, const unsigned int m_elemType, 
        const unsigned int m_nNodesElem,const std::vector<unsigned int> m_elemBuffer, const unsigned int m_mat_id);
        Element(const unsigned int m_id, const unsigned int m_elemType, 
        const unsigned int m_nNodesElem,const unsigned int (&m_elemBuffer)[], const unsigned int m_mat_id);
        
        // Getters, Setters and Operators
        inline unsigned int getNodesElem() const {return nNodesElem;}
        inline unsigned int getID() const{return id;}
        inline unsigned int getMatID() const{return mat_id;}
        inline unsigned int getElemType() const{return elemType;}
        inline double getArea()const{return area;}
        const unsigned int operator()(const int i)const;

        
        // Abstract methods to be implemented by derived classes
        virtual void getInterpolation(Node2 *nodes) = 0;
        virtual void GLPoint(double &xi, double &eta, double &alpha, int i) = 0;
        virtual dMatrix H(double r, double s) = 0;
        virtual dMatrix H_global(double x, double y) = 0;
        // virtual dMatrix Bg(double x, double y);
        virtual dMatrix Bl(double r, double s) = 0;
        virtual dMatrix J(double r, double s) = 0;
        virtual double detJ(double r, double s) = 0;
        virtual dMatrix invJ(double r, double s) = 0;
        virtual dMatrix g2l(double r, double s) = 0;
        virtual dMatrix kEl(double t, Material *mat, bool ept) = 0;
    };

// Derived class for triangular3 nodes element
class TR3 : public Element {
    public:
        TR3(){}
        TR3(const std::vector<unsigned int> m_elemBuffer, unsigned int m_id, const unsigned int m_mat_id);
        TR3(const unsigned int (&m_elemBuffer)[3], unsigned int m_id, const unsigned int m_mat_id);
        ~TR3() {}
        // Implementation of the abstract methods

        void getInterpolation(Node2 *nodes) override;
        dMatrix g2l(double r, double s) override;
        dMatrix H(double x, double y) override;
        dMatrix H_global(double x, double y) override;
        dMatrix Bl(double r, double s) override;
        dMatrix J(double r, double s) override;
        double detJ(double r, double s)override;
        dMatrix invJ(double r, double s) override;
        dMatrix kEl(double t, Material *mat, bool ept) override;
        void GLPoint(double &xi, double &eta, double &alpha, int i) override;

    private:
        Node3 a,b,c;
};

class TR6 : public Element {
    public:
        TR6(){}
        TR6(const std::vector<unsigned int> m_elemBuffer, unsigned int m_id, const unsigned int m_mat_id);
        TR6(const unsigned int (&m_elemBuffer)[6], unsigned int m_id, const unsigned int m_mat_id);
        ~TR6() {}
        // Implementation of the abstract methods

        void getInterpolation(Node2 *nodes) override;
        dMatrix g2l(double x, double y);
        dMatrix H(double r, double s) override;
        dMatrix H_global(double x, double y) override;
        dMatrix Bl(double r, double s) override;
        dMatrix J(double r, double s) override;
        double detJ(double r, double s) override;
        dMatrix invJ(double r, double s) override;
        dMatrix kEl(double t, Material *mat, bool ept) override;
        void GLPoint(double &xi, double &eta, double &alpha, int i) override;

    private:
        Node3 a,b,c;
};

class Mesh {
    private:
        bool EPT;
        double t;
        unsigned int nElem;
        unsigned int nNodes;
        unsigned int nMats;
        unsigned int nFixedNodes;
        unsigned int nPointLoads;
        unsigned int nSurfForces;
        unsigned int nBodyForces;
        unsigned int elemType;
        unsigned int nDofs;
        Node2 *node;
        Element **elem;
        Material *mat;
        std::vector<int> fixed_dofs;
        std::vector<pointLoad> pLoads;
        std::vector<surfForce> sForces;
        std::vector<bodyForce> bForces;
        std::string path;

    public:
        Mesh();
        ~Mesh(){};
        inline unsigned int getnNodes()const{return nNodes;}
        inline unsigned int getnDofs()const{return nDofs;}
        inline unsigned int getnElem()const{return nElem;}
        inline std::vector<int> getFixedDofs()const{return fixed_dofs;}
        void initMesh(std::string path);
        dMatrix calcKEl(const unsigned int i);
        SparseMatrix assemblyGlobalStiffnessMatrix();
        VectorXd getNodalLoads();
        void applySurfaceForces(VectorXd &loads);
        void applyBodyForces(VectorXd &loads);
        void assignBoundaryConditions(SparseMatrix &K, VectorXd &nodalLoads);
        dMatrix getElementStrains(VectorXd x);
        dMatrix getElementStresses(dMatrix strain);
};


class FEM {
public:
    FEM(){};
    ~FEM(){};

    void initFEM(std::string &path);

    void solve();

    void getNodeDisplacements(VectorXd x);

    void outputResults(const std::string& separator);

    void execFEM();
private:
    // Private member variables for storing problem data

    // Mesh data
    Mesh mesh;

    // Private member variables for storing results

    // Displacements
    std::string file;
    VectorXd dofDisplacements;
    dMatrix strains;
    dMatrix stresses;

};
