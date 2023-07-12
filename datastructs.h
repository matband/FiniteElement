#pragma once

#include <vector>
#include <iostream>
#include <ctime>
#include <omp.h>
#include <thread>
#include <chrono>
#include <cmath>
#include <iomanip>

struct sparseMatrixNode {
    int col, row;
    double val;
    sparseMatrixNode* next;
};
typedef sparseMatrixNode* Pt;


class Vector3D {
public:
    // construtor padrão
    Vector3D() : m_x(0.0), m_y(0.0), m_z(0.0) {}

    // construtor com parâmetros
    Vector3D(double x, double y, double z) : m_x(x), m_y(y), m_z(z) {}

    // getters e setters
    void setX(double x);
    void setY(double y);
    void setZ(double z);

    // operações com vetores
    Vector3D operator+(const Vector3D& v) const;
    Vector3D operator-(const Vector3D& v) const;
    Vector3D operator*(double scalar) const;
    double operator*(const Vector3D& v) const;
    Vector3D operator^(const Vector3D& v) const;
    double& operator()(const int i);
    const double& operator()(const int i) const;
    Vector3D& operator=(const double (&arr)[3]);
private:
    double m_x;
    double m_y;
    double m_z;
};
class VectorXd {
public:
    // Construtores
    VectorXd() {}
    VectorXd(int size) : data_(size) {}
    VectorXd(int size, double value) : data_(size, value) {}
    VectorXd(const std::vector<double>& values) : data_(values) {} 
    
    // Métodos públicos
    void del(int index);
    int size() const;
    void resize(int size);
    double& operator[](int i);
    const double& operator[](int i) const;
    double& operator()(const int i);
    const double& operator()(const int i) const;
    VectorXd& operator=(const VectorXd& other);

    // Operações com vetores
    VectorXd operator+(const VectorXd& v) const;
    VectorXd operator-(const VectorXd& v) const;
    double operator*(const VectorXd& v) const;

    // Operações com escalares
    VectorXd operator*(double scalar) const;
    friend VectorXd operator*(double scalar, const VectorXd& v);
    VectorXd operator/(const double scalar) const;
    VectorXd& operator+=(const VectorXd& v);
    VectorXd& operator-=(const VectorXd& v);

private:
    std::vector<double> data_;
};
class SparseMatrix {
private:
    // Estrutura do nó da matriz esparsa

    int rows, cols;
    Pt* rowsList;

public:
    // Construtor da matriz esparsa
    SparseMatrix(int r, int c);

    int getRows() const;
    int getCols() const;

    Pt getRow(int i) const;

    // Methods
    double& addElem(int r, int c, double val);
    void printMatrix();
    void assignBand(const std::vector<double>& bandVals, int k);
    void assign(const std::vector<std::vector<double>>& values);
    double getElem(int r, int c) const;
    void deleteElem(int row, int col);
    void eraseRow(int rowIndex);
    void eraseCol(int colIndex);
    double computeNorm(Pt row) const;
    double computeDotProduct(Pt row1, Pt row2) const;
    Pt copyRow(Pt row) const;
    Pt scaleRow(Pt row, double factor)const;
    void addColumn(const SparseMatrix& columnMatrix);
    void subtractRow(Pt row1, Pt row2, double factor);
    bool luDecomposition(SparseMatrix& L, SparseMatrix& U) const;
    void assignCol(int colIndex, Pt col); 
    VectorXd matrixVectorProduct(const VectorXd& x) const;  
    VectorXd conjugateGradient(const VectorXd& b) const;
    //Operators
    double& operator()(int i, int j);
    const double& operator()(int i, int j) const;
    SparseMatrix operator+(const SparseMatrix& B) const;
    SparseMatrix operator-(const SparseMatrix& B) const;
    SparseMatrix operator*(const double scalar) const;
    SparseMatrix operator*(const SparseMatrix& B) const;
    SparseMatrix operator&(const SparseMatrix& other) const;
    SparseMatrix operator~();
};
class iMatrix {
public:
    // Construtores
    iMatrix(int n_rows, int n_cols) : data_(n_rows, std::vector<int>(n_cols)) {}
    iMatrix() : iMatrix(0, 0) {}
    
    // Métodos públicos
    int rows() const;
    int cols() const;
    void resize(int n_rows, int n_cols);
    void setZero();
    int& operator()(int i, int j);
    const int& operator()(int i, int j) const;
    iMatrix operator+(const iMatrix& other) const;
    iMatrix operator-(const iMatrix& other) const;
    iMatrix operator*(const int scalar) const;
    iMatrix operator%(int scalar) const;
private:
    // Atributos
    std::vector<std::vector<int>> data_;
};
class dMatrix {
public:
    // Construtores
    dMatrix(int n_rows, int n_cols, const std::vector<std::vector<double>>& values);
    dMatrix(int n_rows, int n_cols) : data_(n_rows, std::vector<double>(n_cols,0)) {}
    dMatrix(int n_rows, int n_cols, double initial_value) : data_(n_rows, std::vector<double>(n_cols, initial_value)) {}
    dMatrix() : dMatrix(0, 0) {}

    static dMatrix id(int size);

    // Métodos públicos
    int rows() const;
    int cols() const;
    void resize(int n_rows, int n_cols);
    void setZero();
    double norm() const;
    void printMatrix() const;
    dMatrix block(int start_row, int start_col, int block_rows, int block_cols) const;
    std::vector<double> getRowVector(unsigned int row) const;
    dMatrix getRowMatrix(unsigned int i) const;
    void assignRow(int row, const std::vector<double>& values);

    double& operator()(int i, int j);
    const double& operator()(int i, int j) const;
    dMatrix operator+(const dMatrix& other) const;
    dMatrix operator-(const dMatrix& other) const;

    dMatrix& operator-=(const dMatrix& other);
    dMatrix& operator+=(const dMatrix& other);

    dMatrix operator~() const;
    dMatrix operator*(const dMatrix& other) const;
    dMatrix operator*(const double scalar) const;  
    dMatrix operator/(double scalar) const;
    dMatrix operator/(const dMatrix& other) const;

    template <typename... Args>
    dMatrix& operator<<(const Args&... args);

private:
    // Atributos
    std::vector<std::vector<double>> data_;
    friend std::ostream& operator<<(std::ostream& os, const dMatrix& matrix);
};

class Timer {
public: 
    void start();
    double stop();
private:
    std::chrono::time_point<std::chrono::system_clock> start_time, end_time;
    std::chrono::duration<double> total_time;
};

std::ostream& operator<<(std::ostream& os, const dMatrix& m);
std::ostream& operator<<(std::ostream& os, const iMatrix& m);
std::ostream& operator<<(std::ostream& os, const VectorXd& v);
SparseMatrix operator*(const double scalar, const SparseMatrix &A);
dMatrix operator*(const double scalar, const dMatrix& m);