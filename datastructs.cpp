#include "datastructs.h"

using namespace std;

/////////////////
// VETOR 3D
// getters e setters
void Vector3D::setX(double x) { m_x = x; }
void Vector3D::setY(double y) { m_y = y; }
void Vector3D::setZ(double z) { m_z = z; }

// operações com vetores

Vector3D Vector3D::operator+(const Vector3D& v) const {
    return Vector3D(m_x + v.m_x, m_y + v.m_y, m_z + v.m_z);
}
Vector3D Vector3D::operator-(const Vector3D& v) const {
    return Vector3D(m_x - v.m_x, m_y - v.m_y, m_z - v.m_z);
}
Vector3D Vector3D::operator*(double scalar) const {
    return Vector3D(m_x * scalar, m_y * scalar, m_z * scalar);
}
double Vector3D::operator*(const Vector3D& v) const {
    return m_x * v.m_x + m_y * v.m_y + m_z * v.m_z;
}
Vector3D Vector3D::operator^(const Vector3D& v) const {
    return Vector3D(m_y * v.m_z - m_z * v.m_y,
                    m_z * v.m_x - m_x * v.m_z,
                    m_x * v.m_y - m_y * v.m_x);
}
double& Vector3D::operator()(const int i) {
    if (i == 0) {
        return m_x;
    } else if (i == 1) {
        return m_y;
    } else if (i == 2) {
        return m_z;
    } else {
        throw std::out_of_range("Invalid index for Vector3D");
    }
}
const double& Vector3D::operator()(const int i) const {
    if (i == 0) {
        return m_x;
    } else if (i == 1) {
        return m_y;
    } else if (i == 2) {
        return m_z;
    } else {
        throw std::out_of_range("Invalid index for Vector3D");
    }
}
Vector3D& Vector3D::operator=(const double (&arr)[3]) {
    m_x = arr[0];
    m_y = arr[1];
    m_z = arr[2];
    return *this;
}

/////////////////
// VETOR XD
VectorXd& VectorXd::operator=(const VectorXd& other) {
    data_ = other.data_;
    return *this;
}
void VectorXd::del(int index) {
    if (index < 0 || index >= data_.size()) {
        throw std::out_of_range("Index out of range");
    }
    
    data_.erase(data_.begin() + index);
}
int VectorXd::size() const { return data_.size(); }
void VectorXd::resize(int size) { data_.resize(size); }
double& VectorXd::operator[](int i) { return data_[i]; }
const double& VectorXd::operator[](int i) const { return data_[i]; }
double& VectorXd::operator()(const int i) { return data_[i]; }
const double& VectorXd::operator()(const int i) const { return data_[i]; }
VectorXd VectorXd::operator+(const VectorXd& v) const
{
    // Operador de soma de vetores
    if (size() != v.size()) {
        throw std::invalid_argument("Os vetores devem ter o mesmo tamanho");
    }
    
    VectorXd result(size());
    
    
    for (int i = 0; i < size(); i++) {
        result[i] = data_[i] + v[i];
    }
    
    return result;
}
VectorXd VectorXd::operator-(const VectorXd& v) const
{
    // Operador de subtração de vetores
    if (size() != v.size()) {
        throw std::invalid_argument("Os vetores devem ter o mesmo tamanho");
    }
    
    
    VectorXd result(size());
    
    
    for (int i = 0; i < size(); i++) {
        result[i] = data_[i] - v[i];
    }
    
    return result;
}
double VectorXd::operator*(const VectorXd& v) const
{
    // Operador de produto escalar entre vetores
    if (size() != v.size()) {
        throw std::invalid_argument("Os vetores devem ter o mesmo tamanho");
    }
    
    double result = 0;
    
    
    // #pragma omp parallel for reduction(+:result)
    for (int i = 0; i < size(); i++) {
        result += data_[i] * v[i];
    }
    
    return result;
}
VectorXd VectorXd::operator*(const double scalar) const {
    // Operador de multiplicação por escalar à esquerda
    VectorXd result(size());
    #pragma omp parallel for
    for (int i = 0; i < size(); i++) {
        result[i] = scalar * data_[i];
    }
    return result;
}
VectorXd operator*(const double scalar, const VectorXd& v) {
    // Operador de multiplicação por escalar à direita
    VectorXd result(v.size());
    #pragma omp parallel for
    for (int i = 0; i < v.size(); i++) {
        result[i] = scalar * v[i];
    }
    return result;
}
VectorXd VectorXd::operator/(const double scalar) const {
    // Operador de divisão por escalar
    VectorXd result(size());
    #pragma omp parallel for
    for (int i = 0; i < size(); i++) {
        result[i] = data_[i] / scalar;
    }
    return result;
}
VectorXd& VectorXd::operator+=(const VectorXd& v) {
    if (data_.size() != v.data_.size()) {
        throw std::runtime_error("Vector dimensions must match for += operator");
    }
    for (int i = 0; i < data_.size(); ++i) {
        data_[i] += v.data_[i];
    }
    return *this;
}
VectorXd& VectorXd::operator-=(const VectorXd& v) {
    if (data_.size() != v.data_.size()) {
        throw std::runtime_error("Vector dimensions must match for -= operator");
    }
    for (int i = 0; i < data_.size(); ++i) {
        data_[i] -= v.data_[i];
    }
    return *this;
}

/////////////////
// MATRIZ INTEIROS
int iMatrix::rows() const { return data_.size(); }
int iMatrix::cols() const { return rows() > 0 ? data_[0].size() : 0; }
void iMatrix::resize(int n_rows, int n_cols) { data_.resize(n_rows, vector<int>(n_cols)); }
void iMatrix::setZero() { data_.assign(rows(), vector<int>(cols(), 0)); }
int& iMatrix::operator()(int i, int j) { return data_[i][j]; }
const int& iMatrix::operator()(int i, int j) const { return data_[i][j]; }
iMatrix iMatrix::operator+(const iMatrix& other) const {
    if (rows() != other.rows() || cols() != other.cols()) {
        throw std::invalid_argument("Matrices must have the same dimensions for addition.");
    }
    iMatrix result(rows(), cols());
    for (int i = 0; i < rows(); ++i) {
        for (int j = 0; j < cols(); ++j) {
            result(i, j) = (*this)(i, j) + other(i, j);
        }
    }

    return result;
}
iMatrix iMatrix::operator-(const iMatrix& other) const {
    if (rows() != other.rows() || cols() != other.cols()) {
        throw std::invalid_argument("Matrices must have the same dimensions for addition.");
    }
    iMatrix result(rows(), cols());

    for (int i = 0; i < rows(); ++i) {
        for (int j = 0; j < cols(); ++j) {
            result(i, j) = (*this)(i, j) - other(i, j);
        }
    }

    return result;
}
iMatrix iMatrix::operator*(const int scalar) const {
    iMatrix result(rows(), cols());
    for (int i = 0; i < rows(); ++i) {
        for (int j = 0; j < cols(); ++j) {
            result(i, j) = (*this)(i, j) * scalar;
        }
    }
    return result;
}
iMatrix iMatrix::operator%(int scalar) const {
    iMatrix result(rows(), cols());
    int num_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads(num_threads);

    int chunk_size = rows() / num_threads;
    int start, end;
    for (int i = 0; i < num_threads; ++i) {
        start = i * chunk_size;
        end = (i + 1) * chunk_size;
        if (i == num_threads - 1) {
            end = rows();
        }
        threads[i] = std::thread([this, &result, scalar, start, end](){
            for (int i = start; i < end; ++i) {
                for (int j = 0; j < cols(); ++j) {
                    result(i, j) = (*this)(i, j) * scalar;
                }
            }
        });
    }

    for (int i = 0; i < num_threads; ++i) {
        threads[i].join();
    }

    return result;
}

/////////////////
// MATRIZ DECIMAIS



dMatrix::dMatrix(int n_rows, int n_cols, const std::vector<std::vector<double>>& values)
    : data_(n_rows, std::vector<double>(n_cols)) {
    if (n_rows != values.size() || n_cols != values[0].size()) {
        throw std::invalid_argument("Error");
    }
    
    for (int i = 0; i < n_rows; ++i) {
        for (int j = 0; j < n_cols; ++j) {
            data_[i][j] = values[i][j];
        }
    }
}

dMatrix dMatrix::id(int size) {
    dMatrix identityMatrix(size, size);
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            if (i == j) {
                identityMatrix(i, j) = 1.0;
            } else {
                identityMatrix(i, j) = 0;
            }
        }
    }
    return identityMatrix;
}
void dMatrix::printMatrix() const {
    for (int i = 0; i < rows(); i++) {
        for (int j = 0; j < cols(); j++) {
            std::cout << data_[i][j] << std::setprecision(2)<< " ";
        }
        std::cout << std::endl;
    }
}

int dMatrix::rows() const { return data_.size(); }
int dMatrix::cols() const { return rows() > 0 ? data_[0].size() : 0; }
void dMatrix::resize(int n_rows, int n_cols) { data_.resize(n_rows, vector<double>(n_cols)); }
void dMatrix::setZero() { data_.assign(rows(), vector<double>(cols(), 0)); }
double dMatrix::norm() const {
    double sumOfSquares = 0.0;
    for (int i = 0; i < rows(); ++i) {
        for (int j = 0; j < cols(); ++j) {
            sumOfSquares += data_[i][j] * data_[i][j];
        }
    }
    return std::sqrt(sumOfSquares);
}

dMatrix dMatrix::block(int start_row, int start_col, int block_rows, int block_cols) const {
    dMatrix result(block_rows, block_cols);
    for (int i = 0; i < block_rows; ++i) {
        for (int j = 0; j < block_cols; ++j) {
            result(i, j) = data_[start_row + i][start_col + j];
        }
    }
    return result;
}

std::vector<double> dMatrix::getRowVector(unsigned int row) const {
    if (row >= 0 && row < rows()) {
        return data_[row];
    } else {
        throw std::out_of_range("Invalid row index");
    }
}
dMatrix dMatrix::getRowMatrix(unsigned int i) const {
    dMatrix row(1, cols());
    if (i >= 0 && i < rows()) {
        for (int j = 0; j < cols(); ++j) {
            row(0, j) = data_[i][j];
        }
    }
    return row;
}
void dMatrix::assignRow(int row, const std::vector<double>& values) {
    if (row >= 0 && row < rows() && values.size() == cols()) {
        for (int j = 0; j < cols(); ++j) {
            data_[row][j] = values[j];
        }
    } else {
        throw std::out_of_range("Invalid row index or mismatch in number of values");
    }
}

double& dMatrix::operator()(int i, int j) { return data_[i][j]; }
const double& dMatrix::operator()(int i, int j) const { return data_[i][j]; }
dMatrix dMatrix::operator+(const dMatrix& other) const {
    if (rows() != other.rows() || cols() != other.cols()) {
        throw std::invalid_argument("Matrices must have the same dimensions for addition.");
    }
    dMatrix result(rows(), cols());
    for (int i = 0; i < rows(); ++i) {
        for (int j = 0; j < cols(); ++j) {
            result(i, j) = (*this)(i, j) + other(i, j);
        }
    }

    return result;
}
dMatrix dMatrix::operator-(const dMatrix& other) const {
    if (rows() != other.rows() || cols() != other.cols()) {
        throw std::invalid_argument("Matrices must have the same dimensions for addition.");
    }
    dMatrix result(rows(), cols());

    for (int i = 0; i < rows(); ++i) {
        for (int j = 0; j < cols(); ++j) {
            result(i, j) = (*this)(i, j) - other(i, j);
        }
    }

    return result;
}
dMatrix& dMatrix::operator-=(const dMatrix& other) {

    if (rows() != other.rows() || cols() != other.cols()) {
        throw std::runtime_error("Matrix dimensions do not match");
    }


    for (int i = 0; i < rows(); ++i) {
        for (int j = 0; j < cols(); ++j) {
            data_[i][j] -= other.data_[i][j];
        }
    }

    return *this;
}
dMatrix& dMatrix::operator+=(const dMatrix& other) {

    if (rows() != other.rows() || cols() != other.cols()) {
        throw std::runtime_error("Matrix dimensions do not match");
    }


    for (int i = 0; i < rows(); ++i) {
        for (int j = 0; j < cols(); ++j) {
            data_[i][j] += other.data_[i][j];
        }
    }

    return *this;
}
dMatrix dMatrix::operator~() const {
    dMatrix result(cols(), rows()); 

    for (int i = 0; i < rows(); ++i) {
        for (int j = 0; j < cols(); ++j) {
            result(j, i) = data_[i][j];  
        }
    }

    return result;
}
dMatrix dMatrix::operator*(const dMatrix& other) const {
    if (cols() != other.rows()) {
        throw std::invalid_argument("Matrix dimensions are incompatible for multiplication.");
    }

    dMatrix result(rows(), other.cols());  

    for (int i = 0; i < rows(); ++i) {
        for (int j = 0; j < other.cols(); ++j) {
            double sum = 0.0;
            for (int k = 0; k < cols(); ++k) {
                sum += data_[i][k] * other.data_[k][j];  
            }
            result(i, j) = sum; 
        }
    }

    return result;
}
dMatrix dMatrix::operator*(const double scalar) const {
    dMatrix result(rows(), cols());
    for (int i = 0; i < rows(); ++i) {
        for (int j = 0; j < cols(); ++j) {
            result(i, j) = (*this)(i, j) * scalar;
        }
    }
    return result;
}
dMatrix operator*(const double scalar, const dMatrix& m) {
    dMatrix result(m.rows(), m.cols());
    for (int i = 0; i < m.rows(); ++i) {
        for (int j = 0; j < m.cols(); ++j) {
            result(i, j) = m(i, j) * scalar;
        }
    }
    return result;

}

dMatrix dMatrix::operator/(double scalar) const {

    dMatrix result(rows(), cols());

    for (int i = 0; i < rows(); ++i) {
        for (int j = 0; j < cols(); ++j) {
            result(i, j) = (*this)(i, j) / scalar;
        }
    }

    return result;
}

dMatrix dMatrix::operator/(const dMatrix& other) const {

    if (cols() != other.rows()) {
        throw std::runtime_error("Cannot divide matrices. Incompatible dimensions.");
    }

    int resultRows = rows();
    int resultCols = other.cols();
    dMatrix result(resultRows, resultCols);

    for (int i = 0; i < resultRows; ++i) {
        for (int j = 0; j < resultCols; ++j) {
            double sum = 0.0;
            for (int k = 0; k < cols(); ++k) {
                sum += (*this)(i, k) / other(k, j);
            }
            result(i, j) = sum;
        }
    }

    return result;
}

template <typename... Args>
dMatrix& dMatrix::operator<<(const Args&... args) {
    if (sizeof...(args) != rows()* cols()) {
        throw std::invalid_argument("Number of values does not match matrix dimensions.");
    }
    assignValues(0, 0, args...);
    return *this;
}
// Function to convert std::vector<double> to Pt
Pt convertToPt(const std::vector<double>& vec) {
  Pt pt = nullptr;
  Pt current = nullptr;

  for (int i = 0; i < vec.size(); ++i) {
    if (vec[i] != 0) {
      Pt newNode = new sparseMatrixNode;
      newNode->row = i;
      newNode->col = 0;  // Assuming you want to create a column vector
      newNode->val = vec[i];
      newNode->next = nullptr;

      if (pt == nullptr) {
        pt = newNode;
        current = pt;
      } else {
        current->next = newNode;
        current = current->next;
      }
    }
  }

  return pt;
}

// Function to delete memory allocated for Pt
void deletePt(Pt& pt) {
  Pt current = pt;
  while (current != nullptr) {
    Pt nextNode = current->next;
    delete current;
    current = nextNode;
  }
  pt = nullptr;
}



/////////////////
// MATRIZ ESPARSA
SparseMatrix::SparseMatrix(int r, int c) {
// Construtor da matriz esparsa
    rows = r;
    cols = c;
    rowsList = (Pt*) malloc (r * sizeof(Pt));
    for (int i = 0; i < rows; i++) {
        rowsList[i] = NULL;
    }
}
int SparseMatrix::getRows() const {
    return rows;
}
int SparseMatrix::getCols() const {
    return cols;
}
Pt SparseMatrix::getRow(int i) const {
    if (i < 0 || i >= rows) {
        return NULL;
    }
    return rowsList[i];
}
double& SparseMatrix::addElem(int r, int c, double val) {
    // Função para adicionar elemento na matriz esparsa
    if (r < 0 || r >= rows || c < 0 || c >= cols) {
        std::cerr << "Invalid index" << std::endl;
        exit(1);
    }
    Pt prev = NULL, curr = rowsList[r];
    while (curr != NULL && curr->col < c) {
        prev = curr;
        curr = curr->next;
    }
    if (curr != NULL && curr->col == c) {
        curr->val = val;
        return curr->val;
    }
    Pt newNode = new sparseMatrixNode;
    newNode->row = r;
    newNode->col = c;
    newNode->val = val;
    newNode->next = curr;
    if (prev == NULL)
        rowsList[r] = newNode;
    else
        prev->next = newNode;
    return newNode->val;
}
void SparseMatrix::printMatrix() {
    // Função para imprimir a matriz esparsa
    for (int i = 0; i < rows; i++) {
        Pt curr = rowsList[i];
        for (int j = 0; j < cols; j++) {
            if (curr != NULL && curr->col == j) {
                cout << curr->val << " ";
                curr = curr->next;
            } else {
                cout << "0 ";
            }
        }
        cout << endl;
    }
}
double SparseMatrix::getElem(int r, int c) const {
    if (r < 0 || r >= rows || c < 0 || c > cols) return 0;
    Pt curr = rowsList[r];
    while (curr != NULL && curr->col < c)
        curr = curr->next;
    if (curr != NULL && curr->col == c) return curr->val;
    return 0;
}
double& SparseMatrix::operator()(int i, int j) {
    return addElem(i, j, this->getElem(i, j));
}
const double& SparseMatrix::operator()(int i, int j) const {
    return this->getElem(i, j);
}
SparseMatrix SparseMatrix::operator*(const SparseMatrix& B) const {
        if (cols != B.getRows()) {
        cout << "Error: incompatible matrix dimensions" << endl;
        return SparseMatrix(0, 0);
    }

    SparseMatrix C(rows, B.getCols());
    // #pragma omp parallel for
    for (int i = 0; i < rows; i++) {
        Pt rowA = this->getRow(i);
        while (rowA != NULL) {
            int j = rowA->col;
            double valA = rowA->val;
            Pt rowB = B.getRow(j);
            while (rowB != NULL) {
                int k = rowB->col;
                double valB = rowB->val;
                double valC = valA * valB + C.getElem(i, k);
                C.addElem(i, k, valC);
                rowB = rowB->next;
            }
            rowA = rowA->next;
        }
    }

    return C;
}
SparseMatrix SparseMatrix::operator+(const SparseMatrix& B) const {
    if (rows != B.getRows() || cols != B.getCols()) {
        cout << "Error: incompatible matrix dimensions" << endl;
        return SparseMatrix(0, 0);
    }

    SparseMatrix C(rows, cols);
    for (int i = 0; i < rows; i++) {
        Pt rowA = this->getRow(i);
        Pt rowB = B.getRow(i);
        while (rowA != NULL || rowB != NULL) {
            int j;
            double valA, valB;
            if (rowA == NULL) {
                j = rowB->col;
                valA = 0;
                valB = rowB->val;
                rowB = rowB->next;
            } else if (rowB == NULL) {
                j = rowA->col;
                valA = rowA->val;
                valB = 0;
                rowA = rowA->next;
            } else if (rowA->col < rowB->col) {
                j = rowA->col;
                valA = rowA->val;
                valB = 0;
                rowA = rowA->next;
            } else if (rowB->col < rowA->col) {
                j = rowB->col;
                valA = 0;
                valB = rowB->val;
                rowB = rowB->next;
            } else {
                j = rowA->col;
                valA = rowA->val;
                valB = rowB->val;
                rowA = rowA->next;
                rowB = rowB->next;
            }
            C.addElem(i, j, valA + valB);
        }
    }

    return C;
}
SparseMatrix SparseMatrix::operator-(const SparseMatrix& B) const {
    if (rows != B.getRows() || cols != B.getCols()) {
        cout << "Error: incompatible matrix dimensions" << endl;
        return SparseMatrix(0, 0);
    }

    SparseMatrix C(rows, cols);
    for (int i = 0; i < rows; i++) {
        Pt rowA = this->getRow(i);
        Pt rowB = B.getRow(i);
        while (rowA != NULL || rowB != NULL) {
            int j;
            double valA, valB;
            if (rowA == NULL) {
                j = rowB->col;
                valA = 0;
                valB = rowB->val;
                rowB = rowB->next;
            } else if (rowB == NULL) {
                j = rowA->col;
                valA = rowA->val;
                valB = 0;
                rowA = rowA->next;
            } else if (rowA->col < rowB->col) {
                j = rowA->col;
                valA = rowA->val;
                valB = 0;
                rowA = rowA->next;
            } else if (rowB->col < rowA->col) {
                j = rowB->col;
                valA = 0;
                valB = rowB->val;
                rowB = rowB->next;
            } else {
                j = rowA->col;
                valA = rowA->val;
                valB = rowB->val;
                rowA = rowA->next;
                rowB = rowB->next;
            }
            C.addElem(i, j, valA - valB);
        }
    }

    return C;
}
SparseMatrix SparseMatrix::operator*(const double scalar) const {
    SparseMatrix C(rows, cols);
    // #pragma omp parallel for
    for (int i = 0; i < rows; i++) {
        Pt row = getRow(i);
        while (row != NULL) {
            int j = row->col;
            double val = row->val * scalar;
            C.addElem(i, j, val);
            row = row->next;
        }
    }

    return C;
}
SparseMatrix operator*(const double scalar, const SparseMatrix &A){
    SparseMatrix C(A.getRows(), A.getCols());
    // #pragma omp parallel for
    for (int i = 0; i < A.getRows(); i++) {
        Pt row = A.getRow(i);
        while (row != NULL) {
            int j = row->col;
            double val = row->val * scalar;
            C.addElem(i, j, val);
            row = row->next;
        }
    }

    return C;
}
void SparseMatrix::assignBand(const std::vector<double>& bandVals, int k) {
    k-=2;
    int bandSize = bandVals.size();
    #pragma omp parallel for
    for (int i = 0; i < rows; i++) {
        int startCol = std::max(0, i - k);
        int endCol = std::min(cols - 1, i + k);
        for (int j = startCol; j <= endCol; j++) {
            int bandIndex = j - i + k;
            if (bandIndex >= 0 && bandIndex < bandSize) {
                addElem(i, j, bandVals[bandIndex]);
            }
        }
    }
}
void SparseMatrix::assign(const std::vector<std::vector<double>>& values) {
    if (values.size() != rows || values[0].size() != cols) {
        std::cerr << "Invalid matrix dimensions" << std::endl;
        exit(1);
    }

    const int block_size = 32;  

    for (int bi = 0; bi < rows; bi += block_size) {  
        for (int bj = 0; bj < cols; bj += block_size) {
            int block_rows = std::min(block_size, rows - bi);
            int block_cols = std::min(block_size, cols - bj);


            std::vector<std::vector<double>> block(block_rows, std::vector<double>(block_cols, 0.0));


            for (int i = 0; i < block_rows; ++i) {
                for (int j = 0; j < block_cols; ++j) {
                    block[i][j] = values[bi + i][bj + j];
                }
            }


            for (int i = 0; i < block_rows; ++i) {
                for (int j = 0; j < block_cols; ++j) {
                    addElem(bi + i, bj + j, block[i][j]);
                }
            }
        }
    }
}
SparseMatrix SparseMatrix::operator~() {
    SparseMatrix transposed(cols, rows);

    for (int i = 0; i < rows; i++) {
        Pt curr = rowsList[i];
        while (curr != NULL) {
            int j = curr->col;
            double val = curr->val;
            transposed.addElem(j, i, val);
            curr = curr->next;
        }
    }

    return transposed;
}
void SparseMatrix::eraseRow(int rowIndex) {
    if (rowIndex < 0 || rowIndex >= rows) {
        std::cerr << "Invalid row index" << std::endl;
        return;
    }

    Pt curr = rowsList[rowIndex];
    while (curr != NULL) {
        Pt next = curr->next;
        delete curr;
        curr = next;
    }
    rowsList[rowIndex] = NULL;
}
void SparseMatrix::deleteElem(int row, int col) {
    if (row < 0 || row >= rows || col < 0 || col >= cols) {
        std::cerr << "Invalid row or column index" << std::endl;
        return;
    }

    Pt prev = nullptr;
    Pt curr = rowsList[row];

    if(curr == 0){
        return;
    }
    while (curr != nullptr && curr->col != col) {
        prev = curr;
        curr = curr->next;
    }

    if (curr != nullptr && curr->col == col) {
        if (prev == nullptr) {
            // If the node to be deleted is the first node in the row
            rowsList[row] = curr->next;
        } else {
            // If the node to be deleted is not the first node
            prev->next = curr->next;
        }

        delete curr;
    }
}


void SparseMatrix::eraseCol(int colIndex) {
    #pragma omp parallel for
    for (int i = 0; i < rows; i++) {
        deleteElem(i,colIndex);
    }
}
double SparseMatrix::computeNorm(Pt row) const {
    double norm = 0.0;
    while (row != NULL) {
        norm += row->val * row->val;
        row = row->next;
    }
    return std::sqrt(norm);
}

double SparseMatrix::computeDotProduct(Pt row1, Pt row2) const {
    double dotProduct = 0.0;
    while (row1 != NULL && row2 != NULL) {
        if (row1->col < row2->col) {
            row1 = row1->next;
        } else if (row1->col > row2->col) {
            row2 = row2->next;
        } else {
            dotProduct += row1->val * row2->val;
            row1 = row1->next;
            row2 = row2->next;
        }
    }
    return dotProduct;
}

Pt SparseMatrix::copyRow(Pt row) const {
    Pt newRow = NULL;
    Pt current = NULL;

    while (row != NULL) {
        Pt newNode = new sparseMatrixNode;
        newNode->col = row->col;
        newNode->val = row->val;
        newNode->next = NULL;

        if (newRow == NULL) {
            newRow = newNode;
            current = newNode;
        } else {
            current->next = newNode;
            current = current->next;
        }

        row = row->next;
    }

    return newRow;
}

Pt SparseMatrix::scaleRow(Pt row, double factor)const{
    while (row != NULL) {
        row->val *= factor;
        row = row->next;
    }
    return row;
}

void SparseMatrix::subtractRow(Pt row1, Pt row2, double factor){
    while (row1 != NULL && row2 != NULL) {
        if (row1->col < row2->col) {
            row1 = row1->next;
        } else if (row1->col > row2->col) {
            addElem(row2->col, row2->col, -factor);
            row2 = row2->next;
        } else {
            row1->val -= factor * row2->val;
            row1 = row1->next;
            row2 = row2->next;
        }
    }

}


void SparseMatrix::assignCol(int colIndex, Pt col) {
    if (colIndex < 0 || colIndex >= cols) {
        std::cerr << "Invalid column index" << std::endl;
        exit(1);
    }

    Pt prev = NULL, curr = rowsList[colIndex];
    while (curr != NULL) {
        Pt temp = curr;
        curr = curr->next;
        delete temp;
    }

    rowsList[colIndex] = NULL;

    Pt prevRow = NULL;
    while (col != NULL) {
        Pt newNode = new sparseMatrixNode;
        newNode->col = col->col;
        newNode->val = col->val;
        newNode->next = NULL;

        if (prevRow == NULL) {
            rowsList[colIndex] = newNode;
        } else {
            prevRow->next = newNode;
        }

        prevRow = newNode;

        Pt temp = col;
        col = col->next;
        delete temp;
    }
}

VectorXd SparseMatrix::matrixVectorProduct(const VectorXd& x) const {
    int n = getRows();
    VectorXd result(n);

    for (int i = 0; i < n; ++i) {
        Pt row = getRow(i);
        double innerProduct = 0.0;

        while (row != nullptr) {
            innerProduct += row->val * x[row->col];
            row = row->next;
        }

        result[i] = innerProduct;
    }

    return result;
}

VectorXd SparseMatrix::conjugateGradient(const VectorXd& b) const {
    int n = getRows(); // Size of the matrix

    VectorXd x(n); // Initial solution vectors
    VectorXd r = b; // Residual vector
    VectorXd p = r; // Search direction vector
    double rsold = r * r; // Compute initial residual squared norm

    for (int i = 0; i < 100; ++i) {
        VectorXd Ap = matrixVectorProduct(p); // Compute A * p
        double alpha = rsold / (p * Ap); // Compute step size

        x = x + alpha * p; // Update solution vector
        r = r - alpha * Ap; // Update residual vector

        double rsnew = r * r; // Compute new residual squared norm
        if (rsnew < 10E-6) {
            break; // Convergence criterion reached
        }

        p = r + (rsnew / rsold) * p; // Update search direction vector

        rsold = rsnew; // Update previous residual squared norm
    }

    return x;
}
////////////
// Overloads 
std::ostream& operator<<(std::ostream& os, const VectorXd& v) {
        os << "[ ";
        for (int i = 0; i < v.size(); ++i) {
            os << v(i) << " ";
        }
        os << "]";
        return os;
    }
std::ostream& operator<<(std::ostream& os, const iMatrix& m) {
    for (int i = 0; i < m.rows(); ++i) {
        for (int j = 0; j < m.cols(); ++j) {
            os << m(i, j) << " ";
        }
        os << std::endl;
    }
    return os;
}
std::ostream& operator<<(std::ostream& os, const dMatrix& m) {
    for (int i = 0; i < m.rows(); ++i) {
        for (int j = 0; j < m.cols(); ++j) {
            os << m(i, j) << " ";
        }
        os << std::endl;
    }
    return os;
}

//////////
// Timer

void Timer::start() {
    start_time = std::chrono::system_clock::now();
}
double Timer::stop() {
    end_time = std::chrono::system_clock::now();
    total_time = end_time - start_time;
    return total_time.count(); 
}


