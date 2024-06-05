#include <iostream>
#include <vector>
#include <cmath>
#include <random>

class Matrix {
public:
    int rows, cols;
    std::vector<std::vector<float>> matrix;

    Matrix(int r, int c) : rows(r), cols(c), matrix(r, std::vector<float>(c, 0)) {}

    Matrix(const std::vector<std::vector<float>>& m) : matrix(m), rows(m.size()), cols(m[0].size()) {}

    void output() const {
        for (const auto& row : matrix) {
            for (float val : row) {
                std::cout << val << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }

    Matrix dot(const Matrix& n) const {
        Matrix result(rows, n.cols);

        if (cols == n.rows) {
            for (int i = 0; i < rows; ++i) {
                for (int j = 0; j < n.cols; ++j) {
                    float sum = 0;
                    for (int k = 0; k < cols; ++k) {
                        sum += matrix[i][k] * n.matrix[k][j];
                    }
                    result.matrix[i][j] = sum;
                }
            }
        }
        return result;
    }

    void randomize() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(-1, 1);

        for (auto& row : matrix) {
            for (auto& val : row) {
                val = dis(gen);
            }
        }
    }

    Matrix singleColumnMatrixFromArray(const std::vector<float>& arr) const {
        Matrix n(arr.size(), 1);
        for (size_t i = 0; i < arr.size(); ++i) {
            n.matrix[i][0] = arr[i];
        }
        return n;
    }

    std::vector<float> toArray() const {
        std::vector<float> arr(rows * cols);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                arr[j + i * cols] = matrix[i][j];
            }
        }
        return arr;
    }

    Matrix addBias() const {
        Matrix n(rows + 1, 1);
        for (int i = 0; i < rows; ++i) {
            n.matrix[i][0] = matrix[i][0];
        }
        n.matrix[rows][0] = 1;
        return n;
    }

    Matrix activate() const {
        Matrix n(rows, cols);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                n.matrix[i][j] = relu(matrix[i][j]);
            }
        }
        return n;
    }

    float relu(float x) const {
        return std::max(0.0f, x);
    }

    void mutate(float mutationRate) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0, 1);
        std::normal_distribution<> gauss(0, 1);

        for (auto& row : matrix) {
            for (auto& val : row) {
                if (dis(gen) < mutationRate) {
                    val += gauss(gen) / 5;
                    if (val > 1) val = 1;
                    if (val < -1) val = -1;
                }
            }
        }
    }

    Matrix crossover(const Matrix& partner) const {
        Matrix child(rows, cols);

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> disCol(0, cols - 1);
        std::uniform_int_distribution<> disRow(0, rows - 1);

        int randC = disCol(gen);
        int randR = disRow(gen);

        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                if (i < randR || (i == randR && j <= randC)) {
                    child.matrix[i][j] = matrix[i][j];
                } else {
                    child.matrix[i][j] = partner.matrix[i][j];
                }
            }
        }
        return child;
    }

    Matrix clone() const {
        Matrix clone(rows, cols);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                clone.matrix[i][j] = matrix[i][j];
            }
        }
        return clone;
    }

private:
    float random(float min, float max) const {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(min, max);
        return dis(gen);
    }
};

int main() {
    // Example usage of the Matrix class
    Matrix m(3, 3);
    m.randomize();
    m.output();

    Matrix n(3, 3);
    n.randomize();
    n.output();

    Matrix result = m.dot(n);
    result.output();

    return 0;
}

