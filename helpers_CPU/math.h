#include <Eigen/Dense>

// basic matrix vector multiplication c (+) A*b
template <typename T, int M, int K, int PEQFLAG = 0, int T_A = 0>
void matVMult(T *c, T *A, T *b){
    Eigen::Map<Eigen::Matrix<T,M,K>> A_eigen(A,M,K);
    Eigen::Map<Eigen::Matrix<T,K,1>> b_eigen(b,K,1);
    Eigen::Map<Eigen::Matrix<T,M,1>> c_eigen(c,M,1);
    if(T_A && PEQFLAG){c_eigen.noalias() += A_eigen.transpose()*b_eigen;}
    else if(T_A){c_eigen.noalias() = A_eigen.transpose()*b_eigen;}
    else if(PEQFLAG){c_eigen.noalias() += A_eigen*b_eigen;}
    else{c_eigen.noalias() = A_eigen*b_eigen;}
}

// basic matrix vector multiply C (+)= alpha*A*B but A is a symmetric matrix with only the triangle stored (default UPPER)
template <typename T, int M, int N, int K, int PEQFLAG = 0, int LOWER = 0>
void matMultSym(T *C, int ld_C, T *A, int ld_A, T *B, int ld_B, T alpha = 1.0){
    Eigen::Map<Eigen::Matrix<T,M,K>> A_eigen(A,M,K);
    Eigen::Map<Eigen::Matrix<T,K,N>> B_eigen(B,K,N);
    Eigen::Map<Eigen::Matrix<T,M,N>> C_eigen(C,M,N);
    if(LOWER && PEQFLAG){C_eigen.noalias() += alpha*A_eigen.template selfadjointView<Eigen::Lower>()*B_eigen;}
    else if(LOWER){C_eigen.noalias() = alpha*A_eigen.template selfadjointView<Eigen::Lower>()*B_eigen;}
    else if(PEQFLAG){C_eigen.noalias() += alpha*A_eigen.template selfadjointView<Eigen::Upper>()*B_eigen;}
    else{C_eigen.noalias() = alpha*A_eigen.template selfadjointView<Eigen::Upper>()*B_eigen;}
}

// basic matrix vector multiply c (+)= alpha*A*b but A is a symmetric matrix with only the triangle stored (default UPPER)
template <typename T, int M, int K, int PEQFLAG = 0, int LOWER = 0>
void matVMultSym(T *c, T *A, int ld_A, T *b, T alpha = 1.0){
    Eigen::Map<Eigen::Matrix<T,M,K>> A_eigen(A,M,K);
    Eigen::Map<Eigen::Matrix<T,K,1>> b_eigen(b,K,1);
    Eigen::Map<Eigen::Matrix<T,M,1>> c_eigen(c,M,1);
    if(LOWER && PEQFLAG){c_eigen.noalias() += alpha*A_eigen.template selfadjointView<Eigen::Lower>()*b_eigen;}
    else if(LOWER){c_eigen.noalias() = alpha*A_eigen.template selfadjointView<Eigen::Lower>()*b_eigen;}
    else if(PEQFLAG){c_eigen.noalias() += alpha*A_eigen.template selfadjointView<Eigen::Upper>()*b_eigen;}
    else{c_eigen.noalias() = alpha*A_eigen.template selfadjointView<Eigen::Upper>()*b_eigen;}
}


// Note tha we tried multiple permutations of Eigen (one example below)
// for the Tmat multiplication -- it remained much faster to do our custom math
// template <typename T, bool PEQFLAG = 0, bool T_A = 0>
// void matVMult_Tmat(T *out, T *Tmat, T *vec){
    // Eigen::Map<Eigen::Matrix<T,6,6>> Tmat_eigen(Tmat,6,6);
    // Eigen::Map<Eigen::Matrix<T,6,1>> vec_eigen(vec,6,1);
    // Eigen::Map<Eigen::Matrix<T,6,1>> out_eigen(out,6,1);
    // if(T_A && PEQFLAG){
    //     out_eigen.template topRows<3>().noalias() += Tmat_eigen.transpose().template topRows<3>()*vec_eigen;
    //     out_eigen.template bottomRows<3>().noalias() += Tmat_eigen.transpose().template bottomRightCorner<3,3>()*vec_eigen.template bottomRows<3>();
    // }
    // else if(T_A){
    //     out_eigen.template topRows<3>().noalias() = Tmat_eigen.transpose().template topRows<3>()*vec_eigen;
    //     out_eigen.template bottomRows<3>().noalias() = Tmat_eigen.transpose().template bottomRightCorner<3,3>()*vec_eigen.template bottomRows<3>();
    // }
    // else if(PEQFLAG){
    //     out_eigen.template topRows<3>().noalias() += Tmat_eigen.template topLeftCorner<3,3>()*vec_eigen.template topRows<3>();
    //     out_eigen.template bottomRows<3>().noalias() += Tmat_eigen.template bottomRows<3>()*vec_eigen;
    // }
    // else{
    //     out_eigen.template topRows<3>().noalias() = Tmat_eigen.template topLeftCorner<3,3>()*vec_eigen.template topRows<3>();
    //     out_eigen.template bottomRows<3>().noalias() = Tmat_eigen.template bottomRows<3>()*vec_eigen;
    // }
// }

template <typename T, bool T_A = 0>
T dotProd_Tmat(T *Tmat, T *vec, int r){
   T val = static_cast<T>(0);
   if(!T_A){
      if(r < 3){for (int i = 0; i < 3; i++){val += Tmat[r + 6 * i]*vec[i];}}
      else{for (int i = 0; i < 6; i++){val += Tmat[r + 6 * i]*vec[i];}}
   }
   else{
      if(r >= 3){for (int i = 3; i < 6; i++){val += Tmat[r * 6 + i]*vec[i];}}
      else{for (int i = 0; i < 6; i++){val += Tmat[r * 6 + i]*vec[i];}}
   }
   return val;
}

// basic matrix vector multiply out (+)= Tmat*vec
template <typename T, bool PEQFLAG = 0, bool T_A = 0>
void matVMult_Tmat(T *out, T *Tmat, T *vec){
    // Sparsity pattern is = [data1   0    
    //                        data2   data1]
    for (int r = 0; r < 6; r ++){
        if (PEQFLAG){out[r] += dotProd_Tmat<T,T_A>(Tmat,vec,r);}
        else{out[r] = dotProd_Tmat<T,T_A>(Tmat,vec,r);}
    }
}
