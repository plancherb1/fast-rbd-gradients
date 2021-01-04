// We need to multiply rc of A by b
// but values on exisit in LOWER or UPPER
// thinking in rows then j is column
// therefore in LOWER we need rc <= j
// and in UPPER we need rc >=j else flip
template <typename T, int K, int LOWER = 0>
__device__
T dotProdMvSym_GPU(T *A, int ld_A, T *b, int row){
   T val = static_cast<T>(0);
   #pragma unroll
   for (int j=0; j < K; j++){
      if (LOWER){
         int ind = row < j ? row * ld_A + j : row + ld_A * j;
         val += A[ind] * b[j];
      }
      else{
         int ind = row > j ? row * ld_A + j : row + ld_A * j;
         val += A[ind] * b[j];
      }
   }
   return val;
}


// basic matrix vector multiply D (+)= alpha*A*B (+ beta*C) but A is a symmetric matrix with only the triangle stored (default UPPER)
template <typename T, int M, int N, int K, int PEQFLAG = 0, int LOWER = 0, int T_C = 0, int T_D = 0>
__device__
void matMultSym_GPU(T *D, int ld_D, T *A, int ld_A, T *B, int ld_B, T alpha = 1.0, T *C = nullptr, int ld_C = 0, T beta = 1.0){
   int startx, dx, starty, dy; doubleLoopVals_GPU(&startx, &dx, &starty, &dy);
   #pragma unroll
   for (int ky = starty; ky < N; ky += dy){
      #pragma unroll
      for (int kx = startx; kx < M; kx += dx){
         T val = alpha*dotProdMvSym_GPU<T,K,LOWER>(A,ld_A,&B[ky*ld_B],kx);
         if (C != nullptr){
            int ind_C = T_C ? ky + ld_C*kx : kx + ld_C*ky;
            val +=   beta*C[ind_C];
         }
         int ind_D = T_D ? ky + ld_D*kx : kx + ld_D*ky;
         if (PEQFLAG){D[ind_D] += val;}
         else{D[ind_D] = val;}
      }
   }
}