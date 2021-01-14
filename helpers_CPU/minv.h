#define rcJ_IndCpp(r,c,ld) ((r-1) + (c-1)*ld)
#define krcJ_IndCpp(k,r,c,ld) (rcJ_IndCpp(r,c,ld) + (k-1)*ld*ld)
#define rc_Ind(r,c,ld) (r + c*ld)
#define krc_Ind(k,r,c,ld) (rc_Ind(r,c,ld) + k*ld*ld)

template <typename T, int K>
T dotProd(T *a, int s_a, T *b, int s_b){
   T val = 0;
   for (int j=0; j < K; j++){val += a[s_a * j] * b[s_b * j];}
   return val;
}

// iterated matVMult for C[:,col_i] += alpha*A*B[:,col_i] for all i in some set
// assumes C and B are pointers to col_0 and then we just need the next N cols
template <typename T, int M, int K, int PEQFLAG = 0, int T_A = 0>
void iteratedMatVMult(T *C, int ld_C, T *A, int ld_A, T *B, int ld_B, int col_N, T alpha = 1.0){
   for (int col = 0; col < col_N; col++){
      T *Cc = &C[col*ld_C]; T *Bc = &B[col*ld_B];
      for (int row = 0; row < M; row++){
         if (T_A && PEQFLAG){Cc[row] += alpha*dotProd<T,K>(&A[row*ld_A],1,Bc,1);}
         else if (PEQFLAG){Cc[row] += alpha*dotProd<T,K>(&A[row],ld_A,Bc,1);}
         else if (T_A){Cc[row] = alpha*dotProd<T,K>(&A[row*ld_A],1,Bc,1);}
         else{Cc[row] = alpha*dotProd<T,K>(&A[row],ld_A,Bc,1);}
      }
   }
}

//-------------------------------------------------------------------------------
// Helper Function to Transform a Symmetric Matrix A with the Rotation Matrix E
//-------------------------------------------------------------------------------
//
// Performs the transformation of a symmetric 3x3 matrix A, with the rotation
// matrix E
//    B = E * A * E^T
//
// These calculations are documented in the appendix of Roy's book
//
// -- RobCoGen
//

template <typename T>
void rot_symmetric_EAET(T *B, T *E, T *A){
   T LXX = A[rcJ_IndCpp(1,1,3)] - A[rcJ_IndCpp(3,3,3)];
   T LXY = A[rcJ_IndCpp(1,2,3)]; // same as LYX
   T LYY = A[rcJ_IndCpp(2,2,3)] - A[rcJ_IndCpp(3,3,3)];
   T LZX = 2*A[rcJ_IndCpp(1,3,3)];
   T LZY = 2*A[rcJ_IndCpp(2,3,3)];

   T yXX = E[rcJ_IndCpp(2,1,3)]*LXX + E[rcJ_IndCpp(2,2,3)]*LXY + E[rcJ_IndCpp(2,3,3)]*LZX;
   T yXY = E[rcJ_IndCpp(2,1,3)]*LXY + E[rcJ_IndCpp(2,2,3)]*LYY + E[rcJ_IndCpp(2,3,3)]*LZY;
   T yYX = E[rcJ_IndCpp(3,1,3)]*LXX + E[rcJ_IndCpp(3,2,3)]*LXY + E[rcJ_IndCpp(3,3,3)]*LZX;
   T yYY = E[rcJ_IndCpp(3,1,3)]*LXY + E[rcJ_IndCpp(3,2,3)]*LYY + E[rcJ_IndCpp(3,3,3)]*LZY;

   T v1 = -A[rcJ_IndCpp(2,3,3)];
   T v2 =  A[rcJ_IndCpp(1,3,3)];
   T EvX = E[rcJ_IndCpp(1,1,3)]*v1 + E[rcJ_IndCpp(1,2,3)]*v2;
   T EvY = E[rcJ_IndCpp(2,1,3)]*v1 + E[rcJ_IndCpp(2,2,3)]*v2;
   T EvZ = E[rcJ_IndCpp(3,1,3)]*v1 + E[rcJ_IndCpp(3,2,3)]*v2;

   B[rcJ_IndCpp(1,2,3)] = yXX * E[rcJ_IndCpp(1,1,3)] + yXY * E[rcJ_IndCpp(1,2,3)] + EvZ;
   B[rcJ_IndCpp(1,3,3)] = yYX * E[rcJ_IndCpp(1,1,3)] + yYY * E[rcJ_IndCpp(1,2,3)] - EvY;
   B[rcJ_IndCpp(2,3,3)] = yYX * E[rcJ_IndCpp(2,1,3)] + yYY * E[rcJ_IndCpp(2,2,3)] + EvX;

   T zYY = yXX * E[rcJ_IndCpp(2,1,3)] + yXY * E[rcJ_IndCpp(2,2,3)];
   T zZZ = yYX * E[rcJ_IndCpp(3,1,3)] + yYY * E[rcJ_IndCpp(3,2,3)];
   B[rcJ_IndCpp(1,1,3)] = LXX + LYY - zYY - zZZ + A[rcJ_IndCpp(3,3,3)];
   B[rcJ_IndCpp(2,2,3)] = zYY + A[rcJ_IndCpp(3,3,3)];
   B[rcJ_IndCpp(3,3,3)] = zZZ + A[rcJ_IndCpp(3,3,3)];
   
   // zero out the rest
   B[rcJ_IndCpp(2,1,3)] = static_cast<T>(0);
   B[rcJ_IndCpp(3,1,3)] = static_cast<T>(0);
   B[rcJ_IndCpp(3,2,3)] = static_cast<T>(0);
}

//-------------------------------------------------------------------------------
// Helper Function to Transform a Matrix A with the Rotation Matrix E
//-------------------------------------------------------------------------------
//
// Performs the transformation of a 3x3 matrix A, with the rotation
// matrix E
//    B = E * A * E^T
//
// -- RobCoGen
//
template <typename T>
void rot_EAET(T *B, T *E, T *A){
   T v_4X = A[rcJ_IndCpp(1,1,3)] - A[rcJ_IndCpp(3,3,3)];
   T v_4Y = A[rcJ_IndCpp(1,2,3)];
   T v_5X = A[rcJ_IndCpp(2,1,3)];
   T v_5Y = A[rcJ_IndCpp(2,2,3)] - A[rcJ_IndCpp(3,3,3)];
   T v_6X = A[rcJ_IndCpp(3,1,3)] + A[rcJ_IndCpp(1,3,3)];
   T v_6Y = A[rcJ_IndCpp(3,2,3)] + A[rcJ_IndCpp(2,3,3)];

   T v1 = -A[rcJ_IndCpp(2,3,3)];
   T v2 =  A[rcJ_IndCpp(1,3,3)];
   T EvX = E[rcJ_IndCpp(1,1,3)]*v1 + E[rcJ_IndCpp(1,2,3)]*v2;
   T EvY = E[rcJ_IndCpp(2,1,3)]*v1 + E[rcJ_IndCpp(2,2,3)]*v2;
   T EvZ = E[rcJ_IndCpp(3,1,3)]*v1 + E[rcJ_IndCpp(3,2,3)]*v2;

   T yXX = E[rcJ_IndCpp(1,1,3)]*v_4X + E[rcJ_IndCpp(1,2,3)]*v_5X + E[rcJ_IndCpp(1,3,3)]*v_6X;
   T yXY = E[rcJ_IndCpp(1,1,3)]*v_4Y + E[rcJ_IndCpp(1,2,3)]*v_5Y + E[rcJ_IndCpp(1,3,3)]*v_6Y;
   T yYX = E[rcJ_IndCpp(2,1,3)]*v_4X + E[rcJ_IndCpp(2,2,3)]*v_5X + E[rcJ_IndCpp(2,3,3)]*v_6X;
   T yYY = E[rcJ_IndCpp(2,1,3)]*v_4Y + E[rcJ_IndCpp(2,2,3)]*v_5Y + E[rcJ_IndCpp(2,3,3)]*v_6Y;
   T yZX = E[rcJ_IndCpp(3,1,3)]*v_4X + E[rcJ_IndCpp(3,2,3)]*v_5X + E[rcJ_IndCpp(3,3,3)]*v_6X;
   T yZY = E[rcJ_IndCpp(3,1,3)]*v_4Y + E[rcJ_IndCpp(3,2,3)]*v_5Y + E[rcJ_IndCpp(3,3,3)]*v_6Y;

   B[rcJ_IndCpp(1,1,3)] = yXX*E[rcJ_IndCpp(1,1,3)] + yXY*E[rcJ_IndCpp(1,2,3)] + A[rcJ_IndCpp(3,3,3)];
   B[rcJ_IndCpp(2,2,3)] = yYX*E[rcJ_IndCpp(2,1,3)] + yYY*E[rcJ_IndCpp(2,2,3)] + A[rcJ_IndCpp(3,3,3)];
   B[rcJ_IndCpp(3,3,3)] = yZX*E[rcJ_IndCpp(3,1,3)] + yZY*E[rcJ_IndCpp(3,2,3)] + A[rcJ_IndCpp(3,3,3)];

   B[rcJ_IndCpp(1,2,3)] = yXX*E[rcJ_IndCpp(2,1,3)] + yXY*E[rcJ_IndCpp(2,2,3)] - EvZ;
   B[rcJ_IndCpp(2,1,3)] = yYX*E[rcJ_IndCpp(1,1,3)] + yYY*E[rcJ_IndCpp(1,2,3)] + EvZ;
   B[rcJ_IndCpp(1,3,3)] = yXX*E[rcJ_IndCpp(3,1,3)] + yXY*E[rcJ_IndCpp(3,2,3)] + EvY;
   B[rcJ_IndCpp(3,1,3)] = yZX*E[rcJ_IndCpp(1,1,3)] + yZY*E[rcJ_IndCpp(1,2,3)] - EvY;
   B[rcJ_IndCpp(2,3,3)] = yYX*E[rcJ_IndCpp(3,1,3)] + yYY*E[rcJ_IndCpp(3,2,3)] - EvX;
   B[rcJ_IndCpp(3,2,3)] = yZX*E[rcJ_IndCpp(2,1,3)] + yZY*E[rcJ_IndCpp(2,2,3)] + EvX;
}

//-------------------------------------------------------------------------------
// Helper Function to Compute Articulated Inertia for Revolute Joint
//-------------------------------------------------------------------------------
//
// These voids calculate the spatial articulated inertia of a subtree, in
// the special case of a mass-less handle (see chapter 7 of the RBDA
// book, s7.2.2, equation 7.23)
//
// Two specialized voids are available for the two cases of revolute and
// prismatic joint (which connects the subtree with the mass-less handle), since
// the inertia exhibits two different sparsity patterns.
//
// In:  IA the regular articulated inertia of some subtree
// In:  U the U term for the current joint (cfr. eq. 7.43 of RBDA)
// In:  D the D term for the current joint (cfr. eq. 7.44 of RBDA)
// Out: Ia the articulated inertia for the same subtree, but
//      propagated across the current joint. The matrix is assumed to be
//      already initialized with zeros, at least in the row and column which
//      is known to be zero. In other words, those elements are not computed
//      nor assigned in this void.
//      Note that the constness of the argument is
//      casted away (trick required with Eigen), as this is an output
//      argument.
//
// -- RobCoGen
//
template <typename T>
void compute_Ia_revolute(T *s_Ia, T *s_I, T *s_U, T Dinv){
   // for(int i = 0; i < 36; i++){s_Ia[i] = 0;}
   T s_UD[6];
   for(int i = 0; i < 6; i++){s_UD[i] = s_U[i]*Dinv;}

   s_Ia[rcJ_IndCpp(1,1,6)] = s_I[rcJ_IndCpp(1,1,6)] - s_U[0]*s_UD[0];
   s_Ia[rcJ_IndCpp(1,2,6)] = s_I[rcJ_IndCpp(1,2,6)] - s_U[0]*s_UD[1];
   // s_Ia[rcJ_IndCpp(2,1,6)] = s_I[rcJ_IndCpp(1,2,6)] - s_U[0]*s_UD[1];
   //s_Ia[rcJ_IndCpp(1,3,6)] = s_Ia[rcJ_IndCpp(3,1,6)] = 0 // assumed to be set already
   s_Ia[rcJ_IndCpp(1,3,6)] = static_cast<T>(0);
   s_Ia[rcJ_IndCpp(1,4,6)] = s_I[rcJ_IndCpp(1,4,6)] - s_U[0]*s_UD[3];
   // s_Ia[rcJ_IndCpp(4,1,6)] = s_I[rcJ_IndCpp(1,4,6)] - s_U[0]*s_UD[3];
   s_Ia[rcJ_IndCpp(1,5,6)] = s_I[rcJ_IndCpp(1,5,6)] - s_U[0]*s_UD[4];
   // s_Ia[rcJ_IndCpp(5,1,6)] = s_I[rcJ_IndCpp(1,5,6)] - s_U[0]*s_UD[4];
   s_Ia[rcJ_IndCpp(1,6,6)] = s_I[rcJ_IndCpp(1,6,6)] - s_U[0]*s_UD[5];
   // s_Ia[rcJ_IndCpp(6,1,6)] = s_I[rcJ_IndCpp(1,6,6)] - s_U[0]*s_UD[5];

   s_Ia[rcJ_IndCpp(2,2,6)] = s_I[rcJ_IndCpp(2,2,6)] - s_U[1]*s_UD[1];
   s_Ia[rcJ_IndCpp(2,3,6)] = static_cast<T>(0);
   //s_Ia[rcJ_IndCpp(2,3,6)] = s_Ia[rcJ_IndCpp(3,2,6)] = 0 // assumed to be set already
   s_Ia[rcJ_IndCpp(2,4,6)] = s_I[rcJ_IndCpp(2,4,6)] - s_U[1]*s_UD[3];
   // s_Ia[rcJ_IndCpp(4,2,6)] = s_I[rcJ_IndCpp(2,4,6)] - s_U[1]*s_UD[3];
   s_Ia[rcJ_IndCpp(2,5,6)] = s_I[rcJ_IndCpp(2,5,6)] - s_U[1]*s_UD[4];
   // s_Ia[rcJ_IndCpp(5,2,6)] = s_I[rcJ_IndCpp(2,5,6)] - s_U[1]*s_UD[4];
   s_Ia[rcJ_IndCpp(2,6,6)] = s_I[rcJ_IndCpp(2,6,6)] - s_U[1]*s_UD[5];
   // s_Ia[rcJ_IndCpp(6,2,6)] = s_I[rcJ_IndCpp(2,6,6)] - s_U[1]*s_UD[5];

   // The whole row 3 is assumed to be already set to zero
   s_Ia[rcJ_IndCpp(3,3,6)] = static_cast<T>(0);
   s_Ia[rcJ_IndCpp(3,4,6)] = static_cast<T>(0);
   s_Ia[rcJ_IndCpp(3,5,6)] = static_cast<T>(0);
   s_Ia[rcJ_IndCpp(3,6,6)] = static_cast<T>(0);

   s_Ia[rcJ_IndCpp(4,4,6)] = s_I[rcJ_IndCpp(4,4,6)] - s_U[3]*s_UD[3];
   s_Ia[rcJ_IndCpp(4,5,6)] = s_I[rcJ_IndCpp(4,5,6)] - s_U[3]*s_UD[4];
   // s_Ia[rcJ_IndCpp(5,4,6)] = s_I[rcJ_IndCpp(4,5,6)] - s_U[3]*s_UD[4];
   s_Ia[rcJ_IndCpp(4,6,6)] = s_I[rcJ_IndCpp(4,6,6)] - s_U[3]*s_UD[5];
   // s_Ia[rcJ_IndCpp(6,4,6)] = s_I[rcJ_IndCpp(4,6,6)] - s_U[3]*s_UD[5];

   s_Ia[rcJ_IndCpp(5,5,6)] = s_I[rcJ_IndCpp(5,5,6)] - s_U[4]*s_UD[4];
   s_Ia[rcJ_IndCpp(5,6,6)] = s_I[rcJ_IndCpp(5,6,6)] - s_U[4]*s_UD[5];
   // s_Ia[rcJ_IndCpp(6,5,6)] = s_I[rcJ_IndCpp(5,6,6)] - s_U[4]*s_UD[5];

   s_Ia[rcJ_IndCpp(6,6,6)] = s_I[rcJ_IndCpp(6,6,6)] - s_U[5]*s_UD[5];

   // we now have an upper triangular matrix and need to copy over
   // but actually we only use this in the bleow fun so we can skip this
   // for (int c = 0; c < 6; c++){
   //    for (int r = c+1; r < 6; r ++){s_Ia[rc_Ind(r,c,6)] = s_Ia[rc_Ind(c,r,6)];}
   // }
}

//-------------------------------------------------------------------------------
// Helper Function for Coordinate Transform of Articulated Inertia for Revolute Joint
//-------------------------------------------------------------------------------

//
// These voids perform the coordinate transform of a spatial articulated
// inertia, in the special case of a mass-less handle (see chapter 7 of the RBDA
// book, s7.2.2, equation 7.23)
//
// Two specialized voids are available for the two cases of revolute and
// prismatic joint (which connects the subtree with the mass-less handle), since
// the inertia exhibits two different sparsity patterns.
//
// In:  Ia_A the articulated inertia in A coordinates
// In:  XM a spatial coordinate transform for motion vectors, in the form A_XM_B
//      (that is, mapping forces from A to B coordinates)
// Out: Ia_B the same articulated inertia, but expressed in B
//      coordinates. Note that the constness is casted away (trick required
//      with Eigen)
//
// -- RobCoGen
//
template <typename T>
void ctransform_Ia_revolute(T *Ilam, T *Ia, T *XM){
   // Get the coefficients of the 3x3 rotation matrix B_E_A
   // It is the transpose of the angular-angular block of the spatial transform
   T E[9];
   E[rcJ_IndCpp(1,1,3)] = XM[rcJ_IndCpp(1,1,6)];
   E[rcJ_IndCpp(1,2,3)] = XM[rcJ_IndCpp(2,1,6)];
   E[rcJ_IndCpp(1,3,3)] = XM[rcJ_IndCpp(3,1,6)];
   E[rcJ_IndCpp(2,1,3)] = XM[rcJ_IndCpp(1,2,6)];
   E[rcJ_IndCpp(2,2,3)] = XM[rcJ_IndCpp(2,2,6)];
   E[rcJ_IndCpp(2,3,3)] = XM[rcJ_IndCpp(3,2,6)];
   E[rcJ_IndCpp(3,1,3)] = XM[rcJ_IndCpp(1,3,6)];
   E[rcJ_IndCpp(3,2,3)] = XM[rcJ_IndCpp(2,3,6)];
   E[rcJ_IndCpp(3,3,3)] = XM[rcJ_IndCpp(3,3,6)];
   // Recover the translation vector.
   // The relative position 'r' of the origin of frame B wrt A (in A coordinates).
   // The vector is basically "reconstructed" from the matrix XF, which has
   // this form:
   //    | E   -E rx |
   //    | 0    E    |
   // where 'rx' is the cross product matrix. The strategy is to compute
   // E^T (-E rx) = -rx  , and then get the coordinates of 'r' from 'rx'.
   // This code is a manual implementation of the E transpose multiplication,
   // limited to the elements of interest.
   // Note that this is necessary because, currently, spatial transforms do
   // not carry explicitly the information about the translation vector.
   T rx = XM[rcJ_IndCpp(2,1,6)] * XM[rcJ_IndCpp(6,1,6)] + 
         XM[rcJ_IndCpp(2,2,6)] * XM[rcJ_IndCpp(6,2,6)] + 
         XM[rcJ_IndCpp(2,3,6)] * XM[rcJ_IndCpp(6,3,6)];
   T ry = XM[rcJ_IndCpp(3,1,6)] * XM[rcJ_IndCpp(4,1,6)] + 
         XM[rcJ_IndCpp(3,2,6)] * XM[rcJ_IndCpp(4,2,6)] + 
         XM[rcJ_IndCpp(3,3,6)] * XM[rcJ_IndCpp(4,3,6)];
   T rz = XM[rcJ_IndCpp(1,1,6)] * XM[rcJ_IndCpp(5,1,6)] + 
         XM[rcJ_IndCpp(1,2,6)] * XM[rcJ_IndCpp(5,2,6)] + 
         XM[rcJ_IndCpp(1,3,6)] * XM[rcJ_IndCpp(5,3,6)];
   
   // Angular-angular 3x3 sub-block:
   // compute  I + (Crx + Crx^T)  :
   // Remember that, for a revolute joint, the structure of the matrix is as
   //  follows (note the zeros):
   //    [ia1,  ia2,  0, ia4,  ia5,  ia6 ]
   //    [ia2,  ia7,  0, ia8,  ia9,  ia10]
   //    [0  ,    0,  0,   0,    0,    0 ]
   //    [ia4,  ia8,  0, ia11, ia12, ia13]
   //    [ia5,  ia9,  0, ia12, ia14, ia15]
   //    [ia6, ia10,  0, ia13, ia15, ia16]
   // copying the coefficients results in slightly fewer invocations of the
   //  operator(int,int), in the rest of the void
   T C[9];
   C[rcJ_IndCpp(1,1,3)] = Ia[rcJ_IndCpp(1,4,6)];
   C[rcJ_IndCpp(1,2,3)] = Ia[rcJ_IndCpp(1,5,6)];
   C[rcJ_IndCpp(1,3,3)] = Ia[rcJ_IndCpp(1,6,6)];
   C[rcJ_IndCpp(2,1,3)] = Ia[rcJ_IndCpp(2,4,6)];
   C[rcJ_IndCpp(2,2,3)] = Ia[rcJ_IndCpp(2,5,6)];
   C[rcJ_IndCpp(2,3,3)] = Ia[rcJ_IndCpp(2,6,6)];
   C[rcJ_IndCpp(3,1,3)] = static_cast<T>(0);
   C[rcJ_IndCpp(3,2,3)] = static_cast<T>(0);
   C[rcJ_IndCpp(3,3,3)] = static_cast<T>(0);

   T aux1[9];
   aux1[rcJ_IndCpp(1,1,3)] = Ia[rcJ_IndCpp(1,1,6)] + static_cast<T>(2)*C[rcJ_IndCpp(1,2,3)]*rz - 2*C[rcJ_IndCpp(1,3,3)]*ry;
   aux1[rcJ_IndCpp(2,2,3)] = Ia[rcJ_IndCpp(2,2,6)] + static_cast<T>(2)*C[rcJ_IndCpp(2,3,3)]*rx - 2*C[rcJ_IndCpp(2,1,3)]*rz;
   aux1[rcJ_IndCpp(1,2,3)] = Ia[rcJ_IndCpp(1,2,6)] + C[rcJ_IndCpp(2,2,3)]*rz - C[rcJ_IndCpp(1,1,3)]*rz - C[rcJ_IndCpp(2,3,3)]*ry + C[rcJ_IndCpp(1,3,3)]*rx;
   aux1[rcJ_IndCpp(1,3,3)] = C[rcJ_IndCpp(1,1,3)]*ry - C[rcJ_IndCpp(1,2,3)]*rx;
   aux1[rcJ_IndCpp(2,3,3)] = C[rcJ_IndCpp(2,1,3)]*ry - C[rcJ_IndCpp(2,2,3)]*rx;
   aux1[rcJ_IndCpp(2,1,3)] = static_cast<T>(0);
   aux1[rcJ_IndCpp(3,1,3)] = static_cast<T>(0);
   aux1[rcJ_IndCpp(3,2,3)] = static_cast<T>(0);
   aux1[rcJ_IndCpp(3,3,3)] = static_cast<T>(0);
   
   // compute (- rx M)  (note the minus)
   T M[9];
   M[rcJ_IndCpp(1,1,3)] = Ia[rcJ_IndCpp(4,4,6)];
   M[rcJ_IndCpp(1,2,3)] = Ia[rcJ_IndCpp(4,5,6)];
   M[rcJ_IndCpp(1,3,3)] = Ia[rcJ_IndCpp(4,6,6)];
   M[rcJ_IndCpp(2,1,3)] = Ia[rcJ_IndCpp(4,5,6)];
   M[rcJ_IndCpp(2,2,3)] = Ia[rcJ_IndCpp(5,5,6)];
   M[rcJ_IndCpp(2,3,3)] = Ia[rcJ_IndCpp(5,6,6)];
   M[rcJ_IndCpp(3,1,3)] = Ia[rcJ_IndCpp(4,6,6)];
   M[rcJ_IndCpp(3,2,3)] = Ia[rcJ_IndCpp(5,6,6)];
   M[rcJ_IndCpp(3,3,3)] = Ia[rcJ_IndCpp(6,6,6)];

   T rxM[9];
   rxM[rcJ_IndCpp(1,1,3)] = rz*M[rcJ_IndCpp(1,2,3)] - ry*M[rcJ_IndCpp(1,3,3)];
   rxM[rcJ_IndCpp(2,2,3)] = rx*M[rcJ_IndCpp(2,3,3)] - rz*M[rcJ_IndCpp(1,2,3)];
   rxM[rcJ_IndCpp(3,3,3)] = ry*M[rcJ_IndCpp(1,3,3)] - rx*M[rcJ_IndCpp(2,3,3)];
   rxM[rcJ_IndCpp(1,2,3)] = rz*M[rcJ_IndCpp(2,2,3)] - ry*M[rcJ_IndCpp(2,3,3)];
   rxM[rcJ_IndCpp(2,1,3)] = rx*M[rcJ_IndCpp(1,3,3)] - rz*M[rcJ_IndCpp(1,1,3)];
   rxM[rcJ_IndCpp(1,3,3)] = rz*M[rcJ_IndCpp(2,3,3)] - ry*M[rcJ_IndCpp(3,3,3)];
   rxM[rcJ_IndCpp(3,1,3)] = ry*M[rcJ_IndCpp(1,1,3)] - rx*M[rcJ_IndCpp(1,2,3)];
   rxM[rcJ_IndCpp(2,3,3)] = rx*M[rcJ_IndCpp(3,3,3)] - rz*M[rcJ_IndCpp(1,3,3)];
   rxM[rcJ_IndCpp(3,2,3)] = ry*M[rcJ_IndCpp(1,2,3)] - rx*M[rcJ_IndCpp(2,2,3)];

   // compute  (I + (Crx + Crx^T))  -  (rxM) rx
   aux1[rcJ_IndCpp(1,1,3)] += rxM[rcJ_IndCpp(1,2,3)]*rz - rxM[rcJ_IndCpp(1,3,3)]*ry;
   aux1[rcJ_IndCpp(2,2,3)] += rxM[rcJ_IndCpp(2,3,3)]*rx - rxM[rcJ_IndCpp(2,1,3)]*rz;
   aux1[rcJ_IndCpp(3,3,3)] += rxM[rcJ_IndCpp(3,1,3)]*ry - rxM[rcJ_IndCpp(3,2,3)]*rx;
   aux1[rcJ_IndCpp(1,2,3)] += rxM[rcJ_IndCpp(1,3,3)]*rx - rxM[rcJ_IndCpp(1,1,3)]*rz;
   aux1[rcJ_IndCpp(1,3,3)] += rxM[rcJ_IndCpp(1,1,3)]*ry - rxM[rcJ_IndCpp(1,2,3)]*rx;
   aux1[rcJ_IndCpp(2,3,3)] += rxM[rcJ_IndCpp(2,1,3)]*ry - rxM[rcJ_IndCpp(2,2,3)]*rx;

   // compute  E ( .. ) E^T
   T aux2[9];
   rot_symmetric_EAET(aux2,E,aux1);
   
   // Copy the result, angular-angular block of the output
   Ilam[rcJ_IndCpp(1,1,6)] += aux2[rcJ_IndCpp(1,1,3)];
   Ilam[rcJ_IndCpp(2,2,6)] += aux2[rcJ_IndCpp(2,2,3)];
   Ilam[rcJ_IndCpp(3,3,6)] += aux2[rcJ_IndCpp(3,3,3)];
   Ilam[rcJ_IndCpp(1,2,6)] += aux2[rcJ_IndCpp(1,2,3)];
   Ilam[rcJ_IndCpp(2,1,6)] += aux2[rcJ_IndCpp(1,2,3)];
   Ilam[rcJ_IndCpp(1,3,6)] += aux2[rcJ_IndCpp(1,3,3)];
   Ilam[rcJ_IndCpp(3,1,6)] += aux2[rcJ_IndCpp(1,3,3)];
   Ilam[rcJ_IndCpp(2,3,6)] += aux2[rcJ_IndCpp(2,3,3)];
   Ilam[rcJ_IndCpp(3,2,6)] += aux2[rcJ_IndCpp(2,3,3)];

   // Angular-linear block (and linear-angular block)
   // Calculate E ( C -rxM ) E^T
   //  - note that 'rxM' already contains the coefficients of  (- rx * M)
   //  - for a revolute joint, the last line of C is zero
   rxM[rcJ_IndCpp(1,1,3)] += C[rcJ_IndCpp(1,1,3)];
   rxM[rcJ_IndCpp(1,2,3)] += C[rcJ_IndCpp(1,2,3)];
   rxM[rcJ_IndCpp(1,3,3)] += C[rcJ_IndCpp(1,3,3)];
   rxM[rcJ_IndCpp(2,1,3)] += C[rcJ_IndCpp(2,1,3)];
   rxM[rcJ_IndCpp(2,2,3)] += C[rcJ_IndCpp(2,2,3)];
   rxM[rcJ_IndCpp(2,3,3)] += C[rcJ_IndCpp(2,3,3)];

   T aux3[9];
   rot_EAET(aux3,E,rxM);

   // copy the result, also to the symmetric 3x3 block
   Ilam[rcJ_IndCpp(1,4,6)] += aux3[rcJ_IndCpp(1,1,3)];
   Ilam[rcJ_IndCpp(1,5,6)] += aux3[rcJ_IndCpp(1,2,3)];
   Ilam[rcJ_IndCpp(1,6,6)] += aux3[rcJ_IndCpp(1,3,3)];
   Ilam[rcJ_IndCpp(2,4,6)] += aux3[rcJ_IndCpp(2,1,3)];
   Ilam[rcJ_IndCpp(2,5,6)] += aux3[rcJ_IndCpp(2,2,3)];
   Ilam[rcJ_IndCpp(2,6,6)] += aux3[rcJ_IndCpp(2,3,3)];
   Ilam[rcJ_IndCpp(3,4,6)] += aux3[rcJ_IndCpp(3,1,3)];
   Ilam[rcJ_IndCpp(3,5,6)] += aux3[rcJ_IndCpp(3,2,3)];
   Ilam[rcJ_IndCpp(3,6,6)] += aux3[rcJ_IndCpp(3,3,3)];

   Ilam[rcJ_IndCpp(4,1,6)] += aux3[rcJ_IndCpp(1,1,3)];
   Ilam[rcJ_IndCpp(5,1,6)] += aux3[rcJ_IndCpp(1,2,3)];
   Ilam[rcJ_IndCpp(6,1,6)] += aux3[rcJ_IndCpp(1,3,3)];
   Ilam[rcJ_IndCpp(4,2,6)] += aux3[rcJ_IndCpp(2,1,3)];
   Ilam[rcJ_IndCpp(5,2,6)] += aux3[rcJ_IndCpp(2,2,3)];
   Ilam[rcJ_IndCpp(6,2,6)] += aux3[rcJ_IndCpp(2,3,3)];
   Ilam[rcJ_IndCpp(4,3,6)] += aux3[rcJ_IndCpp(3,1,3)];
   Ilam[rcJ_IndCpp(5,3,6)] += aux3[rcJ_IndCpp(3,2,3)];
   Ilam[rcJ_IndCpp(6,3,6)] += aux3[rcJ_IndCpp(3,3,3)];

   // Linear-linear block
   rot_symmetric_EAET(aux1,E,M);

   Ilam[rcJ_IndCpp(4,4,6)] += aux1[rcJ_IndCpp(1,1,3)];
   Ilam[rcJ_IndCpp(5,5,6)] += aux1[rcJ_IndCpp(2,2,3)];
   Ilam[rcJ_IndCpp(6,6,6)] += aux1[rcJ_IndCpp(3,3,3)];
   
   Ilam[rcJ_IndCpp(4,5,6)] += aux1[rcJ_IndCpp(1,2,3)];
   Ilam[rcJ_IndCpp(4,6,6)] += aux1[rcJ_IndCpp(1,3,3)];
   Ilam[rcJ_IndCpp(5,6,6)] += aux1[rcJ_IndCpp(2,3,3)];

   Ilam[rcJ_IndCpp(5,4,6)] += aux1[rcJ_IndCpp(1,2,3)];
   Ilam[rcJ_IndCpp(6,4,6)] += aux1[rcJ_IndCpp(1,3,3)];
   Ilam[rcJ_IndCpp(6,5,6)] += aux1[rcJ_IndCpp(2,3,3)];
}

//-------------------------------------------------------------------------------
// M inverse helpers
//-------------------------------------------------------------------------------
template <typename T>
void bwdPassMinvFUpdate(T *s_Minv, T *Fi, T *Flami, T *iXlami, T *Ui, T Dinvi, int li){
   bool has_parent = li > 0; int num_children = 6 - li; 
   // (21) Minv[i,i] = Di^-1
   s_Minv[rc_Ind(li,li,NUM_POS)] = Dinvi;
   // (12) Minv[i,subtree(i)] = -Di^-T * SiT * Fi[:,subtree(i)]
   if (num_children > 0){
      for(int i = 0; i < num_children; i++){
         s_Minv[rc_Ind(li,(li+1+i),NUM_POS)] = -Dinvi * Fi[rc_Ind(2,(li+1+i),6)];
      }
   }
   // (14)    Fi[:,subtree(i)] += Ui * Minv[i,subtree(i)]
   // (15)    Flami[:,subtree(i)] = Flami[:,subtree(i)] + lamiXi* * Fi[:,subtree(i)]
   // or we do the following if there is no children then c = 0
   // (19)    Flami[:,i] += lamiXi* * Ui * Minv[i,i]
   for(int c = 0; c <= num_children; c++){
      T Minvc = s_Minv[rc_Ind(li,(li+c),NUM_POS)]; T *Fic = &Fi[rc_Ind(0,(li+c),6)];
      for(int r = 0; r < 6; r++){Fic[r] += Ui[r] * Minvc;}
   }
   // if has parent need to transform and add Fi to Flami
   if (has_parent){iteratedMatVMult<T,6,6,1,1>(Flami,6,iXlami,6,Fi,6,7);}
}

template <typename T>
void bwdPassIUpdate(T *Ilam, T *I, T *Ia, T *X, T *U, T Dinv, int li){
   bool has_parent = li > 0;
   if (has_parent){
      compute_Ia_revolute(Ia, I, U, Dinv); // same as: Ia_r = AI_l7 - U_l7/D_l7 * U_l7.transpose()
      ctransform_Ia_revolute(Ilam, Ia, X);
   }
}

template <typename T>
void Minv_bp(T *s_Minv, T *s_F, T *s_I, T *s_T, T *s_U, T *s_D, T *s_Ia){
   for (int k = NUM_POS - 1; k>=0; k--){
      T *Uk = &s_U[k*6];            T *Dk = &s_D[2*k];               T *Tk = &s_T[36*k];
      T *Ik = &s_I[36*k];           T *Ilamk = &s_I[36*(k-1)];
      T *Fk = &s_F[6*NUM_POS*k];    T *Flamk = &s_F[6*NUM_POS*(k-1)];
      // get Uk and Dk
      for(int i = 0; i < 6; i ++){Uk[i] = Ik[6*2+i]; if(i == 2){Dk[0] = Uk[i]; Dk[1] = 1/Dk[0];}}
      // update Minv and F
      bwdPassMinvFUpdate(s_Minv,Fk,Flamk,Tk,Uk,Dk[1],k);
      // update Ilamk -- note: this is an independent operation from the F and Minv updates
      bwdPassIUpdate(Ilamk,Ik,s_Ia,Tk,Uk,Dk[1],k);
   }
}

// need T s_temp[6]
template <typename T>
void fwdPassUpdate(T *s_Minv, T *s_Fi, T *s_Flami, T *iXlami, T *Ui, T Dinvi, T *s_temp, int li){
   // we have all rotational joints so Si = [0,0,1,0,0,0]
   bool has_parent = (li > 0);         T *Minv_subtree = &s_Minv[NUM_POS*li + li]; 
   T *Fi_subtree = &s_Fi[6*li];        T *Flami_subtree = &s_Flami[6*li];              int N_subtree = NUM_POS-li;
   if (has_parent){
      // (30) Minv[i,subtree(i)] += -D^-1 * Ui^T * iXlami * Flami[:,subtree(i)]
      // Ui^T * iXlami = (iXlami^T * Ui)^T
      matVMult<T,6,6,0,1>(s_temp,iXlami,Ui);
      // again we employ the transpose trick where subtree(i) is li to 6 (base 0) so 7-li rows
      // and we only compute one row of length 7-li and stride with 1
      iteratedMatVMult<T,1,6,1,1>(Minv_subtree,NUM_POS,s_temp,1,Flami_subtree,6,N_subtree,-Dinvi);
   }
   // (32) Fi = Si * Minv[i,subtree(i)]
   // we have all rotational joints so Si = [0,0,1,0,0,0]
   for(int i = 0; i < N_subtree; i++){
      for(int j = 0; j < 6; j++){
         Fi_subtree[6*i+j] = (j == 2) ? Minv_subtree[NUM_POS*i] : static_cast<T>(0);
      }
   }
   // (34) Fi[:,subtree(i)] += iXlami * Flami[:,subtree(i)]
   if (has_parent){iteratedMatVMult<T,6,6,1>(Fi_subtree,6,iXlami,6,Flami_subtree,6,N_subtree);}
}
template <typename T>
void Minv_fp(T *s_Minv, T *s_F, T *s_T, T *s_U, T *s_D, T *s_temp){
   for (int k = 0; k < NUM_POS; k++){
      T *Uk = &s_U[k*6];            T *Dk = &s_D[2*k];               T *Tk = &s_T[36*k];
      T *Fk = &s_F[6*NUM_POS*k];    T *Flamk = &s_F[6*NUM_POS*(k-1)];
      fwdPassUpdate(s_Minv,Fk,Flamk,Tk,Uk,Dk[1],s_temp,k);
   }
}

template <typename T>
void computeMinv(T *d_Minv, T *s_T, T *s_IA){
   T s_F[6*NUM_POS*NUM_POS] = {0};
   T s_U[6*NUM_POS]; T s_D[2*NUM_POS]; // also store 1/Di
   T s_Ia[36]; T s_temp[42];
   Minv_bp<T>(d_Minv,s_F,s_IA,s_T,s_U,s_D,s_Ia);
   Minv_fp<T>(d_Minv,s_F,s_T,s_U,s_D,s_temp);
}