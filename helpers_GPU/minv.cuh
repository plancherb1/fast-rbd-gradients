#define rc_Ind(r,c,ld) (r + c*ld)

__device__ __forceinline__
int Mx3ind_GPU_MINV(int r){return r == 0 ? 1 : (r == 1 ? 0 : (r == 3 ? 4 : 3));} // note this is src r: [v(1), -v(0), 0, v(4), -v(3), 0]^T

template <typename T>
__device__ __forceinline__
T Mx3mul_GPU_MINV(int r){return (r == 0 || r == 3) ? static_cast<T>(1) : ((r == 1 || r == 4) ? static_cast<T>(-1) : static_cast<T>(0));}

template <typename T>
__device__ 
void loadMx3_GPU_MINV(T *dst, T *src, int r, T alpha = 1){
	int Mxr = Mx3ind_GPU_MINV(r); T sgn = Mx3mul_GPU_MINV<T>(r);
	dst[r] = sgn*alpha*src[Mxr];
}

// for multiplying matrix row ind of A by col vector b
// s_a = ld_A and s_b = 1 and pass in &A[ind] and b
// then we get A[ind + ld_A * j] * b[j]
// if transposed then we need to get col ind of A so
// then we need &A[ind*ld_A] and s_a = 1 and s_b = 1
// then we get A[ind * ld_A + j] * b[j]
template <typename T, int K, int T_A = 0>
__device__
T dotProdMv_GPU(T *A, int ld_A, T *b, int ind){
   if(T_A){return dotProd_GPU<T,K>(&A[ind*ld_A],1,b,1);}
   else{return dotProd_GPU<T,K>(&A[ind],ld_A,b,1);}
}

// basic matrix vector multiply d (+)= alpha*A*b (+ beta*c)
template <typename T, int M, int K, int PEQFLAG = 0, int T_A = 0>
__device__
void matVMult_GPU(T *d, T *A, int ld_A, T *b, T alpha = 1.0, T *c = nullptr, T beta = 1.0){
    int start, delta; singleLoopVals_GPU(&start, &delta);
    #pragma unroll
    for (int ind = start; ind < M; ind += delta){
      T val = alpha*dotProdMv_GPU<T,K,T_A>(A, ld_A, b, ind) + (c != nullptr ? beta*c[ind] : static_cast<T>(0));
      if (PEQFLAG){d[ind] += val;}
      else{d[ind] = val;}
   }
}

// iterated matVMult for D[:,col_i] += alpha*A*B[:,col_i] (+ c) for all i in some set
// assumes D and B are pointers to col_0 and then we just need the next N cols
template <typename T, int M, int K, int PEQFLAG = 0, int T_A = 0>
__device__
void iteratedMatVMult_GPU(T *D, int ld_D, T *A, int ld_A, T *B, int ld_B, int col_N, T alpha = 1.0, T *c = nullptr, T beta = 1.0){
   int startx, dx, starty, dy; doubleLoopVals_GPU(&startx, &dx, &starty, &dy);
   for (int col = starty; col < col_N; col += dy){
      T *Dc = &D[col*ld_D]; T *Bc = &B[col*ld_B];
      #pragma unroll
         for (int row = startx; row < M; row += dx){
         T val = alpha*dotProdMv_GPU<T,K,T_A>(A, ld_A, Bc, row) + (c != nullptr ? beta*c[row] : static_cast<T>(0));
         if (PEQFLAG){Dc[row] += val;}
         else{Dc[row] = val;}
      }
   }
}


template <typename T, bool MPC_MODE = false>
__device__
void FD_helpers_GPU(T *s_v, T *s_a, T *s_f, T *s_Minv, T *s_qd, T *s_qdd, T *s_I, T *s_IA, T *s_T, T *s_U, T *s_D, T *s_F, T *s_temp, T *s_fext = nullptr){
	// In GPU mode want to reduce the number of syncs so we intersperce the computations
	//----------------------------------------------------------------------------
	// RNEA Forward Pass / Minv Backward Pass
	//----------------------------------------------------------------------------
	int start, delta; singleLoopVals_GPU(&start,&delta);
	int starty, dy, startx, dx; doubleLoopVals_GPU(&starty,&dy,&startx,&dx);
	for (int k = 0; k < NUM_POS; k++){ int kMinv = NUM_POS - 1 - k;
		// first the first half of the Minv comp fully in parallel aka get Uk and Dk
		T *Uk = &s_U[6*kMinv];      T *Dk = &s_D[2*kMinv];      T *IAk = &s_IA[36*kMinv];  bool has_parent = (kMinv > 0); 
		for(int i = start; i < 6; i += delta){Uk[i] = IAk[6*2+i]; if(i == 2){Dk[0] = Uk[i]; Dk[1] = 1/Dk[0];}}
		// we need to intersperse the functions to reduce syncthreads in the GPU version
		// then the first bit of the RNEA forward pass aka get v and a
		T *lk_v = &s_v[6*k];    T *lk_a = &s_a[6*k];    T *tkXkm1 = &s_T[36*k];    T *Tk = &s_T[36*kMinv];    T *Fk = &s_F[6*NUM_POS*kMinv];
		if (k == 0){
			// l1_v = vJ, for the first kMinvnk of a fixed base robot
			// and the a is just gravity times the last col of the transform and then add qdd in 3rd entry if computing bias term for gradients
			for(int i = start; i < 6; i += delta){
				lk_v[i] = (i == 2) ? s_qd[k] : 0;
				lk_a[i] = MPC_MODE ? static_cast<T>(0) : tkXkm1[6*5+i] * GRAVITY; if(i == 2){lk_a[2] += s_qdd[k];}
			}
			// then we start computing the IA for Minv
			if (has_parent){
				for(int ky = starty; ky < 6; ky += dy){
				   	for(int kx = startx; kx < 6; kx += dx){
						T val = static_cast<T>(0);
						#pragma unroll
						for (int i = 0; i < 6; i++){val += (IAk[kx+6*i] - Uk[i]*Uk[kx]*Dk[1]) * Tk[ky*6+i];}
						s_temp[ky*6+kx] = val;
				   	}
			    }
			}
			// while we load in the update to minv
			int num_children = 6 - kMinv;
			if(__doOnce()){s_Minv[rc_Ind(kMinv,kMinv,NUM_POS)] = Dk[1];}  // (21) Minv[i,i] = Di^-1
			if (num_children > 0){ // (12) Minv[i,subtree(i)] = -Di^-T * SiT * Fi[:,subtree(i)]
				#pragma unroll
				for(int i = start; i < num_children; i += delta){
					s_Minv[rc_Ind(kMinv,(kMinv+1+i),NUM_POS)] = -Dk[1] * Fk[rc_Ind(2,(kMinv+1+i),6)];
				}
			}
			__syncthreads();
			// then we finish the IA
			if (has_parent){
			    T *IAlamk = IAk - 36;
			    for(int ky = starty; ky < 6; ky += dy){
			       	for(int kx = startx; kx < 6; kx += dx){
						T val = static_cast<T>(0);
						#pragma unroll
						for (int i = 0; i < 6; i++){val += Tk[kx*6+i] * s_temp[ky*6+i];}
						IAlamk[ky*6+kx] += val;
			       	}
			    }
			}
			// while we update the F
			#pragma unroll
			for(int c = starty; c <= num_children; c += dy){ // (14)  Fi[:,subtree(i)] = Ui * Minv[i,subtree(i)]
			    T Minvc = s_Minv[rc_Ind(kMinv,(kMinv+c),NUM_POS)]; T *Fkc = &Fk[rc_Ind(0,(kMinv+c),6)];
			    #pragma unroll
			    for(int r = startx; r < 6; r += dx){Fkc[r] += Uk[r] * Minvc;}
			}
		}
	  	else{
			T *lkm1_v = lk_v - 6;    T *lkm1_a = lk_a - 6;    int num_children = 6 - kMinv; 
			// first compute the v and add += s_qd[k] in entry [2]
			for(int r = start; r < 6; r += delta){
				T val = r == 2 ? s_qd[k] : static_cast<T>(0);
				#pragma unroll
				for (int c = 0; c < 6; c++){val += tkXkm1[6*c+r] * lkm1_v[c];}
				lk_v[r] = val; 
			}
			// then we start computing the IA for Minv
			if (has_parent){
				for(int ky = starty; ky < 6; ky += dy){
					for(int kx = startx; kx < 6; kx += dx){
						T val = static_cast<T>(0);
						#pragma unroll
						for (int i = 0; i < 6; i++){val += (IAk[kx+6*i] - Uk[i]*Uk[kx]*Dk[1]) * Tk[ky*6+i];}
						s_temp[ky*6+kx] = val;
					}
				}
			}
			// while we load in the update to minv
			if(__doOnce()){s_Minv[rc_Ind(kMinv,kMinv,NUM_POS)] = Dk[1];}  // (21) Minv[i,i] = Di^-1
			if (num_children > 0){ // (12) Minv[i,subtree(i)] = -Di^-T * SiT * Fi[:,subtree(i)]
				#pragma unroll
				for(int i = start; i < num_children; i += delta){
				   s_Minv[rc_Ind(kMinv,(kMinv+1+i),NUM_POS)] = -Dk[1] * Fk[rc_Ind(2,(kMinv+1+i),6)];
				}
			}
			__syncthreads();
			// Compute the a (Mx3(v) += that with tkXkm1 * lkm1_a) and add qdd in entry [2]
			for(int r = start; r < 6; r += delta){
				loadMx3_GPU_MINV<T>(lk_a,lk_v,r,s_qd[k]);
				T val = r == 2 ? s_qdd[k] : static_cast<T>(0);
				#pragma unroll
				for (int c = 0; c < 6; c++){val += tkXkm1[6*c+r] * lkm1_a[c];}
				lk_a[r] += val; 
			}
			// then we finish the IA
			if (has_parent){
				T *IAlamk = IAk - 36;
				for(int ky = starty; ky < 6; ky += dy){
				   for(int kx = startx; kx < 6; kx += dx){
				      T val = static_cast<T>(0);
				      #pragma unroll
				      for (int i = 0; i < 6; i++){val += Tk[kx*6+i] * s_temp[ky*6+i];}
				      IAlamk[ky*6+kx] += val;
				   }
				}
			}
			// while we update the F
			#pragma unroll
			for(int c = starty; c <= num_children; c += dy){ // (14)  Fi[:,subtree(i)] = Ui * Minv[i,subtree(i)]
				T Minvc = s_Minv[rc_Ind(kMinv,(kMinv+c),NUM_POS)]; T *Fkc = &Fk[rc_Ind(0,(kMinv+c),6)];
				#pragma unroll
				for(int r = startx; r < 6; r += dx){Fkc[r] += Uk[r] * Minvc;}
			}
		}
		__syncthreads();
		// if has parent need to transform and add Fi to Flami
		// (15)    Flami[:,subtree(i)] = Flami[:,subtree(i)] + lamiXi* * Fi[:,subtree(i)]
		if (has_parent){T *Flamk = Fk - 6*NUM_POS; iteratedMatVMult_GPU<T,6,6,1,1>(Flamk,6,Tk,6,Fk,6,7);}
		__syncthreads();
	}
	// we note that we can finally compute the lk_fs fully in parallel so we break out to specific code for this situation
	// we need to compute lk_f = vcrosIv + Ia - fext
	#pragma unroll
	for(int k = starty; k < NUM_POS; k += dy){
		T *lk_v = &s_v[6*k]; T *lk_a = &s_a[6*k]; T *lk_f = &s_f[6*k]; T *lk_fext = &s_fext[6*k]; T *lk_I = &s_I[36*k]; T *lk_temp = &s_temp[6*k];
		#pragma unroll
		for(int r = startx; r < 6; r += dx){
			// first do temp = Iv and start lk_f = Ia - fext
			T val = 0;           T val2 = 0;
			#pragma unroll
			for (int i = 0; i < 6; i++){T currI = lk_I[i*6 + r]; val += currI*lk_v[i]; val2 += currI*lk_a[i];}
			lk_temp[r] = val;   lk_f[r] = val2;  if(s_fext != nullptr){lk_f[r] -= lk_fext[r];}
		}
	}
	__syncthreads();
	// now we can finish off lk_f by += with the vcross*Temp
	#pragma unroll
	for(int k = start; k < NUM_POS; k += delta){
		T *lk_v = &s_v[6*k]; T *lk_f = &s_f[6*k]; T *lk_temp = &s_temp[6*k];
		lk_f[0] += -lk_v[2]*lk_temp[1] + lk_v[1]*lk_temp[2];
		lk_f[1] += lk_v[2]*lk_temp[0] + -lk_v[0]*lk_temp[2];
		lk_f[2] += -lk_v[1]*lk_temp[0] + lk_v[0]*lk_temp[1];
		lk_f[3] += -lk_v[2]*lk_temp[1+3] + lk_v[1]*lk_temp[2+3];
		lk_f[4] += lk_v[2]*lk_temp[0+3] + -lk_v[0]*lk_temp[2+3];
		lk_f[5] += -lk_v[1]*lk_temp[0+3] + lk_v[0]*lk_temp[1+3];
		lk_f[0] += -lk_v[2+3]*lk_temp[1+3] + lk_v[1+3]*lk_temp[2+3];
		lk_f[1] += lk_v[2+3]*lk_temp[0+3] + -lk_v[0+3]*lk_temp[2+3];
		lk_f[2] += -lk_v[1+3]*lk_temp[0+3] + lk_v[0+3]*lk_temp[1+3];
	}
	__syncthreads();

	//----------------------------------------------------------------------------
	// RNEA Backward Pass / Minv forwrd pass
	//----------------------------------------------------------------------------
	for (int k = NUM_POS - 1; k >= 0; k--){ int kMinv = NUM_POS - 1 - k;
		// first for the RNEA
		T *lk_f = &s_f[6*k];     
		if(k > 0){
			T *lkm1_f = &s_f[6*(k-1)];     T *tkXkm1 = &s_T[36*k]; 
			matVMult_GPU<T,6,6,1,1>(lkm1_f,tkXkm1,6,lk_f); // lkm1_f += tkXkm1^T * lk_f
		}
		// then for Minv
		T *Uk = &s_U[6 * kMinv];      T Dinvk = s_D[2*kMinv+1];     T *Tk = &s_T[36*kMinv];
		T *Fk_subtree = &s_F[(NUM_POS+1)*6*kMinv];    T *Flamk_subtree = &s_F[(NUM_POS+1)*6*kMinv - 6*NUM_POS];
		bool has_parent = (kMinv > 0);   int N_subtree = NUM_POS-kMinv;   T *Minv_subtree = &s_Minv[NUM_POS*kMinv + kMinv];
		if (has_parent){ // (30) Minv[i,subtree(i)] += -D^-1 * Ui^T * Tk * Flami[:,subtree(i)]
			matVMult_GPU<T,6,6,0,1>(s_temp,Tk,6,Uk); __syncthreads();
			iteratedMatVMult_GPU<T,1,6,1,1>(Minv_subtree,NUM_POS,s_temp,1,Flamk_subtree,6,N_subtree,-Dinvk);
		}
		for(int i = starty; i < N_subtree; i += dy){ // (32) Fi = Si * Minv[i,subtree(i)]
			for(int j = startx; j < 6; j += dx){
		    	Fk_subtree[6*i+j] = (j == 2) ? Minv_subtree[NUM_POS*i] : static_cast<T>(0);
			}
		}
		__syncthreads();
		if (has_parent){ // (34) Fi[:,subtree(i)] += Tk * Flami[:,subtree(i)]
			iteratedMatVMult_GPU<T,6,6,1>(Fk_subtree,6,Tk,6,Flamk_subtree,6,N_subtree); __syncthreads();
		}
	}
}

template <typename T>
__device__
void FD_helpers_GPU_Minv(T *s_Minv, T *s_IA, T *s_T, T *s_U, T *s_D, T *s_F, T *s_temp, T *s_fext = nullptr){
	//----------------------------------------------------------------------------
	// Minv Backward Pass
	//----------------------------------------------------------------------------
	int start, delta; singleLoopVals_GPU(&start,&delta);
	int starty, dy, startx, dx; doubleLoopVals_GPU(&starty,&dy,&startx,&dx);
	for (int kMinv = NUM_POS-1; kMinv >= 0; kMinv--){bool has_parent = (kMinv > 0); int num_children = 6 - kMinv;
		T *Tk = &s_T[36*kMinv]; T *Fk = &s_F[6*NUM_POS*kMinv]; T *Uk = &s_U[6*kMinv]; T *Dk = &s_D[2*kMinv]; T *IAk = &s_IA[36*kMinv];
		// first the first half of the Minv comp fully in parallel aka get Uk and Dk
		for(int i = start; i < 6; i += delta){Uk[i] = IAk[6*2+i]; if(i == 2){Dk[0] = Uk[i]; Dk[1] = 1/Dk[0];}}
		// do the first half of the IA comp if we have a parent
		if (has_parent){
			for(int ky = starty; ky < 6; ky += dy){
			   	for(int kx = startx; kx < 6; kx += dx){
					T val = static_cast<T>(0);
					#pragma unroll
					for (int i = 0; i < 6; i++){val += (IAk[kx+6*i] - Uk[i]*Uk[kx]*Dk[1]) * Tk[ky*6+i];}
					s_temp[ky*6+kx] = val;
			   	}
		    }
		}
		// (21) Minv[i,i] = Di^-1
		if(__doOnce()){s_Minv[rc_Ind(kMinv,kMinv,NUM_POS)] = Dk[1];}
		// (12) Minv[i,subtree(i)] = -Di^-T * SiT * Fi[:,subtree(i)]
		if (num_children > 0){
			#pragma unroll
			for(int i = start; i < num_children; i += delta){
				s_Minv[rc_Ind(kMinv,(kMinv+1+i),NUM_POS)] = -Dk[1] * Fk[rc_Ind(2,(kMinv+1+i),6)];
			}
		}
		__syncthreads();
		// then we finish the IA (if applicable)
		if (has_parent){
		    T *IAlamk = IAk - 36;
		    for(int ky = starty; ky < 6; ky += dy){
		       	for(int kx = startx; kx < 6; kx += dx){
					T val = static_cast<T>(0);
					#pragma unroll
					for (int i = 0; i < 6; i++){val += Tk[kx*6+i] * s_temp[ky*6+i];}
					IAlamk[ky*6+kx] += val;
		       	}
		    }
		}
		// (14)  Fi[:,subtree(i)] = Ui * Minv[i,subtree(i)]
		#pragma unroll
		for(int c = starty; c <= num_children; c += dy){ 
		    T Minvc = s_Minv[rc_Ind(kMinv,(kMinv+c),NUM_POS)]; T *Fkc = &Fk[rc_Ind(0,(kMinv+c),6)];
		    #pragma unroll
		    for(int r = startx; r < 6; r += dx){Fkc[r] += Uk[r] * Minvc;}
		}
		__syncthreads();
		// (15) Flami[:,subtree(i)] = Flami[:,subtree(i)] + lamiXi* * Fi[:,subtree(i)]
		if (has_parent){ 
			T *Flamk = Fk - 6*NUM_POS;
			iteratedMatVMult_GPU<T,6,6,1,1>(Flamk,6,Tk,6,Fk,6,7);
			__syncthreads();
		}
	}
	//----------------------------------------------------------------------------
	// Minv forwrd pass
	//----------------------------------------------------------------------------
	for (int kMinv = 0; kMinv < NUM_POS; kMinv++){
		// then for Minv
		T *Uk = &s_U[6 * kMinv];      T Dinvk = s_D[2*kMinv+1];     T *Tk = &s_T[36*kMinv];
		T *Fk_subtree = &s_F[(NUM_POS+1)*6*kMinv];    T *Flamk_subtree = &s_F[(NUM_POS+1)*6*kMinv - 6*NUM_POS];
		bool has_parent = (kMinv > 0);   int N_subtree = NUM_POS-kMinv;   T *Minv_subtree = &s_Minv[NUM_POS*kMinv + kMinv];
		if (has_parent){ // (30) Minv[i,subtree(i)] += -D^-1 * Ui^T * Tk * Flami[:,subtree(i)]
			matVMult_GPU<T,6,6,0,1>(s_temp,Tk,6,Uk); __syncthreads();
			iteratedMatVMult_GPU<T,1,6,1,1>(Minv_subtree,NUM_POS,s_temp,1,Flamk_subtree,6,N_subtree,-Dinvk);
		}
		for(int i = starty; i < N_subtree; i += dy){ // (32) Fi = Si * Minv[i,subtree(i)]
			for(int j = startx; j < 6; j += dx){
		    	Fk_subtree[6*i+j] = (j == 2) ? Minv_subtree[NUM_POS*i] : static_cast<T>(0);
			}
		}
		__syncthreads();
		if (has_parent){ // (34) Fi[:,subtree(i)] += Tk * Flami[:,subtree(i)]
			iteratedMatVMult_GPU<T,6,6,1>(Fk_subtree,6,Tk,6,Flamk_subtree,6,N_subtree); __syncthreads();
		}
	}
}