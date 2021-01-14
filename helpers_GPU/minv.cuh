// template <typename T, bool MPC_MODE = false>
// __device__
// void FD_helpers_GPU_Minv(T *s_Minv, T *s_IA, T *s_T, T *s_U, T *s_D, T *s_F, T *s_temp, T *s_fext = nullptr){
// 	//----------------------------------------------------------------------------
// 	// Minv Backward Pass
// 	//----------------------------------------------------------------------------
// 	int start, delta; singleLoopVals_GPU(&start,&delta);
// 	for (int kMinv = NUM_POS - 1; kMinv >= 0; kMinv--){
// 		// start with the first bit of the Minv comp (U,D)
// 		T *Uk = &s_U[6*kMinv]; T *Dk = &s_D[2*kMinv]; T *IAk = &s_IA[36*kMinv]; T *Tk = &s_T[36*kMinv];
// 		for(int i = start; i < 6; i += delta){Uk[i] = IAk[6*2+i]; if(i == 2){Dk[0] = Uk[i]; Dk[1] = 1/Dk[0];}}
// 		__syncthreads();
// 		// if it has a parent (aka not root link) start updating IA by computing the temp val
// 		bool has_parent = kMinv > 0;
// 		if (has_parent){
// 			for(int i = start; i < 36; i += delta){
// 				int c = start / 6; int r = start % 6; T val = static_cast<T>(0);
// 				for (int i = 0; i < 6; i++){val += (IAk[r+6*i] - Uk[i]*Uk[r]*Dk[1]) * Tk[c*6+i];}
// 				s_temp[c*6+r] = val;
// 		    }
// 		}
// 		// Update Minv
// 		// Minv[k,k] = Dk^-1 (stored in Dk[1])
// 		if(LEAD_THREAD){s_Minv[kMinv*NUM_POS+kMinv] = Dk[1];}
// 		// Minv[k,subtree(k)] = -Dk^-T * Sk^T * Fk[:,subtree(k)]
// 		int num_children = 6 - kMinv; T *Fk = &s_F[6*NUM_POS*kMinv];
// 		if (num_children > 0){ 
// 			#pragma unroll
// 			for(int i = start; i < num_children; i += delta){
// 				s_Minv[(kMinv+1+i)*NUM_POS + kMinv] = -Dk[1] * Fk[(kMinv+1+i)*6 + 2];
// 			}
// 		}
// 		__syncthreads();
// 		// then finish the IA += Tk^T * temp
// 		if (has_parent){
// 		    T *IAlamk = IAk - 36;
// 		    for(int ind = start; ind < 36; ind += delta){
// 		    	int c = ind / 6; int r = ind % 6;
// 		    	T *Tkc = &Tk[6*r]; T *tempc = &s_temp[6*c];
// 		    	IAlamk[ind] += dotProd_GPU<T,6,1,1>(Tkc,tempc);
// 		    }
// 		}
// 		// while updating F
// 		// Fk[:,subtree(k)] = Ui * Minv[k,subtree(k)]
// 		#pragma unroll
// 		for(int ind = start; ind <= num_children*6; ind += delta){
// 			int c = ind / 6; int r = ind % 6;
// 		    Fk[(kMinv+c)*6 + r] += Uk[r] * s_Minv[(kMinv+c)*NUM_POS + kMinv];
// 		}
// 		__syncthreads();
// 		// Flamk[:,subtree(k)] = Flamk[:,subtree(k)] + Tk^T * Fk[:,subtree(k)]
// 		if (has_parent){ 
// 			for(int ind = start; ind <= NUM_POS*6; ind += delta){
// 				int c = ind / 6; int r = ind % 6; int cOffset = c*NUM_POS;
// 				T *FkColc = Fk + cOffset; T *FlamkColc = FkColc - 6*NUM_POS; T *TkColr = Tk + 6*r; 
// 				FlamkColc[r] = dotProd_GPU<T,6,6,1>(TkColr,FkColc);
// 			}
// 		}
// 		__syncthreads();
// 	}

// 	//----------------------------------------------------------------------------
// 	// Minv forwrd pass
// 	//----------------------------------------------------------------------------
    
// 	for (int kMinv = 0; kMinv < NUM_POS; kMinv++){
// 		// define variables
// 		int N_subtree = NUM_POS - kMinv; bool has_parent = (kMinv > 0); T *Tk = &s_T[36*kMinv];
// 		T *Fk_subtree = &s_F[(NUM_POS+1)*6*kMinv]; T *Minv_subtree = &s_Minv[NUM_POS*kMinv + kMinv];
// 		if(has_parent){
// 			// Start the update for Minv and F
// 			//    (if_has_parent) Temp = Tk^T * Flamk[:,subtree(i)] // transform columns of Flamk
// 			T *Flamk_subtree = Fk_subtree - 6*NUM_POS;
// 	        for (int ind = start; ind < 6*N_subtree; ind += delta){
// 	        	int c = ind / 6; int r = ind % 6;
// 	        	T *TkCol = &Tk[r*6]; // for transpose
// 	        	T *FlamCol = &Flamk_subtree[6*c];
// 	            s_temp[ind] += dotProd_GPU<T,6,1,1>(TkCol,FlamCol);
// 	        }
// 	        __syncthreads();
// 	        // Minv[k,subtree(k)] += -Dk^-1 * Uk^T * Temp
// 	        T *Uk = &s_U[6 * kMinv];
// 	        for (int ind = start; ind < N_subtree; ind += delta){
// 	        	T nDkinv = -s_D[2*kMinv+1]; T *tempc = s_temp[6*ind];
// 	        	Minv_subtree[NUM_POS*ind] += nDkinv * dotProd_GPU<T,6,1,1>(Uk,tempc);
// 	        }
// 	        __syncthreads();
// 	    }
// 	    // Fk = Sk * Minv[k,subtree(k)] + (if_has_parents)Temp
// 	    //    since Sk = [0,0,1,0,0,0] it just zeros out all but the 3rd entry
// 	    for(int ind = start; ind < 6*N_subtree; ind += delta){
// 			int c = ind / 6; int r = ind % 6;
// 			Fk_subtree[ind] = (r == 2) * Minv_subtree[NUM_POS*c] + has_parent * s_temp[ind]; // no branch
// 		}
// 	    __syncthreads();
// 	}
// }

template <typename T, bool MPC_MODE = false, bool QDD_READY = false>
__device__
void FD_helpers_GPU(T *s_v, T *s_a, T *s_f, T *s_Minv, T *s_qd, T *s_qdd, T *s_I, T *s_IA, T *s_T, T *s_U, T *s_D, T *s_F, T *s_temp, T *s_fext = nullptr){
	// In GPU mode want to reduce the number of syncs so we intersperce the computations
	//----------------------------------------------------------------------------
	// RNEA Forward Pass / Minv Backward Pass
	//----------------------------------------------------------------------------
	int start, delta; singleLoopVals_GPU(&start,&delta);
	for (int k = 0; k < NUM_POS; k++){
		// start with the first bit of the Minv comp (U,D)
		int kMinv = NUM_POS - 1 - k; T *Uk = &s_U[6*kMinv]; T *Dk = &s_D[2*kMinv]; T *IAk = &s_IA[36*kMinv];
		for(int i = start; i < 6; i += delta){Uk[i] = IAk[6*2+i]; if(i == 2){Dk[0] = Uk[i]; Dk[1] = 1/Dk[0];}}
		// then the first bit of the RNEA forward pass aka get v and a (part 1)
		T *lk_v = &s_v[6*k]; T *lkm1_v = lk_v - 6; T *lk_a = &s_a[6*k]; T *lkm1_a = lk_a - 6; T *tkXkm1 = &s_T[36*k];
		if (k == 0){
			// l1_v = vJ, for the first kMinvnk of a fixed base robot and the a is just gravity times the last col of the transform and then add qdd in 3rd entry
			for(int r = start; r < 6; r += delta){
				lk_v[r] = (r == 2) * s_qd[k];
				lk_a[r] = (!MPC_MODE) * tkXkm1[6*5+r] * GRAVITY + QDD_READY * (r == 2) * s_qdd[k];
			}
		}
	  	else{
			// first compute vk = tkXkm1 * lkm1_v + qd[k] in entry [2]
            // at the same time we can compute ak_part1 = tkXkm1 * lkm1_a + qdd[k] in entry [2]
			for(int rInd = start; rInd < 2*6; rInd += delta){
                int r = rInd % 6; T val; T *lkm1; T *lk; T *TkRow = &tkXkm1[r];
                if (rInd < 6){ // v
                    val = (r == 2) * s_qd[k];
                    lkm1 = lkm1_v;
                    lk = lk_v;
                }
                else{ // a_part1
                    val = QDD_READY * (r == 2) * s_qdd[k];
                    lkm1 = lkm1_a;
                    lk = lk_a;
                }
                lk[r] = dotProd_GPU<T,6,6,1>(TkRow,lkm1) + val;
			}
			
		}
		// then keep working on the Minv comp
		// if it has a parent (aka not root link) start updating IA by computing the temp val
		bool has_parent = kMinv > 0; T *Tk = &s_T[36*kMinv];
		if (has_parent){
			for(int rc = start; rc < 36; rc += delta){
				int c = rc / 6; int r = rc % 6; T val = static_cast<T>(0);
				for (int i = 0; i < 6; i++){val += (IAk[r+6*i] - Uk[i]*Uk[r]*Dk[1]) * Tk[c*6+i];}
				s_temp[c*6+r] = val;
		    }
		}
		// Update Minv
		
		// if(LEAD_THREAD){s_Minv[kMinv*NUM_POS+kMinv] = Dk[1];}
		// Minv[k,k] = Dk^-1 (stored in Dk[1]) // first part of subtree
		// Minv[k,subtree(k)] = -Dk^-T * Sk^T * Fk[:,subtree(k)]
		int N_subtree = NUM_POS - kMinv; T *Fk = &s_F[6*NUM_POS*kMinv];
		for(int i = start; i < N_subtree; i += delta){
			s_Minv[(kMinv+i)*NUM_POS + kMinv] = (i == 0)*(Dk[1]) + (i != 0)*(-Dk[1] * Fk[(kMinv+i)*6 + 2]);
		}
		__syncthreads();
		// then finish the a (if applicable)
		if (k != 0){
			// Finish the a = a_part1 + Mx3(v)
			for(int r = start; r < 6; r += delta){
				lk_a[r] += Mx3_GPU<T>(lk_v,r,s_qd[k]);
			}
		}
		// then finish the IA += Tk^T * temp
		if (has_parent){
		    T *IAlamk = IAk - 36;
		    for(int ind = start; ind < 36; ind += delta){
		    	int c = ind / 6; int r = ind % 6;
		    	T *Tkc = &Tk[6*r]; T *tempc = &s_temp[6*c];
		    	IAlamk[ind] += dotProd_GPU<T,6,1,1>(Tkc,tempc);
		    }
		}
		// while updating F
		// Fk[:,subtree(k)] = Ui * Minv[k,subtree(k)]
		for(int ind = start; ind < N_subtree*6; ind += delta){
			int c = ind / 6; int r = ind % 6;
		    Fk[(kMinv+c)*6 + r] += Uk[r] * s_Minv[(kMinv+c)*NUM_POS + kMinv];
		}
		__syncthreads();
		// Flamk[:,subtree(k)] += Tk^T * Fk[:,subtree(k)]
		if (has_parent){ 
			for(int ind = start; ind < N_subtree*6; ind += delta){
				int c = ind / 6; int r = ind % 6;
				T *FkColc = &Fk[(kMinv+c)*6]; T *FlamkColc = FkColc - 6*NUM_POS; T *TkColr = Tk + 6*r; 
				FlamkColc[r] += dotProd_GPU<T,6,1,1>(TkColr,FkColc);
			}
		}
	}
	// we now compute in parallel lk_f = vcrossIv + Ia - fext
    // we first do temp = Iv and f_part1 = Ia - fext
	for(int kr = start; kr < 2*6*NUM_POS; kr += delta){
        int r = kr % 6; int kInd = kr / 6; 
        T val = static_cast<T>(0); T *va; T *ftemp; T *IkRow;
        if (kInd < NUM_POS){ // compute f_part1 = Ia - fext
            int k = kInd;
            IkRow = &s_I[36*k+r];
            va = &s_a[6*k];
            ftemp = &s_f[6*k];
            if(s_fext != nullptr){val -= s_fext[6*k+r];} // apply fext if applicable
        }
        else{ // compute temp = Iv
            int k = kInd - NUM_POS;
            IkRow = &s_I[36*k+r];
            va = &s_v[6*k];
            ftemp = &s_temp[6*k];
        }
        // do the dot product
        ftemp[r] = dotProd_GPU<T,6,6,1>(IkRow,va);
	}
	__syncthreads();
	// now we can finish off f = f_part1 + vcross*Temp
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
	// RNEA Backward Pass (updating f) / Minv forwrd pass
	//----------------------------------------------------------------------------
    
	for (int k = NUM_POS - 1; k >= 0; k--){
		// define variables
		T *lk_f = &s_f[6*k]; T *lkm1_f = &s_f[6*(k-1)]; T *tkXkm1 = &s_T[36*k];
		int kMinv = NUM_POS - 1 - k; int N_subtree = NUM_POS - kMinv; bool has_parent = (kMinv > 0);
		T *Fk_subtree = &s_F[(NUM_POS+1)*6*kMinv]; T *Minv_subtree = &s_Minv[NUM_POS*kMinv + kMinv]; T *Tk = &s_T[36*kMinv];
		if(has_parent){
			// First finish f
			//    (if k > 0) lkm1_f += tkXkm1^T * lk_f
			// And start the update for Minv and F
			//    (if_has_parent) Temp = Tk * Flamk[:,subtree(i)] // transform columns of Flamk
			if (k > 0){
				for (int r = start; r < 6; r += delta){
					lkm1_f[r] += dotProd_GPU<T,6,1,1>(&tkXkm1[r*6],lk_f);
				}
			}
			T *Flamk_subtree = Fk_subtree - 6*NUM_POS;
	        for (int ind = start; ind < 6*N_subtree; ind += delta){
	        	int c = ind / 6; int r = ind % 6;
	            s_temp[ind] = dotProd_GPU<T,6,6,1>(&Tk[r],&Flamk_subtree[6*c]);
	        }
	        __syncthreads();
	        // Minv[k,subtree(k)] += -Dk^-1 * Uk^T * Temp
	        T *Uk = &s_U[6 * kMinv];
	        for (int ind = start; ind < N_subtree; ind += delta){
	        	T nDkinv = -s_D[2*kMinv+1]; T *tempc = &s_temp[6*ind];
	        	Minv_subtree[NUM_POS*ind] += nDkinv * dotProd_GPU<T,6,1,1>(Uk,tempc);
	        }
	        __syncthreads();
	    }
	    else{
	    	for (int r = start; r < 6; r += delta){
	    		T *TkCol = &tkXkm1[r*6]; // for transpose
	            lkm1_f[r] += dotProd_GPU<T,6,1,1>(TkCol,lk_f);
	        }
	    }
	    // Fk = Sk * Minv[k,subtree(k)] + (if_has_parents)Temp
	    //    since Sk = [0,0,1,0,0,0] it just zeros out all but the 3rd entry
	    for(int ind = start; ind < 6*N_subtree; ind += delta){
			int c = ind / 6; int r = ind % 6;
			Fk_subtree[ind] = (r == 2) * Minv_subtree[NUM_POS*c] + has_parent * s_temp[ind]; // no branch
			// printf("c[%d]r[%d] -> MinvTerm[%f] + tempTerm[%f] = Final[%f]\n",c,r,(r == 2) * Minv_subtree[NUM_POS*c],has_parent * s_temp[ind],Fk_subtree[ind]);
		}
	    __syncthreads();
	}
}

template <typename T, bool MPC_MODE = false>
__device__
void FD_helpers_GPU_af_update(T *s_v, T *s_a, T *s_f, T *s_qd, T *s_qdd, T *s_I, T *s_T, T *s_temp, T *s_fext = nullptr){
	//----------------------------------------------------------------------------
	// RNEA Forward Pass (to update a and f with the new qdd)
	//----------------------------------------------------------------------------
	int start, delta; singleLoopVals_GPU(&start,&delta);
	for (int k = 0; k < NUM_POS; k++){
		// then the first bit of the RNEA forward pass aka get v and a
		T *lk_v = &s_v[6*k]; T *lk_a = &s_a[6*k]; T *lkm1_a = lk_a - 6; T *tkXkm1 = &s_T[36*k];
		if (k == 0){
			if (LEAD_THREAD){lk_a[2] += s_qdd[k];} // add qdd in 3rd entry of a
		}
	  	else{
            // we recompute a due to the recursion so we start with
            // a = tkXkm1 * lkm1_a + Mx3(v) + qdd[k] in entry [2]
			for(int r = start; r < 6; r += delta){
                T *TkRow = &tkXkm1[r];
                lk_a[r] = dotProd_GPU<T,6,6,1>(TkRow,lkm1_a) + Mx3_GPU<T>(lk_v,r,s_qd[k]) + (r == 2) * s_qdd[k];
			}
		}
		__syncthreads();
	}
	// we now compute in parallel lk_f = vcrossIv + Ia - fext
    // we first do temp = Iv and f_part1 = Ia - fext
	for(int kr = start; kr < 2*6*NUM_POS; kr += delta){
        int r = kr % 6; int kInd = kr / 6; 
        T val = static_cast<T>(0); T *va; T *ftemp; T *IkRow;
        if (kInd < NUM_POS){ // compute f_part1 = Ia - fext
            int k = kInd;
            IkRow = &s_I[36*k+r];
            va = &s_a[6*k];
            ftemp = &s_f[6*k];
            if(s_fext != nullptr){val -= s_fext[6*k+r];} // apply fext if applicable
        }
        else{ // compute temp = Iv
            int k = kInd - NUM_POS;
            IkRow = &s_I[36*k+r];
            va = &s_v[6*k];
            ftemp = &s_temp[6*k];
        }
        // do the dot product
        ftemp[r] = dotProd_GPU<T,6,6,1>(IkRow,va);
	}
	__syncthreads();
	// now we can finish off f = f_part1 + vcross*Temp
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
	// RNEA Backward Pass (updating f)
	//----------------------------------------------------------------------------
    // lkm1_f += tkXkm1^T * lk_f
	for (int k = NUM_POS - 1; k > 0; k--){
		T *lk_f = &s_f[6*k]; T *lkm1_f = &s_f[6*(k-1)]; T *tkXkm1 = &s_T[36*k]; 
        for (int r = start; r < 6; r += delta){ T *TkCol = &tkXkm1[r*6]; // for transpose
            lkm1_f[r] += dotProd_GPU<T,6,1,1>(TkCol,lk_f);
        }
        __syncthreads();
	}
}


// compute/load T,I
// compute v,a,f,c,minv
// compute qdd
// recompute a,f
// compute gradient