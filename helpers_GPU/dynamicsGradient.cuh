#include <assert.h>
#include <cuda_runtime.h>

#ifndef GRAVITY
   #define GRAVITY 9.81
#endif
#ifndef NUM_POS
   #define NUM_POS 7
#endif

#if TEST_FOR_EQUIVALENCE
    dim3 BlockDimms(1,1);
#else
    // mainly single loops a few double loops where dimy = NUM_POS
    // dimx and single loops ocassionally very high but generally O(NUM_POS)
    // with small constants. As such we set dimx = 32 which is the warp size
    // to avoid divergence of warps during double loops (ideally) and this is
    // not too large that it will overtax the GPU scheduler
    dim3 BlockDimms(32,NUM_POS);
#endif

#ifndef LEAD_THREAD
   #define LEAD_THREAD (threadIdx.x == 0 && threadIdx.y == 0)
#define

__device__ __forceinline__
void singleLoopVals_GPU(int *start, int *delta){*start = threadIdx.x + threadIdx.y*blockDim.x; *delta = blockDim.x*blockDim.y;}
__device__ __forceinline__
void doubleLoopVals_GPU(int *starty, int *dy, int *startx, int *dx){*starty = threadIdx.y; *dy = blockDim.y; *startx = threadIdx.x; *dx = blockDim.x;}
__device__ __forceinline__
int __doOnce(){return threadIdx.x == 0 && threadIdx.y == 0;}

#include <type_traits>
template <typename T, int M, int N>
__device__
void printMat_GPU(T *A, int lda){
    for(int i=0; i<M; i++){
        for(int j=0; j<N; j++){printf("%.6f ",A[i + lda*j]);}
        printf("\n");
    }
} 

__host__ 
void gpuAssert(cudaError_t code, const char *file, const int line, bool abort=true){
    if (code != cudaSuccess){
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort){cudaDeviceReset(); exit(code);}
    }
}
#define gpuErrchk(err) {gpuAssert(err, __FILE__, __LINE__);}

// dot product between two vectors of length K with values spaced s_a, s_b
// since in some cases we will know the spacing at compile time and sometimes it
// varies at runtime we provide all 4 options to give the compiler a better chance
// to fully optimize the code loop (and unroll etc.)
template <typename T, int K, int s_a, int s_b>
__device__
T dotProd_GPU(T *a, T *b){
   T val = static_cast<T>(0);
   #pragma unroll
   for (int j=0; j < K; j++){val += a[s_a * j] * b[s_b * j];}
   return val;
}
template <typename T, int K, int s_a>
__device__
T dotProd_GPU(T *a, T *b, int s_b){
   T val = static_cast<T>(0);
   #pragma unroll
   for (int j=0; j < K; j++){val += a[s_a * j] * b[s_b * j];}
   return val;
}
template <typename T, int K, int s_b>
__device__
T dotProd_GPU(T *a, int s_a, T *b){
   T val = static_cast<T>(0);
   #pragma unroll
   for (int j=0; j < K; j++){val += a[s_a * j] * b[s_b * j];}
   return val;
}
template <typename T, int K>
__device__
T dotProd_GPU(T *a, int s_a, T *b, int s_b){
   T val = static_cast<T>(0);
   #pragma unroll
   for (int j=0; j < K; j++){val += a[s_a * j] * b[s_b * j];}
   return val;
}

template <typename T>
__device__ 
T Mx3_GPU(T *vec, int r, T alpha = 1){
    T sgnAlpha = (r == 1 || r == 4) * -alpha + (r == 0 || r == 3) * alpha; // if r == 2 || r == 5 will set to 0
    int ind = (r == 0) * 1 + (r == 3) * 4 + (r == 4) * 3; // else will be 0 so (r == 1) * 0 is not needed
    return sgnAlpha*vec[ind];
}

template <typename T>
__device__
void serialMx3_GPU(T *Mxvec, T *vec){
    // [v(1), -v(0), 0, v(4), -v(3), 0]^T
    Mxvec[0] = vec[1];
    Mxvec[1] = -vec[0];
    Mxvec[2] = static_cast<T>(0);
    Mxvec[3] = vec[4];
    Mxvec[4] = -vec[3];
    Mxvec[5] = static_cast<T>(0);
}

template <typename T>
__device__
void serialMx3_GPU(T *Mxvec, T *vec, T alpha){
    // [v(1), -v(0), 0, v(4), -v(3), 0]^T
    Mxvec[0] = alpha*vec[1];
    Mxvec[1] = -alpha*vec[0];
    Mxvec[2] = static_cast<T>(0);
    Mxvec[3] = alpha*vec[4];
    Mxvec[4] = -alpha*vec[3];
    Mxvec[5] = static_cast<T>(0);
}

template <typename T>
__device__
void serialFx_GPU(T *Fxmat, T *vec){
    /* 0 -v(2)  v(1)    0  -v(5)  v(4)
    v(2)   0  -v(0)  v(5)    0  -v(3)
    -v(1) v(0)    0  -v(4)  v(3)    0
     0     0     0     0  -v(2)  v(1)
     0     0     0   v(2)    0  -v(0)
     0     0     0  -v(1)  v(0)    0 */
    Fxmat[0] = static_cast<T>(0);
    Fxmat[1] = vec[2];
    Fxmat[2] = -vec[1];
    Fxmat[3] = static_cast<T>(0);
    Fxmat[4] = static_cast<T>(0);
    Fxmat[5] = static_cast<T>(0);
    Fxmat[6] = -vec[2];
    Fxmat[7] = static_cast<T>(0);
    Fxmat[8] = vec[0];
    Fxmat[9] = static_cast<T>(0);
    Fxmat[10] = static_cast<T>(0);
    Fxmat[11] = static_cast<T>(0);
    Fxmat[12] = vec[1];
    Fxmat[13] = -vec[0];
    Fxmat[14] = static_cast<T>(0);
    Fxmat[15] = static_cast<T>(0);
    Fxmat[16] = static_cast<T>(0);
    Fxmat[17] = static_cast<T>(0);
    Fxmat[18] = static_cast<T>(0);
    Fxmat[19] = vec[5];
    Fxmat[20] = -vec[4];
    Fxmat[21] = static_cast<T>(0);
    Fxmat[22] = vec[2];
    Fxmat[23] = -vec[1];
    Fxmat[24] = -vec[5];
    Fxmat[25] = static_cast<T>(0);
    Fxmat[26] = vec[3];
    Fxmat[27] = -vec[2];
    Fxmat[28] = static_cast<T>(0);
    Fxmat[29] = vec[0];
    Fxmat[30] = vec[4];
    Fxmat[31] = -vec[3];
    Fxmat[32] = static_cast<T>(0);
    Fxmat[33] = vec[1];
    Fxmat[34] = -vec[0];
    Fxmat[35] = static_cast<T>(0);
}

#define tz_j1 static_cast<T>(0.1575)
#define tz_j2 static_cast<T>(0.2025)
#define ty_j3 static_cast<T>(0.2045)
#define tz_j4 static_cast<T>(0.2155)
#define ty_j5 static_cast<T>(0.1845)
#define tz_j6 static_cast<T>(0.2155)
#define ty_j7 static_cast<T>(0.081)
template <typename T>
__device__
void updateTransforms_GPU(T *s_T, T *s_sinq, T *s_cosq){
    // note since the TL and BR are identical we can simply load in the TL
    // and then copy it over to the BR in parallel
    if(__doOnce()){
        // Link 0
        s_T[36*0 + 6*0 + 0] = s_cosq[0];
        s_T[36*0 + 6*0 + 1] = -s_sinq[0];
        s_T[36*0 + 6*0 + 3] = -tz_j1 * s_sinq[0];
        s_T[36*0 + 6*0 + 4] = -tz_j1 * s_cosq[0];
        s_T[36*0 + 6*1 + 0] = s_sinq[0];
        s_T[36*0 + 6*1 + 1] = s_cosq[0];
        s_T[36*0 + 6*1 + 3] =  tz_j1 * s_cosq[0];
        s_T[36*0 + 6*1 + 4] = -tz_j1 * s_sinq[0];
        // s_T[36*0 + 6*3 + 3] = s_cosq[0];
        // s_T[36*0 + 6*3 + 4] = -s_sinq[0];
        // s_T[36*0 + 6*4 + 3] = s_sinq[0];
        // s_T[36*0 + 6*4 + 4] = s_cosq[0];
        // Link 1
        s_T[36*1 + 6*0 + 0] = -s_cosq[1];
        s_T[36*1 + 6*0 + 1] = s_sinq[1];;
        s_T[36*1 + 6*1 + 3] = -tz_j2 * s_cosq[1];
        s_T[36*1 + 6*1 + 4] =  tz_j2 * s_sinq[1];
        s_T[36*1 + 6*2 + 0] = s_sinq[1];
        s_T[36*1 + 6*2 + 1] = s_cosq[1];
        // s_T[36*1 + 6*3 + 3] = -s_cosq[1];
        // s_T[36*1 + 6*3 + 4] = s_sinq[1];
        // s_T[36*1 + 6*5 + 3] = s_sinq[1];
        // s_T[36*1 + 6*5 + 4] = s_cosq[1];
        // Link 2
        s_T[36*2 + 6*0 + 0] = -s_cosq[2];
        s_T[36*2 + 6*0 + 1] = s_sinq[2];
        s_T[36*2 + 6*0 + 3] =  ty_j3 * s_sinq[2];
        s_T[36*2 + 6*0 + 4] =  ty_j3 * s_cosq[2];
        s_T[36*2 + 6*2 + 0] = s_sinq[2];
        s_T[36*2 + 6*2 + 1] = s_cosq[2];
        s_T[36*2 + 6*2 + 3] =  ty_j3 * s_cosq[2];
        s_T[36*2 + 6*2 + 4] = -ty_j3 * s_sinq[2];
        // Link 3
        // s_T[36*2 + 6*3 + 3] = -s_cosq[2];
        // s_T[36*2 + 6*3 + 4] = s_sinq[2];
        // s_T[36*2 + 6*5 + 3] = s_sinq[2];
        // s_T[36*2 + 6*5 + 4] = s_cosq[2];
        s_T[36*3 + 6*0 + 0] = s_cosq[3];
        s_T[36*3 + 6*0 + 1] = -s_sinq[3];
        s_T[36*3 + 6*1 + 3] =  tz_j4 * s_cosq[3];
        s_T[36*3 + 6*1 + 4] = -tz_j4 * s_sinq[3];
        s_T[36*3 + 6*2 + 0] = s_sinq[3];
        s_T[36*3 + 6*2 + 1] = s_cosq[3];
        // Link 4
        // s_T[36*3 + 6*3 + 3] = s_cosq[3];
        // s_T[36*3 + 6*3 + 4] = -s_sinq[3];
        // s_T[36*3 + 6*5 + 3] = s_sinq[3];
        // s_T[36*3 + 6*5 + 4] = s_cosq[3];
        s_T[36*4 + 6*0 + 0] = -s_cosq[4];
        s_T[36*4 + 6*0 + 1] = s_sinq[4];
        s_T[36*4 + 6*0 + 3] =  ty_j5 * s_sinq[4];
        s_T[36*4 + 6*0 + 4] =  ty_j5 * s_cosq[4];
        s_T[36*4 + 6*2 + 0] = s_sinq[4];
        s_T[36*4 + 6*2 + 1] = s_cosq[4];
        s_T[36*4 + 6*2 + 3] =  ty_j5 * s_cosq[4];
        s_T[36*4 + 6*2 + 4] = -ty_j5 * s_sinq[4];
        // s_T[36*4 + 6*3 + 3] = -s_cosq[4];
        // s_T[36*4 + 6*3 + 4] = s_sinq[4];
        // s_T[36*4 + 6*5 + 3] = s_sinq[4];
        // s_T[36*4 + 6*5 + 4] = s_cosq[4];
        // Link 5
        s_T[36*5 + 6*0 + 0] = s_cosq[5];
        s_T[36*5 + 6*0 + 1] = -s_sinq[5];
        s_T[36*5 + 6*1 + 3] =  tz_j6 * s_cosq[5];
        s_T[36*5 + 6*1 + 4] = -tz_j6 * s_sinq[5];
        s_T[36*5 + 6*2 + 0] = s_sinq[5];
        s_T[36*5 + 6*2 + 1] = s_cosq[5];
        // s_T[36*5 + 6*3 + 3] = s_cosq[5];
        // s_T[36*5 + 6*3 + 4] = -s_sinq[5];
        // s_T[36*5 + 6*5 + 3] = s_sinq[5];
        // s_T[36*5 + 6*5 + 4] = s_cosq[5];
        // Link 6
        s_T[36*6 + 6*0 + 0] = -s_cosq[6];
        s_T[36*6 + 6*0 + 1] = s_sinq[6];
        s_T[36*6 + 6*0 + 3] =  ty_j7 * s_sinq[6];
        s_T[36*6 + 6*0 + 4] =  ty_j7 * s_cosq[6];
        s_T[36*6 + 6*2 + 0] = s_sinq[6];
        s_T[36*6 + 6*2 + 1] = s_cosq[6];
        s_T[36*6 + 6*2 + 3] =  ty_j7 * s_cosq[6];
        s_T[36*6 + 6*2 + 4] = -ty_j7 * s_sinq[6];
        // s_T[36*6 + 6*3 + 3] = -s_cosq[6];
        // s_T[36*6 + 6*3 + 4] = s_sinq[6];
        // s_T[36*6 + 6*5 + 3] = s_sinq[6];
        // s_T[36*6 + 6*5 + 4] = s_cosq[6];
    }
    __syncthreads();
    int start, delta; singleLoopVals_GPU(&start,&delta);
    // then copy over the TL to the BR
    for(int kcr = 0; kcr < NUM_POS*9; kcr += delta){
        int k = kcr / 9; int cr = kcr % 9; int c = cr / 3; int r = cr % 3;
        T *Tkcr = &s_T[36*k + 6*c + r];
        Tkcr[6*3 + 3] = *Tkcr; // BR is 3 cols and 3 rows over
    }
}

// include Minv Helpers
#include "minv.cuh"

template <typename T, bool MPC_MODE = false>
__device__
void FD_helpers_GPU_vaf(T *s_v, T *s_a, T *s_f, T *s_qd, T *s_qdd, T *s_I, T *s_T, T *s_temp, T *s_fext = nullptr){
	// In GPU mode want to reduce the number of syncs so we intersperce the computations
	//----------------------------------------------------------------------------
	// RNEA Forward Pass
	//----------------------------------------------------------------------------
	int start, delta; singleLoopVals_GPU(&start,&delta);
	for (int k = 0; k < NUM_POS; k++){
		// then the first bit of the RNEA forward pass aka get v and a
		T *lk_v = &s_v[6*k]; T *lkm1_v = lk_v - 6; T *lk_a = &s_a[6*k]; T *lkm1_a = lk_a - 6; T *tkXkm1 = &s_T[36*k];
		if (k == 0){
			// l1_v = vJ, for the first kMinvnk of a fixed base robot and the a is just gravity times the last col of the transform and then add qdd in 3rd entry
			for(int r = start; r < 6; r += delta){
				lk_v[r] = (r == 2) * s_qd[k];
				lk_a[r] = (!MPC_MODE) * tkXkm1[6*5+r] * GRAVITY + (r == 2)* s_qdd[k];
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
                    val = (r == 2) * s_qdd[k];
                    lkm1 = lkm1_a;
                    lk = lk_a;
                }
                lk[r] = dotProd_GPU<T,6,6,1>(TkRow,lkm1) + val;
			}
			__syncthreads();
			// Finish the a = a_part1 + Mx3(v)
			for(int r = start; r < 6; r += delta){
				lk_a[r] += Mx3_GPU<T>(lk_v,r,s_qd[k]);
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

template <typename T>
__device__ 
void computeTempVals_GPU_part1(T *s_Iv, T *s_Ta, T *s_Tv, T *s_I, T *s_T, T *s_v, T *s_a){
    // start with the matMuls: Tk*vlamk, Tk*alamk, Ik*vk
    int start, delta; singleLoopVals_GPU(&start, &delta);
    for (int ind = start; ind < 18*NUM_POS; ind += delta){
        int r = ind % 6; int kind = ind / 6; int k = kind % NUM_POS;
        int ind6 = 6*k; int TmatInd = 6*ind6 + r; 
        T *dst; T *TmatRow; T *vec; // branch on pointer selector
        if(kind < NUM_POS){                 dst = &s_Iv[ind6+r]; TmatRow = &s_I[TmatInd]; vec = &s_v[ind6];}
        else if(kind < 2*NUM_POS && k != 0){dst = &s_Tv[ind6+r]; TmatRow = &s_T[TmatInd]; vec = &s_v[ind6-6];}
        else if(k != 0){                    dst = &s_Ta[ind6+r]; TmatRow = &s_T[TmatInd]; vec = &s_a[ind6-6];}
        else{continue;} // no lambda for k = 0
        // then do the dot product for that matrix row and vector without branching
        *dst = dotProd_GPU<T,6,6,1>(TmatRow,vec);
    }
}

template <typename T>
__device__ 
void computeTempVals_GPU_part2(T *s_Fxv, T *s_Mxv, T *s_Mxf, T *s_MxTa, T *s_vdq, T *s_Ta, T *s_Tv, T *s_v, T *s_f){
	// then sync and do all Mxs: Mx3(fk), Mx3(vk)*qd[k], Mx3(Tv), Mx3(Ta) --- note Mx3(Tv) stored in k col of dVk/dq 
	int start, delta; singleLoopVals_GPU(&start,&delta); T *dst; T *src;
	for (int k = start; k < 5*NUM_POS; k += delta){ // better to branch here on the pointers and then in parallel do (most of) the comp
        int kadj = k % NUM_POS; int ind = 6*kadj; // only NUM_POS of each comp
        if (k < NUM_POS){dst = &s_Mxv[ind]; src = &s_v[ind];}
        else if (k < 2*NUM_POS){dst = &s_Mxf[ind]; src = &s_f[ind];}
        else if (k < 3*NUM_POS){dst = &s_MxTa[ind]; src = &s_Ta[ind];}
        else if (k < 4*NUM_POS){dst = &s_vdq[3*kadj*(kadj+1) + ind]; src = &s_Tv[ind];} // load straight into vdq
        else{dst = &s_Fxv[6*ind]; src = &s_v[ind];}

        if (k < 2*NUM_POS || (k < 4*NUM_POS && ind != 0)){serialMx3_GPU<T>(dst,src);} // no ta or tv for k == 0
        else if (k >= 4*NUM_POS){serialFx_GPU<T>(dst,src);} // this branching is non-ideal but not really a better way to do this
	}
}

template <typename T>
__device__ 
void computeTempVals_GPU_part3(T *s_vdq, T *s_vdqd, T *s_TFxf, T *s_FxvI, T *s_Fxv, T *s_Mxf, T *s_T, T *s_I){
    // then sync before finishing off with Fxvk*Ik, -Tk^T*Mxfk = Tk^T*Fxfk and preload some of Vdu
    // we start by computing Fxv*I across all columns (6) of each k and all k of -T^T*Mxf
    int start, delta; singleLoopVals_GPU(&start,&delta);
    for (int ind = start; ind < 6*(NUM_POS+6*NUM_POS); ind += delta){
        int r = ind % 6; int kInd = ind / 6; T *dst; T *v1; T *v2; int v1Step; T alpha;
        if(kInd < NUM_POS){ //-T^T*Mxf
            alpha = static_cast<T>(-1);
            int ind6 = 6*kInd; 
            dst = &s_TFxf[ind6 + r];
            v1 = &s_T[6*ind6 + 6*r]; v1Step = 1; // columns of s_T because transpose
            v2 = &s_Mxf[ind6]; // vector so step is 1
        }
        else{ // I^T*Fxv^T
            alpha = static_cast<T>(1);
            kInd -= NUM_POS;
            int k = kInd / 6;
            int c = kInd % 6;
            int ind36 = 36*k;
            dst = &s_FxvI[ind36 + 6*c + r];
            v1 = &s_Fxv[ind36 + r]; v1Step = 6; // row of Fxv
            v2 = &s_I[ind36 + 6 * c];// columns of I so step is 1
        }
        /// then do the dot product for those two vectors
        *dst = alpha*dotProd_GPU<T,6,1>(v1,v1Step,v2);
    }
	// also note for the forward pass we can pre-load in part of 
	// dVk/dqd [Tk[:,2] for k-1 col | [0 0 1 0 0 0]T for k col]
	// dVk/dq has the pattern [0 for 0 col | MxTv for k col] but the later was already stored
	for (int kr = start; kr < 6*NUM_POS; kr += delta){
        int r = kr % 6; int k = kr / 6;
        int ind = 3*k*(k+1); int ind6 = 6*k; 
		s_vdq[ind + r] = static_cast<T>(0);
		s_vdqd[ind + ind6 + r] = (r == 2)*static_cast<T>(1); // branch free conditional loads
		if(k > 0){s_vdqd[ind + ind6 - 6 + r] = s_T[ind6*6 + 12 + r];} // need if or will overwrite vdq
	}
}

template <typename T>
__device__
void compute_vdu_GPU(T *s_vdq, T *s_vdqd, T *s_T){
    int start, delta; singleLoopVals_GPU(&start,&delta); T *dst; T *lamkc;
    for (int k = 0; k < NUM_POS; k++){
       	// dVk/dqd has the pattern [Tk*dvlamk/dqd for 0:k-2 cols | Tk[:,2] for k-1 col | [0 0 1 0 0 0]T for k col | 0 for k+1 to NUM_POS col]
        // only need to now load the multuplication into 0:k-2 -- starting with k = 3
        // dVk/dq has the pattern [0 | Tk*dvlank/dq for 1:k-1 cols | MxMat_col3(Tk*vlamk) | 0 for k+1 to NUM_POS col]
        // already loaded the 0 and the MxTv so only 1:k-1
       	int indlamk = 3*k*(k-1); int indk = indlamk + 6*k;
       	for (int ind = start; ind < 6*(k-1+k); ind += delta){ // again we use one loop and diverge on pointer loading and then branch free compute
            int r = ind % 6; int cind = ind / 6; T *TmatRow = &s_T[36*k+r];
            if(cind < k){
                if(cind == 0){continue;} // don't fill in for 0
                int indc = cind*6;     
                lamkc = &s_vdq[indlamk + indc];
                dst = &s_vdq[indk + indc + r];
            }
            else{
                int indc = (cind-k)*6;
                lamkc = &s_vdqd[indlamk + indc];
                dst = &s_vdqd[indk + indc + r];
            }
            *dst = dotProd_GPU<T,6,6,1>(TmatRow,lamkc);;
       	}
       	__syncthreads();
    }
}

template <typename T>
__device__
void compute_adu_GPU_part1(T *s_adq, T *s_adqd, T *s_vdq, T *s_vdqd, T *s_MxTa, T *s_Mxv, T *s_qd){
	// sync then again exploiting sparsity in dv and da we continue
	// dak/du = Tk * dalamk/du + MxMatCol3_oncols(dvk/du)*qd[k] + {for q in col k: MxMatCol3(Tk*alamk)} 
	//                                                          + {for qd in col k: MxMatCol3(vk)}
	// remember we have already computed Mxv, MxTa so add that to Mx3(dvk/du)*qd[k] and store in dak/du in parallel
	int starty, dy, startx, dx; doubleLoopVals_GPU(&starty,&dy,&startx,&dx);
    T *adukc; T *vdukc; T *Mx; bool doMx;
	for (int k = starty; k < NUM_POS; k += dy){int indk = 3*k*(k+1);
      	for (int c = startx; c <= 2*k+1; c += dx){
            // again we follow the branch on pointers and execute in sync pattern
            if (c <= k){ // dq
                int indkc = indk + 6*c;
                adukc = &s_adq[indkc];
                vdukc = &s_vdq[indkc];
                Mx = &s_MxTa[6*k];
                doMx = (c == k && k > 0);
            }
            else{ // dqd
                int indkc = indk + 6*(c-k-1);
                adukc = &s_adqd[indkc];
                vdukc = &s_vdqd[indkc];
                Mx = &s_Mxv[6*k];
                doMx = (c == 2*k+1);
            }
            // MxMatCol3_oncols(dvk/du)*qd[k]
            serialMx3_GPU<T>(adukc,vdukc,s_qd[k]);
            // add the special part if applicable
            if(doMx){
             	#pragma unroll
             	for (int r = 0; r < 6; r++){adukc[r] += Mx[r];}
            }
      	}
	}
}

template <typename T>
__device__
void compute_adu_GPU_part2(T *s_adq, T *s_adqd, T *s_T){
	// then do the loop to sum up the da/dus with the transform
	int start, delta; singleLoopVals_GPU(&start,&delta);
	for (int k = 1; k < NUM_POS; k++){
		int lamkStart = 3*k*(k-1); int sixk = 6*k; int kStart = lamkStart + sixk;
        T *Tk = &s_T[36*k]; T *dst; T *src;
		for (int rc = start; rc < k*6*2; rc += delta){
			int r = rc % 6;
            if(rc < sixk){ // dq
                int cStart = rc - r;
                dst = &s_adq[kStart + cStart];
                src = &s_adq[lamkStart + cStart];
            }
            else{ // dqd
                int cStart = rc - r - 6*k;
                dst = &s_adqd[kStart + cStart];
                src = &s_adqd[lamkStart + cStart];
            }
			T val = static_cast<T>(0); 
			#pragma unroll
			for(int i = 0; i < 6; i++){
				val += Tk[r + 6 * i] * src[i];
			}
			dst[r] += val; 
		}
		__syncthreads();
	}
}

template <typename T>
__device__
void compute_Fxvdu(T *s_Fxvdq, T *s_Fxvdqd, T *s_vdq, T *s_vdqd){
	// we can also precompute FxMat(dv/du) here for use later (expanding each col of dv/du into a 6x6 matrix)
	// this is probably the slowest operation because it is so irregular on the GPU
	int start, delta; singleLoopVals_GPU(&start,&delta); 
    T *dst; T *src; int NUM_COLS = static_cast<int>(NUM_POS*(NUM_POS+1)/2);
	for (int kc = start; kc < 2*NUM_COLS; kc += delta){
        if (kc < NUM_COLS){ // better to branch here on the pointers and then in parallel do the comp
            int indk = 6*kc; int indMk = indk*6;
            dst = &s_Fxvdq[indMk]; src = &s_vdq[indk];
        }
        else{
            int indk = 6*(kc-NUM_COLS); int indMk = indk*6;
            dst = &s_Fxvdqd[indMk]; src = &s_vdqd[indk];
        }
        serialFx_GPU<T>(dst,src);
	}
}

template <typename T>
__device__
void compute_fdu_GPU(T *s_fdq, T *s_fdqd, T *s_adq, T *s_adqd, T *s_vdq, T *s_vdqd, T *s_Fxvdq, T *s_Fxvdqd, T *s_FxvI, T *s_Iv, T *s_I){
	// now we can compute all dfdu in parallel noting that dv and da only exist for c <=k
	// dfk/du = Ik*dak/du + FxMat(vk)*Ik*dvk/du + FxMat(dvk/du across cols)*Ik*vk
	// noting that all of them have a dv or da in each term we know that df only exists for c <=k (in the fp but all NUM_POS in bp)
	// also we already have FxvI, Iv, and Fxdv so we just need to sum the iterated MatVs: I*da + FxvI*dv + Fxdv*Iv
	int starty, dy, startx, dx; doubleLoopVals_GPU(&starty,&dy,&startx,&dx);
	for (int k = starty; k < NUM_POS; k += dy){int ind6 = 6*k; int ind36 = 6*ind6;
	  	T *Ivk = &s_Iv[ind6]; T *Fxdvdu; T *vdu; T *adu; T *fdu;
	  	for (int cr = startx; cr < (k+1)*6*2; cr += dx){ // make sure c <= k
			int c = cr / 6;  int r = cr % 6;
            if(c <= k){ //dq
                Fxdvdu = s_Fxvdq;
                vdu = s_vdq;
                adu = s_adq;
                fdu = s_fdq;
            }  
            else{ //dqd
                c -= k + 1;
                Fxdvdu = s_Fxvdqd;
                vdu = s_vdqd;
                adu = s_adqd;
                fdu = s_fdqd;
            }
            int c6 = c*6; int indkc = 3*k*(k+1) + c6; int indFkcr = 6*indkc + r;
			T *Fxdvdukcr = &Fxdvdu[indFkcr]; T *FxvIkr = &s_FxvI[ind36 + r]; T *Ikr = &s_I[ind36 + r];
            T *vdukc = &vdu[indkc]; T *adukc = &adu[indkc]; T val = static_cast<T>(0);
			#pragma unroll
			for(int i = 0; i < 6; i++){
		        val  +=  Ikr[6 * i] * adukc[i] + FxvIkr[6 * i] * vdukc[i] + Fxdvdukcr[6 * i] * Ivk[i];
     		}
	    	fdu[NUM_POS*ind6 + c6 + r] = val;
	  	}
	}
}

template <typename T>
__device__
void compute_cdu_GPU_part1(T *s_fdq, T *s_fdqd, T *s_TFxf, T *s_T){
	int start, delta; singleLoopVals_GPU(&start,&delta); T *fdukc; T *fdulamkc; bool PEQ_FLAG;
	for (int k = NUM_POS - 1; k > 0; k--){int ind6 = 6*k;
       	for (int rc = start; rc < 2*6*NUM_POS; rc += delta){
            int r = rc % 6; int c = rc / 6; T val = static_cast<T>(0); T *Tkr = &s_T[(ind6 + r) * 6];
            if (c < NUM_POS){ // dq
                int indlamkc = 6*NUM_POS*(k-1) + 6*c; int indkc = indlamkc + 6*NUM_POS;
                fdulamkc = &s_fdq[indlamkc]; fdukc = &s_fdq[indkc];
                PEQ_FLAG = c < k;
                // if c = k (and dq) then += Tk^T*Fx(fk) which we already have computed so just += :)!
                if (c == k){val += s_TFxf[ind6 + r];}
            }
            else{ // dqd
                int indkc = 6*NUM_POS*(k-1) + 6*c; int indlamkc = indkc - 6*NUM_POS;
                fdukc = &s_fdqd[indkc]; fdulamkc = &s_fdqd[indlamkc];
                PEQ_FLAG = (c - NUM_POS) < k;
            }
            // dflamk/du += Tk^T*cols(df/du)
         	#pragma unroll
         	for(int i = 0; i < 6; i++){
               val += Tkr[i] * fdukc[i];
         	}
            // result += if c <= k else set because should be 0 right now anyway
            if (PEQ_FLAG){fdulamkc[r] += val;}
            else {fdulamkc[r] = val;}
       	}
       	__syncthreads();
	}
}

template <typename T, bool VEL_DAMPING = 1>
__device__
void compute_cdu_GPU_part2(T *s_cdu, T *s_fdq, T *s_fdqd){
	int start, delta; singleLoopVals_GPU(&start,&delta);
	// now in full parallel load in dc/du = row 3 of df/du
	for (int rc = start; rc < 2*NUM_POS*NUM_POS; rc += delta){
      	int r = rc % NUM_POS; int c = rc / NUM_POS; // r and c of output (note c = k for fdu)
     	T *dst = &s_cdu[NUM_POS*c + r]; T *src; 
        if (c < NUM_POS){src = &s_fdq[6*NUM_POS*r + 6*c + 2];}
        else{src = &s_fdqd[6*NUM_POS*r + 6*(c-NUM_POS) + 2];}
     	*dst = *src; 
     	if(VEL_DAMPING && (r + NUM_POS == c)){*dst += 0.5;}
   	}
}

template <typename T, bool VEL_DAMPING = 1>
__device__
void inverseDynamicsGradient_GPU(T *s_cdu, T *s_qd, T *s_v, T *s_a, T *s_f, T *s_I, T *s_T){
	// note v,a,fdu (for iiwa) only exist for c < k cols so can compactly fit in (1 + 2 + ... NUM_POS)*6 space
	// this is (NUM_POS)*(NUM_POS+1)/2*6 = 3*NUM_POS*(NUM_POS+1)
	__shared__ T s_vdq[3*NUM_POS*(NUM_POS+1)];
	__shared__ T s_vdqd[3*NUM_POS*(NUM_POS+1)];
	__shared__ T s_adq[3*NUM_POS*(NUM_POS+1)];
	__shared__ T s_adqd[3*NUM_POS*(NUM_POS+1)];
	__shared__ T s_fdq[6*NUM_POS*NUM_POS];
	__shared__ T s_fdqd[6*NUM_POS*NUM_POS];
	// we also need FxMat(vdu)
	__shared__ T s_Fxvdq[18*NUM_POS*(NUM_POS+1)]; // expand vectors into mats
	__shared__ T s_Fxvdqd[18*NUM_POS*(NUM_POS+1)]; // expand vectors into mats

	//----------------------------------------------------------------------------
	// Compute helpers in parallel with few syncs
	//----------------------------------------------------------------------------
	// start by computing the temp vals (note that one can be directly parallel store in vdq and vdqd a couple things)
	__shared__ T s_temp[72*NUM_POS];
	__shared__ T s_temp2[36*NUM_POS];
	T *s_Tv = &s_temp[18];
	T *s_Ta = &s_temp[24*NUM_POS];
	T *s_Fxv = &s_temp[36*NUM_POS];
	T *s_Mxf = &s_temp2[0];
	T *s_Iv = &s_temp2[6*NUM_POS];
	T *s_Mxv = &s_temp2[12*NUM_POS];
	// T *s_MxTv = &s_temp2[18*NUM_POS]; // just storing directly into the dv/dq
	T *s_MxTa = &s_temp2[18*NUM_POS];
	T *s_TFxf = &s_temp2[24*NUM_POS];
	T *s_FxvI = &s_temp[0]; //Tv,Ta done when computing this so reuse
	// start computing all of the sub parts we can compute outside of the loops
	// Mx3(Tk*vlamk), Mx3(Tk*alamk), Mx3(vk)*qd[k], Ik*vk, FxMat(vk)*Ik, Tk^T*Fx3(fk) = -Tk^T*Mx3(fk)
	// start with the matMuls: Tk*vlamk, Tk*alamk, Ik*vk
	computeTempVals_GPU_part1(s_Iv, s_Ta, s_Tv, s_I, s_T, s_v, s_a);
	__syncthreads();
	// then sync and do all Mxs: Mx3(fk), Mx3(vk)*qd[k], Mx3(Tv), Mx3(Ta) --- note Mx3(Tv) stored in k col of dVk/dq 
	// also do FxMat(vk) -- this is probably slow
	computeTempVals_GPU_part2(s_Fxv, s_Mxv, s_Mxf, s_MxTa, s_vdq, s_Ta, s_Tv, s_v, s_f);
	__syncthreads();
	// then sync before finishing off with Fxvk*Ik, -Tk^T*Mxfk = Tk^T*Fxfk and preload some of vdu
	computeTempVals_GPU_part3(s_vdq, s_vdqd, s_TFxf, s_FxvI, s_Fxv, s_Mxf, s_T, s_I);
	__syncthreads();
	//----------------------------------------------------------------------------
	// Forward Pass for dc/u
	// note: dq and dqd are independent and lots of sparsity
	// note2: we have s_FxvI, s_Iv, s_Mxv, s_MxTv, s_MxTa, s_TMxf to use!
	//----------------------------------------------------------------------------
	// first compute the vdu recursion
	compute_vdu_GPU<T>(s_vdq, s_vdqd, s_T);
	__syncthreads();
	// then compute adu -- first the parallel part while saving Fxvdu for later
	compute_adu_GPU_part1<T>(s_adq, s_adqd, s_vdq, s_vdqd, s_MxTa, s_Mxv, s_qd);
	compute_Fxvdu<T>(s_Fxvdq, s_Fxvdqd, s_vdq, s_vdqd);
	__syncthreads();
	// then finish adu with the recursion
	compute_adu_GPU_part2<T>(s_adq, s_adqd, s_T);
	__syncthreads();
	// finally comptue all fdu in parallel
	compute_fdu_GPU<T>(s_fdq, s_fdqd, s_adq, s_adqd, s_vdq, s_vdqd, s_Fxvdq, s_Fxvdqd, s_FxvI, s_Iv, s_I);
	__syncthreads();
	//----------------------------------------------------------------------------
	// Backward Pass for dc/u
	//----------------------------------------------------------------------------
	compute_cdu_GPU_part1<T>(s_fdq, s_fdqd, s_TFxf, s_T);
	__syncthreads();
	compute_cdu_GPU_part2<T,VEL_DAMPING>(s_cdu, s_fdq, s_fdqd);
}

template <typename T>
__device__
void finish_dqdd_GPU(T *s_dqdd, T *s_cdu, T *s_Minv){
    // dqdd = -Minv * cdu
    // we can do all NUM_POS*2 columns of cdu by all NUM_POS rows of Minv in parallel
    int start, delta; singleLoopVals_GPU(&start,&delta);
    for (int rc = start; rc < 2*NUM_POS*NUM_POS; rc += delta){
        int r = rc % NUM_POS; int c = rc / NUM_POS;
        s_dqdd[NUM_POS*c + r] = static_cast<T>(-1)*dotProd_GPU<T,NUM_POS,NUM_POS,1>(&s_Minv[r],&s_cdu[NUM_POS*c]);
    }
}

// different kernel memory transfer sizes
#define SPLIT_VALS (4*NUM_POS + 18*NUM_POS)
#define FUSED_VALS (4*NUM_POS)
#define COMPLETELY_FUSED_VALS (4*NUM_POS + NUM_POS*NUM_POS)
#define CONST_VALS (72*NUM_POS)

template <typename T, int THREADS, int KNOT_POINTS, bool MPC_MODE = false, bool VEL_DAMPING = true>
void forwardDynamicsGradientSetup_forGPU(T *mem_vol_split, T *mem_const, int tid){
   int kStart = KNOT_POINTS/THREADS*tid; int kMax = KNOT_POINTS/THREADS*(tid+1); if(tid == THREADS-1){kMax = KNOT_POINTS;}
   T *mem_vol_splitk = &mem_vol_split[SPLIT_VALS*kStart]; T *mem_const_tid = &mem_const[CONST_VALS*tid];
   for (int k=kStart; k < kMax; k++){
      forwardDynamicsGradientSetup<T,true,MPC_MODE,VEL_DAMPING,true>(&mem_vol_splitk[4*NUM_POS],mem_const_tid,&mem_const_tid[36*NUM_POS],mem_vol_splitk,
                                                                     &mem_vol_splitk[NUM_POS],&mem_vol_splitk[3*NUM_POS],&mem_vol_splitk[2*NUM_POS],nullptr);
      mem_vol_splitk += SPLIT_VALS;
   }
}

template <typename T, int NUM_THREADS, int KNOT_POINTS, bool MPC_MODE = false, bool VEL_DAMPING = true>
__host__
void dynamicsGradientReusableThreaded_start(T *mem_vol_split, T *mem_const, ReusableThreads<NUM_THREADS> *threads){
   void (*f)(T*,T*,int) = &forwardDynamicsGradientSetup_forGPU<T,NUM_THREADS,KNOT_POINTS,MPC_MODE,VEL_DAMPING>;
   for (int tid = 0; tid < NUM_THREADS; tid++){
      threads->addTask(tid, f, std::ref(mem_vol_split), std::ref(mem_const), tid);
   }
   threads->sync();
}

template <typename T, int THREADS, int KNOT_POINTS, bool QDD_MINV_PASSED_IN = false, bool MPC_MODE = false, bool VEL_DAMPING = true, bool EXTERNAL_TI = false>
void forwardDynamicsGradientFinish_forGPU(T *dqdd, T *cdu, T *Minv, int tid){
   int kStart = KNOT_POINTS/THREADS*tid; int kMax = KNOT_POINTS/THREADS*(tid+1); if(tid == THREADS-1){kMax = KNOT_POINTS;}
   int strideMinv = NUM_POS*NUM_POS; int stride = 2*strideMinv;
   T *Minvk = &Minv[strideMinv*kStart]; T *dqddk = &dqdd[stride*kStart]; T *cduk = &cdu[stride*kStart];
   for (int k=kStart; k < kMax; k++){
      // Then finish it off: dqdd/dtau = Minv, dqdd/dqd = -Minv*dc/dqd, dqdd/dq = -Minv*dc/dq --- remember Minv is a symmetric sym UPPER matrix
      matMultSym<T,NUM_POS,2*NUM_POS,NUM_POS>(dqddk,NUM_POS,Minvk,NUM_POS,cduk,NUM_POS,static_cast<T>(-1));
      dqddk += stride; cduk += stride; Minvk += strideMinv;
   }
}

template <typename T, int NUM_THREADS, int KNOT_POINTS, bool MPC_MODE = false, bool VEL_DAMPING = true>
__host__
void dynamicsGradientReusableThreaded_finish(T *dqdd, T *cdu, T *Minv, ReusableThreads<NUM_THREADS> *threads){
   void (*f)(T*,T*,T*,int) = &forwardDynamicsGradientFinish_forGPU<T,NUM_THREADS,KNOT_POINTS,MPC_MODE,VEL_DAMPING>;
   for (int tid = 0; tid < NUM_THREADS; tid++){
      threads->addTask(tid, f, std::ref(dqdd), std::ref(cdu), std::ref(Minv), tid);
   }
   threads->sync();
}

template <typename T, int KNOT_POINTS, bool VEL_DAMPING = 1>
__global__
void dynamicsGradientKernel_split(T *d_cdu, T *d_mem_vol_split, T *d_mem_const){
   int start, delta; singleLoopVals_GPU(&start,&delta);
   __shared__ T s_mem_const[CONST_VALS];  T *s_T = &s_mem_const[0]; T *s_I = &s_mem_const[36*NUM_POS];
   __shared__ T s_mem_vol[SPLIT_VALS]; T *s_q = &s_mem_vol[0]; T *s_qd = &s_mem_vol[NUM_POS];
                                       // T *s_qdd = &s_mem_vol[3*NUM_POS]; T *s_u = &s_mem_vol[2*NUM_POS]; 
                                       T *s_v = &s_mem_vol[4*NUM_POS]; T *s_a = &s_mem_vol[10*NUM_POS]; T *s_f = &s_mem_vol[16*NUM_POS];
   __shared__ T s_sinq[NUM_POS]; __shared__ T s_cosq[NUM_POS]; __shared__ T s_cdu[2*NUM_POS*NUM_POS];
   // Load in const vars to shared mem
   #pragma unroll
   for (int ind = start; ind < 72*NUM_POS; ind += delta){s_mem_const[ind] = d_mem_const[ind];} __syncthreads();
   // then grided thread the knot points
   for (int k = blockIdx.x; k < KNOT_POINTS; k += gridDim.x){
      T *d_mem_volk = &d_mem_vol_split[SPLIT_VALS*k];
      // load in q,qd,qdd,Minv and compute sinq,cosq
      #pragma unroll
      for (int ind = start; ind < SPLIT_VALS; ind += delta){s_mem_vol[ind] = d_mem_volk[ind];}
      #pragma unroll
      for (int ind = start; ind < NUM_POS; ind += delta){s_sinq[ind] = sin(s_q[ind]); s_cosq[ind] = cos(s_q[ind]);} __syncthreads();
      // then update transforms
      updateTransforms_GPU<T>(s_T,s_sinq,s_cosq); __syncthreads();
      // then dc/du
      inverseDynamicsGradient_GPU<T,VEL_DAMPING>(s_cdu,s_qd,s_v,s_a,s_f,s_I,s_T); __syncthreads();
      // Then copy to global
      T *cduk = &d_cdu[k*2*NUM_POS*NUM_POS];
      #pragma unroll
      for (int ind = start; ind < 2*NUM_POS*NUM_POS; ind += delta){cduk[ind] = s_cdu[ind];} __syncthreads();
        
   }
}

template <typename T, int KNOT_POINTS, bool MPC_MODE = false, bool VEL_DAMPING = true>
__global__
void dynamicsGradientKernel_fused(T *d_cdu, T *d_mem_vol_fused, T *d_mem_const){
   int start, delta; singleLoopVals_GPU(&start,&delta);
   __shared__ T s_mem_const[CONST_VALS];  T *s_T = &s_mem_const[0]; T *s_I = &s_mem_const[36*NUM_POS];
   __shared__ T s_mem_vol[FUSED_VALS]; T *s_q = &s_mem_vol[0]; T *s_qd = &s_mem_vol[NUM_POS];
                                       T *s_qdd = &s_mem_vol[3*NUM_POS]; //T *s_u = &s_mem_vol[2*NUM_POS]; 
   __shared__ T s_sinq[NUM_POS];          __shared__ T s_cosq[NUM_POS];          __shared__ T s_temp[6*NUM_POS];
   __shared__ T s_v[6*NUM_POS];           __shared__ T s_a[6*NUM_POS];           __shared__ T s_f[6*NUM_POS];
   __shared__ T s_cdu[2*NUM_POS*NUM_POS];
   // Load in const vars to shared mem
   #pragma unroll
   for (int ind = start; ind < CONST_VALS; ind += delta){s_mem_const[ind] = d_mem_const[ind];} __syncthreads();
   // loop for all knots needed per SM
   for (int k = blockIdx.x; k < KNOT_POINTS; k += gridDim.x){
      T *d_mem_volk = &d_mem_vol_fused[FUSED_VALS*k];
      // load in q,qd,qdd,Minv and compute sinq,cosq
      #pragma unroll
      for (int ind = start; ind < FUSED_VALS; ind += delta){s_mem_vol[ind] = d_mem_volk[ind];}
      #pragma unroll
      for (int ind = start; ind < NUM_POS; ind += delta){s_sinq[ind] = sin(s_q[ind]); s_cosq[ind] = cos(s_q[ind]);} __syncthreads();
      // then update transforms
      updateTransforms_GPU<T>(s_T,s_sinq,s_cosq); __syncthreads();
      // first compute vaf
      FD_helpers_GPU_vaf<T,MPC_MODE>(s_v,s_a,s_f,s_qd,s_qdd,s_I,s_T,s_temp); __syncthreads();
      // then dc/du
      inverseDynamicsGradient_GPU<T,VEL_DAMPING>(s_cdu,s_qd,s_v,s_a,s_f,s_I,s_T); __syncthreads();
      // copy out to RAM
      T *cduk = &d_cdu[k*2*NUM_POS*NUM_POS];
      #pragma unroll
      for (int ind = start; ind < 2*NUM_POS*NUM_POS; ind += delta){cduk[ind] = s_cdu[ind];} __syncthreads();
   }
}

template <typename T, int KNOT_POINTS, bool MPC_MODE = false, bool VEL_DAMPING = true>
__global__
void dynamicsGradientKernel(T *d_dqdd, T *d_mem_vol_completely_fused, T *d_mem_const){
   int start, delta; singleLoopVals_GPU(&start,&delta);
   __shared__ T s_mem_const[CONST_VALS];  T *s_T = &s_mem_const[0]; T *s_I = &s_mem_const[36*NUM_POS];
   __shared__ T s_mem_vol[COMPLETELY_FUSED_VALS]; T *s_q = &s_mem_vol[0]; T *s_qd = &s_mem_vol[NUM_POS];
                                                  T *s_qdd = &s_mem_vol[3*NUM_POS]; T *s_Minv = &s_mem_vol[4*NUM_POS]; //T *s_u = &s_mem_vol[2*NUM_POS]; 
   __shared__ T s_sinq[NUM_POS];          __shared__ T s_cosq[NUM_POS];          __shared__ T s_temp[6*NUM_POS];
   __shared__ T s_v[6*NUM_POS];           __shared__ T s_a[6*NUM_POS];           __shared__ T s_f[6*NUM_POS];
   __shared__ T s_cdu[2*NUM_POS*NUM_POS]; __shared__ T s_dqdd[2*NUM_POS*NUM_POS];
   // Load in const vars to shared mem
   #pragma unroll
   for (int ind = start; ind < CONST_VALS; ind += delta){s_mem_const[ind] = d_mem_const[ind];} __syncthreads();
   // loop for all knots needed per SM
   for (int k = blockIdx.x; k < KNOT_POINTS; k += gridDim.x){
      T *d_mem_volk = &d_mem_vol_completely_fused[COMPLETELY_FUSED_VALS*k];
      // load in q,qd,qdd,Minv and compute sinq,cosq
      #pragma unroll
      for (int ind = start; ind < COMPLETELY_FUSED_VALS; ind += delta){s_mem_vol[ind] = d_mem_volk[ind];}
      #pragma unroll
      for (int ind = start; ind < NUM_POS; ind += delta){s_sinq[ind] = sin(s_q[ind]); s_cosq[ind] = cos(s_q[ind]);} __syncthreads();
      // then update transforms
      updateTransforms_GPU<T>(s_T,s_sinq,s_cosq); __syncthreads();
      // first compute vaf
      FD_helpers_GPU_vaf<T,MPC_MODE>(s_v,s_a,s_f,s_qd,s_qdd,s_I,s_T,s_temp); __syncthreads();
      // then dc/du
      inverseDynamicsGradient_GPU<T,VEL_DAMPING>(s_cdu,s_qd,s_v,s_a,s_f,s_I,s_T); __syncthreads();
      // finally compute the final dqdd
      finish_dqdd_GPU<T>(s_dqdd,s_cdu,s_Minv); __syncthreads();
      // copy out to RAM
      T *dqddk = &d_dqdd[k*2*NUM_POS*NUM_POS];
      #pragma unroll
      for (int ind = start; ind < 2*NUM_POS*NUM_POS; ind += delta){dqddk[ind] = s_dqdd[ind];} __syncthreads();
   }
}

// __shared__ T s_vaf[18*NUM_POS]; __shared__ s_temp[(50 + 6*NUM_POS)*NUM_POS];
// assumes q, qd, u, I, Tbase are all loaded into mem already
template <typename T, bool MPC_MODE = false, bool VEL_DAMPING = true>
__device__
void dynamicsAndGradient_scratch(T *s_dqdd, T *s_cdu, T *s_vaf, T *s_Minv, T *s_q, T *s_qd, T *s_qdd, T *s_u, T *s_I, T *s_T, T *s_scratchMem, T *s_fext = nullptr){
   // compute vaf, Minv, (and qdd) from scratch
   FD_helpers_GPU_scratch<T,MPC_MODE,VEL_DAMPING>(s_vaf,s_Minv,s_q,s_qd,s_qdd,s_u,s_I,s_T,s_scratchMem); __syncthreads();
   // then dc/du
   T *s_v = &s_vaf[0]; T *s_a = &s_v[6*NUM_POS]; T *s_f = &s_a[6*NUM_POS];
   inverseDynamicsGradient_GPU<T,VEL_DAMPING>(s_cdu,s_qd,s_v,s_a,s_f,s_I,s_T); __syncthreads();
   // finally compute the final dqdd
   finish_dqdd_GPU<T>(s_dqdd,s_cdu,s_Minv); __syncthreads();
}
// constMem = I then T
template <typename T, bool MPC_MODE = false, bool VEL_DAMPING = true>
__device__
void dynamicsAndGradient_scratch(T *s_dqdd, T *s_qdd, T *s_q, T *s_qd, T *s_u, T *s_constMem, T *s_fext = nullptr){
   __shared__ T s_cdu[2*NUM_POS*NUM_POS]; __shared__ T s_vaf[18*NUM_POS]; __shared__ T s_Minv[NUM_POS*NUM_POS]; 
   __shared__ T s_scratchMem[(50 + 6*NUM_POS)*NUM_POS]; T *s_I = &s_constMem[0]; T *s_T = &s_I[36*NUM_POS]; 
   dynamicsAndGradient_scratch<T,MPC_MODE,VEL_DAMPING>(s_dqdd,s_cdu,s_vaf,s_Minv,s_q,s_qd,s_qdd,s_u,s_I,s_T,s_scratchMem,s_fext);
}

// __shared__ T s_mem_const[CONST_VALS];  T *s_T = &s_mem_const[0]; T *s_I = &s_mem_const[36*NUM_POS];
template <typename T>
__device__
void dynamicsAndGradient_init(T *d_mem_const, T *s_mem_const){
   // load in constant I and any bit of T we compute as const vs. updated (if applicable)
   int start, delta; singleLoopVals_GPU(&start,&delta);
   for (int ind = start; ind < CONST_VALS; ind += delta){s_mem_const[ind] = d_mem_const[ind];}
}