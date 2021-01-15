/***
nvcc -std=c++11 -o testMinv.exe testMinv.cu -gencode arch=compute_75,code=sm_75 -O3
***/
#define ERROR_TOL 1e-4
#define TEST_FOR_EQUIVALENCE 1
#include "../utils/experiment_helpers.h"
#include "../helpers_CPU/dynamicsGradient.h" // GPU requires CPU for partial kernels
#include "../helpers_GPU/dynamicsGradient.cuh"

template <typename T, bool MPC_MODE = false>
__global__
void helpersGPUKern(T *d_mem_vol_fused, T *d_mem_const){
	int start, delta; singleLoopVals_GPU(&start,&delta);
	__shared__ T s_mem_const[CONST_VALS];  T *s_T = &s_mem_const[0]; T *s_I = &s_mem_const[36*NUM_POS];
	__shared__ T s_mem_vol[COMPLETELY_FUSED_VALS]; T *s_q = &s_mem_vol[0]; T *s_qd = &s_mem_vol[NUM_POS];
                                                  T *s_qdd = &s_mem_vol[3*NUM_POS]; T *s_Minv0 = &s_mem_vol[4*NUM_POS]; //T *s_u = &s_mem_vol[2*NUM_POS]; 
	__shared__ T s_sinq[NUM_POS]; __shared__ T s_cosq[NUM_POS]; __shared__ T s_temp[6*NUM_POS];
	__shared__ T s_v0[6*NUM_POS]; __shared__ T s_a0[6*NUM_POS]; __shared__ T s_f0[6*NUM_POS];
	__shared__ T s_v[6*NUM_POS]; __shared__ T s_a[6*NUM_POS]; __shared__ T s_f[6*NUM_POS];
	__shared__ T s_Minv[NUM_POS*NUM_POS]; __shared__ T s_IA[36*NUM_POS]; T s_U[6*NUM_POS]; T s_D[2*NUM_POS]; T s_F[6*NUM_POS*NUM_POS];
	// Load in const vars to shared mem
	#pragma unroll
	for (int ind = start; ind < CONST_VALS; ind += delta){s_mem_const[ind] = d_mem_const[ind];} __syncthreads();
	// load in q,qd,qdd,Minv and compute sinq,cosq
	#pragma unroll
	for (int ind = start; ind < COMPLETELY_FUSED_VALS; ind += delta){s_mem_vol[ind] = d_mem_vol_fused[ind];}
	#pragma unroll
	for (int ind = start; ind < NUM_POS; ind += delta){s_sinq[ind] = sin(s_q[ind]); s_cosq[ind] = cos(s_q[ind]);} __syncthreads();
	// then update transforms
	updateTransforms_GPU<T>(s_T,s_sinq,s_cosq); __syncthreads();
	// known good vaf
    FD_helpers_GPU_vaf<T,MPC_MODE>(s_v0,s_a0,s_f0,s_qd,s_qdd,s_I,s_T,s_temp); __syncthreads();
	// then call full helpers
	// but first need to copy I into IA
	#pragma unroll
	for (int ind = start; ind < 36*NUM_POS; ind += delta){s_IA[ind] = s_I[ind];} __syncthreads();
	FD_helpers_GPU<T,MPC_MODE,true>(s_v,s_a,s_f,s_Minv,s_qd,s_qdd,s_I,s_IA,s_T,s_U,s_D,s_F,s_temp); __syncthreads();
	// compare vaf and Minv
	if(threadIdx.x == 0 && threadIdx.y == 0){
		bool hasErr = false;
		for(int i = 0; i < 6*NUM_POS; i++){
			T diff = s_v0[i] - s_v[i];
			if (abs(diff) > ERROR_TOL){printf("v diff at i[%d] of [%f] with old[%f] and new[%f]\n",i,diff,s_v0[i],s_v[i]); hasErr = true;}
		}
		for(int i = 0; i < 6*NUM_POS; i++){
			T diff = s_a0[i] - s_a[i];
			if (abs(diff) > ERROR_TOL){printf("a diff at i[%d] of [%f] with old[%f] and new[%f]\n",i,diff,s_a0[i],s_a[i]); hasErr = true;}
		}
		for(int i = 0; i < 6*NUM_POS; i++){
			T diff = s_f0[i] - s_f[i];
			if (abs(diff) > ERROR_TOL){printf("f diff at i[%d] of [%f] with old[%f] and new[%f]\n",i,diff,s_f0[i],s_f[i]); hasErr = true;}
		}
		for(int i = 0; i < NUM_POS*NUM_POS; i++){
			int c = i / NUM_POS; int r = i % NUM_POS;
			T diff; if (r > c){diff = s_Minv0[i] - s_Minv[r*NUM_POS+c];} else{diff = s_Minv0[i] - s_Minv[i];} // its UPPER
			if (abs(diff) > ERROR_TOL){printf("Minv diff at i[%d] of [%f]\n",i,diff); hasErr = true;}
		}
		// printf("Minv old\n");
		// printMat_GPU<T,NUM_POS,NUM_POS>(s_Minv0,NUM_POS);
		// printf("Minv new\n");
		// printMat_GPU<T,NUM_POS,NUM_POS>(s_Minv,NUM_POS);
		if(!hasErr){
			printf("Accurate to TOL of [%f]\n", ERROR_TOL);
		}
	}

}

template <typename T, bool MPC_MODE = false, bool VEL_DAMPING = false>
__global__
void helpersGPUKern2(T *d_mem_vol_fused, T *d_mem_const){
	int start, delta; singleLoopVals_GPU(&start,&delta);
	__shared__ T s_mem_const[CONST_VALS];  T *s_T = &s_mem_const[0]; T *s_I = &s_mem_const[36*NUM_POS];
	__shared__ T s_mem_vol[COMPLETELY_FUSED_VALS]; T *s_q = &s_mem_vol[0]; T *s_qd = &s_mem_vol[NUM_POS];
               T *s_qdd = &s_mem_vol[3*NUM_POS]; T *s_Minv0 = &s_mem_vol[4*NUM_POS]; T *s_u = &s_mem_vol[2*NUM_POS];
    __shared__ T s_v0[6*NUM_POS]; __shared__ T s_a0[6*NUM_POS]; __shared__ T s_f0[6*NUM_POS]; __shared__ T s_qdd0[NUM_POS];
	__shared__ T s_Minv[NUM_POS*NUM_POS]; __shared__ T s_vaf[18*NUM_POS]; __shared__ T s_temp[(50 + 6*NUM_POS)*NUM_POS];
	// Load in const vars to shared mem
	for (int ind = start; ind < CONST_VALS; ind += delta){s_mem_const[ind] = d_mem_const[ind];} __syncthreads();
	// load in q,qd,qdd,Minv,u
	for (int ind = start; ind < COMPLETELY_FUSED_VALS; ind += delta){s_mem_vol[ind] = d_mem_vol_fused[ind];}
	// known good vaf and save copy of qdd (needs T comp)
	T *s_sinq = &s_temp[0]; T *s_cosq = &s_sinq[NUM_POS];
	for (int ind = start; ind < NUM_POS; ind += delta){s_qdd0[ind] = s_qdd[ind];} 
	for (int ind = start; ind < NUM_POS; ind += delta){s_sinq[ind] = sin(s_q[ind]); s_cosq[ind] = cos(s_q[ind]);} __syncthreads();
	updateTransforms_GPU<T>(s_T,s_sinq,s_cosq); __syncthreads();
    FD_helpers_GPU_vaf<T,MPC_MODE>(s_v0,s_a0,s_f0,s_qd,s_qdd,s_I,s_T,s_temp); __syncthreads();
	// then call full helpers form scratch
	FD_helpers_GPU_scratch<T,MPC_MODE,VEL_DAMPING>(s_vaf,s_Minv,s_q,s_qd,s_qdd,s_u,s_I,s_T,s_temp); __syncthreads();
	// compare vaf and Minv and qdd
	if(threadIdx.x == 0 && threadIdx.y == 0){
		bool hasErr = false; T *s_v = &s_vaf[0]; T *s_a = &s_v[6*NUM_POS]; T *s_f = &s_a[6*NUM_POS];
		for(int i = 0; i < NUM_POS; i++){
			T diff = s_qdd0[i] - s_qdd[i];
			if (abs(diff) > ERROR_TOL){printf("qdd diff at i[%d] of [%f] with old[%f] and new[%f]\n",i,diff,s_qdd0[i],s_qdd[i]); hasErr = true;}
		}
		for(int i = 0; i < 6*NUM_POS; i++){
			T diff = s_v0[i] - s_v[i];
			if (abs(diff) > ERROR_TOL){printf("v diff at i[%d] of [%f] with old[%f] and new[%f]\n",i,diff,s_v0[i],s_v[i]); hasErr = true;}
		}
		for(int i = 0; i < 6*NUM_POS; i++){
			T diff = s_a0[i] - s_a[i];
			if (abs(diff) > ERROR_TOL){printf("a diff at i[%d] of [%f] with old[%f] and new[%f]\n",i,diff,s_a0[i],s_a[i]); hasErr = true;}
		}
		for(int i = 0; i < 6*NUM_POS; i++){
			T diff = s_f0[i] - s_f[i];
			if (abs(diff) > ERROR_TOL){printf("f diff at i[%d] of [%f] with old[%f] and new[%f]\n",i,diff,s_f0[i],s_f[i]); hasErr = true;}
		}
		for(int i = 0; i < NUM_POS*NUM_POS; i++){
			int c = i / NUM_POS; int r = i % NUM_POS;
			T diff; if (r > c){diff = s_Minv0[i] - s_Minv[r*NUM_POS+c];} else{diff = s_Minv0[i] - s_Minv[i];} // its UPPER
			if (abs(diff) > ERROR_TOL){printf("Minv diff at i[%d] of [%f]\n",i,diff); hasErr = true;}
		}
		// printf("Minv old\n");
		// printMat_GPU<T,NUM_POS,NUM_POS>(s_Minv0,NUM_POS);
		// printf("Minv new (UPPER)\n");
		// printMat_GPU<T,NUM_POS,NUM_POS>(s_Minv,NUM_POS);
		if(!hasErr){
			printf("Accurate to TOL of [%f]\n", ERROR_TOL);
		}
	}

}

template<typename T, bool MPC_MODE = false, bool VEL_DAMPING = false>
void test(){
	// allocate and load on CPU
	T *h_mem_const = (T *)malloc(CONST_VALS*sizeof(T)); // T,I
	initTransforms(h_mem_const); initInertiaTensors(&h_mem_const[36*NUM_POS]);
	
	traj<T,1> *testTraj = new traj<T,1>;
	for (int k = 0; k < 1; k++){
		for(int i = 0; i < NUM_POS; i++){
			testTraj->knots[k].q[i] = getRand<T>();
			testTraj->knots[k].qd[i] = getRand<T>(); 
			testTraj->knots[k].u[i] = getRand<T>(); 
		}
		forwardDynamics<T,MPC_MODE,VEL_DAMPING,1>(&(testTraj->knots[k]));
	}
	// and load into gpu structured memory (on the CPU)
	T *h_mem_vol_completely_fused = (T *)malloc(COMPLETELY_FUSED_VALS*sizeof(T)); // q,qd,u,qdd,Minv
	for (int k = 0; k < 1; k++){
		T *h_mem_vol_completely_fusedk = &h_mem_vol_completely_fused[COMPLETELY_FUSED_VALS*k];
		for(int i = 0; i < NUM_POS; i++){
			h_mem_vol_completely_fusedk[i] = testTraj->knots[k].q[i];
			h_mem_vol_completely_fusedk[i+NUM_POS] = testTraj->knots[k].qd[i];
			h_mem_vol_completely_fusedk[i+2*NUM_POS] = testTraj->knots[k].u[i];
			h_mem_vol_completely_fusedk[i+3*NUM_POS] = testTraj->knots[k].qdd[i];
		}
		for(int i = 0; i < NUM_POS*NUM_POS; i++){
			h_mem_vol_completely_fusedk[4*NUM_POS + i] = testTraj->knots[k].Minv[i];
		}
	}

	// allocate and copy to GPU
	T *d_mem_const; gpuErrchk(cudaMalloc((void**)&d_mem_const, CONST_VALS*sizeof(T))); // T,I
	T *d_mem_vol_completely_fused; gpuErrchk(cudaMalloc((void**)&d_mem_vol_completely_fused, COMPLETELY_FUSED_VALS*sizeof(T))); // q,qd,u,qdd,Minv
	gpuErrchk(cudaMemcpy(d_mem_const,h_mem_const,CONST_VALS*sizeof(T),cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_mem_vol_completely_fused,h_mem_vol_completely_fused,COMPLETELY_FUSED_VALS*sizeof(T),cudaMemcpyHostToDevice));
	gpuErrchk(cudaDeviceSynchronize());

	// for(int r = 0; r < NUM_POS; r++){
	// 	for(int c = 0; c < NUM_POS; c++){
	// 		printf("%f ",testTraj->knots[0].Minv[c*NUM_POS+r]);
	// 	}
	// 	printf("\n");
	// }
	
	helpersGPUKern<T,MPC_MODE><<<1,BlockDimms>>>(d_mem_vol_completely_fused,d_mem_const);
	gpuErrchk(cudaDeviceSynchronize());

	helpersGPUKern2<T,MPC_MODE><<<1,BlockDimms>>>(d_mem_vol_completely_fused,d_mem_const);
	gpuErrchk(cudaDeviceSynchronize());
	
    // free all
	free(h_mem_const); delete testTraj;
	free(h_mem_vol_completely_fused);
	gpuErrchk(cudaFree(d_mem_const));
	gpuErrchk(cudaFree(d_mem_vol_completely_fused));
}

int main(void){
	test<float,MPC_MODE_GLOBAL>();
}