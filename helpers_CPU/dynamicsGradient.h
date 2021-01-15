/****************************************************************
 * CUDA Rigid Body DYNamics
 *
 * Based on the Joint Space Inversion Algorithm
 * currently special cased for 7dof Kuka Arm
 ****************************************************************/
#ifndef GRAVITY
   #define GRAVITY 9.81
#endif
#ifndef NUM_POS
   #define NUM_POS 7
#endif
#include <thread>
#include <cmath>
#include <cstring>
#include <immintrin.h>

template <typename T>
struct knot {
    T q[NUM_POS];
    T qd[NUM_POS];
    T u[NUM_POS];
    T qdd[NUM_POS];
    T Minv[NUM_POS*NUM_POS];
};

template <typename T, int KNOT_POINTS>
struct traj {
    knot<T> knots[KNOT_POINTS];
};

// debug print
template <typename T, int M, int N>
void printMat(T *A, int lda){
    for(int i=0; i<M; i++){
        for(int j=0; j<N; j++){
            printf("%.6f ",static_cast<double>(A[i + lda*j]));
        }
        printf("\n");
    }
}

template <typename T, int N, bool UPPER = true>
void triangularToDense(T *Mat){
    if(UPPER){
        for(int c = 0; c < N; c++){
            for (int r = c; r < N; r++){
                Mat[c*N+r] = Mat[r*N+c];
            }
        }
    }
    else{
        for(int r = 0; r < N; r++){
            for (int c = r; c < N; c++){
                Mat[c*N+r] = Mat[r*N+c];
            }
        }
    }
}

#include "math.h"
#include "minv.h"

//-------------------------------------------------------------------------------
// INERTIAS AND TRANSFORMS
//-------------------------------------------------------------------------------

//-------------------------------------------------------------------------------
// Build 6x6 Spatial Inertia Tensor
//-------------------------------------------------------------------------------
// I6x6 = I3x3 - MCadj  mc    I3x3 = Ixx   Ixy   Ixz    mc =   0  -m*cz  m*cy
//         mcT          mI           Ixy   Iyy   Iyz         m*cz     0 -m*cx
//                                   Ixz   Iyz   Izz        -m*cy  m*cx  0
// MCadj = -m*(cy*cy+cz*cz) m*cx*cy         m*cx*cz
//          m*cx*cy        -m*(cx*cx+cz*cz) m*cy*cz 
//          m*cx*cz         m*cy*cz        -m*(cx*cx+cy*cy)
template <typename T>
void buildI6x6(T *currI, T Ixx, T Ixy, T Ixz, T Iyy, T Iyz, T Izz, T cx, T cy, T cz, T m){
    // Copy elements from the 3x3 Inertia Tensor and adjust for non-centered COM
    currI[6*0 + 0] =  Ixx + m*(cy*cy+cz*cz);
    currI[6*0 + 1] =  Ixy - m*cx*cy;
    currI[6*0 + 2] =  Ixz - m*cx*cz;
    currI[6*1 + 0] =  currI[6*0 + 1];
    currI[6*1 + 1] =  Iyy + m*(cx*cx+cz*cz);
    currI[6*1 + 2] =  Iyz - m*cy*cz;
    currI[6*2 + 0] =  currI[6*0 + 2];
    currI[6*2 + 1] =  currI[6*1 + 2];
    currI[6*2 + 2] =  Izz + m*(cx*cx+cy*cy);
    // define the mx sections on the BL and RT
    currI[6*0 + 3] = 0;
    currI[6*0 + 4] = -m*cz;
    currI[6*0 + 5] =  m*cy;
    currI[6*1 + 3] =  m*cz;
    currI[6*1 + 4] = 0;
    currI[6*1 + 5] = -m*cx;
    currI[6*2 + 3] = -m*cy;
    currI[6*2 + 4] =  m*cx;
    currI[6*2 + 5] = 0;
    currI[6*3 + 0] = 0;
    currI[6*3 + 1] =  m*cz;
    currI[6*3 + 2] = -m*cy;
    currI[6*4 + 0] = -m*cz;
    currI[6*4 + 1] = 0;
    currI[6*4 + 2] =  m*cx;
    currI[6*5 + 0] =  m*cy;
    currI[6*5 + 1] = -m*cx;
    currI[6*5 + 2] = 0;
    // finally the mass on the BR
    currI[6*3 + 3] = m;
    currI[6*3 + 4] = 0;
    currI[6*3 + 5] = 0;
    currI[6*4 + 3] = 0;
    currI[6*4 + 4] = m;
    currI[6*4 + 5] = 0;
    currI[6*5 + 3] = 0;
    currI[6*5 + 4] = 0;
    currI[6*5 + 5] = m;
}

//-------------------------------------------------------------------------------
// Initialize Spatial Inertia Tensors from Constants
//-------------------------------------------------------------------------------
template <typename T>
void initInertiaTensors(T *I){
    // Per-Robot Inertia Constants (parsed from urdf)
    // -- First Joint/Link Mass, Position, Inertia
    #define l0_Ixx   static_cast<T>(0.121)
    #define l0_Ixy   static_cast<T>(0.0)
    #define l0_Ixz   static_cast<T>(0.0)
    #define l0_Iyy   static_cast<T>(0.116)
    #define l0_Iyz   static_cast<T>(-0.021)
    #define l0_Izz   static_cast<T>(0.0175)
    #define l0_mass  static_cast<T>(5.76)
    #define l0_comx  static_cast<T>(0.0)
    #define l0_comy  static_cast<T>(-0.03)
    #define l0_comz  static_cast<T>(0.12)
    // -- Second Joint/Link Mass, Position, Inertia
    #define l1_Ixx   static_cast<T>(0.0638)
    #define l1_Ixy   static_cast<T>(0.0001)
    #define l1_Ixz   static_cast<T>(0.00008)
    #define l1_Iyy   static_cast<T>(0.0416)
    #define l1_Iyz   static_cast<T>(0.0157)
    #define l1_Izz   static_cast<T>(0.0331)
    #define l1_mass  static_cast<T>(6.35)
    #define l1_comx  static_cast<T>(0.0003)
    #define l1_comy  static_cast<T>(0.059)
    #define l1_comz  static_cast<T>(0.042)
    // -- Third Joint/Link Mass, Position, Inertia
    #define l2_Ixx   static_cast<T>(0.0873)
    #define l2_Ixy   static_cast<T>(0.0)
    #define l2_Ixz   static_cast<T>(0.0)
    #define l2_Iyy   static_cast<T>(0.083)
    #define l2_Iyz   static_cast<T>(0.014)
    #define l2_Izz   static_cast<T>(0.0108)
    #define l2_mass  static_cast<T>(3.5)
    #define l2_comx  static_cast<T>(0.0)
    #define l2_comy  static_cast<T>(0.03)
    #define l2_comz  static_cast<T>(0.13)
    // -- Fourth Joint/Link Mass, Position, Inertia
    #define l3_Ixx   static_cast<T>(0.0368)
    #define l3_Ixy   static_cast<T>(0.0)
    #define l3_Ixz   static_cast<T>(0.0)
    #define l3_Iyy   static_cast<T>(0.02045)
    #define l3_Iyz   static_cast<T>(0.008)
    #define l3_Izz   static_cast<T>(0.02171)
    #define l3_mass  static_cast<T>(3.5)
    #define l3_comx  static_cast<T>(0.0)
    #define l3_comy  static_cast<T>(0.067)
    #define l3_comz  static_cast<T>(0.034)
    // -- Fifth Joint/Link Mass, Position, Inertia
    #define l4_Ixx   static_cast<T>(0.0318)
    #define l4_Ixy   static_cast<T>(0.000007)
    #define l4_Ixz   static_cast<T>(0.000027)
    #define l4_Iyy   static_cast<T>(0.028916)
    #define l4_Iyz   static_cast<T>(0.005586)
    #define l4_Izz   static_cast<T>(0.006)
    #define l4_mass  static_cast<T>(3.5)
    #define l4_comx  static_cast<T>(0.0001)
    #define l4_comy  static_cast<T>(0.021)
    #define l4_comz  static_cast<T>(0.076)
    // -- Sixth Joint/Link Mass, Position, Inertia
    #define l5_Ixx   static_cast<T>(0.0049)
    #define l5_Ixy   static_cast<T>(0.0)
    #define l5_Ixz   static_cast<T>(0.0)
    #define l5_Iyy   static_cast<T>(0.0047)
    #define l5_Iyz   static_cast<T>(0.0)
    #define l5_Izz   static_cast<T>(0.0036)
    #define l5_mass  static_cast<T>(1.8)
    #define l5_comx  static_cast<T>(0.0)
    #define l5_comy  static_cast<T>(0.0006)
    #define l5_comz  static_cast<T>(0.0004)
    // -- Seventh Joint/Link Mass, Position, Inertia
    #define l6_Ixx   static_cast<T>(0.0055)
    #define l6_Ixy   static_cast<T>(0.0)
    #define l6_Ixz   static_cast<T>(0.0)
    #define l6_Iyy   static_cast<T>(0.0055)
    #define l6_Iyz   static_cast<T>(0.0)
    #define l6_Izz   static_cast<T>(0.005)
    #define l6_mass  static_cast<T>(1.2)
    #define l6_comx  static_cast<T>(0.0)
    #define l6_comy  static_cast<T>(0.0)
    #define l6_comz  static_cast<T>(0.02)

    // build intertia matricies with these constants
    buildI6x6<T>(I,l0_Ixx,l0_Ixy,l0_Ixz,l0_Iyy,l0_Iyz,l0_Izz,l0_comx,l0_comy,l0_comz,l0_mass);
    buildI6x6<T>(&I[36],l1_Ixx,l1_Ixy,l1_Ixz,l1_Iyy,l1_Iyz,l1_Izz,l1_comx,l1_comy,l1_comz,l1_mass);
    buildI6x6<T>(&I[36*2],l2_Ixx,l2_Ixy,l2_Ixz,l2_Iyy,l2_Iyz,l2_Izz,l2_comx,l2_comy,l2_comz,l2_mass);
    buildI6x6<T>(&I[36*3],l3_Ixx,l3_Ixy,l3_Ixz,l3_Iyy,l3_Iyz,l3_Izz,l3_comx,l3_comy,l3_comz,l3_mass);
    buildI6x6<T>(&I[36*4],l4_Ixx,l4_Ixy,l4_Ixz,l4_Iyy,l4_Iyz,l4_Izz,l4_comx,l4_comy,l4_comz,l4_mass);
    buildI6x6<T>(&I[36*5],l5_Ixx,l5_Ixy,l5_Ixz,l5_Iyy,l5_Iyz,l5_Izz,l5_comx,l5_comy,l5_comz,l5_mass);
    buildI6x6<T>(&I[36*6],l6_Ixx,l6_Ixy,l6_Ixz,l6_Iyy,l6_Iyz,l6_Izz,l6_comx,l6_comy,l6_comz,l6_mass);
}

//-------------------------------------------------------------------------------
// Build Transforms
//-------------------------------------------------------------------------------
//  X =    R3x3     0
//     -rx*R3x3  R3x3
//
//  Note: due to the fact that there are a lot of fixed joints and link relationships
//        we do not use a formula here but an explicit code generated result
//  Per-Robot Translation Constants
#define tz_j1 static_cast<T>(0.1575)
#define tz_j2 static_cast<T>(0.2025)
#define ty_j3 static_cast<T>(0.2045)
#define tz_j4 static_cast<T>(0.2155)
#define ty_j5 static_cast<T>(0.1845)
#define tz_j6 static_cast<T>(0.2155)
#define ty_j7 static_cast<T>(0.081)
template <typename T>
void buildTransforms(T *s_T, T *s_sinq, T *s_cosq){
    s_T[36*0 + 6*0 + 0] = s_cosq[0];
    s_T[36*0 + 6*0 + 1] = -s_sinq[0];
    s_T[36*0 + 6*0 + 2] = static_cast<T>(0);
    s_T[36*0 + 6*0 + 3] = -tz_j1 * s_sinq[0];
    s_T[36*0 + 6*0 + 4] = -tz_j1 * s_cosq[0];
    s_T[36*0 + 6*0 + 5] = static_cast<T>(0);
    s_T[36*0 + 6*1 + 0] = s_sinq[0];
    s_T[36*0 + 6*1 + 1] = s_cosq[0];
    s_T[36*0 + 6*1 + 2] = static_cast<T>(0);
    s_T[36*0 + 6*1 + 3] =  tz_j1 * s_cosq[0];
    s_T[36*0 + 6*1 + 4] = -tz_j1 * s_sinq[0];
    s_T[36*0 + 6*1 + 5] = static_cast<T>(0);
    s_T[36*0 + 6*2 + 0] = static_cast<T>(0);
    s_T[36*0 + 6*2 + 1] = static_cast<T>(0);
    s_T[36*0 + 6*2 + 2] = static_cast<T>(1.0);
    s_T[36*0 + 6*2 + 3] = static_cast<T>(0);
    s_T[36*0 + 6*2 + 4] = static_cast<T>(0);
    s_T[36*0 + 6*2 + 5] = static_cast<T>(0);
    s_T[36*0 + 6*3 + 0] = static_cast<T>(0);
    s_T[36*0 + 6*3 + 1] = static_cast<T>(0);
    s_T[36*0 + 6*3 + 2] = static_cast<T>(0);
    s_T[36*0 + 6*3 + 3] = s_cosq[0];
    s_T[36*0 + 6*3 + 4] = -s_sinq[0];
    s_T[36*0 + 6*3 + 5] = static_cast<T>(0);
    s_T[36*0 + 6*4 + 0] = static_cast<T>(0);
    s_T[36*0 + 6*4 + 1] = static_cast<T>(0);
    s_T[36*0 + 6*4 + 2] = static_cast<T>(0);
    s_T[36*0 + 6*4 + 3] = s_sinq[0];
    s_T[36*0 + 6*4 + 4] = s_cosq[0];
    s_T[36*0 + 6*4 + 5] = static_cast<T>(0);
    s_T[36*0 + 6*5 + 0] = static_cast<T>(0);
    s_T[36*0 + 6*5 + 1] = static_cast<T>(0);
    s_T[36*0 + 6*5 + 2] = static_cast<T>(0);
    s_T[36*0 + 6*5 + 3] = static_cast<T>(0);
    s_T[36*0 + 6*5 + 4] = static_cast<T>(0);
    s_T[36*0 + 6*5 + 5] = static_cast<T>(1.0);
    s_T[36*1 + 6*0 + 0] = -s_cosq[1];
    s_T[36*1 + 6*0 + 1] = s_sinq[1];;
    s_T[36*1 + 6*0 + 2] = static_cast<T>(0);
    s_T[36*1 + 6*0 + 3] = static_cast<T>(0);
    s_T[36*1 + 6*0 + 4] = static_cast<T>(0);
    s_T[36*1 + 6*0 + 5] = -tz_j2;
    s_T[36*1 + 6*1 + 0] = static_cast<T>(0);
    s_T[36*1 + 6*1 + 1] = static_cast<T>(0);
    s_T[36*1 + 6*1 + 2] = static_cast<T>(1.0);
    s_T[36*1 + 6*1 + 3] = -tz_j2 * s_cosq[1];
    s_T[36*1 + 6*1 + 4] =  tz_j2 * s_sinq[1];
    s_T[36*1 + 6*1 + 5] = static_cast<T>(0);
    s_T[36*1 + 6*2 + 0] = s_sinq[1];
    s_T[36*1 + 6*2 + 1] = s_cosq[1];
    s_T[36*1 + 6*2 + 2] = static_cast<T>(0);
    s_T[36*1 + 6*2 + 3] = static_cast<T>(0);
    s_T[36*1 + 6*2 + 4] = static_cast<T>(0);
    s_T[36*1 + 6*2 + 5] = static_cast<T>(0);
    s_T[36*1 + 6*3 + 0] = static_cast<T>(0);
    s_T[36*1 + 6*3 + 1] = static_cast<T>(0);
    s_T[36*1 + 6*3 + 2] = static_cast<T>(0);
    s_T[36*1 + 6*3 + 3] = -s_cosq[1];
    s_T[36*1 + 6*3 + 4] = s_sinq[1];
    s_T[36*1 + 6*3 + 5] = static_cast<T>(0);
    s_T[36*1 + 6*4 + 0] = static_cast<T>(0);
    s_T[36*1 + 6*4 + 1] = static_cast<T>(0);
    s_T[36*1 + 6*4 + 2] = static_cast<T>(0);
    s_T[36*1 + 6*4 + 3] = static_cast<T>(0);
    s_T[36*1 + 6*4 + 4] = static_cast<T>(0);
    s_T[36*1 + 6*4 + 5] = static_cast<T>(1.0);
    s_T[36*1 + 6*5 + 0] = static_cast<T>(0);
    s_T[36*1 + 6*5 + 1] = static_cast<T>(0);
    s_T[36*1 + 6*5 + 2] = static_cast<T>(0);
    s_T[36*1 + 6*5 + 3] = s_sinq[1];
    s_T[36*1 + 6*5 + 4] = s_cosq[1];
    s_T[36*1 + 6*5 + 5] = static_cast<T>(0);
    s_T[36*2 + 6*0 + 0] = -s_cosq[2];
    s_T[36*2 + 6*0 + 1] = s_sinq[2];
    s_T[36*2 + 6*0 + 2] = static_cast<T>(0);
    s_T[36*2 + 6*0 + 3] =  ty_j3 * s_sinq[2];
    s_T[36*2 + 6*0 + 4] =  ty_j3 * s_cosq[2];
    s_T[36*2 + 6*0 + 5] = static_cast<T>(0);
    s_T[36*2 + 6*1 + 0] = static_cast<T>(0);
    s_T[36*2 + 6*1 + 1] = static_cast<T>(0);
    s_T[36*2 + 6*1 + 2] = static_cast<T>(1.0);
    s_T[36*2 + 6*1 + 3] = static_cast<T>(0);
    s_T[36*2 + 6*1 + 4] = static_cast<T>(0);
    s_T[36*2 + 6*1 + 5] = static_cast<T>(0);
    s_T[36*2 + 6*2 + 0] = s_sinq[2];
    s_T[36*2 + 6*2 + 1] = s_cosq[2];
    s_T[36*2 + 6*2 + 2] = static_cast<T>(0);
    s_T[36*2 + 6*2 + 3] =  ty_j3 * s_cosq[2];
    s_T[36*2 + 6*2 + 4] = -ty_j3 * s_sinq[2];
    s_T[36*2 + 6*2 + 5] = static_cast<T>(0);
    s_T[36*2 + 6*3 + 0] = static_cast<T>(0);
    s_T[36*2 + 6*3 + 1] = static_cast<T>(0);
    s_T[36*2 + 6*3 + 2] = static_cast<T>(0);
    s_T[36*2 + 6*3 + 3] = -s_cosq[2];
    s_T[36*2 + 6*3 + 4] = s_sinq[2];
    s_T[36*2 + 6*3 + 5] = static_cast<T>(0);
    s_T[36*2 + 6*4 + 0] = static_cast<T>(0);
    s_T[36*2 + 6*4 + 1] = static_cast<T>(0);
    s_T[36*2 + 6*4 + 2] = static_cast<T>(0);
    s_T[36*2 + 6*4 + 3] = static_cast<T>(0);
    s_T[36*2 + 6*4 + 4] = static_cast<T>(0);
    s_T[36*2 + 6*4 + 5] = static_cast<T>(1.0);
    s_T[36*2 + 6*5 + 0] = static_cast<T>(0);
    s_T[36*2 + 6*5 + 1] = static_cast<T>(0);
    s_T[36*2 + 6*5 + 2] = static_cast<T>(0);
    s_T[36*2 + 6*5 + 3] = s_sinq[2];
    s_T[36*2 + 6*5 + 4] = s_cosq[2];
    s_T[36*2 + 6*5 + 5] = static_cast<T>(0);
    s_T[36*3 + 6*0 + 0] = s_cosq[3];
    s_T[36*3 + 6*0 + 1] = -s_sinq[3];
    s_T[36*3 + 6*0 + 2] = static_cast<T>(0);
    s_T[36*3 + 6*0 + 3] = static_cast<T>(0);
    s_T[36*3 + 6*0 + 4] = static_cast<T>(0);
    s_T[36*3 + 6*0 + 5] =  tz_j4;
    s_T[36*3 + 6*1 + 0] = static_cast<T>(0);
    s_T[36*3 + 6*1 + 1] = static_cast<T>(0);
    s_T[36*3 + 6*1 + 2] = static_cast<T>(-1.0);
    s_T[36*3 + 6*1 + 3] =  tz_j4 * s_cosq[3];
    s_T[36*3 + 6*1 + 4] = -tz_j4 * s_sinq[3];
    s_T[36*3 + 6*1 + 5] = static_cast<T>(0);
    s_T[36*3 + 6*2 + 0] = s_sinq[3];
    s_T[36*3 + 6*2 + 1] = s_cosq[3];
    s_T[36*3 + 6*2 + 2] = static_cast<T>(0);
    s_T[36*3 + 6*2 + 3] = static_cast<T>(0);
    s_T[36*3 + 6*2 + 4] = static_cast<T>(0);
    s_T[36*3 + 6*2 + 5] = static_cast<T>(0);
    s_T[36*3 + 6*3 + 0] = static_cast<T>(0);
    s_T[36*3 + 6*3 + 1] = static_cast<T>(0);
    s_T[36*3 + 6*3 + 2] = static_cast<T>(0);
    s_T[36*3 + 6*3 + 3] = s_cosq[3];
    s_T[36*3 + 6*3 + 4] = -s_sinq[3];
    s_T[36*3 + 6*3 + 5] = static_cast<T>(0);
    s_T[36*3 + 6*4 + 0] = static_cast<T>(0);
    s_T[36*3 + 6*4 + 1] = static_cast<T>(0);
    s_T[36*3 + 6*4 + 2] = static_cast<T>(0);
    s_T[36*3 + 6*4 + 3] = static_cast<T>(0);
    s_T[36*3 + 6*4 + 4] = static_cast<T>(0);
    s_T[36*3 + 6*4 + 5] = static_cast<T>(-1.0);
    s_T[36*3 + 6*5 + 0] = static_cast<T>(0);
    s_T[36*3 + 6*5 + 1] = static_cast<T>(0);
    s_T[36*3 + 6*5 + 2] = static_cast<T>(0);
    s_T[36*3 + 6*5 + 3] = s_sinq[3];
    s_T[36*3 + 6*5 + 4] = s_cosq[3];
    s_T[36*3 + 6*5 + 5] = static_cast<T>(0);
    s_T[36*4 + 6*0 + 0] = -s_cosq[4];
    s_T[36*4 + 6*0 + 1] = s_sinq[4];
    s_T[36*4 + 6*0 + 2] = static_cast<T>(0);
    s_T[36*4 + 6*0 + 3] =  ty_j5 * s_sinq[4];
    s_T[36*4 + 6*0 + 4] =  ty_j5 * s_cosq[4];
    s_T[36*4 + 6*0 + 5] = static_cast<T>(0);
    s_T[36*4 + 6*1 + 0] = static_cast<T>(0);
    s_T[36*4 + 6*1 + 1] = static_cast<T>(0);
    s_T[36*4 + 6*1 + 2] = static_cast<T>(1.0);
    s_T[36*4 + 6*1 + 3] = static_cast<T>(0);
    s_T[36*4 + 6*1 + 4] = static_cast<T>(0);
    s_T[36*4 + 6*1 + 5] = static_cast<T>(0);
    s_T[36*4 + 6*2 + 0] = s_sinq[4];
    s_T[36*4 + 6*2 + 1] = s_cosq[4];
    s_T[36*4 + 6*2 + 2] = static_cast<T>(0);
    s_T[36*4 + 6*2 + 3] =  ty_j5 * s_cosq[4];
    s_T[36*4 + 6*2 + 4] = -ty_j5 * s_sinq[4];
    s_T[36*4 + 6*2 + 5] = static_cast<T>(0);
    s_T[36*4 + 6*3 + 0] = static_cast<T>(0);
    s_T[36*4 + 6*3 + 1] = static_cast<T>(0);
    s_T[36*4 + 6*3 + 2] = static_cast<T>(0);
    s_T[36*4 + 6*3 + 3] = -s_cosq[4];
    s_T[36*4 + 6*3 + 4] = s_sinq[4];
    s_T[36*4 + 6*3 + 5] = static_cast<T>(0);
    s_T[36*4 + 6*4 + 0] = static_cast<T>(0);
    s_T[36*4 + 6*4 + 1] = static_cast<T>(0);
    s_T[36*4 + 6*4 + 2] = static_cast<T>(0);
    s_T[36*4 + 6*4 + 3] = static_cast<T>(0);
    s_T[36*4 + 6*4 + 4] = static_cast<T>(0);
    s_T[36*4 + 6*4 + 5] = static_cast<T>(1.0);
    s_T[36*4 + 6*5 + 0] = static_cast<T>(0);
    s_T[36*4 + 6*5 + 1] = static_cast<T>(0);
    s_T[36*4 + 6*5 + 2] = static_cast<T>(0);
    s_T[36*4 + 6*5 + 3] = s_sinq[4];
    s_T[36*4 + 6*5 + 4] = s_cosq[4];
    s_T[36*4 + 6*5 + 5] = static_cast<T>(0);
    s_T[36*5 + 6*0 + 0] = s_cosq[5];
    s_T[36*5 + 6*0 + 1] = -s_sinq[5];
    s_T[36*5 + 6*0 + 2] = static_cast<T>(0);
    s_T[36*5 + 6*0 + 3] = static_cast<T>(0);
    s_T[36*5 + 6*0 + 4] = static_cast<T>(0);
    s_T[36*5 + 6*0 + 5] =  tz_j6;
    s_T[36*5 + 6*1 + 0] = static_cast<T>(0);
    s_T[36*5 + 6*1 + 1] = static_cast<T>(0);
    s_T[36*5 + 6*1 + 2] = static_cast<T>(-1.0);
    s_T[36*5 + 6*1 + 3] =  tz_j6 * s_cosq[5];
    s_T[36*5 + 6*1 + 4] = -tz_j6 * s_sinq[5];
    s_T[36*5 + 6*1 + 5] = static_cast<T>(0);
    s_T[36*5 + 6*2 + 0] = s_sinq[5];
    s_T[36*5 + 6*2 + 1] = s_cosq[5];
    s_T[36*5 + 6*2 + 2] = static_cast<T>(0);
    s_T[36*5 + 6*2 + 3] = static_cast<T>(0);
    s_T[36*5 + 6*2 + 4] = static_cast<T>(0);
    s_T[36*5 + 6*2 + 5] = static_cast<T>(0);
    s_T[36*5 + 6*3 + 0] = static_cast<T>(0);
    s_T[36*5 + 6*3 + 1] = static_cast<T>(0);
    s_T[36*5 + 6*3 + 2] = static_cast<T>(0);
    s_T[36*5 + 6*3 + 3] = s_cosq[5];
    s_T[36*5 + 6*3 + 4] = -s_sinq[5];
    s_T[36*5 + 6*3 + 5] = static_cast<T>(0);
    s_T[36*5 + 6*4 + 0] = static_cast<T>(0);
    s_T[36*5 + 6*4 + 1] = static_cast<T>(0);
    s_T[36*5 + 6*4 + 2] = static_cast<T>(0);
    s_T[36*5 + 6*4 + 3] = static_cast<T>(0);
    s_T[36*5 + 6*4 + 4] = static_cast<T>(0);
    s_T[36*5 + 6*4 + 5] = static_cast<T>(-1.0);
    s_T[36*5 + 6*5 + 0] = static_cast<T>(0);
    s_T[36*5 + 6*5 + 1] = static_cast<T>(0);
    s_T[36*5 + 6*5 + 2] = static_cast<T>(0);
    s_T[36*5 + 6*5 + 3] = s_sinq[5];
    s_T[36*5 + 6*5 + 4] = s_cosq[5];
    s_T[36*5 + 6*5 + 5] = static_cast<T>(0);
    s_T[36*6 + 6*0 + 0] = -s_cosq[6];
    s_T[36*6 + 6*0 + 1] = s_sinq[6];
    s_T[36*6 + 6*0 + 2] = static_cast<T>(0);
    s_T[36*6 + 6*0 + 3] =  ty_j7 * s_sinq[6];
    s_T[36*6 + 6*0 + 4] =  ty_j7 * s_cosq[6];
    s_T[36*6 + 6*0 + 5] = static_cast<T>(0);
    s_T[36*6 + 6*1 + 0] = static_cast<T>(0);
    s_T[36*6 + 6*1 + 1] = static_cast<T>(0);
    s_T[36*6 + 6*1 + 2] = static_cast<T>(1.0);
    s_T[36*6 + 6*1 + 3] = static_cast<T>(0);
    s_T[36*6 + 6*1 + 4] = static_cast<T>(0);
    s_T[36*6 + 6*1 + 5] = static_cast<T>(0);
    s_T[36*6 + 6*2 + 0] = s_sinq[6];
    s_T[36*6 + 6*2 + 1] = s_cosq[6];
    s_T[36*6 + 6*2 + 2] = static_cast<T>(0);
    s_T[36*6 + 6*2 + 3] =  ty_j7 * s_cosq[6];
    s_T[36*6 + 6*2 + 4] = -ty_j7 * s_sinq[6];
    s_T[36*6 + 6*2 + 5] = static_cast<T>(0);
    s_T[36*6 + 6*3 + 0] = static_cast<T>(0);
    s_T[36*6 + 6*3 + 1] = static_cast<T>(0);
    s_T[36*6 + 6*3 + 2] = static_cast<T>(0);
    s_T[36*6 + 6*3 + 3] = -s_cosq[6];
    s_T[36*6 + 6*3 + 4] = s_sinq[6];
    s_T[36*6 + 6*3 + 5] = static_cast<T>(0);
    s_T[36*6 + 6*4 + 0] = static_cast<T>(0);
    s_T[36*6 + 6*4 + 1] = static_cast<T>(0);
    s_T[36*6 + 6*4 + 2] = static_cast<T>(0);
    s_T[36*6 + 6*4 + 3] = static_cast<T>(0);
    s_T[36*6 + 6*4 + 4] = static_cast<T>(0);
    s_T[36*6 + 6*4 + 5] = static_cast<T>(1.0);
    s_T[36*6 + 6*5 + 0] = static_cast<T>(0);
    s_T[36*6 + 6*5 + 1] = static_cast<T>(0);
    s_T[36*6 + 6*5 + 2] = static_cast<T>(0);
    s_T[36*6 + 6*5 + 3] = s_sinq[6];
    s_T[36*6 + 6*5 + 4] = s_cosq[6];
    s_T[36*6 + 6*5 + 5] = static_cast<T>(0);
}

template <typename T>
void initTransforms(T *s_T){
    std::memset(s_T,0,36*NUM_POS*sizeof(T));
    s_T[36*0 + 6*2 + 2] = static_cast<T>(1.0);
    s_T[36*0 + 6*5 + 5] = static_cast<T>(1.0);
    s_T[36*1 + 6*0 + 5] = -tz_j2;
    s_T[36*1 + 6*1 + 2] = static_cast<T>(1.0);
    s_T[36*1 + 6*4 + 5] = static_cast<T>(1.0);
    s_T[36*2 + 6*1 + 2] = static_cast<T>(1.0);
    s_T[36*2 + 6*4 + 5] = static_cast<T>(1.0);
    s_T[36*3 + 6*0 + 5] =  tz_j4;
    s_T[36*3 + 6*1 + 2] = static_cast<T>(-1.0);
    s_T[36*3 + 6*4 + 5] = static_cast<T>(-1.0);
    s_T[36*4 + 6*1 + 2] = static_cast<T>(1.0);
    s_T[36*4 + 6*4 + 5] = static_cast<T>(1.0);
    s_T[36*5 + 6*0 + 5] =  tz_j6;
    s_T[36*5 + 6*1 + 2] = static_cast<T>(-1.0);
    s_T[36*5 + 6*4 + 5] = static_cast<T>(-1.0);
    s_T[36*6 + 6*1 + 2] = static_cast<T>(1.0);
    s_T[36*6 + 6*4 + 5] = static_cast<T>(1.0);
}
template <typename T>
void updateTransforms(T *s_T, T *s_sinq, T *s_cosq){
    s_T[36*0 + 6*0 + 0] = s_cosq[0];
    s_T[36*0 + 6*0 + 1] = -s_sinq[0];
    s_T[36*0 + 6*0 + 3] = -tz_j1 * s_sinq[0];
    s_T[36*0 + 6*0 + 4] = -tz_j1 * s_cosq[0];
    s_T[36*0 + 6*1 + 0] = s_sinq[0];
    s_T[36*0 + 6*1 + 1] = s_cosq[0];
    s_T[36*0 + 6*1 + 3] =  tz_j1 * s_cosq[0];
    s_T[36*0 + 6*1 + 4] = -tz_j1 * s_sinq[0];
    s_T[36*0 + 6*3 + 3] = s_cosq[0];
    s_T[36*0 + 6*3 + 4] = -s_sinq[0];
    s_T[36*0 + 6*4 + 3] = s_sinq[0];
    s_T[36*0 + 6*4 + 4] = s_cosq[0];
    s_T[36*1 + 6*0 + 0] = -s_cosq[1];
    s_T[36*1 + 6*0 + 1] = s_sinq[1];;
    s_T[36*1 + 6*1 + 3] = -tz_j2 * s_cosq[1];
    s_T[36*1 + 6*1 + 4] =  tz_j2 * s_sinq[1];
    s_T[36*1 + 6*2 + 0] = s_sinq[1];
    s_T[36*1 + 6*2 + 1] = s_cosq[1];
    s_T[36*1 + 6*3 + 3] = -s_cosq[1];
    s_T[36*1 + 6*3 + 4] = s_sinq[1];
    s_T[36*1 + 6*5 + 3] = s_sinq[1];
    s_T[36*1 + 6*5 + 4] = s_cosq[1];
    s_T[36*2 + 6*0 + 0] = -s_cosq[2];
    s_T[36*2 + 6*0 + 1] = s_sinq[2];
    s_T[36*2 + 6*0 + 3] =  ty_j3 * s_sinq[2];
    s_T[36*2 + 6*0 + 4] =  ty_j3 * s_cosq[2];
    s_T[36*2 + 6*2 + 0] = s_sinq[2];
    s_T[36*2 + 6*2 + 1] = s_cosq[2];
    s_T[36*2 + 6*2 + 3] =  ty_j3 * s_cosq[2];
    s_T[36*2 + 6*2 + 4] = -ty_j3 * s_sinq[2];
    s_T[36*2 + 6*3 + 3] = -s_cosq[2];
    s_T[36*2 + 6*3 + 4] = s_sinq[2];
    s_T[36*2 + 6*5 + 3] = s_sinq[2];
    s_T[36*2 + 6*5 + 4] = s_cosq[2];
    s_T[36*3 + 6*0 + 0] = s_cosq[3];
    s_T[36*3 + 6*0 + 1] = -s_sinq[3];
    s_T[36*3 + 6*1 + 3] =  tz_j4 * s_cosq[3];
    s_T[36*3 + 6*1 + 4] = -tz_j4 * s_sinq[3];
    s_T[36*3 + 6*2 + 0] = s_sinq[3];
    s_T[36*3 + 6*2 + 1] = s_cosq[3];
    s_T[36*3 + 6*3 + 3] = s_cosq[3];
    s_T[36*3 + 6*3 + 4] = -s_sinq[3];
    s_T[36*3 + 6*5 + 3] = s_sinq[3];
    s_T[36*3 + 6*5 + 4] = s_cosq[3];
    s_T[36*4 + 6*0 + 0] = -s_cosq[4];
    s_T[36*4 + 6*0 + 1] = s_sinq[4];
    s_T[36*4 + 6*0 + 3] =  ty_j5 * s_sinq[4];
    s_T[36*4 + 6*0 + 4] =  ty_j5 * s_cosq[4];
    s_T[36*4 + 6*2 + 0] = s_sinq[4];
    s_T[36*4 + 6*2 + 1] = s_cosq[4];
    s_T[36*4 + 6*2 + 3] =  ty_j5 * s_cosq[4];
    s_T[36*4 + 6*2 + 4] = -ty_j5 * s_sinq[4];
    s_T[36*4 + 6*3 + 3] = -s_cosq[4];
    s_T[36*4 + 6*3 + 4] = s_sinq[4];
    s_T[36*4 + 6*5 + 3] = s_sinq[4];
    s_T[36*4 + 6*5 + 4] = s_cosq[4];
    s_T[36*5 + 6*0 + 0] = s_cosq[5];
    s_T[36*5 + 6*0 + 1] = -s_sinq[5];
    s_T[36*5 + 6*1 + 3] =  tz_j6 * s_cosq[5];
    s_T[36*5 + 6*1 + 4] = -tz_j6 * s_sinq[5];
    s_T[36*5 + 6*2 + 0] = s_sinq[5];
    s_T[36*5 + 6*2 + 1] = s_cosq[5];
    s_T[36*5 + 6*3 + 3] = s_cosq[5];
    s_T[36*5 + 6*3 + 4] = -s_sinq[5];
    s_T[36*5 + 6*5 + 3] = s_sinq[5];
    s_T[36*5 + 6*5 + 4] = s_cosq[5];
    s_T[36*6 + 6*0 + 0] = -s_cosq[6];
    s_T[36*6 + 6*0 + 1] = s_sinq[6];
    s_T[36*6 + 6*0 + 3] =  ty_j7 * s_sinq[6];
    s_T[36*6 + 6*0 + 4] =  ty_j7 * s_cosq[6];
    s_T[36*6 + 6*2 + 0] = s_sinq[6];
    s_T[36*6 + 6*2 + 1] = s_cosq[6];
    s_T[36*6 + 6*2 + 3] =  ty_j7 * s_cosq[6];
    s_T[36*6 + 6*2 + 4] = -ty_j7 * s_sinq[6];
    s_T[36*6 + 6*3 + 3] = -s_cosq[6];
    s_T[36*6 + 6*3 + 4] = s_sinq[6];
    s_T[36*6 + 6*5 + 3] = s_sinq[6];
    s_T[36*6 + 6*5 + 4] = s_cosq[6];
}
// I and then T
template <typename T>
__host__
void dynamicsAndGradient_init(T *h_mem_const){
   initInertiaTensors<T>(h_mem_const);
   initTransforms<T>(&h_mem_const[36*NUM_POS]);
}

//----------------------------------------------------------------------------
// RNEA and Helpers
//----------------------------------------------------------------------------
template <typename T>
void motionCrossProductMxCol3(T *s_vx, T *s_v, T qd_k){
    s_vx[0] = s_v[1]*qd_k;
    s_vx[1] = -s_v[0]*qd_k;
    s_vx[2] = 0;
    s_vx[3] = s_v[4]*qd_k;
    s_vx[4] = -s_v[3]*qd_k;
    s_vx[5] = 0;
}

template <typename T>
void vxIv(T *s_ret, T *s_v, T *s_I){
    // first do temp = I*v
    T s_temp[6]; matVMult<T,6,6,0>(s_temp,s_I,s_v); 
    // then do crf(v)*temp
    s_ret[0] = -s_v[2]*s_temp[1] + s_v[1]*s_temp[2];
    s_ret[1] = s_v[2]*s_temp[0] + -s_v[0]*s_temp[2];
    s_ret[2] = -s_v[1]*s_temp[0] + s_v[0]*s_temp[1];
    s_ret[3] = -s_v[2]*s_temp[1+3] + s_v[1]*s_temp[2+3];
    s_ret[4] = s_v[2]*s_temp[0+3] + -s_v[0]*s_temp[2+3];
    s_ret[5] = -s_v[1]*s_temp[0+3] + s_v[0]*s_temp[1+3];
    s_ret[0] += -s_v[2+3]*s_temp[1+3] + s_v[1+3]*s_temp[2+3];
    s_ret[1] += s_v[2+3]*s_temp[0+3] + -s_v[0+3]*s_temp[2+3];
    s_ret[2] += -s_v[1+3]*s_temp[0+3] + s_v[0+3]*s_temp[1+3];
}


template <typename T, bool MPC_MODE = false>
void RNEA_fp(T *s_v, T *s_a, T *s_f, T *s_I, T *s_T, T *s_qd, T *s_qdd = nullptr, T *s_fext = nullptr){
    for (int k = 0; k < NUM_POS; k++){
        T *lk_v = &s_v[6*k];    T *lk_a = &s_a[6*k];    T *tkXkm1 = &s_T[36*k];
        if (k == 0){
            // l1_v = vJ, for the first link of a fixed base robot
            lk_v[0] = static_cast<T>(0);
            lk_v[1] = static_cast<T>(0);
            lk_v[2] = s_qd[k];
            lk_v[3] = static_cast<T>(0);
            lk_v[4] = static_cast<T>(0);
            lk_v[5] = static_cast<T>(0);
            // and the a is just gravity times the last col of the transform and then add qdd in 3rd entry also note that last col is [0,0,0,a,b,c]^T
            lk_a[0] = static_cast<T>(0);
            lk_a[1] = static_cast<T>(0);
            lk_a[2] = (s_qdd != nullptr) * s_qdd[k];
            lk_a[3] = MPC_MODE ? static_cast<T>(0) : tkXkm1[6*5+3] * GRAVITY; // MPC with arm has gravity comp so need to pretend gravity is 0
            lk_a[4] = MPC_MODE ? static_cast<T>(0) : tkXkm1[6*5+4] * GRAVITY;
            lk_a[5] = MPC_MODE ? static_cast<T>(0) : tkXkm1[6*5+5] * GRAVITY;
        }
        else{
            T *lkm1_v = &s_v[6*(k-1)];    T *lkm1_a = &s_a[6*(k-1)];
            // first compute the v
            matVMult_Tmat<T,0,0>(lk_v,tkXkm1,lkm1_v);
            lk_v[2] += s_qd[k]; // add in qd
            // then the a
            motionCrossProductMxCol3<T>(lk_a,lk_v,s_qd[k]); // motion cross product 3rd col of lk_v mulitplied by qd[k]
            matVMult_Tmat<T,1,0>(lk_a,tkXkm1,lkm1_a); // so now we += that with tkXkm1 * lkm1_a
            lk_a[2] += (s_qdd != nullptr) * s_qdd[k]; // add qdd
        }

        // then the f
        T *lk_f = &s_f[6*k];    T *lk_I = &s_I[36*k];
        vxIv<T>(lk_f,lk_v,lk_I); // compute the vxIv of lk_v and lk_I and store in lk_f
        matVMult<T,6,6,1>(lk_f,lk_I,lk_a); // then += that with lk_I * lk_a
        if(s_fext != nullptr){ // then subtract the external force
            T *lk_fext = &s_fext[6*k];
            for (int i = 0; i < 6; i++){lk_f[i] -= lk_fext[i];}    
        }
    }
}

template <typename T>
void RNEA_bp(T *s_tau, T *s_f, T *s_T){
    for (int k = NUM_POS - 1; k >= 0; k--){
        T *lk_f = &s_f[6*k];     if(s_tau != nullptr){s_tau[k] = lk_f[2];}
        if(k > 0){
            T *lkm1_f = &s_f[6*(k-1)];     T *tkXkm1 = &s_T[36*k];
            matVMult_Tmat<T,1,1>(lkm1_f,tkXkm1,lk_f); // lkm1_f += tkXkm1^T * lk_f
        }
    }
}

//-------------------------------------------------------------------------------
// Forward Dynamics and Helpers
//-------------------------------------------------------------------------------

template <typename T, bool VEL_DAMPING = true>
void computeBias(T *s_bias, T *s_c, T *s_tau, T *s_qd){
    // c now in temp so compute bias term (tau-c)
    for (int k = 0; k < NUM_POS; k++){
        if(VEL_DAMPING){s_bias[k] = s_tau[k] - (s_c[k] + 0.5*s_qd[k]);}
        else{s_bias[k] = s_tau[k] - s_c[k];}
    }
}

template <typename T, bool MPC_MODE = false, bool VEL_DAMPING = true, bool DENSE_MINV = false>
void forwardDynamics(T *s_qdd, T *s_tau, T *s_qd, T *s_q, T *d_Minv, T *s_fext = nullptr){
    T s_T[36*NUM_POS]; T s_I[36*NUM_POS];
    // build transforms for this configuration
    T s_sinq[NUM_POS]; T s_cosq[NUM_POS];
    for (int ind = 0; ind < NUM_POS; ind++){s_sinq[ind] = std::sin(s_q[ind]); s_cosq[ind] = std::cos(s_q[ind]);}
    buildTransforms(s_T,s_sinq,s_cosq);
    initInertiaTensors(s_I);
    // then vaf -> bias
    T s_v[6*NUM_POS]; T s_a[6*NUM_POS]; T s_f[6*NUM_POS];
    T s_c[NUM_POS]; T s_bias[NUM_POS];
    RNEA_fp<T,MPC_MODE>(s_v, s_a, s_f, s_I, s_T, s_qd, s_qdd, s_fext); // we leave room to use in gradient with a qdd passed in to compute grad bias
    RNEA_bp<T>(s_c, s_f, s_T);
    computeBias<T,VEL_DAMPING>(s_bias, s_c, s_tau, s_qd);
    // and clear things and compute for Minv
    std::memset(d_Minv,0,NUM_POS*NUM_POS*sizeof(T));
    computeMinv<T>(d_Minv,s_T,s_I); // here we just pass I in as we can safely overwrite it saving a copy
    // now we have an upper triangular matrix in Minv (because symmetric) and the bias term
    // we now need to computer qdd = Minv*bias
    matVMultSym<T,NUM_POS,NUM_POS>(s_qdd,d_Minv,NUM_POS,s_bias);
    if(DENSE_MINV){triangularToDense<T,NUM_POS,1>(d_Minv);}
}

//-------------------------------------------------------------------------------
// FD Gradient and Helpers
//-------------------------------------------------------------------------------

template <typename T, bool PEQFLAG = 0>
void MxMatCol3(T *out, T *vec){
    if (PEQFLAG){
        out[0] += vec[1];
        out[1] += -vec[0];
        out[3] += vec[4];
        out[4] += -vec[3];
    }
    else{
        out[0] = vec[1];
        out[1] = -vec[0];
        out[2] = 0;
        out[3] = vec[4];
        out[4] = -vec[3];
        out[5] = 0;
    }
}

template <typename T, bool PEQFLAG = 0>
void MxMatCol3(T *out, T *vec, T alpha){
    if (PEQFLAG){
        out[0] += vec[1]*alpha;
        out[1] += -vec[0]*alpha;
        out[3] += vec[4]*alpha;
        out[4] += -vec[3]*alpha;
    }
    else{
        out[0] = vec[1]*alpha;
        out[1] = -vec[0]*alpha;
        out[2] = 0;
        out[3] = vec[4]*alpha;
        out[4] = -vec[3]*alpha;
        out[5] = 0;
    }
}

template <typename T, bool PEQFLAG = 0, bool T_FLAG = 0>
void MxMatCol3_Tmat(T *out, T *Tmat, T *vec){
    if (PEQFLAG){
        out[1] += -dotProd_Tmat<T,T_FLAG>(Tmat,vec,0);
        out[0] += dotProd_Tmat<T,T_FLAG>(Tmat,vec,1);
        out[4] += -dotProd_Tmat<T,T_FLAG>(Tmat,vec,3);
        out[3] += dotProd_Tmat<T,T_FLAG>(Tmat,vec,4);
    }
    else{
        out[1] = -dotProd_Tmat<T,T_FLAG>(Tmat,vec,0);
        out[0] = dotProd_Tmat<T,T_FLAG>(Tmat,vec,1);
        out[4] = -dotProd_Tmat<T,T_FLAG>(Tmat,vec,3);
        out[3] = dotProd_Tmat<T,T_FLAG>(Tmat,vec,4);
        out[2] = static_cast<T>(0);
        out[5] = static_cast<T>(0);
    }
}                   

template <typename T>
void computeFxMatTimesVector(T *dstVec, T *srcFxVec, T *srcVec){
    // dstVec = Fx(srxFxVec)*srcVec
    /* 0  -v(2)  v(1)    0  -v(5)  v(4)
    v(2)    0  -v(0)  v(5)    0  -v(3)
    -v(1)  v(0)    0  -v(4)  v(3)    0
     0     0      0     0  -v(2)  v(1)
     0     0      0   v(2)    0  -v(0)
     0     0      0  -v(1)  v(0)    0 */
    dstVec[0] = -srcFxVec[2] * srcVec[1] + srcFxVec[1] * srcVec[2] - srcFxVec[5] * srcVec[4] + srcFxVec[4] * srcVec[5];
    dstVec[1] =  srcFxVec[2] * srcVec[0] - srcFxVec[0] * srcVec[2] + srcFxVec[5] * srcVec[3] - srcFxVec[3] * srcVec[5];
    dstVec[2] = -srcFxVec[1] * srcVec[0] + srcFxVec[0] * srcVec[1] - srcFxVec[4] * srcVec[3] + srcFxVec[3] * srcVec[4];
    dstVec[3] =                                                     -srcFxVec[2] * srcVec[4] + srcFxVec[1] * srcVec[5];
    dstVec[4] =                                                      srcFxVec[2] * srcVec[3] - srcFxVec[0] * srcVec[5];
    dstVec[5] =                                                     -srcFxVec[1] * srcVec[3] + srcFxVec[0] * srcVec[4];
}

template <typename T>
void setToS(T *v){v[0] = static_cast<T>(0); v[1] = static_cast<T>(0); v[2] = static_cast<T>(1); v[3] = static_cast<T>(0); v[4] = static_cast<T>(0); v[5] = static_cast<T>(0);}

template <typename T>
void compute_vafdu(T *s_fdq, T *s_fdqd, T *s_adq, T *s_adqd, T *s_vdq, T *s_vdqd, T *s_vaf, T *s_qd, T *s_T, T *s_I){
    // compute_vafdu(s_fdq,s_fdqd,s_adq,s_adqd,s_vdq,s_vdqd,s_f,s_a,s_v,s_qd,s_T,s_I);
    // taking advantage of sparsity allows us to not try to multiply cols we know are zeros
    // (note that dq and dqd are independent so can walk the tree simulatenously)
    T *s_v = &s_vaf[0]; T *s_a = &s_vaf[6*NUM_POS]; T *s_f = &s_vaf[6*NUM_POS*2];
    for (int k = 0; k < NUM_POS; k++){
        T *vk = &s_v[6*k]; T *vlamk = &s_v[6*(k-1)]; T *alamk = &s_a[6*(k-1)]; T *Tk = &s_T[36*k]; T *Ik = &s_I[36*k];
        // note that we need just one FxvIk and Ivk so lets compute those outside of the column loop
        T Ivk[6]; matVMult<T,6,6>(Ivk,Ik,vk); T FxvIk[36]; for (int c = 0; c < 6; c++){computeFxMatTimesVector<T>(&FxvIk[c*6],vk,&Ik[c*6]);}
        for (int c = 0; c <= k; c++){int indkc = 3*k*(k+1) + 6*c; int indlamkc = indkc - 6*k; int indkc6 = 6*NUM_POS*k + 6*c;
            // for dq all of the first col is 0 identically so this skips k == 0 so lam always exists
            if (c > 0){
                // dVk/dq has the pattern [0 | Tk*dvlamk/dq for 1:k-1 cols | MxMat_col3(Tk*vlamk) | 0 for k+1 to NUM_POS col]
                if (c < k){matVMult_Tmat<T,0,0>(&s_vdq[indkc],Tk,&s_vdq[indlamkc]);}
                else {MxMatCol3_Tmat<T>(&s_vdq[indkc],Tk,vlamk);}
                // dak/dq = Tk * dalamk/du + MxMatCol3_oncols(dvk/du)*qd[k] + {for q in col k: MxMatCol3(Tk*alamk)}
                MxMatCol3<T>(&s_adq[indkc],&s_vdq[indkc],s_qd[k]);
                if (c < k){matVMult_Tmat<T,1,0>(&s_adq[indkc],Tk,&s_adq[indlamkc]);}
                else if(c == k){MxMatCol3_Tmat<T,1>(&s_adq[indkc],Tk,alamk);}
                // dfk/dq = Ik*dak/dq + FxMat(vk)*Ik*dvk/dq + FxMat(dvk/dq across cols)*Ik*vk
                computeFxMatTimesVector<T>(&s_fdq[indkc6],&s_vdq[indkc],Ivk);
                matVMult<T,6,6,1>(&s_fdq[indkc6],Ik,&s_adq[indkc]);
                matVMult<T,6,6,1>(&s_fdq[indkc6],FxvIk,&s_vdq[indkc]);

            }
            // but load the 0 into f for the backward pass
            else{for(int i = 0; i < 6; i++){s_fdq[indkc6 + i] = static_cast<T>(0);}}
            // then dqd where the first col has values
            // dVk/dqd has the pattern [Tk*dvlamk/dqd for 0:k-2 cols | Tk[:,2] for k-1 col | [0 0 1 0 0 0]T for k col | 0 for k+1 to NUM_POS col]
            if (c < k-1){matVMult_Tmat<T,0,0>(&s_vdqd[indkc],Tk,&s_vdqd[indlamkc]);}
            else if (c == k-1){std::memcpy(&s_vdqd[indkc],&Tk[12],6*sizeof(T));}
            else{setToS<T>(&s_vdqd[indkc]);}
            // dak/dqd = Tk * dalamk/du + MxMatCol3_oncols(dvk/du)*qd[k] + {for qd in col k: MxMatCol3(vk)}
            MxMatCol3<T>(&s_adqd[indkc],&s_vdqd[indkc],s_qd[k]);
            if (c < k){matVMult_Tmat<T,1,0>(&s_adqd[indkc],Tk,&s_adqd[indlamkc]);}
            else if(c == k){MxMatCol3<T,1>(&s_adqd[indkc],vk);}
            // dfk/dqd = Ik*dak/dqd + FxMat(vk)*Ik*dvk/dqd + FxMat(dvk/dqd across cols)*Ik*vk
            computeFxMatTimesVector<T>(&s_fdqd[indkc6],&s_vdqd[indkc],Ivk);
            matVMult<T,6,6,1>(&s_fdqd[indkc6],Ik,&s_adqd[indkc]);
            matVMult<T,6,6,1>(&s_fdqd[indkc6],FxvIk,&s_vdqd[indkc]);
        }
    }
}

template <typename T, bool VEL_DAMPING = 1>
void compute_cdu(T *s_cdq, T *s_cdqd, T *s_fdq, T *s_fdqd, T *s_T, T *s_f){
    // taking advantage of sparsity allows us to not try to multiply cols we know are zeros
    // (note that dq and dqd are independent so can walk the tree simulatenously)
    for (int k = NUM_POS - 1; k >= 0; k--){
        T *Tk = &s_T[36*k]; T *fk = &s_f[6*k];
        for (int c = 0; c < NUM_POS; c++){int indkc = 6*NUM_POS*k + 6*c; int indlamkc = indkc - 6*NUM_POS;
            // dc/du = row 3 of df so extract that
            s_cdq[NUM_POS*c + k] = s_fdq[indkc + 2]; s_cdqd[NUM_POS*c + k] = s_fdqd[indkc + 2];
            if(VEL_DAMPING && (c == k)){s_cdqd[NUM_POS*c + k] += 0.5;}
            // then updatre prev f (if not k == 0)
            if (k > 0){
                // dflamk/du += Tk^T*cols(df/du) -- note if c >= k need to set = because not set yet
                if (c < k){
                    matVMult_Tmat<T,1,1>(&s_fdq[indlamkc],Tk,&s_fdq[indkc]);
                    matVMult_Tmat<T,1,1>(&s_fdqd[indlamkc],Tk,&s_fdqd[indkc]);
                }
                else{
                    matVMult_Tmat<T,0,1>(&s_fdq[indlamkc],Tk,&s_fdq[indkc]);
                    matVMult_Tmat<T,0,1>(&s_fdqd[indlamkc],Tk,&s_fdqd[indkc]);   
                    // finally if c = k for f/dq += Tk^T*Fx3(fk)
                    if (c == k){for(int r = 0; r < 6; r++){T *Tkc = &Tk[r * 6]; s_fdq[indlamkc + r] += -fk[1]*Tkc[0] + fk[0]*Tkc[1] - fk[4]*Tkc[3] + fk[3]*Tkc[4];}}
                }
            }
        }  
    }
}

template <typename T, bool VEL_DAMPING = 1>
void inverseDynamicsGradient(T *s_cdq, T *s_cdqd, T *s_qd, T *s_vaf, T *s_T, T *s_I, int k = 0){
    T s_vdq[3*NUM_POS*(NUM_POS+1)];  T s_vdqd[3*NUM_POS*(NUM_POS+1)];
    T s_adq[3*NUM_POS*(NUM_POS+1)];  T s_adqd[3*NUM_POS*(NUM_POS+1)];
    T s_fdq[6*NUM_POS*NUM_POS];      T s_fdqd[6*NUM_POS*NUM_POS];
    compute_vafdu(s_fdq,s_fdqd,s_adq,s_adqd,s_vdq,s_vdqd,s_vaf,s_qd,s_T,s_I);
    compute_cdu<T,VEL_DAMPING>(s_cdq,s_cdqd,s_fdq,s_fdqd,s_T,&s_vaf[6*NUM_POS*2]);
}

template <typename T, bool QDD_MINV_PASSED_IN = false, bool MPC_MODE = false, bool VEL_DAMPING = true, bool EXTERNAL_TI = false>
void forwardDynamicsGradientSetup(T *s_vaf, T *s_T, T *s_I, T *s_q, T *s_qd, T *s_qdd, T *s_u, T *s_Minv, T *s_fext = nullptr){
    T *s_v = &s_vaf[0]; T *s_a = &s_vaf[6*NUM_POS]; T *s_f = &s_vaf[12*NUM_POS];
    T s_sinq[NUM_POS]; T s_cosq[NUM_POS];
    // build transforms and inertia for this configuration (and use external as base if requested)
    for (int ind = 0; ind < NUM_POS; ind++){s_sinq[ind] = std::sin(s_q[ind]); s_cosq[ind] = std::cos(s_q[ind]);}
    if(EXTERNAL_TI){updateTransforms(s_T,s_sinq,s_cosq);}
    else{buildTransforms(s_T,s_sinq,s_cosq); initInertiaTensors(s_I);}
    // start to compute bias term to get v,a,f
    RNEA_fp<T,MPC_MODE>(s_v, s_a, s_f, s_I, s_T, s_qd, s_qdd, s_fext);
    if(QDD_MINV_PASSED_IN){
        RNEA_bp<T>(nullptr,s_f,s_T); // finish f and we are done
    }
    else{
        // we need to keep the bias
        T s_bias[NUM_POS]; T s_c[NUM_POS];
        RNEA_bp<T>(s_c,s_f,s_T);
        computeBias<T,VEL_DAMPING>(s_bias, s_c, s_u, s_qd);
        // then compute Minv
        // note: we copy I to IA as we will modify it and don't want to change I as we will be re-using it
        T s_IA[36*NUM_POS]; std::memset(s_Minv,0,NUM_POS*NUM_POS*sizeof(T));
        std::memcpy(s_IA,s_I,36*NUM_POS*sizeof(T));
        computeMinv<T>(s_Minv,s_T,s_IA);
        // form qdd
        matVMultSym<T,NUM_POS,NUM_POS>(s_qdd,s_Minv,NUM_POS,s_bias);
        // and recompute v,a,f
        RNEA_fp<T,MPC_MODE>(s_v, s_a, s_f, s_I, s_T, s_qd, s_qdd, s_fext);
        RNEA_bp<T>(nullptr,s_f,s_T);
    }
}

template <typename T, bool QDD_MINV_PASSED_IN = 0, bool MPC_MODE = false, bool VEL_DAMPING = true, bool EXTERNAL_TI = false>
void forwardDynamicsGradient_vaf(T *s_vaf, knot<T> *currKnot, T *s_T, T *s_I, T *s_fext = nullptr){
    // dqdd/dtau = Minv and  dqdd/dq(d) = -Minv*dc/dq(d) note: dM term in dq drops out if you compute c with fd qdd per carpentier
    // first compute the dynamics gradient helpers (v,a,f to prep for inverse dynamics gradient) and compute Minv and qdd if needed
    forwardDynamicsGradientSetup<T,QDD_MINV_PASSED_IN,MPC_MODE,VEL_DAMPING,EXTERNAL_TI>(s_vaf,s_T,s_I,currKnot->q,currKnot->qd,currKnot->qdd,currKnot->u,currKnot->Minv,s_fext);
}

template <typename T, bool QDD_MINV_PASSED_IN = 0, bool MPC_MODE = false, bool VEL_DAMPING = true, bool EXTERNAL_TI = false>
void forwardDynamicsGradient_vaf_dcdu(T *s_cdu, knot<T> *currKnot, T *d_T, T *d_I, T *s_fext = nullptr){
    T s_vaf[6*NUM_POS*3]; T s_T[36*NUM_POS]; T s_I[36*NUM_POS];
    // first compute vaf (and T,I)
    forwardDynamicsGradient_vaf<T,QDD_MINV_PASSED_IN,MPC_MODE,VEL_DAMPING,EXTERNAL_TI>(s_vaf,currKnot,s_T,s_I,s_fext);
    // and then dc/du
    inverseDynamicsGradient<T,VEL_DAMPING>(s_cdu,&s_cdu[NUM_POS*NUM_POS],currKnot->qd,s_vaf,s_T,s_I);
}

template <typename T, bool QDD_MINV_PASSED_IN = 0, bool MPC_MODE = false, bool VEL_DAMPING = true, bool EXTERNAL_TI = false>
void forwardDynamicsGradient(T *s_dqdd, knot<T> *currKnot, T *d_T, T *d_I, T *s_fext = nullptr){
    T s_cdu[2*NUM_POS*NUM_POS];
    // first compute c_du from vaf
    forwardDynamicsGradient_vaf_dcdu<T,QDD_MINV_PASSED_IN,MPC_MODE,VEL_DAMPING>(s_cdu,currKnot,d_T,d_I,s_fext);
    // Then finish it off: dqdd/dtau = Minv, dqdd/dqd = -Minv*dc/dqd, dqdd/dq = -Minv*dc/dq --- remember Minv is a symmetric sym UPPER matrix
    matMultSym<T,NUM_POS,2*NUM_POS,NUM_POS>(s_dqdd,NUM_POS,currKnot->Minv,NUM_POS,s_cdu,NUM_POS,static_cast<T>(-1));
}

template <typename T, bool QDD_MINV_PASSED_IN = 0, bool MPC_MODE = false, bool VEL_DAMPING = true, bool EXTERNAL_TI = false>
void forwardDynamicsGradientThreaded(T *dqddk, knot<T> *currKnot, T *d_T, T *d_I, T *s_fext = nullptr){
    // first compute the dynamics gradient helpers (v,a,f to prep for inverse dynamics gradient) and compute Minv and qdd if needed
    T s_vaf[6*NUM_POS*3];
    forwardDynamicsGradientSetup<T,QDD_MINV_PASSED_IN,MPC_MODE,VEL_DAMPING,true>(s_vaf,d_T,d_I,currKnot->q,currKnot->qd,currKnot->qdd,currKnot->u,currKnot->Minv,s_fext);
    // then compute c_du from vaf
    T s_cdu[2*NUM_POS*NUM_POS];
    inverseDynamicsGradient<T,VEL_DAMPING>(s_cdu,&s_cdu[NUM_POS*NUM_POS],currKnot->qd,s_vaf,d_T,d_I);
    // Then finish it off: dqdd/dtau = Minv, dqdd/dqd = -Minv*dc/dqd, dqdd/dq = -Minv*dc/dq --- remember Minv is a symmetric sym UPPER matrix
    matMultSym<T,NUM_POS,2*NUM_POS,NUM_POS>(dqddk,NUM_POS,&(currKnot->Minv[0]),NUM_POS,s_cdu,NUM_POS,static_cast<T>(-1));
}

template <typename T, bool MPC_MODE = false, bool VEL_DAMPING = true, bool DENSE_MINV = false>
void forwardDynamics(knot<T> *currKnot, T *s_fext = nullptr){
    forwardDynamics<T,MPC_MODE,VEL_DAMPING,DENSE_MINV>(currKnot->qdd,currKnot->u,currKnot->qd,currKnot->q,currKnot->Minv,s_fext);
}
