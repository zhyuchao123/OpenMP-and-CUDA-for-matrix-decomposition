// vanilla tile BLAS functions tailored for lower Cholesky factorization
// written by Peter Strazdins, May 20 for COMP4300/8300 Assignment 2
// v1.0 08/05/20

#include <stdio.h>
#include <stdlib.h>
#include <math.h> //sqrt()
#include <assert.h>

#include <cublas_v2.h> 
#include <cusolverDn.h>

#include "semiparCholesky.h"
#include "auxCholesky.h"

static int verbosity = 0;  // output verbosity parameter
static int tuneParam = 0;  // optional tuning parameter

void initSemiParParams(int verbosity_, int tuneParam_) {
  tuneParam = tuneParam_; verbosity = verbosity_;
} //initSemiParParams();

// tile index macro; assumes wT is defined 
#define TIX(i, j) ((i)+(j)*wT)

void cudaHandleError(cudaError_t err, const char *file, int line) {
  if (err != cudaSuccess) {
    printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
    exit(2);
  }
}

// simple serial right-looking Cholesky factorization, 
__device__ __host__
void tile_dpotrfL(int wT, double *A, int *info) {
  // A is one tile size wT*wT
  int i, j, k;
  *info = 0;
  for (k=0; k<wT; k++) {
    double akk = A[TIX(k, k)];
    if (akk <= 0.0) { // A is not +ve definite
      *info = k+1;
      return;
    }
    akk = sqrt(akk);
    // diagnol
    A[TIX(k, k)] = akk; 
    for (i=k+1; i<wT; i++) 
      // lower triangle
      A[TIX(i, k)] /= akk; 
    for (j=k+1; j<wT; j++)
      for (i=j; i<wT; i++)
        // upper triangle
	       A[TIX(i, j)] -= A[TIX(i, k)] * A[TIX(j, k)]; 
       // upper triangle
  } //for (k...)    
} //tile_dpotrfL()



__device__ __host__
void tile_dtrsmRLTN(int wT, double *A, double *B) {
  int i, j, k;
  for (k=0; k<wT; k++) {
    double akk = A[TIX(k, k)];
    for (i=0; i<wT; i++)
      B[TIX(i, k)] /= akk;
    for (i=0; i<wT; i++)
      for (j=k+1; j<wT; j++)
	B[TIX(i, j)] -= B[TIX(i, k)] * A[TIX(j, k)];
  } //for (k...)
} //tile_dtrsmRLTN()


__device__ __host__
void tile_dgemmNT(int wT, double *A, double *B, double *C) {
  int i, j, k;
  for (i=0; i < wT; i++)
    for (j=0; j < wT; j++) {
      double cij = C[TIX(i, j)];
      for (k=0; k < wT; k++)
	       cij -= A[TIX(i, k)] * B[TIX(j, k)];
      C[TIX(i, j)] = cij;
    } //for(j...)
} //tile_dgemmNT()


__global__
void tile_dpotrfL_k(int wT, int nT, double ***AT, int kT, int *info,
		   double *Akk_info) {
  assert (kT < nT);
  assert (blockDim.x == 1  &&  blockDim.y == 1  &&  gridDim.y == 1);
  double *A =  AT[kT][kT];
  tile_dpotrfL(wT, A, info);
  if (*info != 0) //get the value of the diagonal element where it failed
    *Akk_info = A[(*info-1)*wT+*info-1]; 
}

__global__
void tile_dtrsmRLTN_k(int wT, int nT, double ***AT, int kT, int iT) {
  iT += blockIdx.x;
  assert (iT < nT  &&  kT < nT);
  assert (blockDim.x == 1  &&  blockDim.y == 1  &&  gridDim.y == 1);
  double *A = AT[kT][kT], *B = AT[iT][kT];
  tile_dtrsmRLTN(wT, A, B);
}


__global__
void tile_dgemmNT_k(int wT, int nT, double ***AT, int kT, int iT, int jT) {
  iT += blockIdx.x;
  assert (iT < nT  &&  kT < nT  &&  jT < nT);
  assert (blockDim.x == 1  &&  blockDim.y == 1  &&  gridDim.y == 1);
  double *A = AT[iT][kT], *B = AT[jT][kT], *C = AT[iT][jT];
  tile_dgemmNT(wT, A, B, C);
}


void check_dpotrfError(int k, int *info_d, double *Akk_info_d) {
  int info = 0;
  HANDLE_ERROR( cudaMemcpy(&info, info_d, sizeof(int),
			   cudaMemcpyDeviceToHost) );
  if (info != 0) {
    double AkkDiag;
    HANDLE_ERROR( cudaMemcpy(&AkkDiag, Akk_info_d, sizeof(double),
			     cudaMemcpyDeviceToHost) );
    printf("%d: WARNING: dpotrf() failed: tile (%d,%d), element %d=%.3e\n",
	   0, k, k, info, AkkDiag);
  }
} //check_dpotrfError()

void cholesky1x1Block(int nT, int wT, double ***A) {
  int i, j, k;
  int bF = (tuneParam == 0)? nT: tuneParam; //tile blocking factor
  int *info_d; double *Akk_info_d;
  HANDLE_ERROR( cudaMalloc(&info_d, sizeof(int)) );
  HANDLE_ERROR( cudaMalloc(&Akk_info_d, sizeof(double)) );

  for (k=0; k<nT; k++) {
    //printf("calling potrf int, k=%d\n", k);
    tile_dpotrfL_k<<<1,1>>> (wT, nT, A, k, info_d, Akk_info_d);
    check_dpotrfError(k, info_d, Akk_info_d);
    for (i=k+1; i<nT; i+=bF) {
      int nBlks = (i+bF > nT)? nT-i: bF; // get the small
      //printf("calling trsm, k=%d, i=%d\n", k, i);
      tile_dtrsmRLTN_k <<<nBlks,1>>> (wT, nT, A, k, i);
    } //for(i...)
    for (j=k+1; j<nT; j++) {
      for (i=j; i<nT; i+=bF) {
	//printf("calling dgemm, k=%d, i=%d j=%d\n", k, i, j);
      	int nBlks = (i+bF > nT)? nT-i: bF;
      	tile_dgemmNT_k <<<nBlks,1>>> (wT, nT, A, k, i, j);
      } //for (j...)
   } //for (i...)                                                              
  } //for (k...)
  
  HANDLE_ERROR( cudaFree(info_d) );
  HANDLE_ERROR( cudaFree(Akk_info_d) );
} //cholesky1x1Block()
 

//************************ CUBLAS/CUSOLVE AREA **************************/


//col-major indexing of an array of leading dimension N
#define BIX(i,j) ((i)+(j)*N)


#define HANDLE_CUBLAS_ERROR( err ) (cublasHandleError( err, __FILE__, __LINE__))
#define HANDLE_CUSOLVER_ERROR( err ) (cusolverHandleError( err, __FILE__, __LINE__))

void cublasHandleError(cublasStatus_t err, const char *file, int line) {
  if (err != CUBLAS_STATUS_SUCCESS) {
    printf("CUBLAS error %d in %s at line %d\n", err, file, line);
    exit(2);
  }
}

void cusolverHandleError(cusolverStatus_t err, const char *file, int line) {
  if (err != CUSOLVER_STATUS_SUCCESS) {
    printf("CUBLAS error %d in %s at line %d\n", err, file, line);
    exit(2);
  }
}


void choleskyBlockCUBLAS(int N, double *A, int w) {
  int k;
  cublasHandle_t handleCuB; cusolverDnHandle_t handleCuS;
  double dOne = 1.0, dminusOne = -1.0;
  int *info_d; 
  int workSize; double *workSpace;
  
  HANDLE_CUBLAS_ERROR( cublasCreate(&handleCuB) );
  HANDLE_CUSOLVER_ERROR( cusolverDnCreate(&handleCuS) );
  HANDLE_ERROR( cudaMalloc(&info_d, sizeof(int)) );
  
  for (k=0; k<N; k+=w) {
    int info = 0;
    int wk = (k+w <= N)? w: N-k;
    // why a C++-based library makes the caller allocate its own workspaces
    // is beyond me...
    HANDLE_CUSOLVER_ERROR (cusolverDnDpotrf_bufferSize(handleCuS,
						       CUBLAS_FILL_MODE_LOWER,
						       N, &A[BIX(k,k)], N,
						       &workSize) );
    HANDLE_ERROR( cudaMalloc(&workSpace, workSize*sizeof(double)) );  
    HANDLE_CUSOLVER_ERROR( cusolverDnDpotrf(handleCuS,
					    CUBLAS_FILL_MODE_LOWER,
					    wk, &A[BIX(k,k)], N, workSpace,
					    workSize, info_d) );
    HANDLE_ERROR( cudaFree(workSpace) );
    HANDLE_ERROR( cudaMemcpy(&info, info_d, sizeof(int),
    			     cudaMemcpyDeviceToHost) );

    if (info != 0)
      printf("WARNING: dpotrf() failed: element (%d,%d)\n", k+info-1,k+info-1);

    HANDLE_CUBLAS_ERROR( cublasDtrsm(handleCuB, CUBLAS_SIDE_RIGHT,
				     CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T,
				     CUBLAS_DIAG_NON_UNIT, N-k-wk, wk, &dOne,
				     &A[BIX(k,k)], N, &A[BIX(k+wk,k)], N) );

    HANDLE_CUBLAS_ERROR( cublasDsyrk(handleCuB, CUBLAS_FILL_MODE_LOWER,
				     CUBLAS_OP_N, N-k-wk, wk, &dminusOne,
				     &A[BIX(k+wk,k)], N, &dOne,
				     &A[BIX(k+wk,k+wk)], N) );
  } //for (k...)      

  HANDLE_ERROR( cudaFree(info_d) );
  HANDLE_CUBLAS_ERROR( cublasDestroy(handleCuB) );
  HANDLE_CUSOLVER_ERROR( cusolverDnDestroy( handleCuS ));

} //choleskyBlockCUBLAS()

