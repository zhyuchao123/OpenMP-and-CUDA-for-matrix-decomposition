// parallel functions for supporting tiled Cholesky factorization 
// template written by Peter Strazdins, May 20 for COMP4300/8300 Assignment 2
// v1.0 21/05/20

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "auxCholesky.h"
#include "parCholesky.h"
#include "semiparCholesky.h"

static int verbosity = 0;  // output verbosity parameter
static int tuneParam = 0;  // optional tuning parameter

void initParParams(int verbosity_, int tuneParam_) {
  tuneParam = tuneParam_; verbosity = verbosity_; 
} //initParParams();


// tile index macro; assumes wT is defined
#define TIX(i, j) ((i)+(j)*wT)
//############################################################ block

__device__
void tile_dpotrfL_block(int wT, double *A, int *info,int threadId) {
  // the first part divice blocksize wT
  // A is one tile size wT*wT
  int j, k;
  *info = 0;
  for (k=0; k<wT; k++) {
    double akk = A[TIX(k, k)];
    if (akk <= 0.0) { // A is not +ve definite
      *info = k+1;
      return;
    }
    akk = sqrt(akk);
    // diagnol
    if(k==threadId){
      A[TIX(k, k)] = akk; 
    }
    __syncthreads();

    if(threadId>k){
      A[TIX(threadId, k)] /= akk;
    }
    __syncthreads();

    for (j=k+1; j<wT; j++){
      if (threadId>=j)
      {
        A[TIX(threadId, j)] -= A[TIX(threadId, k)] * A[TIX(j, k)]; 
      }
      if(j==k+1){
        __syncthreads();
      }
    }
    // __syncthreads();
  } //for (k...)    
} //tile_dpotrfL_block()

__global__
void tile_dpotrfL_k_block(int wT, int nT, double ***AT, int kT, int *info,
       double *Akk_info) {
  // first kernel for blocksize wT
  int threadId = threadIdx.x;
  assert (kT < nT);
  // change 1 to wT
  assert (blockDim.x == wT  &&  blockDim.y == 1  &&  gridDim.y == 1);
  
  double *A=AT[kT][kT];
  // __shared__ *a_s[33*33];
  // a_s[TIX(threadIdx.x,threadIdx.y)] = A[TIX(threadIdx.x,threadIdx.y)]
  
  tile_dpotrfL_block(wT, A, info,threadId);

  if (*info != 0) //get the value of the diagonal element where it failed
    *Akk_info = A[(*info-1)*wT+*info-1]; 
}


__device__
void tile_dtrsmRLTN_block(int wT, double *A, double *B, int threadId) {
  // the second part for device blocksize wT
  int  j, k;
  for (k=0; k<wT; k++) {
    double akk = A[TIX(k, k)]; // get A 
    // for (i=0; i<wT; i++)
    //   B[TIX(i, k)] /= akk; // 
    // int index = TIX(threadId, k);
    B[TIX(threadId, k)] /= akk;
    __syncthreads();

    for (j=k+1; j<wT; j++)
       B[TIX(threadId, j)] -= B[TIX(threadId, k)] * A[TIX(j, k)];
    // __syncthreads();

  } //for (k...)
} //tile_dtrsmRLTN_block()

__global__ void tile_dtrsmRLTN_k_block(int wT, int nT, double ***AT, int kT, int iT) {
  // the second part for kernel blocksize wT
  iT += blockIdx.x; // 
  int threadId = threadIdx.x;
  assert (iT < nT  &&  kT < nT);
  assert (blockDim.x == wT  &&  blockDim.y == 1  &&  gridDim.y == 1);
  double *A = AT[kT][kT], *B = AT[iT][kT];
  
  tile_dtrsmRLTN_block(wT, A, B, threadId);
  // tile_dtrsmRLTN(wT,A,B);
}
//############################################################ block end
//############################################################ extra start

__global__
void tile_dpotrfL_k_extra(int wT, int nT, double ***AT, int kT, int *info,
       double *Akk_info) {
  // first kernel for blocksize wT
  
  // __shared__ double s_AT[32][32];
  __shared__ double s_AT[32][32];

  assert (kT < nT);
  // change 1 to wT
  assert (blockDim.x == wT  &&  blockDim.y == wT  &&  gridDim.y == 1);
  

  double *A=AT[kT][kT];
  // copy to shared memory

  int id = TIX(threadIdx.x,threadIdx.y);

  s_AT[threadIdx.x][threadIdx.y] = A[id];
    // s_AT[id] = A[id];

  __syncthreads();

  // call the function
  // tile_dpotrfL_extra(wT, (double *)A, info,(double **) s_AT);
//***************************
  int  k;
  *info = 0;
  for (k=0; k<wT; k++) {
    if (threadIdx.x==k && threadIdx.y==k)
    {
      double akk = s_AT[k][k];
      if (akk <= 0.0) { 
        *info = k+1;
        return;
      }
      akk = sqrt(akk);
    // diagnol
      s_AT[k][k] = akk; 
    }
    __syncthreads();


    if (threadIdx.x>k && threadIdx.y==k)
    {
      // __syncthreads();
      s_AT[threadIdx.x][k] /= s_AT[k][k]; 
    }
    __syncthreads();

    if (threadIdx.y>=k+1 && threadIdx.x>=threadIdx.y)
    {

      s_AT[threadIdx.x][threadIdx.y] -= s_AT[threadIdx.x][k] * s_AT[threadIdx.y][k]; 
    }

    __syncthreads();

       // upper triangle
  } //for (k...)    
  //*********************************
  // copy back to A
  A[id] = s_AT[threadIdx.x][threadIdx.y];
  // A[id] = s_AT[id];

  if (*info != 0) //get the value of the diagonal element where it failed
    *Akk_info = A[(*info-1)*wT+*info-1]; 
}




__global__
void tile_dtrsmRLTN_k_extra(int wT, int nT, double ***AT, int kT, int iT) {
  iT += blockIdx.x;
  assert (iT < nT  &&  kT < nT);
  assert (blockDim.x == wT  &&  blockDim.y == wT  &&  gridDim.y == 1);
  double *A = AT[kT][kT], *B = AT[iT][kT];
  int id = TIX(threadIdx.x,threadIdx.y);
  __shared__ double s_AT[32][34];
  __shared__ double s_BT[32][34];
  s_AT[threadIdx.y][threadIdx.x] = A[id];
  s_BT[threadIdx.y][threadIdx.x] = B[id];
  __syncthreads();
  
  int  k;
  for (k=0; k<wT; k++) {
    // double akk = A[TIX(k, k)];
    // for (i=0; i<wT; i++)
    //   B[TIX(i, k)] /= akk;
    if(threadIdx.y==k){
      // double akk = A[TIX(k, k)];
      double akk = s_AT[k][k];

      // B[TIX(threadIdx.x, k)] /= akk;
      s_BT[k][threadIdx.x] /= akk;
    }
    __syncthreads();
    // for (i=0; i<wT; i++)
    //   for (j=k+1; j<wT; j++)
    //     B[TIX(i, j)] -= B[TIX(i, k)] * A[TIX(j, k)];
    if (threadIdx.y>=k+1)
    {
        // B[TIX(threadIdx.x, threadIdx.y)] -= B[TIX(threadIdx.x, k)] * A[TIX(threadIdx.y, k)];
        s_BT[threadIdx.y][threadIdx.x] -= s_BT[k][threadIdx.x] * s_AT[k][threadIdx.y];

    }
    __syncthreads();

  } //for (k...)

   // A[id] = s_AT[threadIdx.y][threadIdx.x];
   B[id] =s_BT[threadIdx.y][threadIdx.x];
}





__global__
void tile_dgemmNT_k_extra(int wT, int nT, double ***AT, int kT, int iT, int jT) {
  // the thrid part for kernel
  iT += blockIdx.x;
  

  assert (iT < nT  &&  kT < nT  &&  jT < nT);
  assert (blockDim.x == wT  &&  blockDim.y == wT  &&  gridDim.y == 1);
  double *A = AT[iT][kT], *B = AT[jT][kT], *C = AT[iT][jT];
  int k;
  int id = TIX(threadIdx.x,threadIdx.y);

  __shared__ double s_AT[32][34];
  __shared__ double s_BT[32][34];
  s_AT[threadIdx.y][threadIdx.x] = A[id];
  s_BT[threadIdx.y][threadIdx.x] = B[id];
  __syncthreads();

  double cij = C[TIX(threadIdx.y, threadIdx.x)];
  for (k=0; k < wT; k++)
     cij -= s_AT[k][threadIdx.y] * s_BT[k][threadIdx.x];
  
  C[TIX(threadIdx.y, threadIdx.x)] = cij;

   // A[id] = s_AT[threadIdx.y][threadIdx.x];
   // B[id] =s_BT[threadIdx.y][threadIdx.x];
}

// ###############extra end##############################





//############# start full parallel######################
__device__
void tile_dpotrfL_full(int wT, double *A, int *info) {
  // A is one tile size wT*wT
  int  k;
  *info = 0;
  for (k=0; k<wT; k++) {
    if (threadIdx.x==k && threadIdx.y==k)
    {
      double akk = A[TIX(k, k)];
      if (akk <= 0.0) { // A is not +ve definite
        *info = k+1;
        return;
      }
      akk = sqrt(akk);
    // diagnol
      A[TIX(k, k)] = akk; 
    }
    __syncthreads();

    if (threadIdx.x>k && threadIdx.y==k)
    {
      // __syncthreads();
      A[TIX(threadIdx.x, k)] /= A[TIX(k, k)]; 
    }
    __syncthreads();


    if (threadIdx.y>=k+1 && threadIdx.x>=threadIdx.y)
    {

      A[TIX(threadIdx.x, threadIdx.y)] -= A[TIX(threadIdx.x, k)] * A[TIX(threadIdx.y, k)]; 
    }

    __syncthreads();

       // upper triangle
  } //for (k...)    
} //tile_dpotrfL()

__global__
void tile_dpotrfL_k_full(int wT, int nT, double ***AT, int kT, int *info,
       double *Akk_info) {
  assert (kT < nT);
  assert (blockDim.x == wT  &&  blockDim.y == wT  &&  gridDim.y == 1);
  double *A =  AT[kT][kT];
  tile_dpotrfL_full(wT, A, info);
  if (*info != 0) //get the value of the diagonal element where it failed
    *Akk_info = A[(*info-1)*wT+*info-1]; 
}

__device__ 
void tile_dtrsmRLTN_full(int wT, double *A, double *B) {
  // block size matching tile size
  int  k;
  for (k=0; k<wT; k++) {
    if(threadIdx.y==k){
      double akk = A[TIX(k, k)];
      B[TIX(threadIdx.x, k)] /= akk;
    }
    __syncthreads();

    if (threadIdx.y>=k+1)
    {
        B[TIX(threadIdx.x, threadIdx.y)] -= B[TIX(threadIdx.x, k)] * A[TIX(threadIdx.y, k)];
    }
    __syncthreads();

  } //for (k...)
} //tile_dtrsmRLTN()

__global__
void tile_dtrsmRLTN_k_full(int wT, int nT, double ***AT, int kT, int iT) {
    // block size matching tile size

  iT += blockIdx.x;
  assert (iT < nT  &&  kT < nT);
  assert (blockDim.x == wT  &&  blockDim.y == wT  &&  gridDim.y == 1);
  double *A = AT[kT][kT], *B = AT[iT][kT];
  tile_dtrsmRLTN_full(wT, A, B);
}



__device__
void tile_dgemmNT_full(int wT, double *A, double *B, double *C) {
  // the thrid part for device
  // block size matching tile size

  int k;
  double cij = C[TIX(threadIdx.y, threadIdx.x)];
  for (k=0; k < wT; k++)
     cij -= A[TIX(threadIdx.y, k)] * B[TIX(threadIdx.x, k)];
  C[TIX(threadIdx.y, threadIdx.x)] = cij;
  // __syncthreads();

} //tile_dgemmNT_block()

__global__
void tile_dgemmNT_k_full(int wT, int nT, double ***AT, int kT, int iT, int jT) {
  // block size matching tile size
  // the thrid part for kernel
  iT += blockIdx.x;
  

  assert (iT < nT  &&  kT < nT  &&  jT < nT);
  assert (blockDim.x == wT  &&  blockDim.y == wT  &&  gridDim.y == 1);
  double *A = AT[iT][kT], *B = AT[jT][kT], *C = AT[iT][jT];
  tile_dgemmNT_full(wT, A, B, C);
}
//#############for full parallel###########################################






void choleskyTileBlock(int useOptMM, int nT, int wT, double ***A) {
  // A is A_d that has been copied to device
  int j, k;
  int *info_d; double *Akk_info_d;
  HANDLE_ERROR( cudaMalloc(&info_d, sizeof(int)) );
  HANDLE_ERROR( cudaMalloc(&Akk_info_d, sizeof(double)) );
  dim3 blk(wT,wT);

  if (useOptMM==1)
  {
    for (k=0; k<nT; k++) {
    tile_dpotrfL_k_extra<<<1,blk>>> (wT, nT, A, k, info_d, Akk_info_d);
    // tile_dpotrfL_k_block <<<1,wT>>> (wT, nT, A, k, info_d, Akk_info_d);
    
    check_dpotrfError(k, info_d, Akk_info_d);
    
    // tile_dtrsmRLTN_k_full<<<nT-k-1,blk>>> (wT, nT, A, k, k+1);
    // tile_dtrsmRLTN_k_block <<<nT-k-1,wT>>> (wT, nT, A, k, k+1);
    tile_dtrsmRLTN_k_extra<<<nT-k-1,blk>>> (wT, nT, A, k, k+1);
    // dim3 blk(wT,wT);
    int deviceNum=1;
    for (j=k+1; j<nT; j+=deviceNum) {
      // tile_dgemmNT_k_full <<<nT-j,blk>>> (wT, nT, A, k, j, j);
      // HANDLE_ERROR(cudaDeviceSynchronize() );
      tile_dgemmNT_k_extra <<<nT-j,blk>>> (wT, nT, A, k, j, j);
    } //for (j...)
  }
}
  else{

        for (k=0; k<nT; k++) {
        tile_dpotrfL_k_full<<<1,blk>>> (wT, nT, A, k, info_d, Akk_info_d);
        // tile_dpotrfL_k_block <<<1,wT>>> (wT, nT, A, k, info_d, Akk_info_d);
        
        check_dpotrfError(k, info_d, Akk_info_d);
        
        tile_dtrsmRLTN_k_full<<<nT-k-1,blk>>> (wT, nT, A, k, k+1);
        // tile_dtrsmRLTN_k_block <<<nT-k-1,wT>>> (wT, nT, A, k, k+1);

        // dim3 blk(wT,wT);
        for (j=k+1; j<nT; j++) {
          tile_dgemmNT_k_full <<<nT-j,blk>>> (wT, nT, A, k, j, j);
        } //for (j...)
      } //for (k...)   /* code */


  }

  
  HANDLE_ERROR( cudaFree(info_d) );
  HANDLE_ERROR( cudaFree(Akk_info_d) );
} //choleskyyTileBlock()


void choleskyExtra(int nT, int wT, double ***A) {
  int s ,j,k;
  int *info_d; double *Akk_info_d;
  HANDLE_ERROR( cudaMalloc(&info_d, sizeof(int)) );
  HANDLE_ERROR( cudaMalloc(&Akk_info_d, sizeof(double)) );
 dim3 blk(wT,wT);
 // cudaStream_t *
 const int streamnum = nT;
 cudaStream_t streams[streamnum];

 for(s=0;s<streamnum;s++){
  cudaStreamCreate(&streams[s]);
 }

 for (k=0; k<nT; k++) {
    tile_dpotrfL_k_extra<<<1,blk,10,streams[0]>>> (wT, nT, A, k, info_d, Akk_info_d);
    // tile_dpotrfL_k_block <<<1,wT>>> (wT, nT, A, k, info_d, Akk_info_d);
    
    check_dpotrfError(k, info_d, Akk_info_d);
    
    // tile_dtrsmRLTN_k_full<<<nT-k-1,blk>>> (wT, nT, A, k, k+1);
    // tile_dtrsmRLTN_k_block <<<nT-k-1,wT,0,streams[0]>>> (wT, nT, A, k, k+1);
    tile_dtrsmRLTN_k_extra<<<nT-k-1,blk,10,streams[0]>>> (wT, nT, A, k, k+1);
    // dim3 blk(wT,wT);
    HANDLE_ERROR(cudaDeviceSynchronize());

    int deviceNum=1;
    for (j=k+1; j<nT; j+=deviceNum) {
      // tile_dgemmNT_k_full <<<nT-j,blk>>> (wT, nT, A, k, j, j);
      // HANDLE_ERROR(cudaDeviceSynchronize() );
      dim3 gid(nT-j,wT);
      // cudaEventRecord(events[], cudaStream_t stream = 0)

      tile_dgemmNT_k_extra <<<nT-j,blk,0,streams[(nT-j)%streamnum]>>> (wT, nT, A, k, j, j);

    } //for (j...)


    HANDLE_ERROR(cudaDeviceSynchronize());

  } //for (k...)   /* code */
  for(s=0;s<streamnum;s++){
  cudaStreamDestroy(streams[s]);
 }
  HANDLE_ERROR( cudaFree(info_d) );
  HANDLE_ERROR( cudaFree(Akk_info_d) );
} //choleskyExtra()


