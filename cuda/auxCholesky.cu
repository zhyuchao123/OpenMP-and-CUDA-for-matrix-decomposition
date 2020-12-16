// auxiliary (serial) functions for supporting tiled Cholesky factorization.
// written by Peter Strazdins, May 20 for COMP4300/8300 Assignment 2 
// v1.0 20/05/20

#include <stdio.h>
#include <stdlib.h>
#include <math.h> //fabs()
#include <assert.h>
#include <string>   //std::string

extern "C" {
#ifdef USE_MKL
#include <mkl_cblas.h>
#else
#include "cblas.h"
#endif
//native Fortran subroutine
int dpotrf_(char *UpLo , int *N, double *A, int *ldA, int *info);
}
#include <string.h> //memset()

#include "auxCholesky.h"
#include "semiparCholesky.h" //tile_dgemm() etc

// tile index macro, ld is the `leading dimension'
#define TIX(i, j, ld) ((i)+(j)*(ld)) /*column major*/

int tileArrayBufSize(size_t eltSize, int nT) {
  return (nT*sizeof(void *) + eltSize*nT*nT);
} //tileArrayBufSize()

//note: Abuf, AeltBuf may be host or device pointers; A must be a host pointer
void initTileArray(char *Abuf, double *AeltBuf, int nT, int wT, double ***A) {
  int i, j;
  //use host memory to access tiles first
  A[0] = (double **) (((char *)A) + nT*sizeof(double *)); //points to 0th row
  for (i = 1; i < nT; i++)
    A[i] = A[i-1] + nT; //points to ith row
  for (i = 0; i < nT; i++) 
    for (j = 0; j < nT; j++) 
      A[i][j] = &AeltBuf[TIX(i, j, nT)*wT*wT];
  //now can set tile pointers on the device
  A[0] = (double **) (Abuf + nT*sizeof(double *));
  for (i = 1; i < nT; i++)
    A[i] = A[i-1] + nT;
} //initTileArray()

		    
// printing out to 2 decimal places will detect most errors, but some will
// require greater precision before a wrong value becomes apparent
static std::string doubleFmt = " %+5.2f";
static std::string doubleSpFmt = "%6s"; // equivalent amount of spaces
static char doubleFmtBuf[32], doubleSpFmtBuf[32];

void setPrintDoublePrecision(int decimalPlaces) {
  sprintf(doubleFmtBuf, " %%+%d.%df", decimalPlaces+3, decimalPlaces);
  sprintf(doubleSpFmtBuf, " %%%ds", decimalPlaces+4);  
  doubleFmt = doubleFmtBuf, doubleSpFmt = doubleSpFmtBuf;
} //setDoublePrecision()

void printDoubleTile(int wT, double *a) {
  int i, j;
  for (i=0; i < wT; i++) {
    for (j=0; j < wT; j++) 
      printf(doubleFmt.c_str(), a[TIX(i, j, wT)]); 
    printf("\n");
  }
} //printDoubleTile()


void initLowerPosDefTileArray(int seed, int N, int nT, int wT, double ***A) {
 int i, j;
 assert (nT == (N+wT-1)/wT);
 for (i = 0; i < nT; i++) 
    for (j = 0; j < nT; j++)
      if (j <= i)
	initLowerPosDefTile(i, j, seed, N, wT, A[i][j]);
      else
 	memset(A[i][j], 0, wT*wT*sizeof(double)); 
} //initLowerPosDefTileArray()


// return a diagonal bias (minimally) sufficient to make a matrix with
// random elements in (-1,+1) positive definite
static double diagBias(int N) {
  return (1.0 + N / 4.0); // the scaling factor 4.0 is empirically determined
}


// seeding the random number generators for every element ensures
// the value of every element depends only on its global index 
// and the seed (i.e. does not depend on wT); this is useful for debugging
// the (serial) tiled algorithm via printing the matrices.
// However seeding is costly, so avoid this when matrix is too large to print
#define N_MAX_PRINT_THRESHOLD 50

void initLowerPosDefTile(int i0, int j0, int seed, int N, int wT, double *a) {
  int i, j;
  int seedOnceOnly = (N > N_MAX_PRINT_THRESHOLD);
  if (seedOnceOnly)
    srand(i0 + j0 + 29*seed);     
  for (i=0; i < wT; i++) 
    for (j=0; j < wT; j++) {
      int iG = i0*wT+i, jG = j0*wT+j;
      double v;
      if (iG < jG) //upper triangular element 
	v = 0.0;
      else if (iG >= N) { //in a padded-out row from when N%wT > 0
	assert (iG < ((N+wT-1)/wT)*wT);
      	v = (double) (iG == jG); //pad out with the identity matrix
      } else {
	if (!seedOnceOnly)
	  srand(iG + jG + 29*seed); 
	v = (2.0 * rand() / RAND_MAX) - 1.0;
	assert (-1.0 <= v && v <= 1.0);
	if (iG == jG)
	  v = fabs(v) + diagBias(N);
      }
      a[TIX(i, j, wT)] = v;
    } //for (j...)
} //initLowerPosDefTile()


double getNrmA(int N) {
  // approximate max row sum of a matrix with each row 
  // having N elements random in [-1,+1] and the
  // diagonal with a bias of diagBias(N) added to
  return (N/2 + diagBias(N));
} //getNrmA()


void printLowerTileArray(int nT, int wT, double ***a) {
  int i0, j0, i, j;
  for (i0=0; i0 < nT; i0++) 
    for (i=0; i < wT; i++) {
      for (j0=0; j0 < nT; j0++) 
	for (j=0; j < wT; j++)
	  if (a[i0][j0] == NULL)
	    printf(doubleSpFmt.c_str(), " ");
	  else
	    printf(doubleFmt.c_str(), a[i0][j0][TIX(i, j, wT)]); 
      printf("\n");
    }
} //printLowerTileArray()


void initVec(int seed, double *x, int N) {
  int i;
  srand(seed);
  for (i=0; i < N; i++) {
    x[i] = (2.0 * rand() / RAND_MAX) - 1.0;
    assert (-1.0 <= x[i] && x[i] <= 1.0);
  }  
} //initVec()


void printVec(std::string name, double *x, int N) {
  int i;
  printf("%s:", name.c_str());
  for (i=0; i < N; i++) 
    printf(doubleFmt.c_str(), x[i]);
  printf("\n");
} //printVec()


void triMatVecMult(int nT, int wT, double ***A, double *x, double *y) {
  int i, j;  
  memset(y, 0, nT*wT*sizeof(double)); // y = 0.0;
  // y = A*x + y
  for (i=0; i<nT; i++) {
    for (j=0; j<nT; j++) { 
      // y[i*wT..i*wT+wT-1] += A[i][j] * x[j*wT..j*wT+wT-1] 
      if (i > j)
	cblas_dgemv(CblasColMajor, CblasNoTrans, wT, wT,
		    1.0, A[i][j], wT, &x[j*wT], 1, 1.0, &y[i*wT], 1); 
      else if (i == j)
	cblas_dsymv(CblasColMajor, CblasLower, wT, 1.0, A[i][i], wT,
		    &x[j*wT], 1, 1.0, &y[i*wT], 1);
      else 
	cblas_dgemv(CblasColMajor, CblasTrans, wT, wT, 
		    1.0, A[j][i], wT, &x[j*wT], 1, 1.0, &y[i*wT], 1); 
    } //for (j...)
  } //for (i...)
} //triMatVecMult()


void triMatVecSolve(int nT, int wT, double ***L, double *y) {
  double *yi;
  int i, j;
  yi = (double *) malloc(sizeof(double) * wT);
  
  // y = L^-1 * y
  for (i=0; i<nT; i++) {
    memset(yi, 0, sizeof(double) * wT); // yi = 0.0;
    for (j=0; j < i; j++) { 
      // yi -= A[i][j] * y[j*wT..j*wT+wT-1] 
      cblas_dgemv(CblasColMajor, CblasNoTrans, wT, wT,
		  -1.0, L[i][j], wT, &y[j*wT], 1, 1.0, yi, 1); 
    } //for (j...)
    
    cblas_daxpy(wT, 1.0, yi, 1, &y[i*wT], 1);  //y[i*wT..i*wT+wT-1] += yi
    //y[i*wT..i*wT+wT-1] = L[i][i]^-1 * y[i*wT..i*wT+wT-1]       
    cblas_dtrsv(CblasColMajor, CblasLower, CblasNoTrans, CblasNonUnit, wT,
		  L[i][i], wT, &y[i*wT], 1);
  } //for (i...)
   
  // y = L^-T * y
  for (i=nT-1; i>=0; i--) {
    memset(yi, 0, sizeof(double) * wT); // yi = 0.0;
    for (j=nT-1; j > i; j--) { 
      // yi -= A[j][i]^T * y[j*wT..j*wT+wT-1] 
      cblas_dgemv(CblasColMajor, CblasTrans, wT, wT,
		  -1.0, L[j][i], wT, &y[j*wT], 1, 1.0, yi, 1); 
    } //for (j...)
    cblas_daxpy(wT, 1.0, yi, 1, &y[i*wT], 1);  //y[i*wT..i*wT+wT-1] += yi
    //y[i*wT..i*wT+wT-1] = L[i][i]^-T * y[i*wT..i*wT+wT-1]       
    cblas_dtrsv(CblasColMajor, CblasLower, CblasTrans, CblasNonUnit, wT,
		L[i][i], wT, &y[i*wT], 1);
  } //for (i...)
   
  free(yi);
} //triMatVecSolve() 

static char Lower = 'L';
void choleskyTileCBLAS(int nT, int wT, double ***A) {
  int i, j, k;
  for (k=0; k<nT; k++) {
    int info = 0;
    dpotrf_(&Lower, &wT, A[k][k], &wT, &info); //Chol. factor A[k][k]
    if (info != 0)
      printf("WARNING: dpotrf() failed: tile (%d,%d), element %d=%.3f\n",
	     k, k, info-1, A[k][k][(info-1)*(wT+1)]);
    for (i=k+1; i<nT; i++) {
      cblas_dtrsm(CblasColMajor, CblasRight, CblasLower, CblasTrans,
                  CblasNonUnit, wT, wT, 1.0, A[k][k], wT, A[i][k], wT);
    } //for(i...)
    for (i=k+1; i<nT; i++) {
      for (j=k+1; j<=i; j++) {
	if (i==j) // only update lower tri. proportion of A[i][i]
	  cblas_dsyrk(CblasColMajor, CblasLower, CblasNoTrans, wT, wT,
		      -1.0, A[i][k], wT, 1.0, A[i][i], wT);
	else
	  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, wT, wT, wT,
		      -1.0, A[i][k], wT, A[j][k], wT, 1.0, A[i][j], wT);
      } //for (j...)
    } //for (i...)
  } //for (k...)
} //choleskyTileCBLAS()


//col-major indexing of an array of leading dimension N
#define BIX(i,j) ((i)+(j)*N) 

void choleskyBlockCBLAS(int N, double *A, int w) {
  int k;
  for (k=0; k<N; k+=w) {
    int info = 0;
    int wk = (k+w <= N)? w: N-k;
    dpotrf_(&Lower, &wk, &A[BIX(k,k)], &N, &info);//Chol factor A[k][k]
    if (info != 0)
      printf("WARNING: dpotrf() failed: element (%d,%d) = %.3f\n",
              k+info-1, k+info-1, A[BIX(k+info-1,k+info-1)]);
    cblas_dtrsm(CblasColMajor, CblasRight, CblasLower, CblasTrans,
		CblasNonUnit, N-k-wk, wk, 1.0, &A[BIX(k,k)], N, 
		&A[BIX(k+wk,k)], N);
    cblas_dsyrk(CblasColMajor, CblasLower, CblasNoTrans, N-k-wk, wk,
		-1.0, &A[BIX(k+wk,k)], N, 1.0, &A[BIX(k+wk,k+wk)], N);
  } //for (k...)
} //choleskyBlockCBLAS()


void choleskyTile(int nT, int wT, double ***A) {
  int i, j, k;
  for (k=0; k<nT; k++) {
    int info = 0;
    tile_dpotrfL(wT, A[k][k], &info); //Cholesky factor A[k][k]
    if (info != 0)
      printf("%d: WARNING: dpotrf() failed: tile (%d,%d), element %d=%.3e\n",
               0, k, k, info, A[k][k][(info-1)*wT+info-1]);
    for (i=k+1; i<nT; i++) {
      tile_dtrsmRLTN(wT, A[k][k], A[i][k]);
    } //for(i...)
    for (i=k+1; i<nT; i++) {
      for (j=k+1; j<=i; j++) {
	tile_dgemmNT(wT, A[i][k], A[j][k], A[i][j]);
      } //for (j...)
    } //for (i...)
  } //for (k...)
} //choleskyTile()


