// auxiliary (serial) functions for supporting tiled Cholesky factorization.
// written by Peter Strazdins, Feb 20 for COMP4300/8300 Assignment 1 
// v1.0 12/05/20

#include <stdio.h>
#include <stdlib.h>
#include <math.h> //fabs()
#include <assert.h>
#ifdef USE_MKL
#include <mkl_cblas.h>
#else
#include <cblas.h>
#endif
#include <string.h> //memset()
#include "auxCholesky.h"

void **allocTileArray(size_t eltSize, int nT) {
  int i;
  char **A = (char **) malloc(nT*sizeof(void *));  
  A[0] = (char *) calloc(eltSize, nT*nT);
  for (i=1; i<nT; i++)
    A[i] = A[i-1] + nT*eltSize;
  return (void **) A;
} //allocTileArray()

void freeTileArray(void **A) {
  free(A[0]);
  free(A);
} //freeTileArray()


void printIntTile(int nT, int **a) {
  int i, j;
  for (i=0; i < nT; i++) {
    for (j=0; j < nT; j++) 
      printf(" %2d", a[i][j]); 
    printf("\n");
  }
} //printIntTile()

// printing out to 2 decimal places will detect most errors, but some will
// require greater precision before a wrong value becomes apparent
static char *doubleFmt = " %+5.2f";
static char *doubleSpFmt = "%6s"; // equivalent amount of spaces
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
      printf(doubleFmt, a[i*wT+j]); 
    printf("\n");
  }
} //printDoubleTile()


void initLowerPosDefTileArray(int seed, int N, int nT, int wT, double ***A) {
 int i, j;
 assert (nT == (N+wT-1)/wT);
 for (i = 0; i < nT; i++) {
    for (j = 0; j <= i; j++) {
      assert(A[i][j] == NULL); //expected from allocTileArray()
      A[i][j] = (double *) malloc(wT*wT*sizeof(double));
      assert(A[i][j] != NULL);
      initLowerPosDefTile(i, j, seed, N, wT, A[i][j]);
    }
  }
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
      if (iG < jG) //upper triangular element 
	       a[i*wT+j] = 0.0;
      else if (iG >= N) { //in a padded-out row from when N%wT > 0
	       assert (iG < ((N+wT-1)/wT)*wT);
      	 a[i*wT+j] = (double) (iG == jG); //pad out with the identity matrix
      } else {
      	if (!seedOnceOnly)
      	  srand(iG + jG + 29*seed); 

      	a[i*wT+j] = (2.0 * rand() / RAND_MAX) - 1.0;
      	assert (-1.0 <= a[i*wT+j] && a[i*wT+j] <= 1.0);
        
      	if (iG == jG)
      	  a[i*wT+j] = fabs(a[i*wT+j]) + diagBias(N);
      }
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
	    printf(doubleSpFmt, " ");
	  else
	    printf(doubleFmt, a[i0][j0][i*wT+j]); 
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


void printVec(char *name, double *x, int N) {
  int i;
  printf("%s:", name);
  for (i=0; i < N; i++) 
    printf(doubleFmt, x[i]);
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
	cblas_dgemv(CblasRowMajor, CblasNoTrans, wT, wT,
		    1.0, A[i][j], wT, &x[j*wT], 1, 1.0, &y[i*wT], 1); 
      else if (i == j)
	cblas_dsymv(CblasRowMajor, CblasLower, wT, 1.0, A[i][i], wT,
		    &x[j*wT], 1, 1.0, &y[i*wT], 1);
      else 
	cblas_dgemv(CblasRowMajor, CblasTrans, wT, wT, 
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
      cblas_dgemv(CblasRowMajor, CblasNoTrans, wT, wT,
		  -1.0, L[i][j], wT, &y[j*wT], 1, 1.0, yi, 1); 
    } //for (j...)
    
    cblas_daxpy(wT, 1.0, yi, 1, &y[i*wT], 1);  //y[i*wT..i*wT+wT-1] += yi
    //y[i*wT..i*wT+wT-1] = L[i][i]^-1 * y[i*wT..i*wT+wT-1]       
    cblas_dtrsv(CblasRowMajor, CblasLower, CblasNoTrans, CblasNonUnit, wT,
		  L[i][i], wT, &y[i*wT], 1);
  } //for (i...)
   
  // y = L^-T * y
  for (i=nT-1; i>=0; i--) {
    memset(yi, 0, sizeof(double) * wT); // yi = 0.0;
    for (j=nT-1; j > i; j--) { 
      // yi -= A[j][i]^T * y[j*wT..j*wT+wT-1] 
      cblas_dgemv(CblasRowMajor, CblasTrans, wT, wT,
		  -1.0, L[j][i], wT, &y[j*wT], 1, 1.0, yi, 1); 
    } //for (j...)
    cblas_daxpy(wT, 1.0, yi, 1, &y[i*wT], 1);  //y[i*wT..i*wT+wT-1] += yi
    //y[i*wT..i*wT+wT-1] = L[i][i]^-T * y[i*wT..i*wT+wT-1]       
    cblas_dtrsv(CblasRowMajor, CblasLower, CblasTrans, CblasNonUnit, wT,
		L[i][i], wT, &y[i*wT], 1);
  } //for (i...)
   
  free(yi);
} //triMatVecSolve() 
