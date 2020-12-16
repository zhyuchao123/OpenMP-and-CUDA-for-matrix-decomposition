// parallel functions for supporting tiled Cholesky factorization checking
// template written by Peter Strazdins, May 20 for COMP4300/8300 Assignment 2
// v1.0 08/05/20

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h> //memset()
#include <omp.h>
#ifdef USE_MKL
#include <mkl_cblas.h>
#else
#include <cblas.h>
#endif

#include "checkCholesky.h"
#include "auxCholesky.h"

static int verbosity = 0;  // output verbosity parameter
static int tuneParam = 0;  // optional tuning parameter
static int id, nthreads;   // OpenMP parameters, thread id, number of threads
#pragma omp threadprivate(id)

void initCheckParams(int verbosity_, int tuneParam_) {
  tuneParam = tuneParam_; verbosity = verbosity_;
  nthreads = omp_get_max_threads();
  #pragma omp parallel
  id = omp_get_thread_num();
} //initCheckParams()


void triMatVecMultPar(int nT, int wT, int **ownerIdTile,
		      double ***A, double *x, double *y) {
  memset(y, 0, nT*wT*sizeof(double)); // y = 0.0;
  // y = A*x + y
  int i=0,j=0;
#pragma omp parallel default(shared) firstprivate(i,j)
  {  

    #pragma omp for collapse(2) schedule(dynamic,2) reduction(+:y[0:nT*wT])
    // collapse nested for loop to reduce branch keep y[0:nT*nT] atomic
    for (i=0; i<nT; i++) {
          for (j=0; j<nT; j++) { 
                // y[i*wT..i*wT+wT-1] += A[i][j] * x[j*wT..j*wT+wT-1] 
       
              if (i > j){
                {
                  cblas_dgemv(CblasRowMajor, CblasNoTrans, wT, wT,
                  1.0, A[i][j], wT, &x[j*wT], 1, 1.0, &y[i*wT], 1); 
                }

                }else if (i == j){
           
                  cblas_dsymv(CblasRowMajor, CblasLower, wT, 1.0, A[i][i], wT,
                  &x[j*wT], 1, 1.0, &y[i*wT], 1);
                
              }else{
          
                    cblas_dgemv(CblasRowMajor, CblasTrans, wT, wT, 
                  1.0, A[j][i], wT, &x[j*wT], 1, 1.0, &y[i*wT], 1); 
                }

            
              } //for (j...)
      } //for (i...)

  }
} //triMatVecMult()

void triMatVecMultPar_bak(int nT, int wT, int **ownerIdTile,
          double ***A, double *x, double *y) {
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

void triMatVecSolvePar(int nT, int wT, int **ownerIdTile, double ***L,
		       double *y) {
  double *yi;
  int i, j;
  yi = (double *) malloc(sizeof(double) * wT);
  
  // y = L^-1 * y
  
    for (i=0; i<nT; i++) {
      memset(yi, 0, sizeof(double) * wT); // yi = 0.0;
      // double *yi_temp;
      // yi_temp = (double *) malloc(sizeof(double) * wT);
      #pragma omp parallel for default(shared) firstprivate(i) private(j) reduction(-:yi[0:wT])
        for (j=0; j < i; j++) { 
          // yi -= L[i][j] * y[j*wT..j*wT+wT-1] 
          // printf("my id is %d\n",omp_get_thread_num());
          cblas_dgemv(CblasRowMajor, CblasNoTrans, wT, wT,
          -1.0, L[i][j], wT, &y[j*wT], 1, 1.0, yi, 1); 
          // yi_temp =yi;
        } //for (j...)
      
      //y[i*wT..i*wT+wT-1] += yi
      cblas_daxpy(wT, 1.0, yi, 1, &y[i*wT], 1);  

      //y[i*wT..i*wT+wT-1] = L[i][i]^-1 * y[i*wT..i*wT+wT-1]       
      cblas_dtrsv(CblasRowMajor, CblasLower, CblasNoTrans, CblasNonUnit, wT,
        L[i][i], wT, &y[i*wT], 1);
    } //for (i...)


  
   
  // y = L^-T * y
  for (i=nT-1; i>=0; i--) {
    memset(yi, 0, sizeof(double) * wT); // yi = 0.0;
    #pragma omp parallel for default(shared) firstprivate(i) private(j) reduction(-:yi[0:wT])
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


void triMatVecSolvePar_bak(int nT, int wT, int **ownerIdTile, double ***L,
           double *y) {
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