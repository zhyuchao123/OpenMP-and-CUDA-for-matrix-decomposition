// parallel functions for supporting tiled Cholesky factorization 
// template written by Peter Strazdins, May 20 for COMP4300/8300 Assignment 2
// v1.1 16/05/20

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <omp.h>
#ifdef USE_MKL
#include <mkl_cblas.h>
#else
#include <cblas.h>
#endif
//native Fortran subroutine
int dpotrf_(char *UpLo , int *N, double *A, int *ldA, int *info);
#include <string.h> //memset()

#include "auxCholesky.h"
#include "parCholesky.h"

static int randDist;       // use random distribution; 
static int blockDist;      // use block distribution; 
static int seed;           // random number generator seed        
static int P, Q;           // if -b or no -r, use a PxQ process grid
static int useExtra;       // if set, initOwners() and
                           // initLowerPosDefTileArrayOwner() may assume
                           // choleskyExtra() will be called 
static int verbosity = 0;  // output verbosity parameter          
static int tuneParam = 0;  // optional tuning parameter           
static int id, nthreads;   // OpenMP parameters, thread id, number of threads
#pragma omp threadprivate(id)

void initParParams(int randDist_, int seed_, int blockDist_, int P_, int Q_, 
		   int verbosity_, int tuneParam_, int useExtra_) {
  randDist = randDist_; seed = seed_; blockDist = blockDist_; P = P_, Q = Q_;
  tuneParam = tuneParam_; verbosity = verbosity_; useExtra = useExtra_;
  nthreads = omp_get_max_threads();
  #pragma omp parallel
  id = omp_get_thread_num();
} //initParParams();


void choleskyLoop(int nT, int wT, double ***A) {
  printf("the avaliable threads are %d\n", nthreads );
  int i, j, k;

  for (k=0; k<nT; k++) {
    int info = 0;
    dpotrf_("Upper", &wT, A[k][k], &wT, &info); //Cholesky factor A[k][k]
    if (info != 0)
      printf("%d: WARNING: dpotrf() failed: tile (%d,%d), diagonal %d=%.3f\n",
               0, k, k, info-1, A[k][k][(info-1)*(wT+1)]);
    // printf("seperate line----------------------\n");

    #pragma omp parallel num_threads(nthreads) 
    {
    #pragma omp for schedule(static,1)
    
        for (i=k+1; i<nT; i++) {
      //int myid = omp_get_thread_num();
      // printf("my id is %i\n",omp_get_thread_num());
      cblas_dtrsm(CblasRowMajor, CblasRight, CblasLower, CblasTrans,
                  CblasNonUnit, wT, wT, 1.0, A[k][k], wT, A[i][k], wT);

    } //for(i...)
    // }
  

    // #pragma omp parallel num_threads(nthreads) 
    // {
      #pragma omp for schedule(dynamic) private(j)
        for (i=k+1; i<nT; i++) {
        //#pragma omp for default(shared) schedule(dynamic) private(j)
        for (j=k+1; j<=i; j++) {
          // printf("my id is %i\n",omp_get_thread_num());

          if (i==j) // only update lower tri. proportion of A[i][i]
            cblas_dsyrk(CblasRowMajor, CblasLower, CblasNoTrans, wT, wT,
                  -1.0, A[i][k], wT, 1.0, A[i][i], wT);
          else
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, wT, wT, wT,
                  -1.0, A[i][k], wT, A[j][k], wT, 1.0, A[i][j], wT);
        } //for (j...)
      } //for (i...)}
    
    }
    
  } //for (k...)
} //choleskyLoop()


void initLowerPosDefTileArrayOwner(int **ownerIdTile, int seed, int N,
				   int nT, int wT, double ***A) {

  // initLowerPosDefTileArray(seed, N, nT, wT, A);
 int i, j;
 assert (nT == (N+wT-1)/wT);
 #pragma omp parallel private(i,j)
 {
    // printf("id %d Parallel success? %i\n", id, omp_in_parallel());
    for (i = 0; i < nT; i++) {
        for (j = 0; j <= i; j++) {
          if (id==ownerIdTile[i][j])
          {
            assert(A[i][j] == NULL); //expected from allocTileArray()
            A[i][j] = (double *) malloc(wT*wT*sizeof(double));
            assert(A[i][j] != NULL);
            initLowerPosDefTile(i, j, seed, N, wT, A[i][j]);
          }
        }
    }

 }

} //initLowerPosDefTileArrayOwner()


void choleskyRegion(int **ownerIdTile, int nT, int wT, double ***A) {
  int i, j, k;

  for (k=0; k<nT; k++) {
    #pragma omp parallel default(shared) private(i,j) firstprivate(k)                                                  
    { 

      // if (id==0)
      // {
      //   printf("id 0 Parallel success? %i\n", omp_in_parallel());
      // }
      if (id == ownerIdTile[k][k]) { //see above note on dpotrf_()
        int info = 0;
        dpotrf_("Upper", &wT, A[k][k], &wT, &info); //Cholesky factor A[k][k]
        if (info != 0)
  	       printf("%d: WARNING: dpotrf() failed: tile (%d,%d), element %d=%.3f\n",
  	       id, k, k, info, A[k][k][(info-1)*(wT+1)]);
        }// end if 


        #pragma omp barrier
        // #pragma omp flush
        for (i=k+1; i<nT; i++) {
          if (id == ownerIdTile[i][k]) { //A[i][k] = A[i][k] * A[k][k]^-T
            cblas_dtrsm(CblasRowMajor, CblasRight, CblasLower, CblasTrans, 
              CblasNonUnit, wT, wT, 1.0, A[k][k], wT, A[i][k], wT);
          }
        } //for(i...)

        #pragma omp barrier
        // #pragma omp flush

        for (i=k+1; i<nT; i++) {
          for (j=k+1; j<=i; j++) {
            if (id == ownerIdTile[i][j]) { // A[i,j] -=  A[i][k] * A[j][k]^T
              if (i==j) // only update lower tri. proportion of A[i][i]
                cblas_dsyrk(CblasRowMajor, CblasLower, CblasNoTrans, wT, wT, 
                -1.0, A[i][k], wT, 1.0, A[i][i], wT);
              else
                cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, wT, wT, wT, 
                    -1.0, A[i][k], wT, A[j][k], wT, 1.0, A[i][j], wT);
            }
          } //for (j...)
        } //for (i...)         
        #pragma omp barrier
        // #pragma omp flush

    }  

  } //for (k...)
} //choleskyRegion()


void choleskyExtra(int **ownerIdTile, int nT, int wT, double ***A) {
    int i, j, k;

    #pragma omp parallel firstprivate(i,j,k)

    {

      #pragma omp single 
      {

      for (k=0; k<nT; k++) {

        #pragma omp task untied depend(inout:A[k][k][0:wT*wT])
        {
          int info = 0;

          dpotrf_("Upper", &wT, A[k][k], &wT, &info); //Cholesky factor A[k][k]
          if (info != 0)
             printf("%d: WARNING: dpotrf() failed: tile (%d,%d), element %d=%.3f\n",
             id, k, k, info, A[k][k][(info-1)*(wT+1)]);
        }

        for(i=k+1;i<nT;i++){
          #pragma omp task untied depend(in:A[k][k][0:wT*wT]) depend(inout:A[i][k][0:wT*wT],A[i][i][0:wT*wT])
          {
            cblas_dtrsm(CblasRowMajor, CblasRight, CblasLower, CblasTrans, 
              CblasNonUnit, wT, wT, 1.0, A[k][k], wT, A[i][k], wT);
            cblas_dsyrk(CblasRowMajor, CblasLower, CblasNoTrans, wT, wT, 
                              -1.0, A[i][k], wT, 1.0, A[i][i], wT);
          }
        }
        // for (i = k+1; i < nT; i++)
        // {
        //       #pragma omp task untied depend(in:A[i][k][0:wT*wT]) depend(inout:)
        //   {
         
        //   }
        // }

        for (i=k+1; i<nT; i++) {
          for (j=k+1; j<i; j++) {
              #pragma omp task untied depend(in:A[i][k][0:wT*wT],A[j][k][0:wT*wT]) depend(inout:A[i][j][0:wT*wT])
              {
                // #pragma omp taskyield
                cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, wT, wT, wT, 
                    -1.0, A[i][k], wT, A[j][k], wT, 1.0, A[i][j], wT);

              }
       
          } //for (j...)


    }// end for i=k+1
  }// end for k

  }//end master
}// end parallel region

} //choleskyExtra()



// this probably need not be modified, unless for the extra algorithm
void initOwners(int **owner, int nT) {
  int i, j;
  int bi = nT/P, ri = nT%P, ci = 0, ni = 0;
  if (randDist)
    srand(seed);
  for (i=0; i < nT; i++) {
      int bj = nT/Q, rj = nT%Q, cj = 0, nj = 0;      
      for (j=0; j < nT; j++) {
        if (j > i)
  	       owner[i][j] = -1;
        else {
        	if (randDist)
        	  owner[i][j] = rand() % nthreads;
        	else if (blockDist)
        	  owner[i][j] = ni*Q + nj; //(i/(nT/P))*Q + j/(nT/Q) sometimes too big
        	else 
        	  owner[i][j] = (i % P)*Q + j%Q; 
        	assert (0 <= owner[i][j]  &&  owner[i][j] < nthreads);
        }
      cj++;
      if (cj == bj + (rj>0))
  	     nj++, cj=0, rj--;
      } //for(j...) 
    ci++;
    if (ci == bi + (ri>0))
        ni++, ci = 0, ri--;
  } //for (i...)
} //initOwners()

