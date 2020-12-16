// semi-parallel tile BLAS functions for lower Cholesky factorization
// written by Peter Strazdins, May 20 for COMP4300/8300 Assignment 2
// v1.0 20/05/20

#define HANDLE_ERROR( err ) (cudaHandleError( err, __FILE__, __LINE__ ))
void cudaHandleError(cudaError_t err, const char *file, int line);

// initializes parameters for the other functions below;
// these may assume that this function has been called first.
void initSemiParParams(int verbosity, int tuneParam);


//these functions operate on column-major tiled square (wT x wT) arrays

// pre: A is symmetric +ve definite matrix stored in the lower triangular half
// post: performs the cholesky factorization of A (lower triangular variant)
__device__ __host__
void tile_dpotrfL(int wT, double *A, int *info);

// A is a lower-triangular tile, non-unit triangular matrix
// B = B * A^-T (right-side, transpose variant)
__device__ __host__
void tile_dtrsmRLTN(int wT, double *A, double *B);

// C -= A*B^T (A non-transposed, B transposed variant)
__device__ __host__
void tile_dgemmNT(int wT, double *A, double *B, double *C);


// in the following, A has nT x nT tiles of size wT x wT 
// Cholesky factorize A[k][k]  (A[k][k] is lower-triangular)) 
__global__
void tile_dpotrfL_k(int wT, int nT, double ***A, int k, int *info,
		    double *Akk_info);

// A[i][k] = A[i][k] * A[k][k]^-T (A[k][k is lower tri., non-unit diagonal)
__global__
void tile_dtrsmRLTN_k(int wT, int nT, double ***A, int k, int i);

// A[i][j] -= A[i][k] * A[j][k]^T
__global__
void tile_dgemmNT_k(int wT, int nT, double ***A, int k, int i, int j);

// convenience function to check for error after calling dportf on tile [k][k]
void check_dpotrfError(int k, int *info_d, double *Akk_info_d);


// pre:  A is a tiled array of doubles, allocated on the device
// post: perform the cholesky factorization of A
void cholesky1x1Block(int nT, int wT, double ***A);

// pre: A is an N x N column-major matrix, w > 0
// post: performs a cholesky factorization using CUBLAS/CUSOLVE functions
//       using a blocking factor of w
void choleskyBlockCUBLAS(int N, double *A, int w);
