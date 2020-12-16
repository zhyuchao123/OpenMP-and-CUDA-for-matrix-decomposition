// parallel tiled Cholesky solver test program
// written by Peter Strazdins, May 20 for COMP4300/8300 Assignment 2
// v1.0 20/05/20

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>   //getopt()
#include <math.h>     //fabs()
#include <sys/time.h> //gettimeofday()
#include <assert.h>
#include <string>   //std::string

extern "C" { //suppress C++ name mangling on C modules
#ifdef USE_MKL
#include <mkl_cblas.h>
#else
#include "cblas.h"
#endif
}
#include "auxCholesky.h"
#include "semiparCholesky.h"
#include "parCholesky.h"

#define USAGE   "testCholesky [-s s] [-v v] [-t t] [-B|-S|-O|-X] N [wT]"
#define DEFAULTS "s=v=t=0 wT=N"
#define CONSTRAINTS "N>=0, 1<=wT<=N"
#define OPTCHARS "s:v:t:BSOX"

static int N;                  // matrix size
static int wT = 0;             // tile width
static int seed = 0;           // s, above; random number generator seed
static int tuneParam = 0;      // t, above; optional tuning parameter
static int verbosity = 0;      // v, above; output verbosity parameter
                               // default: use wT x wT thread blocks
static int useCublas = 0;      // if set, use CBLAS algorithm (-B)
static int useSingleCore = 0;  // if set, use single core thread blocks (-S)
static int useOptMM = 0;       // if set, use optimized DGEMM kernel
static int useExtra = 0;       // if set, use `extra' algorithm


// print a usage message for this program and exit with a status of 1
void usage(std::string msg) {
  printf("testCholesky: %s\n", msg.c_str());
  printf("\tusage: %s\n\tdefault values: %s\n", USAGE, DEFAULTS);
  fflush(stdout);
  exit(2);
}

void getArgs(int argc, char *argv[]) {
  extern char *optarg; // points to option argument (for -p option)
  extern int optind;   // index of last option parsed by getopt()
  extern int opterr;
  char optchar;        // option character returned my getopt()
  opterr = 0;          // suppress getopt() error message for invalid option
  while ((optchar = getopt(argc, argv, OPTCHARS)) != -1) {
    // extract next option from the command line     
    switch (optchar) {
    case 'B':
      useCublas = 1;
      break;
    case 'S':
      useSingleCore = 1;
      break;
    case 'O':
      useOptMM = 1;
      break;
    case 'X':
      useExtra = 1;
      break;
    case 's':
      if (sscanf(optarg, "%d", &seed) != 1) // invalid integer 
	usage("bad value for s");
      break;
    case 'v':
      if (sscanf(optarg, "%d", &verbosity) != 1) // invalid integer 
	usage("bad value for v");
      break;
    case 't':
      if (sscanf(optarg, "%d", &tuneParam) != 1) // invalid integer 
	usage("bad value for t");
      break;
    default:
      usage("unknown option");
      break;
    } //switch 
   } //while

  if (useExtra)
    useOptMM = useSingleCore = useCublas = 0;
  else if (useOptMM)
    useSingleCore = useCublas = 0;
  else if (useSingleCore)
    useCublas = 0;

  if (optind < argc) {
    if (sscanf(argv[optind], "%d", &N) != 1)
      usage("bad value for N");
    if (N < 0)
      usage("N must be >= 0");
  } else
    usage("missing N");
  wT = N;
  if (optind+1 < argc) {
    if (sscanf(argv[optind+1], "%d", &wT) != 1) 
      usage("bad value for wT");
    if (wT < 1 || wT > N)
      usage("wT must be in range 1..N");
  }
} //getArgs()


void printTime(std::string stage, double nOps, std::string metric, double t) {  
  printf("%s time %.2es, rate %.2e = %.1f %s\n",
	 stage.c_str(), t, nOps / t, nOps / t, metric.c_str()); 
} //printTime()


#define MAX_RESID 1.0    // largest acceptable normalized residual
#define EPSILON 2.0E-16  // machine precision for double 

//return wall time in seconds
static double get_wtime() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return(1.0*tv.tv_sec + 1.0e-6*tv.tv_usec);
}


int main(int argc, char** argv) {
  double ***A; int nT; // A stores local tiles of global array's nTxnT tiles 
  char *Abuf; double *AeltBuf; //buffers for tile array A and its elements
  double *x, *y;       // replicated Nx1 vectors used for checking
  double t;            // for recording time
  int blockFactorCUBLAS;
  double ***A_d, *AeltBuf_d; char *Abuf_d; //device copies of A and Abuf
  
  getArgs(argc, argv);

  if (useCublas) // does not support tiled algorithms, must use a single tile
    nT = N, blockFactorCUBLAS = wT, wT=N;

  printf("GPU Cholesky factorization of a %dx%d matrix with %dx%d tiles\n", 
	 N, N, wT, wT);
  if (useExtra)
    printf("\t with extra alg\n");
  else if (useOptMM)
    printf("\t using opt. MM on %dx%d thread blocks\n", wT, wT);
  else if (useSingleCore)
    printf("\t using 1x1 thread blocks\n");
  else if (useCublas)
    printf("\t using CUDA BLAS with a blocking factor  = %d\n", 
	   blockFactorCUBLAS);
  else
    printf("\t using %dx%d thread blocks\n", wT, wT);
  if (seed != 0 || tuneParam != 0)
    printf("\tWith random seed %d, tuning parameter %d\n", seed, tuneParam);

  initSemiParParams(verbosity, tuneParam);
  initParParams(verbosity, tuneParam);
  setPrintDoublePrecision(2); //print doubles to 2 decimal places

  nT = (N + wT-1) / wT;
  int ABufSize = tileArrayBufSize(sizeof(double *), nT);
  int AeltSize = sizeof(double) * nT * wT * nT * wT;
  Abuf = (char *) malloc(ABufSize);  assert (Abuf != NULL);
  AeltBuf = (double *) malloc(AeltSize);  assert (AeltBuf != NULL);
  A = (double ***) Abuf;
  initTileArray(Abuf, AeltBuf, nT, wT, A);

  initLowerPosDefTileArray(seed, N, nT, wT, A);
  if (verbosity > 2)
    printLowerTileArray(nT, wT, A);

  x = (double *) malloc(nT*wT * sizeof(double));
  y = (double *) malloc(nT*wT * sizeof(double));
  assert (x != NULL && y != NULL);
  initVec(seed, x, nT*wT);
  if (verbosity > 1)
    printVec("x", x, nT*wT);

  triMatVecMult(nT, wT, A, x, y); // y = A*x
  if (verbosity > 1)
    printVec("y", y, nT*wT);
  
  HANDLE_ERROR( cudaMalloc(&Abuf_d, ABufSize) );
  HANDLE_ERROR( cudaMalloc(&AeltBuf_d, AeltSize) ); 
  char *Abuf_h = (char *) malloc(ABufSize); assert (Abuf_h != NULL);
  initTileArray(Abuf_d, AeltBuf_d, nT, wT, (double***) Abuf_h);
  A_d = (double ***) Abuf_d;
  
  t = get_wtime();
  HANDLE_ERROR( cudaMemcpy(Abuf_d, Abuf_h, ABufSize, cudaMemcpyHostToDevice) );
  HANDLE_ERROR( cudaMemcpy(AeltBuf_d, AeltBuf, AeltSize,
			   cudaMemcpyHostToDevice) );

  printTime("Copy to device", 1.0e-09 * AeltSize, "GB/s", get_wtime() - t); 
  
  t = get_wtime();
  if (useExtra)
    choleskyExtra(nT, wT, A_d);
  else if (useOptMM)
    choleskyTileBlock(useOptMM, nT, wT, A_d);
  else if (useCublas)
    choleskyBlockCUBLAS(N, AeltBuf_d, blockFactorCUBLAS);
  else if (useSingleCore)
    cholesky1x1Block(nT, wT, A_d);
  else
    choleskyTileBlock(0, nT, wT, A_d);
  HANDLE_ERROR( cudaDeviceSynchronize() );
  printTime("Factorization", 1.0e-09/3.0 * N*N*N, "GFLOPS", get_wtime() - t);
    
  t = get_wtime();
  HANDLE_ERROR( cudaMemcpy(AeltBuf, AeltBuf_d, AeltSize,
  			   cudaMemcpyDeviceToHost) );
  printTime("Copy to host", 1.0e-09 * AeltSize, "GB/s", get_wtime() - t);
  if (verbosity > 1)
    printLowerTileArray(nT, wT, A);    

  triMatVecSolve(nT, wT, A, y); // y = A^-1*y

  if (verbosity > 1)
    printVec("x'", y, nT*wT);
  if (verbosity > 1)
    printVec(" x", x, nT*wT);
  { // y should now be ~= x
    double resid, normX = fabs(x[cblas_idamax(N, x, 1)]), normX_;
    cblas_daxpy(N, -1.0, y, 1, x, 1); //x = x - y, should now be ~= 0
    normX_ = fabs(x[cblas_idamax(N, x, 1)]);
    resid = normX_ / (getNrmA(N) * normX * EPSILON);
    printf("%sed residual check: %1.2e Norms=%1.2e %1.2e\n",
	   (resid > MAX_RESID || resid != resid /*true for +/-NaN*/)? 
	   "FAIL": "PASS", resid, normX, normX_);
  }

  // free all data; note buffer overwrites may cause these to crash
  HANDLE_ERROR( cudaFree(Abuf_d) );
  HANDLE_ERROR( cudaFree(AeltBuf_d) ); 
  free(y); free(x);
  free(AeltBuf);
  free(Abuf); free(Abuf_h);

  return 0;
} //main()

