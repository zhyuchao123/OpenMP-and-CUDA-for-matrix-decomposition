// parallel tiled Cholesky solver test program
// written by Peter Strazdins, May 20 for COMP4300/8300 Assignment 2
// v1.0 13/05/20

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h> //getopt()
#include <math.h>   //fabs()
#include <assert.h>
#include <omp.h>
#ifdef USE_MKL
#include <mkl_cblas.h>
#else
#include <cblas.h>
#endif   
#include "auxCholesky.h"
#include "checkCholesky.h"
#include "parCholesky.h"

#define USAGE   "OMP_NUN_THREADS=p testCholesky [-r|-b] [-R|-X] [-C] [-s s] [-p P] [-v v] [-t t] N [wT]"
#define DEFAULTS "P=1 s=v=t=0 wT=N"
#define CONSTRAINTS "1<=P<=p, N>=0, 1<=wT<=N"
#define OPTCHARS "rbp:s:v:t:RXC"

static int N;                  // matrix size
static int wT = 0;             // tile width
static int randDist = 0;       // use a random distribution; set if -r given
static int seed = 0;           // s, above; random number generator seed
static int blockDist = 0;      // use a block distribution; set if -b given
                               // use a cyclic distribution, if neither -r, -b
static int nthreads;           // =p, above
static int P=1, Q;             // use a PxQ logical process grid, Q = p/P
                               // (not applicable for the random distribution)
static int tuneParam = 0;      // t, above; optional tuning parameter
static int useRegion = 0;  // if set, use parallel region-based algorithm
static int useExtra = 0;       // if set, use `extra' algorithm
static int parCheck = 0;       // if set, parallelize checking functions
static int verbosity = 0;      // v, above; output verbosity parameter



// print a usage message for this program and exit with a status of 1
void usage(char *msg) {
  printf("testCholesky: %s\n", msg);
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
    case 'r':
      randDist = 1;
      break;
    case 'b':
      blockDist = 1;
      break;
    case 'R':
      useRegion = 1;
      break;
    case 'X':
      useExtra = 1;
      break;
    case 'C':
      parCheck = 1;
      break;
    case 'p':
      if (sscanf(optarg, "%d", &P) != 1) // invalid integer 
	usage("bad value for P");
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

  if (!randDist && (P <= 0 || nthreads < P))
    usage("P must be in range 1..p");
  Q = nthreads / P;

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


void printTime(char * stage, double gflops, double t) {  
  printf("%s time %.2es, GFLOPs rate=%.2e (per thread %.2e)\n",
	 stage, t, gflops / t,  gflops / t / nthreads); 
} //printTime()


#define MAX_RESID 1      // largest acceptable normalized residual
#define EPSILON 2.0E-16  // machine precision for double 

#define MKL_NUM_THREADS "MKL_NUM_THREADS"
#define OMP_NUM_THREADS "OMP_NUM_THREADS"

int main(int argc, char** argv) {
  double ***A; int nT; // A stores local tiles of global array's nTxnT tiles 
  int **ownerIdTile;         // ownerIdTile[i][j] give id of tile (i,j)
  double *x, *y;       // replicated Nx1 vectors used for checking
  double t;            // for recording time
  int i, j;
  
  nthreads = omp_get_max_threads(); 
  getArgs(argc, argv);
    
  printf("Cholesky factor of a %dx%d matrix with %dx%d tiles ", N, N, wT, wT);
  if (!useExtra && !useRegion)
    printf("scheduled by OpenMP over %d threads\n", nthreads);
  else if (randDist)
    printf("owned randomly over %d threads\n", nthreads);
  else
    printf("%s owned over %dx%d threads\n",
	   blockDist? "blockwise": "cyclically", P, Q);
  if (useExtra || parCheck)
    printf("\t with %s%s\n", 
	   useExtra? "extra alg, ": "", parCheck? "parallel checking": "");
  if (seed != 0 || tuneParam != 0)
    printf("\tWith random seed %d, tuning parameter %d\n", seed, tuneParam);
  if (getenv(MKL_NUM_THREADS) != NULL) {
    int ncores = atoi(getenv(MKL_NUM_THREADS));
    printf("Running MKL with %s=%d cores\n", MKL_NUM_THREADS, ncores);    
    if (nthreads > 1 && ncores > 1) 
      printf("WARNING: %s=%d is also set. May get strange results!\n",
	     OMP_NUM_THREADS, nthreads);
    nthreads *= ncores; 
    assert (nthreads > 0);
  }

  initParParams(randDist, seed, blockDist, P, Q, verbosity, tuneParam,
		useExtra);
  initCheckParams(verbosity, tuneParam);
  setPrintDoublePrecision(2); //print doubles to 2 decimal places
  
  nT = (N + wT-1) / wT;
  ownerIdTile = (int **) allocTileArray(sizeof(int), nT);
  if (useRegion || useExtra)
    initOwners(ownerIdTile, nT);
  if (verbosity > 0)
    printIntTile(nT, ownerIdTile);

  A = (double ***) allocTileArray(sizeof(double *), nT);
  if (useExtra || useRegion) 
    initLowerPosDefTileArrayOwner(ownerIdTile, seed, N, nT, wT, A);
  else
    initLowerPosDefTileArray(seed, N, nT, wT, A);
  if (verbosity > 2)
    printLowerTileArray(nT, wT, A);

  x = (double *) malloc(nT*wT * sizeof(double));
  y = (double *) malloc(nT*wT * sizeof(double));
  assert (x != NULL && y != NULL);
  initVec(seed, x, nT*wT);
  if (verbosity > 1)
    printVec("x", x, nT*wT);

  t = omp_get_wtime();
  if (parCheck)
    triMatVecMultPar(nT, wT, ownerIdTile, A, x, y); // y = A*x
  else
    triMatVecMult(nT, wT, A, x, y); // y = A*x
  printTime("Generate RHS", 1.0e-09 * 2.0 * N * N, omp_get_wtime() - t);
  if (verbosity > 1)
    printVec("y", y, nT*wT);
  
  t = omp_get_wtime();
  if (useExtra)
    choleskyExtra(ownerIdTile, nT, wT, A);
  else if (useRegion)
    choleskyRegion(ownerIdTile, nT, wT, A);
  else
    choleskyLoop(nT, wT, A);
  
  printTime("Factorization", 1.0e-09/3.0 * N * N * N, omp_get_wtime() - t);
  if (verbosity > 1)
    printLowerTileArray(nT, wT, A);    

  t = omp_get_wtime();
  if (parCheck)
    triMatVecSolvePar(nT, wT, ownerIdTile, A, y); // y = A^-1*y 
  else
    triMatVecSolve(nT, wT, A, y); // y = A^-1*y
  printTime("Backsolve", 1.0e-09 * 2.0 * N * N, omp_get_wtime() - t);

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
  free(y); free(x);
  for (i = 0; i < nT; i++)
    for (j = 0; j < nT; j++) 
      if (A[i][j] != NULL) 
	free(A[i][j]);
  freeTileArray((void**) A);
  freeTileArray((void**) ownerIdTile);

  return 0;
} //main()

