// GPU parallel functions for supporting tiled Cholesky factorization 
// written by Peter Strazdins, May 20 for COMP4300/8300 Assignment 2
// v1.0 21/05/20

// initializes parameters for the other functions below; 
// these may assume that this function has been called first.
void initParParams(int verbosity, int tuneParam);

// the following assume that A is a tiled array of doubles, allocated on the
// device, and perform a cholesky factorization on the device

// uses w x w thread blocks, where wT is a multiple of w, e.g. wT = w;
// if useOptMM, use an optimized DGEMM kernel
void choleskyTileBlock(int useOptMM, int nT, int wT, double ***A);

// extra algorithm: applies an optimization not coverd in choleskyTileBlock()
void choleskyExtra(int nT, int wT, double ***A);
