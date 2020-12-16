// parallel functions for supporting tiled Cholesky factorization checking
// written by Peter Strazdins, May 20 for COMP4300/8300 Assignment 2
// v1.0 13/05/20

// initializes parameters for the other functions below;
// these may assume that this function has been called first.
void initCheckParams(int verbosity, int tuneParam);

//pre:  A stores the local tiles of a global symmetric matrix stored in the
//      lower triangular array of nT x nT tiles of size wT; 
//      x and y are replicated vectors of size nT*wT
//post: y = A*x                              
void triMatVecMultPar(int nT, int wT, int **ownerIdTile,
		      double ***A, double *x, double *y); 

//pre:  L stores the local tiles of a global lower triangular array of 
//      nT x nT tiles of size wT; 
//      y is a replicated vector of size nT*wT
//post: y = (L^-1)^T * (L^-1) * y;
void triMatVecSolvePar(int nT, int wT, int **ownerIdTile,
		       double ***L, double *y);
