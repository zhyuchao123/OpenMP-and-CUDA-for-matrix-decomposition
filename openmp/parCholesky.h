// parallel functions for supporting tiled Cholesky factorization 
// written by Peter Strazdins, May 20 for COMP4300/8300 Assignment 2
// v1.0 13/05/20

// initializes parameters for the other functions below; 
// these may assume that this function has been called first.
void initParParams(int randDist, int seed, int blockDist, int P, int Q,
		   int verbosity, int tuneParam, int useExtra);

// sets ownerIdTile[nT][nT] to the id of the tiles of a lower
// triangular matrix (upper triangular elements should be set to -1),
// according to a random (seeded by seed), if randDist was set,
// block, if blockDist was set, or otherwise a cyclic distribution. 
// For block and cyclic, a PxQ process grid is used, where p = nthreads / P;
// the ordering is row-major with respect to id.
// If useExtra was set, the distribution for choleskyExtra() should be set;
// this may be different (or the same) as the above distributions.
// Note: assumes useRegion or useExtra was set
void initOwners(int **ownerIdTile, int nT);

// OpenMP loop implementation, OpenMP schedules tile updates
void choleskyLoop(int nT, int wT, double ***A);

// in the following, ownerIdTile[nT][nT] contains the OpenMP id of the 
// `owners' of the nT x nT tiles of A, which are each of size wT x wT. 
// They may assume initOwners(ownerIdTile, nT) has been called 
// to set ownerIdTile[nT][nT].

// initializes A[][] as per initLowerPosDefTileArray() but in parallel
// with A[i][j] allocated and initialized on thread ownerIdTile[i][j]
void initLowerPosDefTileArrayOwner(int **ownerIdTile, int seed, int N,
				   int nT, int wT, double ***A); 

// parallel region implementation, ownerIdTile[][] determines tile updates
void choleskyRegion(int **ownerIdTile, int nT, int wT, double ***A);

// extra implementation, ownerIdTile[][] may determine tile updates
void choleskyExtra(int **ownerIdTile, int nT, int wT, double ***A);
