#include <stdio.h>
#include <omp.h>

int main(void) {
  // int i=1, j=2, k;

// #pragma omp parallel for default(shared) lastprivate(i)
  // for (k = 0; k < 4; k++) { 
  //   printf("Initial value in parallel of i %i and j %i\n", i, j);
  //   i = i+99;
  //   j = j+99;
  //   printf("Final value in parallel of i %i and j %i\n", i, j);
  // }
  int i=0,j=0;
  #pragma omp parallel for default(shared) collapse(2) schedule(dynamic)
  {

    for (i = 0; i < 10; i++)
  {
    /* code */
        for (j = 0; j < 10; j++ )
        {
          int id = omp_get_thread_num();
          printf("my id is %i\n", id);
        }

    
  }
  }
  printf(" Final value of i %i and j %i \n", i, j);
  return 0;
}
