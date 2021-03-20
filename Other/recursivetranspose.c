/* %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   %%%%
   %%%% This program file is part of the book and course
   %%%% "Parallel Computing in Science and Engineering"
   %%%% by Victor Eijkhout, copyright 2013-2021
   %%%%
   %%%% Recursive transposition through subdivided communicators
   %%%%
   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
*/

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <mpi.h>

int intsqrt( int n ) {
  int root = sqrt(n);
  if (root*root!=n) {
    printf("Detected non-square grid of %d procs\n",n);
    MPI_Abort(1,MPI_COMM_WORLD);
  }
  return root;
}

/*
 * Gather all data to print out the current state of affairs
 */
void trace_current_state(MPI_Comm comm,double *dat,int size,int level) {
  int nprocs,procno;
  MPI_Comm_size(comm,&nprocs);
  MPI_Comm_rank(comm,&procno);
  if (procno==0) {
    double *global_data = (double*) malloc( nprocs*sizeof(double) );
    MPI_Gather( dat,1,MPI_DOUBLE, global_data,1,MPI_DOUBLE, 0,comm );
    int side = intsqrt(nprocs);
    printf("Level %d:\n",level);
    for (int row=0; row<side; row++) {
      for (int col=0; col<side; col++)
	printf("%4d",(int)( *(global_data+row*side+col) ));
      printf("\n");
    }
    free(global_data);
  } else {
    MPI_Gather( dat,1,MPI_DOUBLE, NULL,1,MPI_DOUBLE, 0,comm );
  }
}

/*
 * Transpose the data on the current communicator
 * which is a subcommunicator of MPI_COMM_WORLD
 */
void transpose_on_comm(MPI_Comm global_comm,MPI_Comm current_comm,double *A,int size,int level) {
  int mytid,ntids,gtid;

  trace_current_state(global_comm,A,size,level);

  MPI_Comm_size(current_comm,&ntids);
  MPI_Comm_rank(current_comm,&mytid);

  if (ntids==1) {
    // All done!
  } else {
    int grid_side = intsqrt(ntids);
    int // compute (row,col) coordinates of this process
      row = mytid/grid_side,
      col = mytid%grid_side;
    int  // compute top/bottom left/right block coordinates
      block_i = row/(grid_side/2),
      block_j = col/(grid_side/2);
    assert(block_i==0 || block_i==1);
    assert(block_j==0 || block_j==1);
    int // compute group 0,1,2,3
      group = 2*block_i + block_j;
    if (group==1 || group==2 ) {
      // off-diagonal blocks first exchange data
      int other;
      if (block_i==0)
	other = mytid + (grid_side/2)*grid_side - grid_side/2;
      else
	other = mytid - (grid_side/2)*grid_side + grid_side/2;
      //printf("%d = (%d,%d) in block (%d,%d) paired with %d\n",mytid,row,col,block_i,block_j,other);
      assert(other>=0 && other<ntids);
      double Atmp = *A;
      MPI_Sendrecv( &Atmp,1,MPI_DOUBLE,other,0, A,1,MPI_DOUBLE,other,0, current_comm,MPI_STATUS_IGNORE);
    }
    /*
     * Find subcommunicators for the 4 blocks and recurse
     */
    MPI_Comm new_comm;
    MPI_Comm_split(current_comm,group,mytid,&new_comm);
    transpose_on_comm(global_comm,new_comm,A,size,level+1);
  }
  return;
}

int main(int argc,char **argv) {
  MPI_Comm comm;

  MPI_Init(&argc,&argv);
  comm = MPI_COMM_WORLD;
  int procno;
  MPI_Comm_rank(comm,&procno);

  int nlocal_elements=1;
  double *A = (double*) malloc(nlocal_elements*sizeof(double));
  A[0] = procno;
  transpose_on_comm( /* global communicator:  */ comm,
		     /* current communicator: */ comm,
		     /* data:  */ A,nlocal_elements,
		     /* level: */ 0);

  MPI_Finalize();
  return 0;
}
