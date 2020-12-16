# Makefile for COMP4300/8300 Assignment 2
# Peter Strazdins, RSCS ANU, 19 May 20
.SUFFIXES:
.PRECIOUS: %.o

HDRS=auxCholesky.h parCholesky.h semiparCholesky.h
OBJS=semiparCholesky.o auxCholesky.o parCholesky.o 
PROG=testCholesky

# on stugpu2,-lcusolve is in  /usr/local/cuda-10.1/lib64/
NVLIBS=-L/usr/local/cuda-10.1/lib64/ -lcusolver -L/usr/lib/x86_64-linux-gnu -lcublas -lcuda -lcudart

CC=gcc
NVCC=nvcc

CCFLAGS=-O3

//LDFLAGS=-llapack -lblas

# the following is for the rather sketchy OpenBLAS installation on stugpu2
BLASLIB=/usr/lib/x86_64-linux-gnu/openblas/
LDFLAGS=$(BLASLIB)/liblapack.so.3 $(BLASLIB)/libblas.so.3
# only g++ and not nvcc will link these .so.3 files

all: $(PROG)

.SUFFIXES:
.PRECIOUS: %.o

%: %.o $(OBJS)
	g++ -o $* $*.o $(OBJS) $(NVLIBS) $(LDFLAGS) -lm 
%.o: %.cu $(HDRS)
	$(NVCC) $(CCFLAGS) -c $*.cu

%.o: %.c $(HDRS)
	$(CC) -Wall $(CCFLAGS) -c $*.c

clean:
	rm -f *.o $(PROG)
