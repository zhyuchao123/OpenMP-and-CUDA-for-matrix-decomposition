# Makefile for COMP4300/8300 Assignment 2
# Peter Strazdins, RSCS ANU, May 20
.SUFFIXES:
.PRECIOUS: %.o

HDRS=auxCholesky.h checkCholesky.h parCholesky.h
OBJS=auxCholesky.o checkCholesky.o parCholesky.o
PROG=testCholesky

#MKL_LIB=parallel
MKL_LIB=sequential

HOST=$(shell hostname | awk -F- '{print $$1}')
ifeq ($(HOST),gadi) # must use MKL BLAS
CC=icc
CCFLAGS=-O3 -qopenmp -DUSE_MKL -mkl=$(MKL_LIB)
LDFLAGS=-qopenmp -mkl=$(MKL_LIB) 
else # assume OpenBLAS
CC=gcc
CCFLAGS=-O3 -fopenmp
LDFLAGS=-fopenmp -llapack -lblas 
endif

all: $(PROG)


%: %.o $(OBJS)
	$(CC) -o $* $*.o $(OBJS) $(LDFLAGS) -lm 
%.o: %.c $(HDRS)
	$(CC) -Wall $(CCFLAGS) -c $*.c
clean:
	rm -f *.o $(PROG)
