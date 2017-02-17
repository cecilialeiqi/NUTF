CXX ?= g++
CFLAGS = -O3  
#CFLAGS = -g  
LIBS = ./eigen/
all:  train  simulate_sparse
train:
		$(CXX) $(CFLAGS) -std=c++0x -I $(LIBS) -lgomp  check_in.cpp  -o train
		./train test.txt 14 6 3 5 5
simulate:
		$(CXX) $(CFLAGS) -std=c++0x -I $(LIBS) -lgomp  simulate_sparse.cpp  -o simulate_sparse
		./simulate_sparse 10000 500 200 20 10 20

