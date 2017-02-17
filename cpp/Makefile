CXX ?= g++
CFLAGS = -O3  
#CFLAGS = -g  
LIBS = ./eigen/
all:  train  simulate_sparse
train:
		$(CXX) $(CFLAGS) -std=c++0x -I $(LIBS) -lgomp  check_in.cpp  -o train
simulate:
		$(CXX) $(CFLAGS) -std=c++0x -I $(LIBS) -lgomp  simulate_sparse.cpp  -o simulate

