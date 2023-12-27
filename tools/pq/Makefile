.PHONY: all

FLAGS = -Wall -fopenmp -O3  
INC = -I/home/teaonly/opt/faiss/include -I./dkm -I../../install/include 
LINK = -L/home/teaonly/opt/faiss/lib \
	   -L/usr/lib/x86_64-linux-gnu \
	   -lgomp 

all: pq 

pq: pq.cpp dkm/dkm.hpp dkm/dkm_parallel.hpp
	g++ $(FLAGS) -o $@ $< $(INC) $(LINK)

clean:
	rm -f pq 
