.PHONY: all

all: install 

FLAGS = -Wall -O3  

install: corex_kernels.hpp
	mkdir -p ../../install/lib	
	cp corex_kernels.hpp ../../install/include/
	cp libcorex_kernels.a ../../install/lib/

clean:
	rm -f *.o *.a
