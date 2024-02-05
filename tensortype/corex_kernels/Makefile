.PHONY: all

all: install 

FLAGS = -D__ILUVATAR__  --cuda-gpu-arch=${COREX_ARCH} -fPIE -Wall -O3 -I/usr/local/corex/include

libcorex_kernels.a: fill.o causal_mask.o
	ar rcs $@ $? 

install: corex_kernels.hpp libcorex_kernels.a
	mkdir -p ../../install/lib	
	cp corex_kernels.hpp ../../install/include/
	cp libcorex_kernels.a ../../install/lib/

fill.o: fill.cu
	clang $(FLAGS) -c -o $@ $<

causal_mask.o: causal_mask.cu
	clang $(FLAGS) -c -o $@ $<

clean:
	rm -f *.o *.a
