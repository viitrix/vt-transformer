.PHONY: all

all: libdcu_kernels.a

libdcu_kernels.a: fill.o causal_mask.o quantize.o embed.o
	ar rcs $@ $? 

fill.o: fill.cpp
	hipcc -c -o $@ $<

causal_mask.o: causal_mask.cpp
	hipcc -c -o $@ $<

quantize.o: quantize.cpp
	hipcc -c -o $@ $<

embed.o: embed.cpp
	hipcc -c -o $@ $<

install: libdcu_kernels.a dcu_kernels.hpp
	cp libdcu_kernels.a ../../install/lib/
	cp dcu_kernels.hpp ../../install/include/

clean:
	rm -f *.o *.a
