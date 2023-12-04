.PHONY: all

all: libdcu_kernels.a

FLAGS = -Wall -O3 

libdcu_kernels.a: fill.o causal_mask.o quantize.o embed.o convert.o pointwise.o rmsnorm.o rotary_embed.o
	ar rcs $@ $? 

fill.o: fill.cpp
	hipcc $(FLAGS) -c -o $@ $<

causal_mask.o: causal_mask.cpp
	hipcc $(FLAGS) -c -o $@ $<

quantize.o: quantize.cpp
	hipcc $(FLAGS) -c -o $@ $<

embed.o: embed.cpp
	hipcc $(FLAGS) -c -o $@ $<

convert.o: convert.cpp
	hipcc $(FLAGS) -c -o $@ $<

pointwise.o: pointwise.cpp
	hipcc $(FLAGS) -c -o $@ $<

rmsnorm.o: rmsnorm.cpp
	hipcc $(FLAGS) -c -o $@ $<

rotary_embed.o: rotary_embed.cpp
	hipcc $(FLAGS) -c -o $@ $<

install: libdcu_kernels.a dcu_kernels.hpp
	cp libdcu_kernels.a ../../install/lib/
	cp dcu_kernels.hpp ../../install/include/

clean:
	rm -f *.o *.a
