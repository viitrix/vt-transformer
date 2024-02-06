.PHONY: all

all: install 

FLAGS = -D__ILUVATAR__  --cuda-gpu-arch=${COREX_ARCH} -fPIE -Wall -O3 -I/usr/local/corex/include

libcorex_kernels.a: fill.o causal_mask.o quantize.o \
					embed.o convert.o pointwise.o \
					rmsnorm.o rotary_embed.o gelu.o \
					silu_product.o transpose_0213.o add_bias.o \
					layernorm.o softmax.o sampling.o
	ar rcs $@ $? 

install: corex_kernels.hpp libcorex_kernels.a
	mkdir -p ../../install/lib	
	cp corex_kernels.hpp ../../install/include/
	cp libcorex_kernels.a ../../install/lib/

fill.o: fill.cu
	clang++ $(FLAGS) -c -o $@ $<

causal_mask.o: causal_mask.cu
	clang++ $(FLAGS) -c -o $@ $<

quantize.o: quantize.cu
	clang++ $(FLAGS) -c -o $@ $<

embed.o: embed.cu
	clang++ $(FLAGS) -c -o $@ $<

convert.o: convert.cu
	clang++ $(FLAGS) -c -o $@ $<

pointwise.o: pointwise.cu
	clang++ $(FLAGS) -c -o $@ $<

rmsnorm.o: rmsnorm.cu
	clang++ $(FLAGS) -c -o $@ $<

rotary_embed.o: rotary_embed.cu
	clang++ $(FLAGS) -c -o $@ $<

gelu.o: gelu.cu
	clang++ $(FLAGS) -c -o $@ $<

silu_product.o: silu_product.cu
	clang++ $(FLAGS) -c -o $@ $<

transpose_0213.o: transpose_0213.cu
	clang++ $(FLAGS) -c -o $@ $<

add_bias.o: add_bias.cu
	clang++ $(FLAGS) -c -o $@ $<

layernorm.o: layernorm.cu
	clang++ $(FLAGS) -c -o $@ $<

softmax.o: softmax.cu
	clang++ $(FLAGS) -c -o $@ $<

sampling.o: sampling.cu
	clang++ $(FLAGS) -c -o $@ $<

clean:
	rm -f *.o *.a
