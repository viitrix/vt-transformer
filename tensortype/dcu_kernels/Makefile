.PHONY: all

all: libdcu_kernels.a

libdcu_kernels.a: kr_fill.o
	ar rcs $@ $? 

kr_fill.o: kr_fill.cpp
	hipcc -c -o $@ $<

install: libdcu_kernels.a dcu_kernels.hpp
	cp libdcu_kernels.a ../../install/lib/
	cp dcu_kernels.hpp ../../install/include/

clean:
	rm -f *.o *.a
