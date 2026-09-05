.PHONY: all

FLAGS = -Wall -Wno-strict-aliasing -std=c++17 -O3 -D_USING_DEVICE_HIP_  
INC = -I../../install/include -I${HIP_DIR}/include
LINK = -L../../install/lib \
       -L${HIP_DIR}/lib \
       -ltensortype -lhip_kernels -lgalaxyhip -lhipblas

all: repl 

repl: repl.cpp
	g++ $(FLAGS) -o $@ $< $(INC) $(LINK)

run: repl 
	LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${HIP_DIR}/lib:${VT_SOURCE}/install/lib ./repl $(target)

clean:
	rm -f repl 
