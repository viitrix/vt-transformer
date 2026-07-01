.PHONY: all

FLAGS = -Wall  -std=c++17 -O3 -Wno-maybe-uninitialized -Wno-strict-aliasing -Wno-uninitialized -fopenmp \
		-D_USING_DEVICE_HIP_

INC = -I. -I../install/include \
	  -I${HIP_DIR}/include 

all: libtensortype.a

libtensortype.a: context.o common.o dag.o  tensortype.o  host_tensor.o hip_tensor.o nn_operators.o nn_kvcache.o
	 ar rcs $@ $? 

tensortype.o: computing.hpp common.hpp tensortype.hpp tensortype.cpp host_tensor.hpp cuda_tensor.hpp
	g++ $(FLAGS) -c -o $@ $(INC) tensortype.cpp

context.o: computing.hpp common.hpp tensortype.hpp context.cpp 
	g++ $(FLAGS) -c -o $@ $(INC) context.cpp

common.o: common.hpp common.cpp 
	g++ $(FLAGS) -c -o $@ $(INC) common.cpp

dag.o: computing.hpp common.hpp tensortype.hpp dag.hpp dag.cpp
	g++ $(FLAGS) -c -o $@ $(INC) dag.cpp

nn_operators.o: computing.hpp common.hpp tensortype.hpp context.hpp nn_operators.cpp
	g++ $(FLAGS) -c -o $@ $(INC) nn_operators.cpp

nn_kvcache.o: computing.hpp common.hpp tensortype.hpp context.hpp nn_kvcache.cpp
	g++ $(FLAGS) -c -o $@ $(INC) nn_kvcache.cpp

host_tensor.o: computing.hpp common.hpp tensortype.hpp context.hpp host_tensor.hpp host_tensor.cpp
	g++ $(FLAGS) -c -o $@ $(INC) host_tensor.cpp

hip_tensor.o: computing.hpp common.hpp tensortype.hpp context.hpp hip_tensor.hpp hip_tensor.cpp
	g++ $(FLAGS) -c -o $@ $(INC) hip_tensor.cpp

install: libtensortype.a
	cp *.hpp ../install/include
	mkdir -p ../install/lib
	cp $< ../install/lib

clean:
	rm -f *.a *.o
