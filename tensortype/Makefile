.PHONY: all

FLAGS = -Wall -O3 -Wno-maybe-uninitialized -fopenmp \
		-D_USING_DEVICE_DNNL_ \
		-D_USING_DEVICE_DCU_ \
		-D_USING_DEVICE_CUDA_

INC = -I. -I../install/include -I../install/include/kernels \
	  -I/usr/local/cuda/include \
	  -I/home/teaonly/opt/nccl/include \
	  -I/home/teaonly/opt/cudnn/include  

all: libtensortype.a

libtensortype.a: context.o tensortype.o host_tensor.o dnnl_tensor.o cuda_tensor.o dcu_tensor.o dag.o nn_operators.o nn_kvcache.o
	 ar rcs $@ $? 

context.o: context.hpp context.cpp
	g++ $(FLAGS) -c -o $@ $(INC) context.cpp

tensortype.o: computing.hpp tensortype.hpp tensortype.cpp
	g++ $(FLAGS) -c -o $@ $(INC) tensortype.cpp

host_tensor.o: computing.hpp tensortype.hpp host_tensor.hpp host_tensor.cpp
	g++ $(FLAGS) -c -o $@ $(INC) host_tensor.cpp

dnnl_tensor.o: computing.hpp tensortype.hpp host_tensor.hpp dnnl_tensor.hpp dnnl_tensor.cpp
	g++ $(FLAGS) -c -o $@ $(INC) dnnl_tensor.cpp

dcu_tensor.o: computing.hpp tensortype.hpp dcu_tensor.hpp dcu_tensor.cpp
	g++ $(FLAGS) -c -o $@ $(INC) dcu_tensor.cpp

cuda_tensor.o: computing.hpp tensortype.hpp cuda_tensor.hpp cuda_tensor.cpp
	g++ $(FLAGS) -c -o $@ $(INC) cuda_tensor.cpp

dag.o: tensortype.hpp dag.hpp dag.cpp
	g++ $(FLAGS) -c -o $@ $(INC) dag.cpp

nn_operators.o: computing.hpp tensortype.hpp dag.hpp nn_operators.cpp
	g++ $(FLAGS) -c -o $@ $(INC) nn_operators.cpp

nn_kvcache.o: computing.hpp tensortype.hpp dag.hpp nn_kvcache.cpp
	g++ $(FLAGS) -c -o $@ $(INC) nn_kvcache.cpp

install: libtensortype.a
	cp *.hpp ../install/include
	cp $< ../install/lib

clean:
	rm -f *.a *.o
