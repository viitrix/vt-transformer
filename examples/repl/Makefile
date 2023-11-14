.PHONY: all

FLAGS = -Wall -O3 -D_USING_DEVICE_CUDA_ -DUSING_DEVICE_DNNL_ 
INC = -I../../install/include \
	  -I/usr/local/cuda/include \
	  -I/home/teaonly/opt/nccl/include \
	  -I/home/teaonly/opt/cudnn/include 
LINK = -L../../install/lib \
	   -L/usr/local/cuda/lib64 \
       -L/home/teaonly/opt/nccl/lib \
	   -L/home/teaonly/opt/cudnn/lib \
	   -L/usr/lib/x86_64-linux-gnu \
	   -ltensortype -lcuda_kernels -lnccl -lcudnn -lcudart -lcublas -lcublasLt 

all: repl 

repl: repl.cpp
	g++ $(FLAGS) -o $@ $< $(INC) $(LINK)

run: repl 
	LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/home/teaonly/opt/nccl/lib:/home/teaonly/opt/dnnl/lib:/home/teaonly/opt/cudnn/lib:/usr/lib/x86_64-linux-gnu ./repl

clean:
	rm -f repl 
