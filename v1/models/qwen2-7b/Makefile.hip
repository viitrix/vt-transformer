.PHONY: all

FLAGS = -Wall -std=c++17 -O3 -D_USING_DEVICE_HIP_ 
INC = -I../../install/include \
	  -I${HIP_DIR}/include \

LINK = -L../../install/lib \
	   -L${HIP_DIR}/lib \
	   -L/usr/lib/x86_64-linux-gnu -lreadline \
	   -ltokenizer_combo -ltokenizers_c \
       -ltensortype -lhip_kernels -lgalaxyhip -lhipblas

all: chat

chat: chat.cpp memory.hpp ../../install/lib/libtensortype.a 
	g++ $(FLAGS) -o $@ $< $(INC) $(LINK)

run16: chat 
	LD_LIBRARY_PATH=${LD_LIBRARY_PATH}::${HIP_DIR}/lib:${VT_SOURCE}/install/lib ./chat ./inference_fp16.dag

clean:
	rm -f chat 
