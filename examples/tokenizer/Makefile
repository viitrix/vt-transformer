.PHONY: all

FLAGS = -Wall -O3 -D_USING_DEVICE_CUDA_ -DUSING_DEVICE_DNNL_ 
INC = -I../../install/include 

LINK = -L../../install/lib -ltokenizer_combo -ltokenizers_c

all: tokenizer 

tokenizer: tokenizer.cpp
	g++ $(FLAGS) -o $@ $< $(INC) $(LINK)

clean:
	rm -f tokenizer 
