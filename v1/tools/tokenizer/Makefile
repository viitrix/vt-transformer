.PHONY: all

FLAGS = -Wall -O3 -DUSING_DEVICE_DNNL_ 
INC = -I../../install/include 

LINK = -L../../install/lib -L${SENTENCEPIECE_DIR}/lib -ltokenizer_combo -ltokenizers_c

all: tokenizer image 

tokenizer: tokenizer.cpp
	g++ $(FLAGS) -o $@ $< $(INC) $(LINK)

image: image.cpp
	g++ $(FLAGS) -o $@ $< $(INC) $(LINK)

test_image: image
	LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:../../install/lib ./image ./demo.jpg

clean:
	rm -f tokenizer image 
