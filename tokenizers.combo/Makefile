.PHONY: all

FLAGS = -Wall -O3 -Wno-maybe-uninitialized

INC = -I. -I../install/include -I/public/home/ac56jm0dor/opt/sentencepiece/include

all: libtokenizer_combo.a

libtokenizer_combo.a: bloomz.o baichuan.o qwen.o
	 ar rcs $@ $?	

bloomz.o: bloomz.cpp
	g++ $(FLAGS) -c -o $@ $(INC) $<

baichuan.o: baichuan.cpp
	g++ $(FLAGS) -c -o $@ $(INC) $<

qwen.o: qwen.cpp
	g++ $(FLAGS) -c -o $@ $(INC) $<

install: libtokenizer_combo.a
	mkdir -p ../install/lib
	cp rust/target/release/libtokenizers_c.so ../install/lib/
	cp libtokenizer_combo.a ../install/lib/
	cp tokenizer_combo.hpp ../install/include

clean:
	rm -f *.a *.o
