.PHONY: all

FLAGS = -Wall -O3 -Wno-maybe-uninitialized

INC = -I. -I../install/include 

all: install

libtokenizer_combo.a: qwen.o llama3.o 
	 ar rcs $@ $?	

qwen.o: qwen.cpp
	g++ $(FLAGS) -c -o $@ $(INC) $<

llama3.o: llama3.cpp
	g++ $(FLAGS) -c -o $@ $(INC) $<

rust/target/release/libtokenizers_c.so:
	cd rust;cargo build --release

install: libtokenizer_combo.a rust/target/release/libtokenizers_c.so
	mkdir -p ../install/lib
	cp rust/target/release/libtokenizers_c.so ../install/lib/
	cp libtokenizer_combo.a ../install/lib/
	cp tokenizer_combo.hpp ../install/include

clean:
	rm -f *.a *.o
