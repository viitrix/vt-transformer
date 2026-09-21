.PHONY: all

all:
	mkdir -p build
	cd build;cmake .. -DCMAKE_INSTALL_PREFIX=../../../install -DCMAKE_BUILD_TYPE=Release
	cd build;make install -j4
	mkdir -p bitsandbytes/build
	cd bitsandbytes/build; cmake ..  -DCMAKE_INSTALL_PREFIX=../../../../install 
	cd bitsandbytes/build; make install -j4

clean:
	rm -rf build
	rm -rf bitsandbytes/build

