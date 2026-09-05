.PHONY: all clean

CXX      = g++
CXXFLAGS = -Wall -O2 -std=c++17 -Iopt
LIBS     = -lzmq

VT_DIR    = ../../vt-transformer
CORE_DIR  = $(VT_DIR)/core
CUDA_HOME ?= /usr/local/cuda

all: mock backend

mock: backend_mock.cpp opt/json.hpp
	$(CXX) $(CXXFLAGS) -o $@ backend_mock.cpp $(LIBS)

backend: backend_cuda.cpp impl/qwen3.cpp impl/qwen3.hpp $(CORE_DIR)/vt.cpp $(CORE_DIR)/vt_cuda.cpp $(CORE_DIR)/vt.hpp $(CORE_DIR)/vt_cuda.hpp
	$(CXX) $(CXXFLAGS) -I$(VT_DIR) -I$(CUDA_HOME)/include -o $@ \
	    backend_cuda.cpp impl/qwen3.cpp $(CORE_DIR)/vt.cpp $(CORE_DIR)/vt_cuda.cpp \
	    -L$(CUDA_HOME)/lib64 -lcudart -lcublas -lcublasLt $(LIBS)

clean:
	rm -f mock backend
