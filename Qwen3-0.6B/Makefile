.PHONY: all clean

CXX      = g++
CXXFLAGS = -Wall -O2 -std=c++17 -Iopt
LIBS     = -lzmq

all: mock

mock: src/mock.cpp opt/json.hpp
	$(CXX) $(CXXFLAGS) -o $@ src/mock.cpp $(LIBS)

clean:
	rm -f mock
