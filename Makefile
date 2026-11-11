CXX      ?= g++
CXXFLAGS ?= -O2 -std=c++17 -Wall -Wextra
CPPFLAGS += -Iinclude
EXE      := $(if $(filter Windows_NT,$(OS)),.exe,)
HEADERS  := $(wildcard include/cuckoo/*.hpp)

all: bin/check$(EXE) bin/cuckoo_fail$(EXE)

bin:
	mkdir -p bin

bin/check$(EXE): src/check.cpp $(HEADERS) | bin
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -o $@ src/check.cpp

bin/cuckoo_fail$(EXE): src/cuckoo_fail.cpp $(HEADERS) | bin
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -o $@ src/cuckoo_fail.cpp

check: bin/check$(EXE)
	./bin/check$(EXE)

clean:
	rm -rf bin

.PHONY: all check clean

# ---- fast/: 64-bit batch engine (same source for CUDA, plain host and AVX-512 IFMA host) ----
FASTFLAGS ?= -O3 -std=c++17 -Wall -Wextra
fast: bin/cuckoo_fast_cpu$(EXE)
bin/cuckoo_fast_cpu$(EXE): fast/cuckoo_fast.cu fast/ifma.inc | bin
	$(CXX) $(FASTFLAGS) -x c++ fast/cuckoo_fast.cu -o $@
fast-ifma: bin/cuckoo_fast_ifma$(EXE)
bin/cuckoo_fast_ifma$(EXE): fast/cuckoo_fast.cu fast/ifma.inc | bin
	$(CXX) $(FASTFLAGS) -mavx2 -mavx512f -mavx512dq -mavx512vl -mavx512ifma -x c++ fast/cuckoo_fast.cu -o $@
fast-gpu: bin/cuckoo_fast_gpu$(EXE)
bin/cuckoo_fast_gpu$(EXE): fast/cuckoo_fast.cu | bin
	nvcc -O3 -std=c++17 fast/cuckoo_fast.cu -o $@
