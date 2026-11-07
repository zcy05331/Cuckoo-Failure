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
