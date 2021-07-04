include ./example.mk

CC=mpic++
# CFLAGS= -g -c
BUILDDIR = ./build
LDIR =
OBJ = main.o

%.o: %.cpp
	$(CC) -O3 -c --std=c++14 -o $@ $< $(INCLUDE_PATH)

gray_scott: $(OBJ)
	$(CC) -o $(BUILDDIR)/$@ $^ $(CFLAGS) $(LIBS_PATH) $(LIBS)

all: $(BUILDDIR)/gray_scott

run: all
	mpirun --oversubscribe -np 4 $(BUILDDIR)/gray_scott

.PHONY: clean all run

clean:
	rm -f *.o *~ core $(BUILDDIR)/gray_scott

