#use these variable to set if we will use mpi or not 

AR=ar
ARFLAGS=-qs
RANLIB=ranlib

EXECUTABLE = wpimd_morse.x wpimd_morse_pes.x

CXX = c++

HOME = .
.SUFFIXES: .cc  


CFLAGS = -O2  -ffast-math    -std=c++11  
FFLAGS = -O2
LIBS   = -larmadillo



all	: wpimd_morse.x wpimd_morse_pes.x


wpimd_morse.x : wpimd_morse.o
	$(CXX)   $(FLAGS) -o  wpimd_morse.x wpimd_morse.o $(LIBS) 

wpimd_morse_pes.x : wpimd_morse_pes.o
	$(CXX)   $(FLAGS) -o  wpimd_morse_pes.x wpimd_morse_pes.o $(LIBS) 

clean:
	rm *.o *~ *.x

.cc.o :
	$(CXX) $(CFLAGS) -c $< -o $@

# DO NOT DELETE
