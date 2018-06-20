#use these variable to set if we will use mpi or not 

AR=ar
ARFLAGS=-qs
RANLIB=ranlib

EXECUTABLE = wpimd1.x pimd1_t300k.x pimd1_t100k.x

CXX = c++
HOME = .
.SUFFIXES: .cc  


CFLAGS = -O2  -ffast-math    -std=c++11  
LIBS   = -larmadillo


MD_SRC = wpimd1.cc pimd1_t300k.cc pimd1_t100k.cc

MD_OBJ=  wpimd1.o  pimd1_t300k.o  pimd1_t100k.o

all	: wpimd1.x  pimd1_t300k.x  pimd1_t100k.x


wpimd1.x : wpimd1.o 
	$(CXX)   $(FLAGS) -o  wpimd1.x wpimd1.o $(LIBS) 

pimd1_t300k.x : pimd1_t300k.o 
	$(CXX)   $(FLAGS) -o  pimd1_t300k.x pimd1_t300k.o $(LIBS) 

pimd1_t100k.x : pimd1_t100k.o 
	$(CXX)   $(FLAGS) -o  pimd1_t100k.x pimd1_t100k.o $(LIBS) 


clean:
	rm *.o *~ *.x

.cc.o :
	$(CXX) $(CFLAGS) -c $< -o $@


# DO NOT DELETE
