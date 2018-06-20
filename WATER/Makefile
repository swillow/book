#use these variable to set if we will use mpi or not 

AR=ar
ARFLAGS=-qs
RANLIB=ranlib

EXECUTABLE = wpimd_ps.x wpimd_ps2.x

CXX = c++
F90 = gfortran
HOME = .
.SUFFIXES: .cc  .f90


CFLAGS = -O2  -ffast-math    -std=c++11  
FFLAGS = -O2
LIBS   = -larmadillo


MD_SRC = wpimd_ps.cc wpimd_ps2.cc nasa_mod.f90 nasa.f90

MD_OBJ=  wpimd_ps.o wpimd_ps2.o nasa_mod.o nasa.o

all	: wpimd_ps.x wpimd_ps2.x 


wpimd_ps.x : wpimd_ps.o nasa_mod.o nasa.o
	$(CXX)   $(FLAGS) -o  wpimd_ps.x wpimd_ps.o nasa_mod.o nasa.o $(LIBS)  -lgfortran

wpimd_ps2.x : wpimd_ps2.o nasa_mod.o nasa.o
	$(CXX)   $(FLAGS) -o  wpimd_ps2.x wpimd_ps2.o nasa_mod.o nasa.o $(LIBS)  -lgfortran

clean:
	rm *.o *~ *.x

.cc.o :
	$(CXX) $(CFLAGS) -c $< -o $@

nasa.o : nasa.f90
	$(F90)  $(FFLAGS) -c $< 

nasa_mod.o : nasa_mod.f90
	$(F90)  $(FFLAGS) -c $< 

# DO NOT DELETE
