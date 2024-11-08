
FC = gfortran

OBJECTS = wanr2k.o 

LIBDIR = C:\MSYS2\mingw64\lib

TARGET = wanr2k

$(TARGET): $(OBJECTS)
	$(FC) -o $@   $^ $(LIBDIR)\liblapack.a $(LIBDIR)\libblas.a

%.o: %.f90
	$(FC) -c $<

clean:
	rm -f $(OBJECTS) $(TARGET)

test:  hamtools.o test.o
	$(FC) -o test test.o hamtools.o $(LIBDIR)\liblapack.a $(LIBDIR)\libblas.a 
