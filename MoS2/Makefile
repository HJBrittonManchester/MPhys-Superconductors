
FC = gfortran

OBJECTS = wanr2k.o 

LIBDIR = C:\msys64\ucrt64\lib

TARGET = wanr2k

$(TARGET): $(OBJECTS)
	$(FC) -o $@   $^ $(LIBDIR)\liblapack.a $(LIBDIR)\libblas.a

%.o: %.f90
	$(FC) -c $<

clean:
	rm -f $(OBJECTS) $(TARGET)

test: test.o hamtools.o
	$(FC) -o test test.o hamtools.o C:\msys64\ucrt64\lib\liblapack.a C:\msys64\ucrt64\lib\libblas.a 