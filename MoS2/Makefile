
FC = gfortran

OBJECTS = test.o hamtools.o

LIBDIR = C:\msys64\ucrt64\lib

TARGET = test

$(TARGET): $(OBJECTS)
	$(FC) -o $@   $^ $(LIBDIR)\liblapack.a $(LIBDIR)\libblas.a

%.o: %.f90
	$(FC) -c $<

clean:
	rm -f $(OBJECTS) $(TARGET)


wanr2k: wanr2k.o
	$(FC) -o wanr2k wanr2k.o $(LIBDIR)\liblapack.a $(LIBDIR)\libblas.a 

