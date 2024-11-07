FC = gfortran -O3 -ffree-line-length-0

OBJECTS = my_module.o test1.o

TARGET = test1

MOD = mm.mod

%.o: %.f90
	$(FC)  -c $<

# %.mod: %.f90
# 	$(FC)  -c $<

clean:
	rm -f $(OBJECTS) $(TARGET) $(MOD)

test: $(OBJECTS)
	$(FC) -o $(TARGET) $(OBJECTS)
