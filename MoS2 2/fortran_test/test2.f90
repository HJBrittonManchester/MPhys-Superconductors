program test2

    use mm
    ! include "test3.f90"

    implicit none

    integer*4:: test_var = 10
    integer*4:: test_var2 = 11
    integer*4 new_var
    integer*4 test1func

    print*, "test_var =", test_var

    print*, "my_var =", my_var

    new_var = test1func(test_var, test_var2)
    
    call test1sub(1, 2)
    print *, "output of function is", new_var

end