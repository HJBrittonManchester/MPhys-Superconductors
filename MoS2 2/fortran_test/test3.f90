function test1func(variable1, variable2)

    use mm

    implicit none

    integer*4:: variable1
    integer*4:: variable2
    integer*4:: test1func

    print*, "variable1 =",  variable1

    print*, "variable2 =", variable2

    print*, "my_var =", my_var

    test1func = variable1 + variable2

end function test1func