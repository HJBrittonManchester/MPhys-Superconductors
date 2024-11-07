
  function matsufreq(T, m)
    integer*4 m
    real*8 T, matsufreq
    real*8, parameter::pi = 4.D0*DATAN(1.D0)
      
    matsufreq = (2*m+1)* pi * (T*kb)
  end function matsufreq
