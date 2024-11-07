function greens(freq, ham, nk)

  use params

    complex*16 Ham(nk,nb,nb)
    real*8 freq
    complex*16 det(nk), greens(nk,nb,nb)
    integer nk


    det = (dcmplx(0d0,freq) - Ham(:,1,1)) * (dcmplx(0d0,freq) - Ham(:,2,2)) - Ham(:,1,2)*Ham(:,2,1)

    greens(:,1,1) = (dcmplx(0d0,freq) - Ham(:,2,2))  / det
    greens(:,2,2) = (dcmplx(0d0,freq) - Ham(:,1,1)) / det
    greens(:,1,2) = +Ham(:,1,2) / det
    greens(:,2,1) = +Ham(:,2,1) / det

end function greens
