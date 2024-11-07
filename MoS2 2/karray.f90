subroutine karray_sub(karray,nk,resolution)

    use params

    implicit none

    integer*4 alpha, beta, resolution, nk
    ! real*8 karray(3,resolution*resolution), pi
    real*8, allocatable:: karray(:,:)
    real*8 :: pi

    allocate(karray(3,nk))

    pi=4.0d0*atan(1.0d0)

    !-------------Generate kpoints
    karray = 0d0
    do alpha = 1,resolution
    do beta = 1,resolution
        karray(1,(beta-1) * resolution + alpha) = alpha / real(resolution, 8)
        karray(2,(beta-1) * resolution+alpha) = beta / real(resolution, 8)
    enddo
    enddo

    karray = karray * pi * 2
  
    end subroutine karray_sub
  