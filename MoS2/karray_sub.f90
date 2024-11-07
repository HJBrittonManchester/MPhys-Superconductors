!----- used to return an (3,resolution,resolution) array of k points
subroutine karray_sub(karray)

    use params
  
      integer*4 alpha, beta
      ! real*8 karray(3,resolution*resolution), pi
      real*8, allocatable:: karray(:,:)
      real*8 :: pi
  
      allocate(karray(3,resolution*resolution))
        
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
  