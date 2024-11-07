subroutine validkpoints_sub(ham, kpoints, nb, nk, nk_valid, threshold)

    implicit none
  
    integer*4 nk, nb, nk_valid, ki, threshold
    real*8 kpoints(3,nk)
    real*8, allocatable:: validkpoints(:,:)

    real*8, allocatable:: ene(:,:), rwork(:)
    complex*16,allocatable:: work(:)
    integer*4  lwork,info
    
    real*8 epsilonk(nk)
    complex*16 ham(nb,nb,nk)
  
  
    !------ get hamiltonian
    allocate(ene(nb,nk))
    ene = 0d0
  
    !------- diagonalise
    lwork=max(1,2*nb-1)
    allocate(work(max(1,lwork)),rwork(max(1,3*nb-2)))
  
    do ki=1,nk
      call zheev('V','U',nb,ham(:,:,ki),nb,ene(:,ki),work,lwork,rwork,info)
    enddo
  
    !------- get average of all energy bands at k-point
    epsilonk = 0d0
  
    epsilonk = (ene(1,:) + ene(2,:))/2
  
     !----- find valid kpoints
    do ki = 1, nk
      if ( abs(epsilonk(ki)) < threshold ) then
        nk_valid = nk_valid + 1
        call AddToList(validkpoints,kpoints(:,ki))
      end if
    end do

    print*, validkpoints
  
end subroutine validkpoints_sub
  