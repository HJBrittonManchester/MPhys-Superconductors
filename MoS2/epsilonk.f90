
subroutine epsilonk_sub(ham)

  implicit none

  use params

  !----- variables
  real*8, allocatable:: ene(:,:), rwork(:)
  integer*4  lwork,info
  real*8 epsilonk(resolution*resolution)
  complex*16,allocatable:: work(:)
  complex*16 Ham(resolution*resolution,nb,nb)
  integer ki


  !------ get hamiltonian
  allocate(ene(nb,resolution*resolution))
  ene = 0d0

  !------- diagonalise

  lwork=max(1,2*nb-1)
  allocate(work(max(1,lwork)),rwork(max(1,3*nb-2)))

  do ki=1,resolution*resolution
    call zheev('V','U',nb,Ham(ki,:,:),nb,ene(:,ki),work,lwork,rwork,info)
  enddo

  !------- get average of all energy bands at k-point
  epsilonk = 0d0

  epsilonk = (ene(1,:) + ene(2,:))/2

   !----- find valid kpoints
  do ki = 1, resolution*resolution
    if ( abs(energy(ki)) < .1 ) then
      nk = nk +1
      call AddToList(validkpoints,kpoints(:,ki))
    end if
  end do

end subroutine epsilonk_sub
