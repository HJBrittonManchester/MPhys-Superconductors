Program test
  ! modules
  use hamtools

  implicit none

  ! define variables
  real*8, allocatable:: kpoints(:,:), energy(:), validkpoints(:,:)
  complex*16, allocatable:: ham(:,:,:,:), gf(:,:,:,:), newham(:,:,:)
  real*8 T, chi0, s
  integer ki, nk
  character(len=16) efmt, kfmt
  character(len=4) intstring
  real*8 singlek(3)
  complex*16 oldsampleham(2,2), sampleham(2,2)

  allocate(kpoints(3,resolution*resolution),energy(resolution*resolution))
  allocate(ham(resolution,resolution,2,2),gf(resolution,resolution,2,2),newham(resolution*resolution,2,2))

  !-------------- Define formatting of output files
  write(intstring,'(i4)')resolution
  efmt = '(' // intstring // '(F10.6,1X))'
  write(intstring,'(i4)') 3* resolution
  kfmt = '(' // intstring // '(F10.6,1X))'

  ! cotrol flow
  print *, 'Hello world' 

  call realham()


  kpoints = karray()
  !open(100,file="kpoints.dat")
  !write(100,kfmt) kpoints(:,:)
  !close(100)
  !write(*,*) kpoints(1,:,1)


  energy = epsilonk(hamk(kpoints,resolution*resolution))
  !open(101,file="energy.dat")
  !write(101,efmt) energy(:)
  !close(101)

  !----- find valid kpoints
  do ki = 1, resolution*resolution
    if ( abs(energy(ki)) < 2 ) then
      nk = nk +1
      call AddToList(validkpoints,kpoints(:,ki))
    end if
  end do

  write(*,*)nk

  s = susc(6.5d0,0d0,nk,validkpoints)

  write(*,*)s

  !ham = Hamkold(+1)
  !oldsampleham = ham(2,1,:,:)
  !write(*,*)oldsampleham

  !newham = hamk(kpoints)
  !write(*,*)kpoints(1,:)
  !sampleham = newham(4,:,:)
  !write(*,*)sampleham


  !gf = greens(0d0)
  !write(*,*) gf

  !chi0 = susc(6.5d0,0d0)
  !write(*,*)chi0


  

  !hamksingle = hamk(singlek)
  !write(*,*) hamksingle


  !-------- must be done to clear memory
  deallocate(rvec, hamr, ndeg)

end 