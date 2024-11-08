Program test
  ! modules
  use hamtools

  implicit none

  ! define variables
  real*8, allocatable:: kpoints(:,:), energy(:), validkpoints(:,:), real_k(:,:)
  complex*16, allocatable:: ham(:,:,:,:), gf(:,:,:,:), newham(:,:,:)
  real*8 T, chi0, v, Hu, Hl, deltaU, deltaL, Hprev, deltaPrev
  integer ki, nk, it, steps
  character(len=16) efmt, kfmt
  character(len=4) intstring
  real*8 singlek(3)
  complex*16 oldsampleham(2,2), sampleham(2,2)

  allocate(kpoints(3,resolution*resolution),energy(resolution*resolution), real_k(3, resolution*resolution))
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

  !write(*,*) kpoints(1,:,1)

  !energy = epsilonk(hamk(kpoints, resolution*resolution, 0d0))

  write(*,*)kpoints(:,resolution*resolution)


  real_k(1,:) = kpoints(2,:) * 2.0d0 / 3d0  + kpoints(1,:) * 1d0/ 3d0
  real_k(2,:) = kpoints(1,:)
  real_k(3,:) = 0d0

  write(*,*)real_k(:,resolution*resolution)

  open(100,file="kpoints.dat")
  write(100,kfmt) real_k(:,:)
  close(100)

  energy = eps(real_k,resolution*resolution)
  open(101,file="energy.dat")
  write(101,efmt) energy(:)
  close(101)

  !----- find valid kpoints
  nk = 0
  do ki = 1, resolution*resolution
    if ( abs(energy(ki)) < threshold ) then
      nk = nk +1
      call AddToList(validkpoints,kpoints(:,ki))
    end if
  end do

  write(*,*)nk

  chi0 = susc(6.5d0,0d0,nk,validkpoints)

  v = 1/chi0
  write(*,*)v

  

 
  !-------- must be done to clear memory
  deallocate(rvec, hamr, ndeg)

end 