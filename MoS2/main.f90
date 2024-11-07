Program main

  use params ! module including all variables

  implicit none
  
  ! allocate variables - doesnt work inside module
  allocate(kpoints(3,nk))
  allocate(hamr(nk,2,2), hamk(nk,2,2))

  call realham_sub()

  call karray_sub(kpoints)

  call hamk_sub()


  !define the functions
  ! real*8 karray(3,resolution*resolution)
  ! complex*16 hamk(nk,2,2)
  ! real*8 epsilonk(resolution*resolution)
  ! real*8 susc

  ! allocate(karray(3,resolution*resolution))
  ! allocate(kpoints(3,resolution*resolution),energy(resolution*resolution))
  ! allocate(ham(resolution,resolution,2,2),gf(resolution,resolution,2,2),newham(resolution*resolution,2,2))

  !-------------- writing output files
  ! write(intstring,'(i4)')resolution
  ! efmt = '(' // intstring // '(F10.6,1X))'
  ! write(intstring,'(i4)') 3* resolution
  ! kfmt = '(' // intstring // '(F10.6,1X))'

  !open(100,file="kpoints.dat")
  !write(100,kfmt) kpoints(:,:)
  !close(100)
  !write(*,*) kpoints(1,:,1)

  !open(101,file="energy.dat")
  !write(101,efmt) energy(:)
  !close(101)

  ! control flow
  ! print *, 'Hello world' 

  

  

 !kpoints = karray()


  energy = epsilonk(hamk(kpoints,resolution*resolution,0d0))

  call hamk_sub(hamk,hamr,kpoints,resolution*resolution,0d0,0d0,0d0) !edits the value of hamk

  call energy_sub(hamk)



 

  !write(*,*)nk

  chi0 = susc(6.5d0,0d0,nk,validkpoints)

  v = 1/chi0
  write(*,*)v

  

 
  !-------- must be done to clear memory
  deallocate(rvec, hamr, ndeg)

end 