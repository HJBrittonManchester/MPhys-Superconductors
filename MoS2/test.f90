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
  real*8 singlek(3), tolerance
  complex*16 oldsampleham(2,2), sampleham(2,2)

  allocate(kpoints(3,resolution*resolution),energy(resolution*resolution), real_k(3, resolution*resolution))
  allocate(ham(resolution,resolution,2,2),gf(resolution,resolution,2,2),newham(resolution*resolution,2,2))

  tolerance = 1d-4

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

  energy = epsilonk(hamk(kpoints, resolution*resolution, 0d0))

  !write(*,*)kpoints(:,resolution*resolution)


  real_k(1,:) = kpoints(2,:) * 2.0d0 / sqrt(3d0)  + kpoints(1,:) * 1d0/ sqrt(3d0)
  real_k(2,:) = kpoints(1,:)
  real_k(3,:) = 0d0

  !write(*,*)real_k(:,resolution*resolution)

  open(100,file="kpoints.dat")
  !write(100,kfmt) real_k(:,:)
  close(100)

  !energy = ep(real_k,resolution*resolution)
  open(101,file="energy.dat")
  !write(101,efmt) energy(:)
  close(101)

  !write(*,*)energy

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


  !!! brackets


  open(90,file="phase.dat")

  do it=1, 10
    !------ braketing
    steps = 1
    T = 6.5d0 - (it -1) * 65d-2
      ! guesses
    Hu = 80d0
    Hl = -5d0

    !write(*,*) 
    deltaL = 1 - v *susc(T, Hl, nk, validkpoints)
    deltaU = 1 - v *susc(T, Hu, nk, validkpoints)
    deltaPrev = 0.0d0
    Hprev = 0d0


    

    111  write(*,*)steps


    !write(*,*) deltaL
    !write(*,*) deltaU

    if(deltaU < 0 .and. steps == 1) then
      write(*,*)"Upper too low"
      goto 888
    elseif(deltaL > 0 .and. steps==1) then
      write(*,*)"lower too high"
      goto 888

    end if
    


    if(deltaL > 0 .and. steps> 1) then
    !--- Hl over stepped the 0 point
    Hu = Hl
    deltaU = deltaL
    deltaL = deltaPrev
    Hl = Hprev
    
    elseif(deltaU < 0 .and. steps> 1 ) then
      !--- Hu over stepped the 0 point
      Hl = Hu
      deltaL = deltaU
      deltaU = deltaPrev
      HU = Hprev
    end if


    if(deltaL + deltaU >= 0) then

      !----- update previous values
      Hprev = Hu
      deltaPrev = deltaU

      !----- find new Hu
      Hu = (Hu + Hl) / 2d0
      deltaU = 1- v* susc(T,Hu,nk,validkpoints)


    elseif(deltaL+deltaU < 0) then 
      !----- update previous values
      Hprev = Hl
      deltaPrev = deltaL

      !----- find new Hu
      Hl = (Hu + Hl) / 2d0
      deltaL = 1 - v* susc(T,Hl,nk,validkpoints)

    end if


    if ( abs(deltaU) < tolerance ) then
      write(90,'(2(F10.6,1X))') Hu, T
      goto 999
    elseif(abs(deltaL) < tolerance) then
      write(90,'(2(F10.6,1X))') Hl, T
      goto 999
    end if

    steps = steps + 1



    !write(*,*)steps

    if(steps > 20) then
      write(90,'(2(F10.6,1X))') Hl, T
      goto 999
    endif

    goto 111


    !------ 
    888 write(*,*)"invalid initial range of H"
    999 write(*,*)"end"
  enddo

  close(90)





  

 
  !-------- must be done to clear memory
  deallocate(rvec, hamr, ndeg)

end 