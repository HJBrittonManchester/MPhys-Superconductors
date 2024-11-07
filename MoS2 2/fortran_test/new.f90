module mm
    implicit none
  
    !---------- user defined parameters
    character(len=80):: prefix="MoS2"
    integer*4:: resolution= 140
    real*8,parameter::ef= .91d0 ! original value is 4.18903772 ! Fermi energy
    real*8, parameter::kb = 8.6173d-5
    real*8, parameter::mub = 5.7883817982d-5
    integer, parameter:: freqmax = 500 !
  
  
    !---------- global variables
    integer*4,allocatable:: ndeg(:) ! weighting used in FT
    real*8,allocatable:: rvec(:,:) ! list of realspace vectors
    integer*4 nb, nr ! number of bands, number of real space points
    complex*16, allocatable:: hamr(:,:,:)
  
    
  contains
  
  
    !--------------------- Defining the workflow ------------------------
    !---  1. Create the real space hamiltonian with realham
    !---  2. Calculate k-space hamiltonian with of whole BZ with hamk
    !---  3. Diagonalise with epsilon to find average energy
    !---  4. Create the array of valid kpoints
    !---  5. calcualte GF and then susc for these
  
  
    !----- used to return an (3,resolution,resolution) array of k points
    function karray()
      integer*4 alpha, beta
      real*8 karray(3,resolution*resolution), pi2
      
      pi2=4.0d0*atan(1.0d0)*2.0d0
  
      !-------------Generate kpoints
      karray = 0d0
      do alpha = 1,resolution
        do beta = 1,resolution
          karray(1,(beta-1) * resolution + alpha) = alpha / real(resolution, 8)
          karray(2,(beta-1) * resolution+alpha) = beta / real(resolution, 8)
        enddo
      enddo
  
      karray = karray * pi2
    end function karray
  
    function hamk(k, nk, H)
      integer j, i, ki, nk
      real*8 k(3,nk), phase, H
      complex*16 hamk( nk, 2,2)
  
  
      !------ this fixes bug of crazy large Im part of (1,1)
      hamk = (0d0, 0d0)
      do ki=1,nk
        do j=1,nr
          phase=0.0d0
          do i=1,3
              phase=phase+k(i,ki)*rvec(i,j)
              
          enddo
          hamk(ki,:,:)=hamk(ki, :,:)+Hamr(:,:,j)*dcmplx(cos(phase),-sin(phase))/float(ndeg(j))
        enddo
      enddo
      !------ adjust fermi surface 
  
      hamk(:,1,1) = hamk(:,1,1) - dcmplx(ef,0d0)
      hamk(:,2,2) = hamk(:,2,2) - dcmplx(ef,0d0)
  
      !------- add H field term  along x axis
      hamk(:, 1, 2) = hamk(:,1,2) - H * mub
      hamk(:, 2, 1) = hamk(:,2,1) - H * mub
      
    end function hamk
  
    subroutine realham()
      character(len=80) hamil_file
      integer*4 ir, i, j, i1,i2
      real*8 a,b
  
  
      !--------- Ammend file name
      write(hamil_file,'(a,a)')trim(adjustl(prefix)),"_hr.dat"
  
  
      !------------- Read real-space hamiltonian
  
      open(99,file=trim(adjustl(hamil_file)),err=333)
      read(99,*)
      read(99,*)nb,nr
      allocate(rvec(3,nr),hamr(nb,nb,nr),ndeg(nr))
      read(99,*)ndeg
      do ir=1,nr
          do i=1,nb
            do j=1,nb
                read(99,*)rvec(1,ir),rvec(2,ir),rvec(3,ir),i1,i2,a,b
                hamr(i1,i2,ir)=dcmplx(a,b)
            enddo
          enddo
      enddo
  
  
      if(.true.) goto 444
      333   write(*,'(3a)')'ERROR: input file "',trim(adjustl(hamil_file)),' not found'
      stop
      444 close(99)
  
    end subroutine realham
  
  
    function greens(freq, ham, nk)
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
  
    function matsufreq(T, m)
      integer*4 m
      real*8 T, matsufreq
      real*8, parameter::pi = 4.D0*DATAN(1.D0)
  
      matsufreq = (2*m+1)* pi * (T*kb)
    end function matsufreq
  
    function susc(T, H, nk, k)
      complex*16 Hamp(nk,2,2), Hamn(nk,2,2)
      real*8 T, H, freq
      real*8 susc, susc_arr(nk), k(3, nk)
      integer m, nk
      complex*16 gfp(nk,2,2), gfn(nk,2,2)
  
      
      Hamp = hamk(k, nk, H)
      Hamn = Hamk(-k, nk, H)
  
      susc = 0d0
      do m=-freqmax,freqmax
        freq = matsufreq(T,m)
        gfp = greens(freq, Hamp, nk)
        gfn = greens(-freq, Hamn, nk)
  
  
        susc_arr = -real((kb* T *( gfp(:,1,1) * gfn(:,2,2) - gfp(:,1,2)*gfn(:,2,1))))
  
        susc = susc + sum(susc_arr)
      enddo
  
      susc = susc / (resolution*resolution)
  
    end function susc
  
  
    function epsilonk(ham)
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
  
    end function epsilonk
  
    
   !----------- Must be used for adding vectors to list of vectors
    subroutine AddToList(list, element)
  
      IMPLICIT NONE
  
      integer :: i, isize
      real*8, intent(in) :: element(3)
      real*8, dimension(:,:), allocatable, intent(inout) :: list
      real*8, dimension(:,:), allocatable :: clist
  
  
      if(allocated(list)) then
          isize = size(list)/3
          allocate(clist(3,isize+1))
          do i=1,isize          
          clist(:,i) = list(:,i)
          end do
          clist(:,isize+1) = element
  
          deallocate(list)
          call move_alloc(clist, list)
  
      else
          allocate(list(3,1))
          list(:,1) = element
      end if
  
  
  end subroutine AddToList
  
  
  end module mm
  
  Program test
    ! modules
    use mm
  
    implicit none
  
    ! define variables
    real*8, allocatable:: kpoints(:,:), energy(:), validkpoints(:,:)
    complex*16, allocatable:: ham(:,:,:,:), gf(:,:,:,:), newham(:,:,:)
    real*8 T, chi0, v, Hu, Hl, deltaU, deltaL, Hprev, deltaPrev
    integer ki, nk, it, steps
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
  
  
  
    energy = epsilonk(hamk(kpoints,resolution*resolution,0d0))
    !open(101,file="energy.dat")
    !write(101,efmt) energy(:)
    !close(101)
  
    !----- find valid kpoints
    do ki = 1, resolution*resolution
      if ( abs(energy(ki)) < .1 ) then
        nk = nk +1
        call AddToList(validkpoints,kpoints(:,ki))
      end if
    end do
  
    !write(*,*)nk
  
    chi0 = susc(6.5d0,0d0,nk,validkpoints)
  
    v = 1/chi0
    write(*,*)v
  
    
  
   
    !-------- must be done to clear memory
    deallocate(rvec, hamr, ndeg)
  
  end 