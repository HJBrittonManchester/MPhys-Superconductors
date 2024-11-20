module hamtools
  implicit none

  !---------- user defined parameters
  character(len=80):: prefix="MoS2"
  integer*4:: resolution= 200
  real*8,parameter::ef= .915d0 ! original value is 4.18903772 ! Fermi energy
  real*8, parameter::kb = 8.6173d-5
  real*8, parameter::mub = 5.7883817982d-5
  integer, parameter:: freqmax = 1000 !
  real*8, parameter::threshold =0.022

  real*8, parameter:: pi2 = 4.0d0*atan(1.0d0)*2.0d0

  !-------------- For the bands
  integer,parameter::nkpath=4,np=100
  integer*4,parameter::nklist=(nkpath-1)*np+1

  !----------- toy hamiltonian parameters
  real*8, parameter:: t1 = 0.146d0
  real*8, parameter:: t2 = -0.4d0 * t1
  real*8, parameter:: t3 = 0.25d0 * t1
  real*8, parameter:: chem = -0.75d0

  real*8, parameter:: a = 3.25d0 ! lattice const

  real*8, parameter:: az = 5.77350269d-04
  real*8, parameter:: ar = az * 9.03566289d-05
  real*8, parameter:: beta = 2.75681159d1

  real*8, parameter:: thekpoint(3) = [0d0,2d0 / 3d0 * pi2, 0d0]




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
    real*8 karray(3,resolution*resolution)
    
    

    !-------------Generate kpoints
    karray = 0d0
    do alpha = 1,resolution
      do beta = 1,resolution
        karray(1,(beta-1) * resolution + alpha) = (alpha-1d0) / real(resolution, 8)
        karray(2,(beta-1) * resolution+ alpha) = (beta-1d0) / real(resolution, 8)
      enddo
    enddo

    karray = karray * pi2
  end function karray

  function klist()

    character(len=80) nnkp,line
    real*8 klist(3,1:nklist),xk(nklist),kpath(3,np),bvec(3,3),ktemp1(3),ktemp2(3),xkl(nkpath), jk
    real*8,parameter:: third = 1d0 / 3d0
    integer i, j, k
    character(len=30)::klabel(nkpath)



    write(nnkp,'(a,a)')trim(adjustl(prefix)),".nnkp"

!---------------  reciprocal vectors
    open(98,file=trim(adjustl(nnkp)))
111   read(98,'(a)')line
    if(trim(adjustl(line)).ne."begin recip_lattice") goto 111
    
    read(98,*)bvec
!---------------kpath
    data kpath(:,1) /     0.0d0,      0.0d0,    0.0d0/  !G
    data kpath(:,2) /     0.5d0,      0.0d0,    0.0d0/  !M
    data kpath(:,3) /     third,      third,    0.0d0/  !K
    data kpath(:,4) /     0.0d0,      0.0d0,    0.0d0/  !G


    data klabel     /'G','M','K','G'/

    ktemp1(:)=(kpath(1,1)-kpath(1,2))*bvec(:,1)+(kpath(2,1)-kpath(2,2))*bvec(:,2)+(kpath(3,1)-kpath(3,2))*bvec(:,3)

!      xk(1)= 0d0 !-sqrt(dot_product(ktemp1,ktemp1))
    xk(1)= -sqrt(dot_product(ktemp1,ktemp1))
    xkl(1)=xk(1)
    

    k=0
    ktemp1=0d0
    do i=1,nkpath-1
     do j=1,np
      k=k+1
      jk=dfloat(j-1)/dfloat(np)
      klist(:,k)=kpath(:,i)+jk*(kpath(:,i+1)-kpath(:,i))
      ktemp2=klist(1,k)*bvec(:,1)+klist(2,k)*bvec(:,2)+klist(3,k)*bvec(:,3)
      if(k.gt.1) xk(k)=xk(k-1)+sqrt(dot_product(ktemp2-ktemp1,ktemp2-ktemp1))
      if(j.eq.1) xkl(i)=xk(k)
      ktemp1=ktemp2
     enddo
    enddo
    klist(:,nklist)=kpath(:,nkpath)
    ktemp2=klist(1,nklist)*bvec(:,1)+klist(2,nklist)*bvec(:,2)+klist(3,nklist)*bvec(:,3)
    xk(nklist)=xk(nklist-1)+sqrt(dot_product(ktemp2-ktemp1,ktemp2-ktemp1))
    xkl(nkpath)=xk(nklist)
!      write(*,*)klist
    klist=klist*pi2

    write(*,*)klist

  end function klist

  function hamk(k, nk, H, theta, phi)
    integer j, i, ki, nk
    real*8 k(3,nk), phase, H, theta, phi
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

    hamk(:,1,1) = hamk(:,1,1) - dcmplx(ef,0d0) -  H * mub * cos(theta)
    hamk(:,2,2) = hamk(:,2,2) - dcmplx(ef,0d0) + H * mub * cos(theta)

    !------- add H field term  along x axis
    hamk(:, 1, 2) = hamk(:,1,2) -  H * mub * dcmplx(cos(phi), sin(phi)) * sin(theta)
    hamk(:, 2, 1) = hamk(:,2,1) - H * mub * dcmplx(cos(phi), -sin(phi)) * sin(theta)
    
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

  function eps(k, nk)
    integer nk
    real*8 eps(nk), k(3,nk)

    eps = t1 * (cos(k(2,:)) + 2d0 * cos(sqrt(3d0)/2d0 * k(1,:))*cos(k(2,:)/2d0))
    eps = eps + t2*( cos(sqrt(3d0) * k(1,:)) + 2* cos(sqrt(3d0)/2d0 * k(1,:))*cos(3d0/2d0 * k(2,:)))
    eps = eps + t3*(cos(2d0 * k(2,:)) + 2* cos(sqrt(3d0)*k(1,:))*cos(k(2,:)))

    eps = 2 * eps - chem

  end function eps

  function f(k,nk)
    integer nk
    real*8 k(3,nk), f(nk)

    f = abs(sin(k(2,:)) - 2 * cos(sqrt(3d0)/2d0 * k(1,:))*sin(k(2,:)/2d0))
    

  end function f

  function bigF(k, nk)
    integer nk
    real*8 k(3,nk), bigF(nk)

    bigF = beta * tanh(f(thekpoint, 1) - f(k,nk)) - 1


  end function bigF

  function zeem(k, nk)
    integer nk
    real*8 zeem(nk), k(3,nk) 

    zeem = az * (sin(k(2,:)) - 2 * cos(sqrt(3d0)/2d0 * k(1,:))*sin(k(2,:)/2d0)) * F(k,nk)

  end function zeem

  function rash(k, nk)
    integer nk
    real*8 k(3,nk), a(nk), b(nk)
    complex*16 rash(nk)

    a = -sin(k(2,:)) - cos(sqrt(3d0)/2d0 * k(1,:))*sin(k(2,:)/2d0)
    b = sqrt(3d0) * sin(sqrt(3d0)/2d0 * k(1,:))*cos(k(2,:)/2d0)
    rash = ar * dcmplx(a,b)

    end function rash



  function toyHamK(k, nk, H)
    integer*4 nk
    complex*16 toyHamK(nk, 2, 2)
    real*8 k(3, nk), H, real_k(3, nk)

    real_k(1,:) = k(2,:) * 2.0d0 /sqrt( 3d0)  + k(1,:) * 1d0/sqrt(3d0)
    real_k(2,:) = k(1,:)
    real_k(3,:) = 0d0

    toyHamK = (0d0, 0d0)


    toyHamK(:, 1, 1) = eps(real_k, nk) + zeem(real_k, nk)
    toyHamK(:, 1, 2) = conjg(rash(real_k,nk)) - mub * H
    toyHamK(:, 2, 1) = rash(real_k,nk) - mub * H
    toyHamK(:,2,2) = eps(real_k,nk) - zeem(real_k,nk)



  end function toyHamK


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

    matsufreq = (2d0*m+1)* pi * (T*kb)
  end function matsufreq

  function susc(T, H, theta, phi, nk, k)
    complex*16 Hamp(nk,2,2), Hamn(nk,2,2)
    real*8 T, H, freq, theta, phi
    real*8 susc, susc_arr(nk), k(3, nk)
    integer m, nk, mcount
    complex*16 gfp(nk,2,2), gfn(nk,2,2)

    
    Hamp = hamk(k, nk, H, theta, phi)
    Hamn = hamk(-k, nk, H, theta, phi)

    susc = 0d0
    mcount = 0
    do m=-freqmax,freqmax
      freq = matsufreq(T,m)
      gfp = greens(freq, Hamp, nk)
      gfn = greens(-freq, Hamn, nk)

      mcount = mcount + 1
      susc_arr = -kb* T *( gfp(:,1,1) * gfn(:,2,2) - gfp(:,1,2)*gfn(:,2,1))

      !write(*,*) sum(susc_arr)

      susc = susc + sum(susc_arr)
    enddo

    !write(*,*) susc

    susc = susc / (resolution*resolution)

  end function susc


  function epsilonk(ham, size)
    !----- variables
    real*8, allocatable:: ene(:,:), rwork(:)
    integer*4  lwork,info, size
    real*8 epsilonk(2, size)
    complex*16,allocatable:: work(:)
    complex*16 Ham(size,nb,nb)
    integer ki


    !------ get hamiltonian
    allocate(ene(nb,size))
    ene = 0d0

    !------- diagonalise

    lwork=max(1,2*nb-1)
    allocate(work(max(1,lwork)),rwork(max(1,3*nb-2)))

    do ki=1,size
      call zheev('V','U',nb,Ham(ki,:,:),nb,ene(:,ki),work,lwork,rwork,info)
    enddo

    !------ projections
    


    !------- get average of all energy bands at k-point

    epsilonk = ene

  end function epsilonk


  function proj_z(eigs)

    end function proj

  
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


end module hamtools