! subroutine realham_sub(hamr,nb,nr,ndeg,rvec)
subroutine realham_sub()

    use params

    implicit none
    
    character(len=80) hamil_file
    integer*4 ir, i, j, i1,i2
    real*8 a,b

    character(len=80), parameter:: prefix="MoS2"
    ! integer*4,allocatable:: ndeg(:) ! weighting used in FT
    ! real*8,allocatable:: rvec(:,:) ! list of realspace vectors
    ! integer nb, nr ! number of bands, number of real space points, number of k points
    ! complex*16, allocatable:: hamr(:,:,:) ! real space hamiltonian

    !--------- Ammend file name
    write(hamil_file,'(a,a)')trim(adjustl(prefix)),"_hr.dat"

    !------------- Read real-space hamiltonian
    open(99,file=trim(adjustl(hamil_file)),err=444)
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


    if(.true.) goto 333
    444   write(*,'(3a)')'ERROR: input file "',trim(adjustl(hamil_file)),' not found'
    stop
    333 close(99)
  
end subroutine realham_sub
  