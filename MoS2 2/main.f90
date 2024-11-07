Program main

    use params

    !--------------------- Defining the workflow ------------------------
    !---  1. Create the real space hamiltonian with realham
    !---  2. Calculate k-space hamiltonian with of whole BZ with hamk
    !---  3. Diagonalise with epsilon to find average energy
    !---  4. Create the array of valid kpoints
    !---  5. calcualte GF and then susc for these

    implicit none

    !------ parameters for the program
    ! character(len=80), parameter:: prefix="MoS2"
    integer*4,parameter:: resolution= 140
    real*8,parameter:: threshold= 0.01
    real*8,parameter::ef= .91d0 ! original value is 4.18903772 ! Fermi energy
    real*8, parameter::kb = 8.6173d-5
    real*8, parameter::mub = 5.7883817982d-5
    integer, parameter:: freqmax = 500 

    !------- global variables
    ! integer*4,allocatable:: ndeg(:) ! weighting used in FT
    ! real*8,allocatable:: rvec(:,:) ! list of realspace vectors
    ! integer nb, nr!, nk, nk_valid! number of bands, number of real space points, number of k points
    ! complex*16, allocatable:: hamr(:,:,:)!, hamk(:,:,:) ! real space and k space hamiltonian
    !real*8, allocatable:: kpoints(:,:), energy(:), validkpoints(:,:) ! for filtering points around ef

    !nb = 2
    !nk = resolution*resolution

    !------- read real space hamiltonian
    ! call realham_sub(hamr,nb,nr,ndeg,rvec)
    call realham_sub()

    !------- create (3,resolution,resolution) array of k points spanning bz
    call karray_sub(kpoints,nk,resolution)

    !------- take fourier transform
    ! call hamk_sub(hamk,hamr,kpoints,nk,nr,nb)

    !------- generate valid k points

    deallocate(hamr,rvec,ndeg)

end