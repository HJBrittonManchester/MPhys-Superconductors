module params

  implicit none

  !---------- user defined parameters
  character(len=80), parameter:: prefix="MoS2"
  integer*4,parameter:: resolution= 140
  real*8,parameter::ef= .91d0 ! original value is 4.18903772 ! Fermi energy
  real*8, parameter::kb = 8.6173d-5
  real*8, parameter::mub = 5.7883817982d-5
  integer, parameter:: freqmax = 500 

  !---------- global variables given values by reading in the hamiltonian
  integer*4,allocatable:: ndeg(:) ! weighting used in FT
  real*8,allocatable:: rvec(:,:) ! list of realspace vectors
  integer*4 nb, nr! number of bands, number of real space points, number of k points
  complex*16, allocatable:: hamr(:,:,:), hamk(:,:,:)
  real*8, allocatable:: kpoints(:,:), energy(:), validkpoints(:,:)
  !complex*16, allocatable:: ham(:,:,:,:), gf(:,:,:,:), newham(:,:,:)
  !----------
  real*8 T, chi0, v, Hu, Hl, deltaU, deltaL, Hprev, deltaPrev
  integer ki, nk, it, steps
  character(len=16) efmt, kfmt
  character(len=4) intstring
  real*8 singlek(3)
  complex*16 oldsampleham(2,2), sampleham(2,2)



  !--------------------- Defining the workflow ------------------------
  !---  1. Create the real space hamiltonian with realham
  !---  2. Calculate k-space hamiltonian with of whole BZ with hamk
  !---  3. Diagonalise with epsilon to find average energy
  !---  4. Create the array of valid kpoints
  !---  5. calcualte GF and then susc for these

end module params