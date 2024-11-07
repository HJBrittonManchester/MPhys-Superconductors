module params

    implicit none

    integer*4,allocatable:: ndeg(:) ! weighting used in FT
    real*8,allocatable:: rvec(:,:), kpoints(:,:), energy(:), validkpoints(:,:) ! list of realspace vectors
    integer nb, nr!, nk, nk_valid! number of bands, number of real space points, number of k points
    complex*16, allocatable:: hamr(:,:,:)
    ! real*8, allocatable:: kpoints(:,:), energy(:), validkpoints(:,:) ! for filtering points around ef

end module params