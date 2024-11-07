subroutine hamk_sub(hamk, hamr, k, nk) !, H_mag, theta, phi)

  use params 

  implicit none

  integer j, i, ki, nk
  real*8 phase, k(3,nk)!, H_mag, theta, phi
  ! complex*16, allocatable:: hamk(:,:,:), hamr(:,:,:)
  complex*16 hamk(nk,2,2), hamr(nk,2,2)
  real*8 phase!, H
  ! real*8, allocatable:: k(:,:)

  ! allocate(hamk(nk,2,2))
  ! allocate(hamr(nk,2,2))
  ! allocate(k(3,nk))

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

  ! put these in a different file
  !------ adjust fermi surface 
  ! hamk(:,1,1) = hamk(:,1,1) - dcmplx(ef,0d0)
  ! hamk(:,2,2) = hamk(:,2,2) - dcmplx(ef,0d0)

  !------- add H field term  along x axis
  ! hamk(:, 1, 2) = hamk(:,1,2) - H * mub
  ! hamk(:, 2, 1) = hamk(:,2,1) - H * mub
  
end subroutine hamk_sub
