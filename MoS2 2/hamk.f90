subroutine hamk_sub(hamk, hamr, k, nk, nr, nb) !, H_mag, theta, phi)
  
    implicit none
  
    integer*4 j, i, ki, nk, nr, nb, ndeg(nr)
    real*8 phase, k(3,nk), rvec(3,nr)
    complex*16 hamr(nb,nb,nr), hamk(nb,nb,nk) ! real space and k space hamiltonian
  
    hamk = (0d0, 0d0)
    do ki=1,nk
      do j=1,nr
        phase=0.0d0
        do i=1,3
            phase=phase+k(i,ki)*rvec(i,j)
            
        enddo
        hamk(:,:,ki)=hamk(:,:,ki) + hamr(:,:,j)*dcmplx(cos(phase),-sin(phase))/float(ndeg(j))
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
  