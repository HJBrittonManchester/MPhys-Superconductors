      Program Wannier_band_structure
      Implicit None
!--------to be midified by the usere
      character(len=80):: prefix="MoS2"
      integer,parameter::nkpath=4,np=100
      real*8,parameter::ef= 0.915 !4.18903772
!------------------------------------------------------
      integer*4 ik,ikmax,ir
      real*8 kz
      character(len=30)::klabel(nkpath)
      character(len=80) hamil_file,nnkp,line
      integer*4,parameter::nk=(nkpath-1)*np+1
      integer*4 i,j,k,nr,i1,i2,nb,lwork,info
      real*8,parameter::third=1d0/3d0!,kz=0d0
      real*8 phase,pi2,jk,a,b
      real*8 klist(3,1:nk),xk(nk),kpath(3,np),bvec(3,3),ktemp1(3),ktemp2(3),xkl(nkpath)
      real*8,allocatable:: rvec(:,:),ene(:,:),rwork(:)
      integer*4,allocatable:: ndeg(:)
      complex*16,allocatable:: Hk(:,:),Hamr(:,:,:),work(:)
      real*8 temp_proj_z,temp_proj_x
!------------------------------------------------------
      write(hamil_file,'(a,a)')trim(adjustl(prefix)),"_hr.dat"
      write(nnkp,'(a,a)')      trim(adjustl(prefix)),".nnkp"

      pi2=4.0d0*atan(1.0d0)*2.0d0

!---------------  reciprocal vectors
      open(98,file=trim(adjustl(nnkp)),err=333)
111   read(98,'(a)')line
      if(trim(adjustl(line)).ne."begin recip_lattice") goto 111
      
      read(98,*)bvec
!---------------kpath
      data kpath(:,1) /     0.0d0,      0.0d0,    0.0d0/  !G
      data kpath(:,2) /     0.5d0,      0.0d0,    0.0d0/  !M
      data kpath(:,3) /     third,      third,    0.0d0/  !K
      data kpath(:,4) /     0.0d0,      0.0d0,    0.0d0/  !G

!      open(77,file='tmp')
 !     read(77,*)ik,ikmax
 !     kz=float(ik)*0.5d0/float(ikmax)
 !     kpath(3,:)=kz

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
      klist(:,nk)=kpath(:,nkpath)
      ktemp2=klist(1,nk)*bvec(:,1)+klist(2,nk)*bvec(:,2)+klist(3,nk)*bvec(:,3)
      xk(nk)=xk(nk-1)+sqrt(dot_product(ktemp2-ktemp1,ktemp2-ktemp1))
      xkl(nkpath)=xk(nk)
      write(*,*)klist
      klist=klist*pi2

      !write(*,*)klist(:,:np+1)

!------read H(R)
      open(99,file=trim(adjustl(hamil_file)),err=444)
      open(100,file='band.dat')
      read(99,*)
      read(99,*)nb,nr
      allocate(rvec(3,nr),Hk(nb,nb),Hamr(nb,nb,nr),ndeg(nr),ene(nb,nk))
      read(99,*)ndeg
      do ir=1,nr
         do i=1,nb
            do j=1,nb
               read(99,*)rvec(1,ir),rvec(2,ir),rvec(3,ir),i1,i2,a,b
               hamr(i1,i2,ir)=dcmplx(a,b)
            enddo
         enddo
      enddo

     lwork=max(1,2*nb-1)
     allocate(work(max(1,lwork)),rwork(max(1,3*nb-2)))

!---- Fourrier transform H(R) to H(k)
     open(104, file="zproj.dat")
      ene=0d0
      do k=1,nk
         HK=(0d0,0d0)
         do j=1,nr

            phase=0.0d0
            do i=1,3
               phase=phase+klist(i,k)*rvec(i,j)
            enddo

            Hk(:,:)=Hk(:,:)+Hamr(:,:,j)*dcmplx(cos(phase),-sin(phase))/float(ndeg(j))

         enddo

         HK(1,1) = HK(1,1) - ef
         HK(2,2) = HK(2,2) - ef

         call zheev('V','U',nb,Hk,nb,ene(:,k),work,lwork,rwork,info)

         
         
         temp_proj_z = real( Hk(2,1)*conjg(Hk(2,1)) - Hk(1,2)*conjg(Hk(1,2)) )

         write(104,*)temp_proj_z
            
      

         !write(*,*)Hk
         
      enddo

      close(104)
      
      do i=1,nb
         do k=1,nk
           write(100,'(2(x,f12.6))') xk(k),ene(i,k)
         enddo
           write(100,*)
           write(100,*)
      enddo
      stop
333   write(*,'(3a)')'ERROR: input file "',trim(adjustl(nnkp)),' not found'
      stop
444   write(*,'(3a)')'ERROR: input file "',trim(adjustl(hamil_file)),' not found'
      stop

      end

