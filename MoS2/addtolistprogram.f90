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
