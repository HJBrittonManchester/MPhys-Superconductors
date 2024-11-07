function susc(T, H, nk, k)
    complex*16 Hamp(nk,2,2), Hamn(nk,2,2)
    real*8 T, H, freq
    real*8 susc, susc_arr(nk), k(3, nk)
    integer m, nk
    complex*16 gfp(nk,2,2), gfn(nk,2,2)

    
    Hamp = hamk(k, nk, H)
    Hamn = Hamk(-k, nk, H)

    susc = 0d0
    do m=-int(freqmax),int(freqmax)
      freq = matsufreq(T,m)
      gfp = greens(freq, Hamp, nk)
      gfn = greens(-freq, Hamn, nk)


      susc_arr = real(-kb* T *( gfp(:,1,1) * gfn(:,2,2) - gfp(:,1,2)*gfn(:,2,1)))

      susc = susc + sum(susc_arr)
    enddo

    susc = susc / (resolution*resolution)

  end function susc
