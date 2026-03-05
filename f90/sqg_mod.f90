! ==============================================================
!  sqg_mod.f90  -  Surface Quasi-Geostrophic turbulence model
!  Serial version, Fortran 95
!  Translated from sqg.py (pyfftw serial version)
!
!  Physics
!  -------
!  r(1) = +r  (bottom Ekman),  r(2) = -r  (lid Ekman)
!  3/2-rule dealiasing via specpad / spectrunc
!  Additive hyperdiffusion: -(1/diff_efold)*(k/kc)^p * pvspec
!  RK4 time stepping
!
!  FFT backend
!  -----------
!  Uses FFTW3 single-precision (fftwf_*) called via the
!  iso_c_binding interface included in fftw3.f03.
!  Plans are created once in sqg_init and reused.
!
!  Dependencies
!  ------------
!    FFTW3 single-precision  (-lfftw3f)
!
!  Compile:
!    gfortran -std=f95 -O3 -march=native \
!             sqg_mod.f90 sqg_main.f90   \
!             -lfftw3f -lnetcdff -lnetcdf -lm
! ==============================================================

module sqg_mod
  use, intrinsic :: iso_c_binding
  implicit none

  include 'fftw3.f03'        ! provides fftwf_* interfaces

  private

  ! ---- precision kinds ----------------------------------------
  integer, parameter, public :: sp = c_float
  integer, parameter, public :: dp = c_double
  integer, parameter, public :: cp = c_float_complex

  ! ---- public interface ---------------------------------------
  public :: sqg_init, sqg_finalize
  public :: sqg_advance, sqg_timestep
  public :: sqg_t, sqg_N, sqg_f, sqg_U, sqg_L, sqg_H
  public :: sqg_nsq, sqg_tdiab, sqg_dt
  public :: sqg_diff_efold, sqg_diff_order, sqg_r

  ! ==============================================================
  !  Module-level model state  (all private)
  ! ==============================================================

  ! Grid sizes
  integer :: N_            ! global grid size
  integer :: Nc_           ! N/2 + 1
  integer :: N_pad_        ! 3*N/2  (padded grid)
  integer :: Nc_pad_       ! 3*N/4 + 1

  ! Physical parameters
  real(sp) :: f_, nsq_, L_, H_, U_
  real(sp) :: r_(2)
  real(sp) :: tdiab_, diff_efold_, theta0_, g_, dt_
  integer  :: diff_order_
  real(dp) :: t_

  ! Spectral operator arrays  (N_ x Nc_)
  real(sp), allocatable :: ksqlsq_(:,:)
  real(dp), allocatable :: Hovermu_(:,:), tanhmu_(:,:), sinhmu_(:,:)
  real(sp), allocatable :: hyperdiff_(:,:)
  complex(cp), allocatable :: ik_(:,:), il_(:,:)

  ! Padded spectral operator arrays  (N_pad_ x Nc_pad_)
  complex(cp), allocatable :: ik_pad_(:,:), il_pad_(:,:)

  ! Model spectral state  (2 x N_ x Nc_)
  complex(cp), allocatable :: pvspec_(:,:,:)
  complex(cp), allocatable :: pvspec_eq_(:,:,:)

  ! FFTW plans (held as opaque C pointers)
  type(c_ptr) :: plan_fwd_        ! N_ x N_       r2c
  type(c_ptr) :: plan_bwd_        ! N_ x N_       c2r
  type(c_ptr) :: plan_fwd_pad_    ! N_pad_ x N_pad_  r2c
  type(c_ptr) :: plan_bwd_pad_    ! N_pad_ x N_pad_  c2r

  ! FFTW-aligned persistent work buffers
  real(sp),    pointer :: buf_r_(:,:)        => null()  ! N_ x N_
  complex(cp), pointer :: buf_c_(:,:)        => null()  ! N_ x Nc_
  real(sp),    pointer :: buf_r_pad_(:,:)    => null()  ! N_pad_ x N_pad_
  complex(cp), pointer :: buf_c_pad_(:,:)    => null()  ! N_pad_ x Nc_pad_

contains

  ! ==============================================================
  !  sqg_init
  !  pv(2,N,N)  -  initial PV field
  ! ==============================================================
  subroutine sqg_init(pv, N_in, f, nsq, L, H, U, r,   &
                      tdiab, diff_order, diff_efold,    &
                      theta0, g, dt, tstart)

    integer,  intent(in) :: N_in
    real(sp), intent(in) :: pv(2, N_in, N_in)
    real(sp), intent(in) :: f, nsq, L, H, U, r
    real(sp), intent(in) :: tdiab
    integer,  intent(in) :: diff_order
    real(sp), intent(in) :: diff_efold, theta0, g, dt
    real(dp), intent(in) :: tstart

    ! locals
    type(c_ptr) :: cptr
    integer     :: i, j, ki, li
    real(sp)    :: pi, kv, lv, k_raw, l_raw, ktot, ktotcutoff
    real(sp)    :: l_fund, mu0, amp
    real(dp)    :: mu_d
    real(sp), allocatable :: pvbar(:,:,:)
    real(sp), allocatable :: k1d(:), l1d(:), k1d_pad(:), l1d_pad(:)

    ! ---- store parameters ------------------------------------
    N_          = N_in
    Nc_         = N_/2 + 1
    N_pad_      = (3*N_) / 2
    Nc_pad_     = (3*N_) / 4 + 1
    f_          = f
    nsq_        = nsq
    L_          = L
    H_          = H
    U_          = U
    r_(1)       =  r
    r_(2)       = -r
    tdiab_      = tdiab
    diff_order_ = diff_order
    diff_efold_ = diff_efold
    theta0_     = theta0
    g_          = g
    dt_         = dt
    t_          = tstart

    pi = 4.0_sp * atan(1.0_sp)

    ! ---- allocate operator arrays ----------------------------
    allocate(ksqlsq_(N_, Nc_), Hovermu_(N_, Nc_))
    allocate(tanhmu_(N_, Nc_), sinhmu_(N_, Nc_))
    allocate(hyperdiff_(N_, Nc_))
    allocate(ik_(N_, Nc_), il_(N_, Nc_))
    allocate(ik_pad_(N_pad_, Nc_pad_), il_pad_(N_pad_, Nc_pad_))
    allocate(pvspec_(2, N_, Nc_), pvspec_eq_(2, N_, Nc_))

    ! ---- 1-D wavenumber arrays (rfftfreq / fftfreq style) ----
    ! k: rfftfreq  index 0 .. Nc_-1
    allocate(k1d(Nc_))
    do i = 1, Nc_
      k1d(i) = real(i-1, sp)
    end do

    ! l: fftfreq  index 0..N/2, then -N/2+1..-1
    allocate(l1d(N_))
    do i = 1, N_
      li = i - 1
      if (li > N_/2) li = li - N_
      l1d(i) = real(li, sp)
    end do

    ! padded k: rfftfreq on N_pad_
    allocate(k1d_pad(Nc_pad_))
    do i = 1, Nc_pad_
      k1d_pad(i) = real(i-1, sp)
    end do

    ! padded l: fftfreq on N_pad_
    allocate(l1d_pad(N_pad_))
    do i = 1, N_pad_
      li = i - 1
      if (li > N_pad_/2) li = li - N_pad_
      l1d_pad(i) = real(li, sp)
    end do

    ! ---- fill 2-D operator arrays ----------------------------
    ktotcutoff = pi * real(N_, sp) / L_

    do j = 1, Nc_
      kv = 2.0_sp * pi * k1d(j) / L_
      do i = 1, N_
        lv = 2.0_sp * pi * l1d(i) / L_

        ksqlsq_(i,j) = kv*kv + lv*lv
        ik_(i,j)     = cmplx(0.0_sp,  kv, kind=cp)
        il_(i,j)     = cmplx(0.0_sp,  lv, kind=cp)

        ! mu in double to avoid sinh overflow
        mu_d = sqrt(real(ksqlsq_(i,j), dp)) &
             * sqrt(real(nsq_, dp))          &
             * real(H_, dp) / real(f_, dp)
        if (mu_d < epsilon(1.0_sp)) mu_d = epsilon(1.0_sp)
        Hovermu_(i,j) = real(real(H_, dp) / mu_d, sp)
        tanhmu_(i,j)  = real(tanh(mu_d), sp)
        sinhmu_(i,j)  = real(sinh(mu_d), sp)

        ktot              = sqrt(ksqlsq_(i,j))
        hyperdiff_(i,j)   = -(1.0_sp/diff_efold_) &
                            * (ktot/ktotcutoff)**real(diff_order_, sp)
      end do
    end do

    ! padded wavenumber operators
    do j = 1, Nc_pad_
      kv = 2.0_sp * pi * k1d_pad(j) / L_
      do i = 1, N_pad_
        lv = 2.0_sp * pi * l1d_pad(i) / L_
        ik_pad_(i,j) = cmplx(0.0_sp, kv, kind=cp)
        il_pad_(i,j) = cmplx(0.0_sp, lv, kind=cp)
      end do
    end do

    deallocate(k1d, l1d, k1d_pad, l1d_pad)

    ! ---- FFTW aligned work buffers ---------------------------
    cptr = fftwf_alloc_real(int(N_*N_,       c_size_t))
    call c_f_pointer(cptr, buf_r_,     [N_, N_])

    cptr = fftwf_alloc_complex(int(N_*Nc_,    c_size_t))
    call c_f_pointer(cptr, buf_c_,     [N_, Nc_])

    cptr = fftwf_alloc_real(int(N_pad_*N_pad_, c_size_t))
    call c_f_pointer(cptr, buf_r_pad_, [N_pad_, N_pad_])

    cptr = fftwf_alloc_complex(int(N_pad_*Nc_pad_, c_size_t))
    call c_f_pointer(cptr, buf_c_pad_, [N_pad_, Nc_pad_])

    ! ---- FFTW plans (MEASURE for best run-time performance) --
    plan_fwd_ = fftwf_plan_dft_r2c_2d( &
        N_, N_, buf_r_, buf_c_, FFTW_MEASURE)

    plan_bwd_ = fftwf_plan_dft_c2r_2d( &
        N_, N_, buf_c_, buf_r_, FFTW_MEASURE)

    plan_fwd_pad_ = fftwf_plan_dft_r2c_2d( &
        N_pad_, N_pad_, buf_r_pad_, buf_c_pad_, FFTW_MEASURE)

    plan_bwd_pad_ = fftwf_plan_dft_c2r_2d( &
        N_pad_, N_pad_, buf_c_pad_, buf_r_pad_, FFTW_MEASURE)

    ! ---- basic-state PV pvbar --------------------------------
    allocate(pvbar(2, N_, N_))
    l_fund = 2.0_sp * pi / L_
    mu0    = l_fund * sqrt(nsq_) * H_ / f_
    amp    = -(mu0 * 0.5_sp * U_ / (l_fund * H_)) &
             * cosh(0.5_sp*mu0) / sinh(0.5_sp*mu0)

    do j = 1, N_
      do i = 1, N_
        ! pvbar depends only on x (second index in Python = column = j here)
        pvbar(1,i,j) = amp * cos(l_fund * real(j-1, sp) * L_ / real(N_, sp))
        pvbar(2,i,j) = pvbar(1,i,j)
      end do
    end do

    ! ---- initial spectral state ------------------------------
    do i = 1, 2
      call do_rfft2(pvbar(i,:,:), pvspec_eq_(i,:,:))
      call do_rfft2(pv(i,:,:),    pvspec_(i,:,:))
    enddo

    deallocate(pvbar)
  end subroutine sqg_init

  ! ==============================================================
  !  sqg_finalize
  ! ==============================================================
  subroutine sqg_finalize()
    call fftwf_destroy_plan(plan_fwd_)
    call fftwf_destroy_plan(plan_bwd_)
    call fftwf_destroy_plan(plan_fwd_pad_)
    call fftwf_destroy_plan(plan_bwd_pad_)
    call fftwf_free(c_loc(buf_r_(1,1)))
    call fftwf_free(c_loc(buf_c_(1,1)))
    call fftwf_free(c_loc(buf_r_pad_(1,1)))
    call fftwf_free(c_loc(buf_c_pad_(1,1)))
    if (allocated(ksqlsq_))    deallocate(ksqlsq_)
    if (allocated(Hovermu_))   deallocate(Hovermu_)
    if (allocated(tanhmu_))    deallocate(tanhmu_)
    if (allocated(sinhmu_))    deallocate(sinhmu_)
    if (allocated(hyperdiff_)) deallocate(hyperdiff_)
    if (allocated(ik_))        deallocate(ik_)
    if (allocated(il_))        deallocate(il_)
    if (allocated(ik_pad_))    deallocate(ik_pad_)
    if (allocated(il_pad_))    deallocate(il_pad_)
    if (allocated(pvspec_))    deallocate(pvspec_)
    if (allocated(pvspec_eq_)) deallocate(pvspec_eq_)
  end subroutine sqg_finalize

  ! ==============================================================
  !  sqg_advance  -  step forward ntimesteps, return physical PV
  ! ==============================================================
  subroutine sqg_advance(ntimesteps, pv_out)
    integer,  intent(in)  :: ntimesteps
    real(sp), intent(out) :: pv_out(2, N_, N_)
    integer :: n
    do n = 1, ntimesteps
      call sqg_timestep()
    end do
    do n = 1, 2
      call do_irfft2(pvspec_(n,:,:), pv_out(n,:,:))
    enddo
  end subroutine sqg_advance

  ! ==============================================================
  !  sqg_timestep  -  4th-order Runge-Kutta
  ! ==============================================================
  subroutine sqg_timestep()
    complex(cp), allocatable :: k1(:,:,:), k2(:,:,:), k3(:,:,:), k4(:,:,:)
    complex(cp), allocatable :: tmp(:,:,:)
    integer :: sz2, sz3

    allocate(k1(2,N_,Nc_), k2(2,N_,Nc_), k3(2,N_,Nc_), k4(2,N_,Nc_))
    allocate(tmp(2,N_,Nc_))

    call gettend(pvspec_, k1)

    tmp = pvspec_ + 0.5_sp * dt_ * k1
    call gettend(tmp, k2)

    tmp = pvspec_ + 0.5_sp * dt_ * k2
    call gettend(tmp, k3)

    tmp = pvspec_ + dt_ * k3
    call gettend(tmp, k4)

    pvspec_ = pvspec_ + (dt_/6.0_sp)*(k1 + 2.0_sp*k2 + 2.0_sp*k3 + k4)
    t_ = t_ + real(dt_, dp)

    deallocate(k1, k2, k3, k4, tmp)
  end subroutine sqg_timestep

  ! ==============================================================
  !  Accessors (public)
  ! ==============================================================
  function sqg_t()           result(v); real(dp) :: v; v = t_;          end function
  function sqg_N()           result(v); integer  :: v; v = N_;          end function
  function sqg_f()           result(v); real(sp) :: v; v = f_;          end function
  function sqg_U()           result(v); real(sp) :: v; v = U_;          end function
  function sqg_L()           result(v); real(sp) :: v; v = L_;          end function
  function sqg_H()           result(v); real(sp) :: v; v = H_;          end function
  function sqg_nsq()         result(v); real(sp) :: v; v = nsq_;        end function
  function sqg_tdiab()       result(v); real(sp) :: v; v = tdiab_;      end function
  function sqg_dt()          result(v); real(sp) :: v; v = dt_;         end function
  function sqg_diff_efold()  result(v); real(sp) :: v; v = diff_efold_; end function
  function sqg_diff_order()  result(v); integer  :: v; v = diff_order_; end function
  function sqg_r(k)          result(v)
    integer, intent(in) :: k; real(sp) :: v; v = r_(k)
  end function

  ! ==============================================================
  !  Private procedures
  ! ==============================================================

  ! ------------------------------------------------------------
  !  invert  -  boundary PV -> streamfunction (spectral)
  ! ------------------------------------------------------------
  subroutine invert(pvspec_in, psispec)
    complex(cp), intent(in)  :: pvspec_in(2, N_, Nc_)
    complex(cp), intent(out) :: psispec(2, N_, Nc_)
    integer :: i, j

    do j = 1, Nc_
      do i = 1, N_
        psispec(1,i,j) = Hovermu_(i,j) * &
          ( pvspec_in(2,i,j)/sinhmu_(i,j) - pvspec_in(1,i,j)/tanhmu_(i,j) )
        psispec(2,i,j) = Hovermu_(i,j) * &
          ( pvspec_in(2,i,j)/tanhmu_(i,j) - pvspec_in(1,i,j)/sinhmu_(i,j) )
      end do
    end do
  end subroutine invert

  ! ------------------------------------------------------------
  !  specpad  -  zero-pad to 3/2 grid, multiply by 2.25
  ! ------------------------------------------------------------
  subroutine specpad(specarr, specarr_pad)
    complex(cp), intent(in)  :: specarr(2, N_, Nc_)
    complex(cp), intent(out) :: specarr_pad(2, N_pad_, Nc_pad_)
    integer :: k, i, j
    integer :: nh, nh_pad

    nh     = N_  / 2
    nh_pad = N_pad_ / 2

    specarr_pad = cmplx(0.0_sp, 0.0_sp, kind=cp)

    do k = 1, 2
      ! positive-l rows  1 .. N/2
      do i = 1, nh
        do j = 1, nh
          specarr_pad(k, i, j) = 2.25_sp * specarr(k, i, j)
        end do
      end do

      ! negative-l rows  N_pad-N/2+1 .. N_pad
      do i = 1, nh
        do j = 1, nh
          specarr_pad(k, N_pad_-nh+i, j) = 2.25_sp * specarr(k, N_-nh+i, j)
        end do
      end do

      ! negative Nyquist column (j = N/2+1 in 1-based = N/2 in 0-based)
      ! Python: specarr_pad[:,0:N/2, N/2] = conj(2.25*specarr[:,0:N/2,-1])
      !         specarr_pad[:,-N/2:, N/2] = conj(2.25*specarr[:,-N/2:,-1])
      do i = 1, nh
        specarr_pad(k, i,          nh+1) = conjg(2.25_sp * specarr(k, i,      Nc_))
        specarr_pad(k, N_pad_-nh+i, nh+1) = conjg(2.25_sp * specarr(k, N_-nh+i, Nc_))
      end do
    end do
  end subroutine specpad

  ! ------------------------------------------------------------
  !  spectrunc  -  truncate padded spectral array back to N
  ! ------------------------------------------------------------
  subroutine spectrunc(specarr_pad, specarr)
    complex(cp), intent(in)  :: specarr_pad(2, N_pad_, Nc_pad_)
    complex(cp), intent(out) :: specarr(2, N_, Nc_)
    integer :: k, i, j, nh

    nh = N_ / 2
    specarr = cmplx(0.0_sp, 0.0_sp, kind=cp)

    do k = 1, 2
      ! positive-l rows
      do i = 1, nh
        do j = 1, nh
          specarr(k, i, j) = specarr_pad(k, i, j)
        end do
      end do
      ! negative-l rows
      do i = 1, nh
        do j = 1, nh
          specarr(k, N_-nh+i, j) = specarr_pad(k, N_pad_-nh+i, j)
        end do
      end do
    end do
  end subroutine spectrunc

  ! ------------------------------------------------------------
  !  xyderiv  -  x/y derivatives on the dealiased padded grid
  ! ------------------------------------------------------------
  subroutine xyderiv(specarr, xderiv, yderiv)
    complex(cp), intent(in)  :: specarr(2, N_, Nc_)
    real(sp),    intent(out) :: xderiv(2, N_pad_, N_pad_)
    real(sp),    intent(out) :: yderiv(2, N_pad_, N_pad_)

    complex(cp), allocatable :: pad(:,:,:), xs(:,:,:), ys(:,:,:)
    integer :: k

    allocate(pad(2,N_pad_,Nc_pad_))
    allocate(xs (2,N_pad_,Nc_pad_))
    allocate(ys (2,N_pad_,Nc_pad_))

    call specpad(specarr, pad)

    do k = 1, 2
      xs(k,:,:) = ik_pad_ * pad(k,:,:)
      ys(k,:,:) = il_pad_ * pad(k,:,:)
      call do_irfft2_pad(xs(k,:,:), xderiv(k,:,:))
      call do_irfft2_pad(ys(k,:,:), yderiv(k,:,:))
    end do

    deallocate(pad, xs, ys)
  end subroutine xyderiv

  ! ------------------------------------------------------------
  !  gettend  -  compute dpvspec/dt
  ! ------------------------------------------------------------
  subroutine gettend(pvspec_in, dpvdt)
    complex(cp), intent(in)  :: pvspec_in(2, N_, Nc_)
    complex(cp), intent(out) :: dpvdt(2, N_, Nc_)

    complex(cp), allocatable :: psispec(:,:,:), jspec_pad(:,:,:), jspec(:,:,:)
    real(sp),    allocatable :: psix(:,:,:), psiy(:,:,:)
    real(sp),    allocatable :: pvx(:,:,:),  pvy(:,:,:)
    real(sp),    allocatable :: jacobian(:,:,:)
    integer :: k, i, j

    allocate(psispec(2,N_,Nc_))
    allocate(psix(2,N_pad_,N_pad_), psiy(2,N_pad_,N_pad_))
    allocate(pvx (2,N_pad_,N_pad_), pvy (2,N_pad_,N_pad_))
    allocate(jacobian(2,N_pad_,N_pad_))
    allocate(jspec_pad(2,N_pad_,Nc_pad_), jspec(2,N_,Nc_))

    call invert(pvspec_in, psispec)
    call xyderiv(psispec,   psix, psiy)
    call xyderiv(pvspec_in, pvx,  pvy)

    ! Jacobian  J(psi,pv) = psi_x * pv_y - psi_y * pv_x
    jacobian = psix * pvy - psiy * pvx

    ! Forward FFT of Jacobian then truncate
    do k = 1, 2
      call do_rfft2_pad(jacobian(k,:,:), jspec_pad(k,:,:))
    enddo
    call spectrunc(jspec_pad, jspec)

    ! Tendency:
    !   (pvspec_eq - pvspec_in)/tdiab
    !   - jspec
    !   + r(k)*ksqlsq*psispec     (Ekman)
    !   + hyperdiff*pvspec_        (hyperdiffusion, uses current pvspec_)
    do k = 1, 2
      do j = 1, Nc_
        do i = 1, N_
          dpvdt(k,i,j) = (pvspec_eq_(k,i,j) - pvspec_in(k,i,j)) / tdiab_ &
                        - jspec(k,i,j)                                      &
                        + cmplx(r_(k)*ksqlsq_(i,j), 0.0_sp, kind=cp)       &
                          * psispec(k,i,j)                                  &
                        + cmplx(hyperdiff_(i,j), 0.0_sp, kind=cp)           &
                          * pvspec_(k,i,j)
        end do
      end do
    end do

    deallocate(psispec, psix, psiy, pvx, pvy, jacobian, jspec_pad, jspec)
  end subroutine gettend

  ! ============================================================
  !  Low-level FFT wrappers  (use persistent aligned buffers +
  !  new-array execute API so plans are never recreated)
  ! ============================================================

  ! -- forward r2c on N_ x N_ grid --
  subroutine do_rfft2(grid_in, spec_out)
    real(sp),    intent(in)  :: grid_in(N_, N_)
    complex(cp), intent(out) :: spec_out(N_, Nc_)
    buf_r_ = grid_in
    call fftwf_execute_dft_r2c(plan_fwd_, buf_r_, buf_c_)
    spec_out = buf_c_
  end subroutine do_rfft2

  ! -- backward c2r on N_ x N_ grid, normalised --
  subroutine do_irfft2(spec_in, grid_out)
    complex(cp), intent(in)  :: spec_in(N_, Nc_)
    real(sp),    intent(out) :: grid_out(N_, N_)
    real(sp) :: norm
    norm   = 1.0_sp / real(N_*N_, sp)
    buf_c_ = spec_in
    call fftwf_execute_dft_c2r(plan_bwd_, buf_c_, buf_r_)
    grid_out = buf_r_ * norm
  end subroutine do_irfft2

  ! -- forward r2c on padded N_pad_ x N_pad_ grid --
  subroutine do_rfft2_pad(grid_in, spec_out)
    real(sp),    intent(in)  :: grid_in(N_pad_, N_pad_)
    complex(cp), intent(out) :: spec_out(N_pad_, Nc_pad_)
    buf_r_pad_ = grid_in
    call fftwf_execute_dft_r2c(plan_fwd_pad_, buf_r_pad_, buf_c_pad_)
    spec_out = buf_c_pad_
  end subroutine do_rfft2_pad

  ! -- backward c2r on padded grid, normalised --
  subroutine do_irfft2_pad(spec_in, grid_out)
    complex(cp), intent(in)  :: spec_in(N_pad_, Nc_pad_)
    real(sp),    intent(out) :: grid_out(N_pad_, N_pad_)
    real(sp) :: norm
    norm       = 1.0_sp / real(N_pad_*N_pad_, sp)
    buf_c_pad_ = spec_in
    call fftwf_execute_dft_c2r(plan_bwd_pad_, buf_c_pad_, buf_r_pad_)
    grid_out = buf_r_pad_ * norm
  end subroutine do_irfft2_pad

end module sqg_mod
