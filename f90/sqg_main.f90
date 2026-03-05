! ==============================================================
!  sqg_main.f90  -  Main program for the serial SQG model
!  Fortran 95, equivalent of the __main__ block in sqg.py
!
!  Compile:
!    gfortran -std=f95 -O3 -march=native \
!             sqg_mod.f90 sqg_main.f90   \
!             -lfftw3f -lnetcdff -lnetcdf -lm
!  Run:
!    ./sqg_mpi
! ==============================================================

program sqg_main
  use sqg_mod
  use netcdf
  implicit none

  ! ---- model parameters ------------------------------------
  integer,  parameter :: N         = 64
  real(sp), parameter :: dt_param  = 1800.0_sp
  real(sp), parameter :: diff_efold = 86400._sp
  integer,  parameter :: norder    = 8
  real(sp), parameter :: r         = 0.0_sp
  real(sp), parameter :: nsq       = 1.0e-4_sp
  real(sp), parameter :: f         = 1.0e-4_sp
  real(sp), parameter :: g_param   = 9.8_sp
  real(sp), parameter :: theta0    = 300.0_sp
  real(sp), parameter :: H         = 10.0e3_sp
  real(sp), parameter :: U         = 20.0_sp
  real(sp), parameter :: L         = 20.0e6_sp
  real(sp), parameter :: tdiab     = 10.0_sp * 86400.0_sp
  real(sp), parameter :: scalefact = f * theta0 / g_param

  real(dp), parameter :: outputinterval = 6.0_dp  * 3600.0_dp
  real(dp), parameter :: tmax           = 300.0_dp * 86400.0_dp

  ! ---- local variables -------------------------------------
  integer  :: ntimesteps, nout, iy, ix, k
  real(sp) :: pv(2, N, N), pv_out(2, N, N)
  real(sp) :: pvmin, pvmax, pi2, x_val, y_val, blob
  real(dp) :: t_now

  ! NetCDF handles
  integer :: ncid, var_pv, var_x, var_y, var_z, var_t
  integer :: dim_x, dim_y, dim_z, dim_t
  integer :: pv_dims(4)
  real(sp) :: xc(N), yc(N), zc(2), tval(1)
  integer  :: start4(4), count4(4), start1(1)

  ! Random seed
  integer :: seed(8)

  pi2 = 8.0_sp * atan(1.0_sp)   ! 2*pi

  ! ---- initial PV ------------------------------------------
  ! Gaussian noise
  !seed = 42
  !call random_seed(put=seed)
  !call random_number(pv)
  !pv = (pv - 0.5_sp) * 200.0_sp    ! rough N(0,100)
  pv = 0

  ! Isolated blob on lid (component 2)
  do iy = 1, N
    y_val = real(iy-1, sp) * pi2 / real(N, sp)
    do ix = 1, N
      x_val = real(ix-1, sp) * pi2 / real(N, sp)
      blob  = 2000.0_sp &
            * (sin(0.5_sp*x_val)**20) * (sin(y_val)**20)
      pv(2,iy,ix) = pv(2,iy,ix) + blob
    end do
  end do

  ! Remove area mean from each level
  do k = 1, 2
    pv(k,:,:) = pv(k,:,:) - sum(pv(k,:,:)) / real(N*N, sp)
  end do

  ! ---- initialise model ------------------------------------
  call sqg_init(pv, N, f, nsq, L, H, U, r, tdiab, &
                norder, diff_efold, theta0, g_param, dt_param, 0.0_dp)

  ntimesteps = int(outputinterval / sqg_dt())
  write(*,'(a,i4,a,f6.0,a,i4)') &
    'SQG model: N=', sqg_N(), '  dt=', real(sqg_dt()), &
    ' s  ntimesteps=', ntimesteps
  print *,'min/max initial pv',scalefact*minval(pv),scalefact*maxval(pv)

  ! ---- create NetCDF output file ---------------------------
  call nc_check( nf90_create('sqg.nc', ior(NF90_CLOBBER,NF90_NETCDF4), ncid) )

  ! global attributes
  call nc_check( nf90_put_att(ncid, NF90_GLOBAL, 'r',          sqg_r(1)) )
  call nc_check( nf90_put_att(ncid, NF90_GLOBAL, 'f',          sqg_f()) )
  call nc_check( nf90_put_att(ncid, NF90_GLOBAL, 'U',          sqg_U()) )
  call nc_check( nf90_put_att(ncid, NF90_GLOBAL, 'L',          sqg_L()) )
  call nc_check( nf90_put_att(ncid, NF90_GLOBAL, 'H',          sqg_H()) )
  call nc_check( nf90_put_att(ncid, NF90_GLOBAL, 'g',          g_param) )
  call nc_check( nf90_put_att(ncid, NF90_GLOBAL, 'theta0',     theta0) )
  call nc_check( nf90_put_att(ncid, NF90_GLOBAL, 'nsq',        sqg_nsq()) )
  call nc_check( nf90_put_att(ncid, NF90_GLOBAL, 'tdiab',      sqg_tdiab()) )
  call nc_check( nf90_put_att(ncid, NF90_GLOBAL, 'dt',         sqg_dt()) )
  call nc_check( nf90_put_att(ncid, NF90_GLOBAL, 'diff_efold', sqg_diff_efold()) )
  call nc_check( nf90_put_att(ncid, NF90_GLOBAL, 'diff_order', sqg_diff_order()) )

  ! dimensions
  call nc_check( nf90_def_dim(ncid, 'x', N,             dim_x) )
  call nc_check( nf90_def_dim(ncid, 'y', N,             dim_y) )
  call nc_check( nf90_def_dim(ncid, 'z', 2,             dim_z) )
  call nc_check( nf90_def_dim(ncid, 't', NF90_UNLIMITED, dim_t) )

  ! coordinate variables
  call nc_check( nf90_def_var(ncid, 'x', NF90_FLOAT, [dim_x], var_x) )
  call nc_check( nf90_def_var(ncid, 'y', NF90_FLOAT, [dim_y], var_y) )
  call nc_check( nf90_def_var(ncid, 'z', NF90_FLOAT, [dim_z], var_z) )
  call nc_check( nf90_def_var(ncid, 't', NF90_FLOAT, [dim_t], var_t) )
  call nc_check( nf90_put_att(ncid, var_x, 'units', 'meters') )
  call nc_check( nf90_put_att(ncid, var_y, 'units', 'meters') )
  call nc_check( nf90_put_att(ncid, var_z, 'units', 'meters') )
  call nc_check( nf90_put_att(ncid, var_t, 'units', 'seconds') )

  ! PV variable (t, z, y, x) with deflate compression
  ! Note: NetCDF-Fortran stores in column-major order, so the
  ! dimension list is reversed relative to the Python (row-major) order:
  !   Python:   pv(t, z, y, x)  -> dims = [dim_t, dim_z, dim_y, dim_x]
  !   Fortran:  pv(x, y, z, t)  -> dims = [dim_x, dim_y, dim_z, dim_t]
  pv_dims = [dim_x, dim_y, dim_z, dim_t]
  call nc_check( nf90_def_var(ncid, 'pv', NF90_FLOAT, pv_dims, var_pv) )
  call nc_check( nf90_def_var_deflate(ncid, var_pv, &
                 shuffle=1, deflate=1, deflate_level=1) )
  call nc_check( nf90_put_att(ncid, var_pv, 'units', 'K') )

  call nc_check( nf90_enddef(ncid) )

  ! write coordinate data
  do ix = 1, N
    xc(ix) = real(ix-1, sp) * sqg_L() / real(N, sp)
    yc(ix) = xc(ix)
  end do
  zc(1) = 0.0_sp; zc(2) = sqg_H()
  call nc_check( nf90_put_var(ncid, var_x, xc) )
  call nc_check( nf90_put_var(ncid, var_y, yc) )
  call nc_check( nf90_put_var(ncid, var_z, zc) )

  ! ---- time loop -------------------------------------------
  nout = 1
  do while (sqg_t() < tmax)
    call sqg_advance(ntimesteps, pv_out)
    t_now = sqg_t()

    pvmin = minval(pv_out)
    pvmax = maxval(pv_out)
    write(*,'(a,f10.2,a,f12.4,a,f12.4)') &
      'hr=', real(t_now)/3600.0, &
      '  min/max pv  ', scalefact*pvmin, '  ', scalefact*pvmax

    ! Write pv at this output time.
    ! Fortran layout pv_out(comp, y, x) maps to NetCDF pv(x, y, z, t).
    start4 = [1, 1, 1, nout]
    count4 = [N, N, 2, 1]
    call nc_check( nf90_put_var(ncid, var_pv, pv_out, &
                   start=start4, count=count4) )

    start1 = [nout]
    tval   = [real(t_now, sp)]
    call nc_check( nf90_put_var(ncid, var_t, tval, start=start1) )
    call nc_check( nf90_sync(ncid) )

    nout = nout + 1
  end do

  call nc_check( nf90_close(ncid) )
  call sqg_finalize()

contains

  subroutine nc_check(rc)
    integer, intent(in) :: rc
    if (rc /= NF90_NOERR) then
      write(*,*) 'NetCDF error: ', trim(nf90_strerror(rc))
      stop 1
    end if
  end subroutine nc_check

end program sqg_main
