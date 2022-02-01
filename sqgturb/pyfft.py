import numpy as np
import pyfftw

class Fouriert(object):
    """
    wrapper class for spectral transform operations

    Jeffrey S. Whitaker <jeffrey.s.whitaker@noaa.gov>
    """
    def __init__(self,N,L,threads=1):
        """initialize
        N: number of grid points (spectral truncation x 2)
        L: domain size"""
        self.L = L
        self.threads = threads
        self.N = N
        self.Nt = 3*N//2
        # set up pyfftw objects for transforms
        self.rfft2=pyfftw.builders.rfft2(pyfftw.empty_aligned((2,self.Nt,self.Nt), dtype='float32'),\
                                          axes=(-2, -1), threads=threads)
        self.irfft2=pyfftw.builders.irfft2(pyfftw.empty_aligned((2,self.Nt,self.Nt//2+1), dtype='complex64'),\
                                          axes=(-2, -1), threads=threads)
        self.rfft2_2d=pyfftw.builders.rfft2(pyfftw.empty_aligned((self.Nt,self.Nt), dtype='float32'),\
                                          axes=(-2, -1), threads=threads)
        self.irfft2_2d=pyfftw.builders.irfft2(pyfftw.empty_aligned((self.Nt,self.Nt//2+1), dtype='complex64'),\
                                          axes=(-2, -1), threads=threads)
        # spectral stuff
        dk = 2.*np.pi/self.L
        k =  dk*np.arange(0.,self.N//2+1)
        l =  dk*np.append( np.arange(0.,self.N//2),np.arange(-self.N//2,0.) )
        k, l = np.meshgrid(k, l)
        self.k = k.astype(np.float32)
        self.l = l.astype(np.float32)
        ksqlsq = self.k ** 2 + self.l ** 2
        self.ksqlsq = ksqlsq
        self.ik = (1.0j * k).astype(np.complex64)
        self.il = (1.0j * l).astype(np.complex64)
        self.lap = -ksqlsq.astype(np.complex64)
        lapnonzero = self.lap != 0.
        self.invlap = np.zeros_like(self.lap)
        self.invlap[lapnonzero] = 1./self.lap[lapnonzero]
    def grdtospec(self,data):
        """compute spectral coefficients from gridded data"""
        if data.ndim==2:
            dataspec = self.rfft2_2d(data)
        else:
            dataspec = self.rfft2(data)
        return self.spectrunc(dataspec)
    def spectogrd(self,dataspec):
        """compute gridded data from spectral coefficients"""
        dataspec_tmp = self.specpad(dataspec)
        if dataspec_tmp.ndim==2:
            data =  self.irfft2_2d(dataspec_tmp)
        else:
            data =  self.irfft2(dataspec_tmp)
        return np.array(data,copy=True)
    def getgrad(self, dataspec):
        return self.spectogrd(self.ik*dataspec), self.spectogrd(self.il*dataspec)
    def specpad(self, specarr):
        # pad spectral arrays with zeros to get
        # interpolation to 3/2 larger grid using inverse fft.
        # take care of normalization factor for inverse transform.
        if specarr.ndim == 3:
            specarr_pad = np.zeros((2, self.Nt, self.Nt// 2 + 1), specarr.dtype)
            specarr_pad[:, 0 : self.N // 2, 0 : self.N // 2] = (
                specarr[:, 0 : self.N // 2, 0 : self.N // 2]
            )
            specarr_pad[:, -self.N // 2 :, 0 : self.N // 2] = (
                specarr[:, -self.N // 2 :, 0 : self.N // 2]
            )
            # include negative Nyquist frequency.
            specarr_pad[:, 0 : self.N // 2, self.N // 2] = np.conjugate(
                specarr[:, 0 : self.N // 2, -1]
            )
            specarr_pad[:, -self.N // 2 :, self.N // 2] = np.conjugate(
                specarr[:, -self.N // 2 :, -1]
            )
        elif specarr.ndim==2:
            specarr_pad = np.zeros((self.Nt, self.Nt// 2 + 1), specarr.dtype)
            specarr_pad[0 : self.N // 2, 0 : self.N // 2] = (
                specarr[0 : self.N // 2, 0 : self.N // 2]
            )
            specarr_pad[-self.N // 2 :, 0 : self.N // 2] = (
                specarr[-self.N // 2 :, 0 : self.N // 2]
            )
            # include negative Nyquist frequency.
            specarr_pad[0 : self.N // 2, self.N // 2] = np.conjugate(
                specarr[0 : self.N // 2, -1]
            )
            specarr_pad[-self.N // 2 :, self.N // 2] = np.conjugate(
                specarr[-self.N // 2 :, -1]
            )
        else:
            raise IndexError('specarr must be 2d or 3d')
        return 2.25*specarr_pad
    def spectrunc(self, specarr):
        # truncate spectral array using 2/3 rule.
        if specarr.ndim == 3:
            specarr_trunc = np.zeros((2, self.N, self.N // 2 + 1), specarr.dtype)
            specarr_trunc[:, 0 : self.N // 2, 0 : self.N // 2] = specarr[
                :, 0 : self.N // 2, 0 : self.N // 2
            ]
            specarr_trunc[:, -self.N // 2 :, 0 : self.N // 2] = specarr[
                :, -self.N // 2 :, 0 : self.N // 2
            ]
        elif specarr.ndim == 2:
            specarr_trunc = np.zeros((self.N, self.N // 2 + 1), specarr.dtype)
            specarr_trunc[0 : self.N // 2, 0 : self.N // 2] = specarr[
                0 : self.N // 2, 0 : self.N // 2
            ]
            specarr_trunc[-self.N // 2 :, 0 : self.N // 2] = specarr[
                -self.N // 2 :, 0 : self.N // 2
            ]
        else:
            raise IndexError('specarr must be 2d or 3d')
        return specarr_trunc/2.25
