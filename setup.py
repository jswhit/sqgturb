from setuptools import setup
short_desc = "A program for simulating surface quasi-geostropic turbulence"
setup(
  name = 'sqgturb',
  version = '0.1',
  description = short_desc,
  author = 'Jeff Whitaker',
  author_email = 'jeffrey dot s dot whitaker at noaa dot gov',
  url = 'https://github.com/jswhit/sqgturb',
  packages = ['sqgturb'],
  requires = ['numpy','scipy','mpi4py','mpi4py_fft']
)








