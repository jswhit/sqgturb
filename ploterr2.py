import matplotlib.pyplot as plt
import numpy as np
data1 = np.loadtxt('run_ensemble_amp0p3_N256_N64_fixedinc.out')
data2 = np.loadtxt('run_ensemble_noai_enkf_amp0p3.out')
plt.plot(data1[:,0],data1[:,1],label='error (upscale truth init)')
plt.plot(data2[:,0],data2[:,1],label='error (EnKF init)')
plt.title('error')
plt.legend()
plt.savefig('err.png')
plt.show()
