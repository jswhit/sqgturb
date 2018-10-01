import matplotlib.pyplot as plt
import numpy as np
data = np.loadtxt('run_ensemble_amp0p3.out')
plt.plot(data[:,0],data[:,1],label='error')
plt.plot(data[:,0],data[:,3],label='spread')
plt.title('0.3 x additive increments')
plt.legend()
plt.savefig('run_ensemble_amp0p3.png')
plt.show()
