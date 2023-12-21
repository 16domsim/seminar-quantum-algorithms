import numpy as np
import math
import matplotlib.pyplot as plt


# Signal rotation operator
def W(a):
    return np.array([[a, 1j*math.sqrt(1-a**2)], [1j*math.sqrt(1-a**2), a]])

# Signal processing operator
def S(phi):
       return np.array([[np.e**(1j*phi),0], [0, np.e**(-1j*phi)]])

# Qantum Signal Processing
def QSP(phis, a):
      ret = S(phis[0])
      for i in range(1, len(phis)):
            ret = np.matmul(ret, W(a))
            ret = np.matmul(ret, S(phis[i]))
      return ret


print("Welcome to the Quantum Signal Processing Simulator")

# Phases
phis = [np.pi/2, -1/2 * np.arccos(-1/4),  np.arccos(-1/4), 0, - np.arccos(-1/4), 1/2 * np.arccos(-1/4)]

# x-values
interval = np.arange(start=-np.pi, stop=np.pi, step=0.01)

# Qubits
a = [np.cos(-1/2*theta) for theta in interval]

# Polynomial
p = [np.square(np.abs(QSP(phis, q)[0][0])) for q in a]

# plotting
plt.plot(interval, p)
plt.xlabel(r"$\theta$")
plt.ylabel(r"$| \langle 0| U_{\vec{\phi}} |0 \rangle |^2$")
plt.show()





      
    