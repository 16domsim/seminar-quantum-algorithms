import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm, sqrtm
from qiskit.quantum_info.operators import Pauli
import pyqsp
from pyqsp.angle_sequence import QuantumSignalProcessingPhases

# Define the Pauli operators
Z = Pauli('Z')
X = Pauli('X')
Y = Pauli('Y')

# Block encoding of Hermitian matrix
def B(H):
    return np.kron(Z, H) + np.kron(X, sqrtm(np.eye(H.shape[0]) - H @ H))

# Phase difference between QSP and QSVT
def P(d):
    return np.array([np.pi/4] + (d-1)*[np.pi/2] + [np.pi/4])

# Quantum Eigenvalue Transformation
def QEVT(phis, H):
    U = B(H)
    result = np.kron(expm(1j*phis[0]*Z.to_matrix()), np.eye(H.shape[0]))
    for k in range(1, len(phis)):
        result = result @ (U @ np.kron(expm(1j*phis[k]*Z.to_matrix()), np.eye(H.shape[0])))
    return result

# Define the Hamiltonian matrix
H = 0.1 * np.kron(Z.to_matrix(), np.kron(Y.to_matrix(), Y.to_matrix())) + \
    0.4 * np.kron(X.to_matrix(), np.kron(Y.to_matrix(), Y.to_matrix())) - \
    0.2 * np.kron(Y.to_matrix(), np.kron(Y.to_matrix(), Y.to_matrix())) + \
    0.05 * np.kron(np.eye(2), np.kron(Z.to_matrix(), X.to_matrix())) + \
    0.45 * np.kron(np.eye(2), np.kron(np.eye(2), np.eye(2)))

# Normalize  Hamiltonian
H = H / (1.25 * np.linalg.norm(H))

# delta = 0.3, d = 30
coeffs, scale = pyqsp.poly.PolyEigenstateFiltering().generate(30, 0.3, return_scale=True)
seq = QuantumSignalProcessingPhases(coeffs, signal_operator="Wz")

# x-values
interval = np.arange(start=0, stop=1, step=0.001)

# Polynomial
p = QEVT(seq - P(len(seq) - 1), H)[0:9, 0:9]

# Eigenvalues
ev, v = np.linalg.eigh(p.astype(np.complex128)) 

# Apply global phase
ev = ev*((-1j)**(len(seq)-1))

plt.plot(interval, 0.9 * (np.sign(interval + 0.5*0.3) - np.sign(interval - 0.5*0.3)) / 2, label="Target function")
plt.plot(interval, np.polynomial.Polynomial(coeffs)(interval), label="Re[Poly(a)]")
plt.plot(ev, ".")
plt.xticks(np.arange(len(ev)), ['$\lambda ' + str(i+1) + '$' for i in range(len(ev))]) 
plt.title("Eigenvalues after QEVT")
plt.legend(loc="upper right")
plt.xlabel("a")
plt.ylabel("Response")
plt.show()
