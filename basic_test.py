import numpy as np
from qiskit.chemistry.drivers import PySCFDriver, UnitsType, Molecule
from qiskit.chemistry.transformations import FermionicTransformation, FermionicQubitMappingType
from qiskit.aqua.algorithms import NumPyMinimumEigensolver
from qiskit import Aer
from qiskit.aqua import QuantumInstance
from qiskit.chemistry.algorithms.ground_state_solvers.minimum_eigensolver_factories import VQEUCCSDFactory
from qiskit.aqua.algorithms import VQE
from qiskit.chemistry.algorithms.ground_state_solvers import GroundStateEigensolver
from qiskit.aqua.operators import I, X, Z

def construct_jastrow_factor(num_qubits : int,
                             alpha: np.array,
                             beta: np.array):

    # construct jastrow factor
    jastrow_factor = (1*I)
    for i in range(num_qubits-1):
        jastrow_factor = jastrow_factor^I
    
    print("Number of qubits is {}".format(num_qubits))
    # First term construction
    for i in range(num_qubits):
        for k in range(num_qubits):
            if k == 0:
                if i==k:
                    p_string = -alpha[i] * Z
                else:
                    p_string = -alpha[i] * I
            else:
                if i == k:
                    p_string = p_string^Z
                else:
                    p_string = p_string^I
        jastrow_factor += p_string

    # Second term construction                
    for i in range(num_qubits):
        for j in range(i+1, num_qubits):
            for k in range(num_qubits):
                if k == 0:
                    if i == 0:
                        p_string = -beta[i, j] * Z
                    else:
                        p_string = -beta[i, j] * I
                else:
                    if i == k or j == k:
                        p_string = p_string^Z
                    else:
                        p_string = p_string^I
            
            jastrow_factor += p_string
    return jastrow_factor


molecule = Molecule(geometry=[['H', [0., 0., 0.]],
                              ['H', [0., 0., 0.735]]],
                     charge=0, multiplicity=1)
driver = PySCFDriver(molecule = molecule, unit=UnitsType.ANGSTROM, basis='sto3g')
transformation = FermionicTransformation(qubit_mapping=FermionicQubitMappingType.JORDAN_WIGNER, two_qubit_reduction=False)

# numpy_solver = NumPyMinimumEigensolver()
# calc = GroundStateEigensolver(transformation, numpy_solver)
# res = calc.solve(driver)
# print(res)

vqe_solver = VQEUCCSDFactory(QuantumInstance(Aer.get_backend('qasm_simulator')))
calc = GroundStateEigensolver(transformation, vqe_solver)
res = calc.solve(driver)
# print(res)

num_parameters = vqe_solver._vqe.var_form.num_parameters
print("Number of parameters:", num_parameters)

seed = np.random.seed(43)
parameters = np.random.rand(num_parameters)

# circuit = vqe_solver._vqe.var_form.construct_circuit(parameters)
# print(circuit.decompose())


# This is the qubit Hamiltonian consisting of Pauli strings with coefficients in front
qubit_ham = calc.transformation.transform(driver)
num_qubits = qubit_ham[0].num_qubits
print("Qubit Hamiltonian")
print(qubit_ham[0])

alpha = np.random.rand(num_qubits)
beta = np.random.rand(num_qubits, num_qubits)        
jastrow_factor = construct_jastrow_factor(num_qubits, alpha, beta)

# print("\nqubit_op ")
# for op in vqe_solver._vqe.var_form._hopping_ops[0].paulis:
#     print(op)
print("\njastrow\n", jastrow_factor)
# print("\nproduct\n", jastrow_factor@vqe_solver._vqe.var_form._hopping_ops[0])
# print("\nproduct\n", jastrow_factor@vqe_solver._vqe.var_form._hopping_ops[0].paulis[0][1])
# print("\nproduct\n", (jastrow_factor@vqe_solver._vqe.var_form._hopping_ops[0].paulis[1][1]))