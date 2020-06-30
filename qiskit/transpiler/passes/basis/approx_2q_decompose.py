# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Expand 2-qubit unitaries into a 2-qubit basis, optionally approximating."""

from qiskit.quantum_info.synthesis import TwoQubitBasisDecomposer
from qiskit.circuit import QuantumRegister
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.dagcircuit import DAGCircuit


class Approx2qDecompose(TransformationPass):
    """Expand 2-qubit unitaries into a 2-qubit basis, optionally approximating."""

    def __init__(self, basis=None, fidelity=1, backend_prop=None):
        super().__init__()
        self.basis = basis
        self.fidelity = fidelity
        self.backend_prop = backend_prop

    def run(self, dag):
        qr = QuantumRegister(2)
        print('running')
        for node in dag.named_nodes("unitary"):
            print('unitary')
            if len(node.qargs) != 2:
                continue
            decomposer = TwoQubitBasisDecomposer(self.basis)
            rule = decomposer(node.op, basis_fidelity=self.fidelity)
            print(self.fidelity)
            decomposition = DAGCircuit()
            decomposition.add_qreg(rule[0][1][0].register)
            for inst in rule:
                decomposition.apply_operation_back(*inst)

            dag.substitute_node_with_dag(node, decomposition)
        return dag
