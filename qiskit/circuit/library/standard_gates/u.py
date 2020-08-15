# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Two-pulse single-qubit gate."""

import numpy
from qiskit.circuit.controlledgate import ControlledGate
from qiskit.circuit.gate import Gate
from qiskit.circuit.quantumregister import QuantumRegister


class UGate(Gate):
    r"""Generic single-qubit rotation gate with 3 Euler angles.
    Implemented using two X90 pulses on IBM Quantum systems:
    .. math::
        U(\theta, \phi, \lambda) =
            RZ(\phi - \pi/2) RX(\pi/2) RZ(\pi - \theta) RX(\pi/2) RZ(\lambda - \pi/2)
    **Circuit symbol:**
    .. parsed-literal::
             ┌──────────┐
        q_0: ┤ U(ϴ,φ,λ) ├
             └──────────┘
    **Matrix Representation:**
    .. math::
        \newcommand{\th}{\frac{\theta}{2}}
        U(\theta, \phi, \lambda) =
            \begin{pmatrix}
                \cos(\th)          & -e^{i\lambda}\sin(\th) \\
                e^{i\phi}\sin(\th) & e^{i(\phi+\lambda)}\cos(\th)
            \end{pmatrix}
    **Examples:**
    .. math::
        U\left(\theta, -\frac{\pi}{2}, \frac{pi}{2}\right) = RX(\theta)
    .. math::
        U(\theta, 0, 0) = RY(\theta)
    """

    def __init__(self, theta, phi, lam, label=None):
        """Create new U gate."""
        super().__init__('u', 1, [theta, phi, lam], label=label)

    def inverse(self):
        r"""Return inverted U gate.
        :math:`U(\theta,\phi,\lambda)^{\dagger} =U3(-\theta,-\phi,-\lambda)`)
        """
        return UGate(-self.params[0], -self.params[2], -self.params[1])

    def control(self, num_ctrl_qubits=1, label=None, ctrl_state=None):
        """Return a (mutli-)controlled-U3 gate.
        Args:
            num_ctrl_qubits (int): number of control qubits.
            label (str or None): An optional label for the gate [Default: None]
            ctrl_state (int or str or None): control state expressed as integer,
                string (e.g. '110'), or None. If None, use all 1s.
        Returns:
            ControlledGate: controlled version of this gate.
        """
        if num_ctrl_qubits == 1:
            gate = CUGate(self.params[0], self.params[1], self.params[2], 0,
                          label=label, ctrl_state=ctrl_state)
            gate.base_gate.label = self.label
            return gate
        return super().control(num_ctrl_qubits=num_ctrl_qubits, label=label, ctrl_state=ctrl_state)

    def _define(self):
        """Alias for U3 until U becomes a basis gate."""
        from qiskit.circuit.quantumcircuit import QuantumCircuit
        q = QuantumRegister(1, 'q')
        qc = QuantumCircuit(q)
        qc.u3(self.params[0], self.params[1], self.params[2], q[0])
        self.definition = qc

    def to_matrix(self):
        """Return a numpy.array for the U gate."""
        theta, phi, lam = [float(param) for param in self.params]
        return numpy.array([
            [
                numpy.cos(theta / 2),
                -numpy.exp(1j * lam) * numpy.sin(theta / 2)
            ],
            [
                numpy.exp(1j * phi) * numpy.sin(theta / 2),
                numpy.exp(1j * (phi + lam)) * numpy.cos(theta / 2)
            ]
        ], dtype=complex)
