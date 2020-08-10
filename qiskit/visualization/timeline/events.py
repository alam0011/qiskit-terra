# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""Event management on qubit timeline.
"""
from collections import namedtuple
from typing import Union, List

from qiskit import circuit
from qiskit.converters import circuit_to_dag
from qiskit.transpiler.instruction_durations import InstructionDurations


InstData = namedtuple('InstData', 't0 operand duration regs')


class RegisterEvents:
    """Register-wise instruction data-set."""

    _auxiliary_type = circuit.Barrier,

    def __init__(self,
                 reg: Union[circuit.Qubit, circuit.Clbit],
                 inst_data: List[InstData]):
        """Create new event object.
        Args:
            reg: Register object of this event.
            inst_data: Time associated operands.
        """
        self.reg = reg
        self.instructions = inst_data

    @classmethod
    def parse_program(cls,
                      scheduled_circuit: circuit.QuantumCircuit,
                      inst_durations: InstructionDurations,
                      reg: Union[circuit.Qubit, circuit.Clbit]):
        """Build new RegisterEvents from scheduled circuit.
        Args:
            scheduled_circuit: Input scheduled circuit object to draw.
            inst_durations: Instruction duration object.
            reg: Target register object.
        """
        dag = circuit_to_dag(scheduled_circuit)
        nodes = list(dag.topological_op_nodes())

        t0 = 0
        instructions = []
        for node in nodes:
            associated_regs = [qarg for qarg in node.qargs] + [carg for carg in node.cargs]
            if reg not in associated_regs:
                continue

            if node.op.name not in ['delay']:
                duration = inst_durations.get(node.op.name, node.qargs)
            else:
                duration = node.op.duration

            instructions.append(InstData(t0, node.op, duration, associated_regs))

            t0 += duration

        return RegisterEvents(reg, instructions)

    def is_empty(self) -> bool:
        """Return if there is any real instruction on this qubit.
        `Barrier` and `Delay` are not counted as a real instruction.
        """
        for inst_data in self.instructions:
            if not isinstance(inst_data.operand, (*self._auxiliary_type,
                                                  circuit.Delay)):
                return False

        return True

    def gates(self) -> List[InstData]:
        """Return Gate type instructions.
        """
        instruction_to_return = []
        for inst_data in self.instructions:
            if not isinstance(inst_data.operand, self._auxiliary_type):
                instruction_to_return.append(inst_data)

        return instruction_to_return

    def barriers(self) -> List[InstData]:
        """Return Barrier type instructions.
        """
        instructions_to_return = []
        for inst_data in self.instructions:
            if isinstance(inst_data.operand, circuit.Barrier) \
                    and inst_data.regs.index(self.reg) == 0:
                instructions_to_return.append(inst_data)

        return instructions_to_return

    def register_links(self) -> List[InstData]:
        """Return multi-register gate links.
        """
        links = []
        for inst_data in self.instructions:
            if not isinstance(inst_data.operand, self._auxiliary_type) \
                    and len(inst_data.regs) > 1 and inst_data.regs.index(self.reg) == 0:
                xpos = inst_data.t0 + 0.5 * inst_data.duration
                new_inst_data = InstData(xpos,
                                         inst_data.operand,
                                         inst_data.duration,
                                         inst_data.regs)
                links.append(new_inst_data)

        return links
