# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# pylint: disable=invalid-name

"""
Timeline visualization.
"""
from collections import OrderedDict
from typing import Optional

from qiskit import QuantumCircuit
from qiskit.circuit import Barrier, Delay
from qiskit.transpiler.instruction_durations import InstructionDurations
from qiskit.visualization.exceptions import VisualizationError
from qiskit.visualization.timeline.core_drawer import DrawDataContainer
from qiskit.visualization.timeline.events import RegisterEvents
from qiskit.visualization.timeline.style import QiskitTimelineStyle


def timeline_drawer(
        circuit: QuantumCircuit,
        inst_durations: InstructionDurations,
        filename: str = None,
        style: QiskitTimelineStyle = None,
        output: str = 'mpl',
        interactive: bool = False,
        reverse_bits: bool = False,
        idle_wires: bool = True,
        show_barrier: bool = True,
        show_delay: bool = True,
        ax: Optional = None):
    """Draw a scheduled circuit timeline to different formats (set by output parameter):
    **mpl**: image with color rendered purely in Python.
    Args:
        circuit: The quantum circuit to draw. The circuit should be transpiled to
            scheduled circuit with embedded gate times.
        inst_durations: ``InstructionDurations`` object to specify the duration of
            quantum gate instruction.
        filename: File path to save image to.
        style: Stylesheet instance for the timeline drawer.
            See :py:class:`~qiskit.visualization.timeline.style.QiskitTimelineStyle`.
        output: Select the output method to use for drawing the timeline.
            Valid choices are ``mpl``. By default the ``mpl`` drawer is used unress a user config
            file has an alternative backend set as the default.
            If the output kwarg is set, the backend will always be used over the default in a
            user config gile.
        interactive: When set ``True`` show the circuit in a new window
            (for `mpl` this depends on the matplotlib backend being used
            supporting this).
        reverse_bits: When set to ``True`` reverse the bit order inside
            registers for the output visualization.
        idle_wires: When set to ``True`` include idle wires (wires with no gate elements)
            in output visualization. ``Delay`` s are recognized as idle element.
        show_barrier: When set to ``True`` include ``Barrier`` s in output visualization.
        show_delay: When set to ``True`` include ``Delay`` s in output visualization.
        ax: (matplotlib.axes.Axes): An optional Axes object to be used for
            the visualization output. If ``None`` is specified a new matplotlib
            Figure will be created and used. Additionally, if specified there
            will be no returned Figure since it is redundant. This is only used
            when the ``output`` kwarg is set to use the ``mpl`` backend. It
            will be silently ignored with all other outputs.
    Returns:
        :class:`matplotlib.figure`
        * `matplotlib.figure.Figure` (output='mpl')
            a matplotlib figure object for the circuit diagram.
    Example:
        .. jupyter-execute::
            from qiskit import QuantumCircuit, transpile
            from qiskit.transpiler.instruction_durations import InstructionDurations
            from qiskit.test.mock.backends import FakeParis
            from qiskit.visualization.timeline_visualization import timeline_drawer
            qc = QuantumCircuit(2)
            qc.delay(500, 1)
            qc.h(0)
            qc.cx(0,1)
            sc = transpile(qc, FakeParis(), scheduling_method='alap')
            inst_dur = InstructionDurations.from_backend(FakeParis())
            timeline_drawer(sc, inst_dur, idle_wires=False)
    """

    # parse scheduled circuit
    qubits = circuit.qubits
    clbits = circuit.clbits

    regs = qubits + clbits if not reverse_bits else qubits[::-1] + clbits[::-1]
    register_events = OrderedDict()
    for reg in regs:
        register_events[reg] = RegisterEvents.parse_program(circuit, inst_durations, reg)

    # create data container
    container = DrawDataContainer(style=style or QiskitTimelineStyle())
    container.program_duration = circuit.duration

    # create drawing IRs
    container.update(register_events)

    # registers to draw
    if not idle_wires:
        registers_to_show = []
        for reg, register_event in register_events.items():
            if not register_event.is_empty():
                registers_to_show.append(reg)
    else:
        registers_to_show = regs

    container.arrange_registers(registers_to_show)

    # remove instructions
    removes = []
    if not show_barrier:
        removes.append(Barrier.__name__)
    if not show_delay:
        removes.append(Delay.__name__)

    container.remove_drawings(removes)

    # output
    if output == 'mpl':
        try:
            from qiskit.visualization.timeline.matplotlib import output_matplotlib
        except ImportError:
            raise VisualizationError('Matplitlib is not installed. ',
                                     'Try pip install matplotlib to use this format.')
        image = output_matplotlib(container, ax=ax, filename=filename)
        if image and interactive:
            image.show()
    else:
        raise VisualizationError('The output format %s is not supported.' % output)

    return image
