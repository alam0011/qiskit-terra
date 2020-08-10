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
"""This module includes `DrawDataContainer` that decouples the drawing backend from the
framework of Qiskit.
The data container consists of multiple internal representation of
drawing elements associated with its appearance on the drawing canvas.
  - :py:class:`~qiskit.visualization.timeline.drawing_objects.LineData`
  - :py:class:`~qiskit.visualization.timeline.drawing_objects.TextData`
  - :py:class:`~qiskit.visualization.timeline.drawing_objects.BoxData`
  - :py:class:`~qiskit.visualization.timeline.drawing_objects.InfoBoxData`
  - :py:class:`~qiskit.visualization.timeline.drawing_objects.InfoDotData`
At the level of those drawing IRs, quantum gates and its timeline are decomposed into
objects that represent just shape and coordinates on the canvas without context of program.
This design makes the drawing framework more robust to the code change on the Qiskit side.
The generated IRs are pure python objects and we can prepare appropriate unittest without
image file dependency. In addition, because the IRs contain sufficient information for
drawing a shape of element, we don't need to write program parser for each drawing backend.
We just need to make interface that takes `DrawDataContainer` and calls proper drawing
function from the drawing backend, i.e. `matplotlib`, according to the data type of the given IR.
The `DrawDataContainer` is initialized with stylesheet and IRs are generated with
`.update()` method with dictionary of register and register events.
"""
from collections import OrderedDict
from typing import Dict, List, Union

import numpy as np

from qiskit import circuit
from qiskit.visualization.timeline import drawing_objects
from qiskit.visualization.timeline.events import RegisterEvents, InstData
from qiskit.visualization.timeline.style import QiskitTimelineStyle


class DrawDataContainer:
    """Data container of IRs."""

    def __init__(self, style: QiskitTimelineStyle):
        """Create new data container for this figure canvas.
        Args:
            style: Stylesheet for this drawing.
        """
        self.style = style
        self.position_table = {}
        self.drawing_objs = []

        self.program_duration = 0
        self.num_channel = 0

    def update(
            self,
            reg_events: Dict[Union[circuit.Qubit, circuit.Clbit], RegisterEvents]) -> None:
        """Update drawing objects with given program.
        Args:
            reg_events: Dictionary of register-wise event object.
        """
        # set position table
        for ind, bit in enumerate(reg_events.keys()):
            self.position_table[bit] = -ind

        # create IRs
        for bit, events in reg_events.items():
            self._create_gates(events)
            self._create_barrier(events)
            self._create_register_labels(bit)
            self._create_timeline(bit)

        # create register coupling
        if self.style.show_multi_qubit_link:
            links = []
            for events in reg_events.values():
                register_links = events.register_links()
                # sort registers by y position
                for register_link in register_links:
                    new_link = InstData(
                        t0=register_link.t0,
                        operand=register_link.operand,
                        duration=register_link.duration,
                        regs=sorted(register_link.regs, key=lambda x: self.position_table[x])
                    )
                    links.append(new_link)

            # shift lines if there is overlap
            aligned_links = self._gate_link_overlap_check(links)

            for aligned_link in aligned_links:
                self._create_register_link(aligned_link)

    def reset(self) -> None:
        """Recover visible property of drawing objects.
        """
        for drawing_obj in self.drawing_objs:
            drawing_obj.visible = True

    def arrange_registers(self, registers: List[Union[circuit.Qubit, circuit.Clbit]]):
        """Rearrange register positions.
        Args:
            registers: Ordered register list to show.
        """
        # overwrite position table
        for reg in self.position_table.keys():
            if reg not in registers:
                self.position_table[reg] = None
            else:
                self.position_table[reg] = -registers.index(reg)

        # update drawing IRs
        for drawing_obj in self.drawing_objs:
            if len(drawing_obj.bind) > 1:
                new_ys = [self.position_table[reg] for reg in drawing_obj.bind]
                if None in new_ys:
                    drawing_obj.visible = False
                else:
                    drawing_obj.offset = new_ys
            else:
                new_y = self.position_table[drawing_obj.bind[0]]
                if new_y is None:
                    drawing_obj.visible = False
                else:
                    drawing_obj.offset = new_y

        self.num_channel = len(registers)

    def remove_drawings(self, removes: List[str]) -> None:
        """Remove drawings with specified ``data_type``.
        Args:
            removes: List of ``data_type`` that is removed from the canvas.
        """
        for drawing_obj in self.drawing_objs:
            if drawing_obj.data_type in removes:
                drawing_obj.visible = False

    def _gate_link_overlap_check(self,
                                 links: List[InstData]) -> List[InstData]:
        """Check overlap of multi-register gate links and shift the position if overlapped.
        Args:
            links: List of register link information to draw.
        """
        groups = []
        while len(links) > 0:
            ref_link = links.pop()
            group = [ref_link]
            # overlap check among links
            for ind in reversed(range(len(links))):
                if np.abs(ref_link.t0 - links[ind].t0) < self.style.link_h_spacing:
                    y0_min = self.position_table[ref_link.regs[0]]
                    y0_max = self.position_table[ref_link.regs[-1]]
                    y1_min = self.position_table[links[ind].regs[0]]
                    y1_max = self.position_table[links[ind].regs[-1]]
                    if not ((y0_min - y1_min) * (y0_max - y1_max) > 0 and
                            (y0_min - y1_max) * (y0_max - y1_min) > 0):
                        group.append(links.pop(ind))
            groups.append(group)

        aligned_links = []
        for group in groups:
            # overlapped links are horizontally shifted.
            if len(group) > 1:
                xpos_mean = np.mean([link.t0 for link in group])

                # sort links by y position
                sorted_links = sorted(group,
                                      key=lambda x: self.position_table[x.regs[-1]],
                                      reverse=True)

                x0 = xpos_mean - 0.5 * self.style.link_h_spacing * (len(group) - 1)
                for ind, sorted_link in enumerate(sorted_links):
                    new_link = InstData(t0=x0 + ind * self.style.link_h_spacing,
                                        operand=sorted_link.operand,
                                        duration=sorted_link.duration,
                                        regs=sorted_link.regs)
                    aligned_links.append(new_link)
            else:
                aligned_links.append(group[0])

        return aligned_links

    def _create_gates(self,
                      events: RegisterEvents) -> None:
        """Update info box object for gates.
        Args:
            events: A event object of register.
        """
        for inst_data in events.gates():
            # get matrix
            if hasattr(inst_data.operand, 'to_matrix'):
                unitary = str(inst_data.operand.to_matrix())
            else:
                unitary = None

            # get label
            if hasattr(inst_data.operand, 'label'):
                label = inst_data.operand.label
            else:
                label = None

            # get parameters
            if hasattr(inst_data.operand, 'params'):
                params = inst_data.operand.params
            else:
                params = None

            meta_data = OrderedDict(
                name=inst_data.operand.name,
                label=label,
                t0=inst_data.t0,
                duration=inst_data.duration,
                qasm=inst_data.operand.qasm(),
                unitary=unitary,
                parameters=params
            )

            gate_name = self.style.gate_name_mapper(inst_data.operand.name)
            if len(inst_data.regs) > 1:
                oper_string = '{0} [{1}]'.format(gate_name, inst_data.regs.index(events.reg))
            else:
                oper_string = gate_name

            if inst_data.duration > 0:
                # generate box type drawing
                styles = {
                    'zorder': self.style.gate_box_z_order,
                    'facecolor': self.style.gate_color_mapper(inst_data.operand.name),
                    'edgecolor': self.style.gate_box_edge_color,
                    'textcolor': self.style.gate_text_color,
                    'box_edge_width': self.style.gate_box_side_width,
                    'fontsize': self.style.gate_text_size,
                    'alpha': self.style.gate_alpha,
                }

                front_string = '{0}\n[{1}]'.format(oper_string, inst_data.duration)

                drawing = drawing_objects.InfoBoxData(
                    data_type=inst_data.operand.__class__.__name__, bind=events.reg,
                    x=inst_data.t0, y=-0.5 * self.style.gate_height,
                    width=inst_data.duration,
                    height=self.style.gate_height,
                    text=front_string,
                    offset=self.position_table[events.reg],
                    meta=meta_data, styles=styles)
            else:
                # generate dot type drawing
                styles = {
                    'zorder': self.style.gate_dot_z_order,
                    'markercolor': self.style.gate_color_mapper(inst_data.operand.name),
                    'markeredgecolor': self.style.gate_dot_edge_color,
                    'textcolor': self.style.gate_text_color,
                    'markersize': self.style.gate_dot_marker_size,
                    'fontsize': self.style.gate_text_size,
                    'alpha': self.style.gate_alpha,
                }

                front_string = '{0}'.format(oper_string)

                drawing = drawing_objects.InfoDotData(
                    data_type=inst_data.operand.__class__.__name__, bind=events.reg,
                    x=inst_data.t0, y=-0,
                    text=front_string,
                    offset=self.position_table[events.reg],
                    meta=meta_data, styles=styles)

            if drawing not in self.drawing_objs:
                self.drawing_objs.append(drawing)

    def _create_register_labels(self,
                                bit: Union[circuit.Qubit, circuit.Clbit]) -> None:
        """Update text object for register labels.
        Args:
            bit: Register object to draw.
        """
        label_string = r'${0}_{{{1}}}$'.format(bit.register.name, bit.index)

        if self.style.show_initial_state:
            label_string = r'{0} $|0\rangle$'.format(label_string)

        styles = {
            'zorder': self.style.register_label_z_order,
            'fontsize': self.style.register_label_fontsize,
            'va': 'center',
            'ha': 'right'
        }

        label = drawing_objects.TextData(
            data_type='RegisterLabel', bind=bit,
            x=-self.style.register_label_h_offset, y=0,
            offset=self.position_table[bit],
            text=label_string, styles=styles)

        if label not in self.drawing_objs:
            self.drawing_objs.append(label)

    def _create_timeline(self,
                         bit: Union[circuit.Qubit, circuit.Clbit]) -> None:
        """Update box object for timeline.
        Args:
            bit: Register object to draw.
        """
        styles = {
            'zorder': self.style.timeline_z_order,
            'facecolor': self.style.timeline_facecolor,
            'edgecolor': self.style.timeline_edgecolor,
            'alpha': self.style.timeline_alpha,
        }

        box = drawing_objects.BoxData(
            data_type='Timeline', bind=bit,
            x=0, y=-0.5 * self.style.gate_height,
            offset=self.position_table[bit],
            width=self.program_duration, height=self.style.gate_height,
            styles=styles)

        if box not in self.drawing_objs:
            self.drawing_objs.append(box)

    def _create_register_link(self,
                              link: InstData) -> None:
        """Update line object for qubit link.
        Args:
            link: Register link.
        """
        styles = {
            'zorder': self.style.link_z_order,
            'color': self.style.gate_color_mapper(link.operand.name),
            'linewidth': self.style.link_linewidth,
            'alpha': self.style.link_alpha,
            'marker': self.style.link_marker
        }

        for reg1, reg2 in zip(link.regs[:-1], link.regs[1:]):
            v1 = 0.5 * self.style.gate_height
            v2 = - 0.5 * self.style.gate_height
            line = drawing_objects.LineData(
                data_type='RegisterLink', bind=[reg1, reg2],
                x=[link.t0, link.t0], y=[v1, v2],
                offset=[self.position_table[reg1], self.position_table[reg2]],
                styles=styles)

            if line not in self.drawing_objs:
                self.drawing_objs.append(line)

    def _create_barrier(self,
                        events: RegisterEvents) -> None:
        """Updata line object for barrier. Generate multiple lines for each register.
        Args:
            events: A event object of register.
        """
        for inst_data in events.barriers():
            styles = {
                'zorder': self.style.barrier_z_order,
                'color': self.style.barrier_color,
                'linewidth': self.style.barrier_linewidth,
                'linestyle': self.style.barrier_line_style,
                'alpha': self.style.barrier_alpha,
            }
            for reg in inst_data.regs:
                drawing = drawing_objects.LineData(
                    data_type=inst_data.operand.__class__.__name__, bind=reg,
                    x=[inst_data.t0, inst_data.t0],
                    y=[-0.5, 0.5],
                    offset=self.position_table[reg],
                    styles=styles)

                if drawing not in self.drawing_objs:
                    self.drawing_objs.append(drawing)
