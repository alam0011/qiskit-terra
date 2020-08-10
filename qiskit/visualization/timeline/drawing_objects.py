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
"""This module includes drawing IRs that represent shape and coordinate of drawing.
Note that a drawing IR must not be designed based on the context of quantum circuit.
For example, when we want to draw a conditional gate, which is represented by a box
and a line from the box to classical register, we cannot define `ConditionalGate` IR.
Such high-context data representation may induce a complex drawing procedure on
drawer side, which is generally hard to prepare appropriate unittest.
Instead, the conditional gate in above example should be represented by combination of
``InfoBoxData``  and ``LineData`` IRs. Those IRs just represent primitive shape and
most of drawing frameworks can support drawing such shapes with native function.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Union

import numpy as np

from qiskit import circuit


class ElementaryData(ABC):
    """Abstract class of visualization intermediate representation."""
    def __init__(self,
                 data_type: str,
                 bind: Union[Union[circuit.Qubit, circuit.Clbit],
                             List[Union[circuit.Qubit, circuit.Clbit]]],
                 offset: Union[float, List[float]],
                 visible: bool,
                 styles: Dict[str, Any]):
        """Create new visualization IR.
        Args:
            data_type: String representation of this drawing object.
            bind: Register object bound to this drawing.
            offset: Offset coordinate of vertical axis.
            visible: Set ``True`` to show the component to the canvas.
        """
        if not isinstance(bind, tuple):
            if isinstance(bind, list):
                bind = tuple(bind)
            else:
                bind = (bind, )

        self.data_type = data_type
        self.offset = offset
        self.visible = visible
        self.bind = bind
        self.styles = styles

    @property
    @abstractmethod
    def data_key(self):
        pass

    def __repr__(self):
        return "{}(data_key={})".format(self.__class__.__name__, self.data_key)

    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.data_key == other.data_key


class LineData(ElementaryData):
    """Drawing IR of line type object."""
    def __init__(self,
                 data_type: str,
                 bind: Union[Union[circuit.Qubit, circuit.Clbit],
                             List[Union[circuit.Qubit, circuit.Clbit]]],
                 x: List[float],
                 y: List[float],
                 offset: Union[float, List[float]],
                 visible: bool = True,
                 styles: Dict[str, Any] = None):
        """Create new line IR.
        Args:
            data_type: String representation of this drawing object.
            bind: Register object bound to this drawing.
            x: List of x-coordinate value of the line.
            y: List of y-coordinate value of the line.
            offset: Offset coordinate of vertical axis.
            visible: Set ``True`` to show the component to the canvas.
        """
        self.x = np.array(x)
        self.y = np.array(y)

        super().__init__(
            data_type=data_type,
            bind=bind,
            offset=offset,
            visible=visible,
            styles=styles)

    @property
    def data_key(self):
        return str(hash((self.__class__.__name__,
                         self.data_type,
                         self.bind,
                         tuple(self.x),
                         tuple(self.y))))


class TextData(ElementaryData):
    """Drawing IR of text type object."""
    def __init__(self,
                 data_type: str,
                 bind: Union[Union[circuit.Qubit, circuit.Clbit],
                             List[Union[circuit.Qubit, circuit.Clbit]]],
                 x: float,
                 y: float,
                 offset: Union[float, List[float]],
                 text: str,
                 visible: bool = True,
                 styles: Dict[str, Any] = None):
        """Create new text IR.
        Args:
            data_type: String representation of this drawing object.
            bind: Register object bound to this drawing.
            x: Value of x-coordinate value of the line.
            y: Value of y-coordinate value of the line.
            offset: Offset coordinate of vertical axis.
            text: String to show.
            visible: Set ``True`` to show the component to the canvas.
        """
        self.x = x
        self.y = y
        self.text = text

        super().__init__(
            data_type=data_type,
            bind=bind,
            offset=offset,
            visible=visible,
            styles=styles)

    @property
    def data_key(self):
        return str(hash((self.__class__.__name__,
                         self.data_type,
                         self.bind,
                         self.text,
                         self.x,
                         self.y)))


class BoxData(ElementaryData):
    """Drawing IR of box type object."""
    def __init__(self,
                 data_type: str,
                 bind: Union[Union[circuit.Qubit, circuit.Clbit],
                             List[Union[circuit.Qubit, circuit.Clbit]]],
                 x: float,
                 y: float,
                 width: float,
                 height: float,
                 offset: Union[float, List[float]],
                 visible: bool = True,
                 styles: Dict[str, Any] = None):
        """Create new line IR.
        Args:
            data_type: String representation of this drawing object.
            bind: Register object bound to this drawing.
            x: Value of x-coordinate value of the line.
            y: Value of y-coordinate value of the line.
            width: Width of box.
            height: Height of box.
            offset: Offset coordinate of vertical axis.
            visible: Set ``True`` to show the component to the canvas.
        """
        self.x = x
        self.y = y
        self.width = width
        self.height = height

        super().__init__(
            data_type=data_type,
            bind=bind,
            offset=offset,
            visible=visible,
            styles=styles)

    @property
    def data_key(self):
        return str(hash((self.__class__.__name__,
                         self.data_type,
                         self.bind,
                         self.x,
                         self.y,
                         self.width,
                         self.height)))


class InfoBoxData(ElementaryData):
    """Drawing IR of box type object that contains information."""
    def __init__(self,
                 data_type: str,
                 bind: Union[Union[circuit.Qubit, circuit.Clbit],
                             List[Union[circuit.Qubit, circuit.Clbit]]],
                 x: float,
                 y: float,
                 width: float,
                 height: float,
                 text: str,
                 offset: Union[float, List[float]],
                 visible: bool = True,
                 styles: Dict[str, Any] = None,
                 meta: Dict[str, Any] = None):
        """Create new line IR.
        Args:
            data_type: String representation of this drawing object.
            bind: Register object bound to this drawing.
            x: Value of x-coordinate value of the line.
            y: Value of y-coordinate value of the line.
            width: Width of box.
            height: Height of box.
            text: Text to show in the front panel of box.
            offset: Offset coordinate of vertical axis.
            visible: Set ``True`` to show the component to the canvas.
            meta: Metadata of this component. This information can be shown in interactive drawer.
        """
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.text = text
        self.meta = meta

        super().__init__(
            data_type=data_type,
            bind=bind,
            offset=offset,
            visible=visible,
            styles=styles)

    @property
    def data_key(self):
        return str(hash((self.__class__.__name__,
                         self.data_type,
                         self.bind,
                         self.text,
                         self.x,
                         self.y,
                         self.width,
                         self.height)))


class InfoDotData(ElementaryData):
    """Drawing IR of scatter type object that contains information."""
    def __init__(self,
                 data_type: str,
                 bind: Union[Union[circuit.Qubit, circuit.Clbit],
                             List[Union[circuit.Qubit, circuit.Clbit]]],
                 x: float,
                 y: float,
                 text: str,
                 offset: Union[float, List[float]],
                 visible: bool = True,
                 styles: Dict[str, Any] = None,
                 meta: Dict[str, Any] = None):
        """Create new line IR.
        Args:
            data_type: String representation of this drawing object.
            bind: Register object bound to this drawing.
            x: Value of x-coordinate value of the line.
            y: Value of y-coordinate value of the line.
            text: Text to show in the front panel of box.
            offset: Offset coordinate of vertical axis.
            visible: Set ``True`` to show the component to the canvas.
            meta: Metadata of this component. This information can be shown in interactive drawer.
        """
        self.x = x
        self.y = y
        self.text = text
        self.meta = meta

        super().__init__(
            data_type=data_type,
            bind=bind,
            offset=offset,
            visible=visible,
            styles=styles)

    @property
    def data_key(self):
        return str(hash((self.__class__.__name__,
                         self.data_type,
                         self.bind,
                         self.text,
                         self.x,
                         self.y)))
