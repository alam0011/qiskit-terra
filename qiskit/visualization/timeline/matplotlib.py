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
"""
Matplotlib plotter interface.
"""
from typing import Optional, Tuple, Union

import numpy as np
from matplotlib import axes, figure, get_backend
from matplotlib import pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle

from qiskit.visualization.exceptions import VisualizationError
from qiskit.visualization.timeline import drawing_objects
from qiskit.visualization.timeline.core_drawer import DrawDataContainer


def output_matplotlib(data: DrawDataContainer,
                      ax: Optional[axes.Axes] = None,
                      filename: Optional[str] = None) -> Union[figure.Figure, axes.Axes]:
    """Generate matplotlib image data.
    Args:
        data: Data container of drawing objects.
        ax: Matplotlib axis object.
        filename: Path of file to save the output image.
    Raises:
        VisualizationError: When invalid object is thrown.
    """
    # generate axis
    if ax is None:
        fig = plt.figure()
        fig.set_size_inches(data.program_duration * data.style.canvas_inch_per_unit_h,
                            data.num_channel * data.style.canvas_inch_per_unit_v)
        _ax = fig.add_subplot(111)
    else:
        fig = None
        _ax = ax

    for drawing_obj in data.drawing_objs:
        if not drawing_obj.visible:
            continue

        style = drawing_obj.styles

        # output info box
        if isinstance(drawing_obj, drawing_objects.InfoBoxData):
            # draw box
            box_style = {
                'zorder': style.get('zorder', 3),
                'facecolor': style.get('facecolor', '#FFFFFF'),
                'edgecolor': style.get('edgecolor', '#000000'),
                'alpha': style.get('alpha', 1)
            }
            x, y1, y2 = box_outline(x=drawing_obj.x, y=drawing_obj.y,
                                    width=drawing_obj.width, height=drawing_obj.height,
                                    ew=style.get('box_edge_width'))
            _ax.fill_between(x, y1 + drawing_obj.offset, y2 + drawing_obj.offset, **box_style)

            # draw text
            text_style = {
                'zorder': style.get('zorder', 4),
                'fontsize': style.get('fontsize', 10),
                'color': style.get('textcolor', '#000000')
            }
            _ax.text(x=drawing_obj.x + 0.5 * drawing_obj.width,
                     y=drawing_obj.y + drawing_obj.offset + 0.5 * drawing_obj.height,
                     s=drawing_obj.text, va='center', ha='center', **text_style)

        # output info dot
        elif isinstance(drawing_obj, drawing_objects.InfoDotData):
            # draw dot
            dot_style = {
                'zorder': style.get('zorder', 3),
                'c': style.get('markercolor', '#000000'),
                's': style.get('markersize', 10) ** 2,
                'edgecolors': style.get('markeredgecolor', '#000000'),
                'alpha': style.get('alpha', 1)
            }
            _ax.scatter(drawing_obj.x,
                        drawing_obj.y + drawing_obj.offset + 0.25,
                        marker='v', **dot_style)

            # draw text
            text_style = {
                'zorder': style.get('zorder', 4),
                'fontsize': style.get('fontsize', 10),
                'color': style.get('textcolor', '#000000')
            }
            _ax.text(x=drawing_obj.x,
                     y=drawing_obj.y + drawing_obj.offset + 0.35,
                     s=drawing_obj.text, va='bottom', ha='center', **text_style)

        # output plane box
        elif isinstance(drawing_obj, drawing_objects.BoxData):
            rect = Rectangle(xy=(drawing_obj.x, drawing_obj.y + drawing_obj.offset),
                             width=drawing_obj.width,
                             height=drawing_obj.height)
            pc = PatchCollection([rect], **style)
            _ax.add_collection(pc)

        # output line
        elif isinstance(drawing_obj, drawing_objects.LineData):
            _ax.plot(drawing_obj.x, drawing_obj.y + drawing_obj.offset, **style)

        # output text
        elif isinstance(drawing_obj, drawing_objects.TextData):
            _ax.text(x=drawing_obj.x,
                     y=drawing_obj.y + drawing_obj.offset,
                     s=drawing_obj.text, **style)

        else:
            VisualizationError('Object %s is not valid drawing object', str(drawing_obj))

    # format figure
    _ax.axis('off')
    _ax.set_ylim(-data.num_channel + 0.5, 0.5)

    if ax is None:
        # save figure
        if filename:
            fig.savefig(filename, dpi=data.style.save_dpi, bbox_inches='tight')
        # close figure
        if get_backend() in ['module://ipykernel.pylab.backend_inline', 'nbAgg']:
            plt.close(fig)

        return fig
    else:
        return ax


def box_outline(x: float,
                y: float,
                width: float,
                height: float,
                ew: float = 10) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create outline of info box.
    Args:
        x: Start x-coordinate of the box.
        y: Start x-coordinate of the box.
        width: Width of the box.
        height: Height of the box.
        ew: Width of edge.
    """
    edge_resolution = 100

    def edge(gain):
        return 1 / (1 + np.exp(-gain * np.linspace(-ew, ew, edge_resolution)))

    x = np.concatenate([np.linspace(x, x + 2 * ew, edge_resolution),
                        np.linspace(x + width - 2 * ew, x + width, edge_resolution)])
    risefall = np.concatenate([edge(ew/100), edge(-ew/100)])

    y1 = y + 0.5 * height + 0.5 * height * risefall / max(risefall)
    y2 = y + 0.5 * height - 0.5 * height * risefall / max(risefall)

    return x, y1, y2
