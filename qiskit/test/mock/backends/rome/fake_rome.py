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

"""
Fake Rome device (5 qubit).
"""

import os
import json

from qiskit.providers.models import QasmBackendConfiguration, BackendProperties
from qiskit.test.mock.fake_backend import FakeBackend


class FakeRome(FakeBackend):
    """A fake 5 qubit backend."""

    def __init__(self):
        dirname = os.path.dirname(__file__)
        filename = "conf_rome.json"
        with open(os.path.join(dirname, filename), "r") as f_conf:
            conf = json.load(f_conf)

        configuration = QasmBackendConfiguration.from_dict(conf)
        configuration.backend_name = 'fake_rome'
        self._defaults = None
        self._properties = None
        super().__init__(configuration)

    def properties(self):
        """Returns a snapshot of device properties as recorded on 04/28/20.
        """
        if not self._properties:
            dirname = os.path.dirname(__file__)
            filename = "props_rome.json"
            with open(os.path.join(dirname, filename), "r") as f_prop:
                props = json.load(f_prop)
            self._properties = BackendProperties.from_dict(props)
        return self._properties
