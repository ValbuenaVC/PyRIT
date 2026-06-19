# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Exception classes for the Inspect AI bridge.
"""

from pyrit.exceptions.exception_classes import PyritException


class InspectBridgeError(PyritException):
    """
    Exception raised for errors in the Inspect AI bridge.

    Raised when the bridge encounters a configuration problem, such as a missing
    ``pyrit`` model provider or ``MemoryAdapter`` hook registration. When the cause
    is a missing registration, the message will mention running ``InspectInitializer``.
    """

    def __init__(self, *, message: str = "Inspect bridge error", status_code: int = 500) -> None:
        """
        Initialize an InspectBridgeError.

        Args:
            message (str): Human-readable error description.
            status_code (int): HTTP-style status code for the error.

        """
        super().__init__(status_code=status_code, message=message)
