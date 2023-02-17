# -*- coding: UTF-8 -*-
import os


__all__       = ["__author__", "__copyright__", "__email__", "__license__"]
__author__    = "Alexandre D'Hondt"
__email__     = "alexandre.dhondt@gmail.com"
__copyright__ = ("A. D'Hondt", 2021)
__license__   = "gpl-3.0"
with open(os.path.join(os.path.dirname(__file__), "VERSION.txt")) as f:
    __version__ = f.read().strip()

