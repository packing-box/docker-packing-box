#!/usr/bin/python3
# -*- coding: UTF-8 -*-
import csv
import mdv
from tinyscript import *


__script__    = "packing-box"
__author__    = "Alexandre D'Hondt"
__email__     = "alexandre.dhondt@gmail.com"
__copyright__ = ("A. D'Hondt", 2021)
__license__   = "gpl-3.0"


BANNER_FONT       = "starwars"
LISTS_LOCATION    = "/root/.%s"
MARKDOWN_TEMPLATE = """
This Docker image is a ready-to-use platform for making datasets of packed and not packed executables, especially for training machine learning models.

"""


if __name__ == '__main__':
    initialize()
    md = MARKDOWN_TEMPLATE
    tools_section = "## Tools\n\n**Tool** | **Description**\n---|---\n"
    with open(LISTS_LOCATION % "tools.csv") as f:
        preader = csv.reader(f, delimiter=";")
        for row in preader:
            tools_section += "|".join(row) + "\n"
    md += tools_section + "\n\n"
    packers_section = "## Packers\n\n**Packer** | **URL** | **Targets** | **Status**\n---|---|:---:|:---:\n"
    with open(LISTS_LOCATION % "packers.csv") as f:
        preader = csv.reader(f, delimiter=";")
        for row in preader:
            row[-1] = "✓✗"[shutil.which(row[0].lower().strip()) is None]
            packers_section += "|".join(row) + "\n"
    md += packers_section
    print(mdv.main(md))

