# -*- coding: UTF-8 -*-
from ....helpers.utils import execute_and_get_values_list


__all__ = ["pefeats"]


def pefeats(executable):
    """ This uses pefeats to extract 119 features from PE files. """
    return [None if x == -1. else x for x in execute_and_get_values_list(f"pefeats \'{executable}\'")]

