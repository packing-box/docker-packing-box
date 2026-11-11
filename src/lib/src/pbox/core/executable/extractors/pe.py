# -*- coding: UTF-8 -*-
from ....helpers.utils import execute_and_get_values_list


__all__ = ["pefeats"]


def pefeats(executable):
    """ This uses pefeats to extract 119 features from PE files.

    With the 'portable' PE parser (pe_parser option), the Python port of pefeats (pbox.portable) is used instead of the
     pefeats tool, as in the standalone tools exported from trained models. """
    if config['pe_parser'] == "portable":
        from ....portable.extractors.pefeats import pefeats_values, PefeatsError
        try:
            values = pefeats_values(str(executable))
        except PefeatsError:
            values = None   # as when pefeats fails
    else:
        values = execute_and_get_values_list(f"pefeats \'{executable}\'")
    return [None if x == -1. else x for x in values]
