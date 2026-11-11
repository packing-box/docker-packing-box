# -*- coding: UTF-8 -*-
""" Terminal user interface (Textual) for managing the abstractions of the current workspace of the Packing Box:
     items (analyzers, detectors, packers, unpackers), definitions (algorithms, alterations, features, scenarios),
     datasets, models, experiments and the configuration. """
import argparse


__all__ = ["main"]


def main(argv=None):
    from .app import CATEGORIES, PackingBoxApp
    p = argparse.ArgumentParser("pbox-tui", description="Manage the abstractions (detectors, packers, datasets, ...) "
                                "of the current workspace of the Packing Box (the default one or the open experiment)")
    p.add_argument("category", nargs="?", default="detectors", choices=list(CATEGORIES),
                   help="category to show first (default: detectors)")
    a = p.parse_args(argv)
    PackingBoxApp(category=a.category).run()
