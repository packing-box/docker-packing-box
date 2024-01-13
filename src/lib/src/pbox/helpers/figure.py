# -*- coding: UTF-8 -*-
from tinyscript import functools
from tinyscript.helpers import Path


__all__ = ["figure_path", "mpl", "plt", "save_figure"]


def _set_params(*a):
    plt.rcParams['figure.dpi'] = config['dpi']
    plt.rcParams['figure.titlesize'] = tfs = config['title_font_size']
    plt.rcParams['xtick.labelsize'] = plt.rcParams['ytick.labelsize'] = tfs - 2
    plt.rcParams['figure.titleweight'] = "bold"
    plt.rcParams['font.family'] = config['font_family']
    plt.rcParams['image.cmap'] = config['colormap']
    plt.set_cmap(config['colormap'])
    if config['dark_mode']:
        plt.style.use(['dark_background', 'presentation'])

#FIXME: lazy loading of mpl throws an error with plt
#lazy_load_module("matplotlib", alias="mpl")
import matplotlib as mpl
"""
Traceback (most recent call last):
  File "/home/user/.opt/tools/model", line 122, in <module>
    getattr(Model(**vars(args)), args.command)(**vars(args))
  File "/home/user/.local/lib/python3.11/site-packages/pbox/core/model/__init__.py", line 631, in visualize
    fig = viz_func(self.classifier, **params)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/user/.local/lib/python3.11/site-packages/pbox/core/model/visualization.py", line 27, in _wrapper
    fig = f(*a, **kw)
          ^^^^^^^^^^^
  File "/home/user/.local/lib/python3.11/site-packages/pbox/core/model/visualization.py", line 113, in image_clustering
    plt.rcParams['axes.labelsize'] = 16
    ^^^^^^^^^^^^
  File "/home/user/.local/lib/python3.11/site-packages/tinyscript/__conf__.py", line 48, in _load
    glob[alias] = glob[module] = m = import_module(*((module, ) if relative is None else ("." + module, relative)))
                                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.11/importlib/__init__.py", line 126, in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "<frozen importlib._bootstrap>", line 1204, in _gcd_import
  File "<frozen importlib._bootstrap>", line 1176, in _find_and_load
  File "<frozen importlib._bootstrap>", line 1147, in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 690, in _load_unlocked
  File "<frozen importlib._bootstrap_external>", line 940, in exec_module
  File "<frozen importlib._bootstrap>", line 241, in _call_with_frames_removed
  File "/home/user/.local/lib/python3.11/site-packages/matplotlib/pyplot.py", line 52, in <module>
    import matplotlib.colorbar
  File "/home/user/.local/lib/python3.11/site-packages/matplotlib/colorbar.py", line 19, in <module>
    from matplotlib import _api, cbook, collections, cm, colors, contour, ticker
ModuleNotFoundError: No module named 'mpl'
"""
lazy_load_module("matplotlib.pyplot", alias="plt", postload=_set_params)


def figure_path(filename, format=None):
    # fix the extension if not specified
    if Path(filename).extension[1:] not in IMG_FORMATS:
        filename = f"{filename}.{format or config['format']}"
    # at this point, 'filename' is either a string with eventual separators for subfolders if the figure is to be sorted
    #  in the scope of an experiment
    try:
        # by convention, if we are saving files in the workspace of an experiment, using separators will sort the figure
        #  according to the defined structure (e.g. if an experiment has multiple datasets, it is cleaner to sort its
        #  figures with a tree structure that uses datasets' names)
        filename = config['experiment'].joinpath("figures", filename)
        filename.dirname.mkdir(exist_ok=True, parents=True)
    # key 'experiment' does not exist as ~/.packing-box/experiment.env is not set (not in an experiment)
    except KeyError:
        # by convention, if the given path is not absolute and we are not saving files in the workspace of an open
        #  experiment, we shall remain in the current folder, hence replacing separators by underscores
        filename = Path(str(filename).replace("/", "_"))
    return filename


def save_figure(f):
    """ Decorator for computing the path of a figure and plotting it, given the filename returned by the wrapped
         function ; put it in the "figures" subfolder of the current experiment's folder if relevant. """
    @functools.wraps(f)
    def _wrapper(*a, **kw):
        import matplotlib.pyplot
        l = getattr(a[0], "logger", null_logger)
        l.info("Preparing plot data...")
        try:
            imgs = f(*a, **kw)
        except KeyError:
            l.error(f"Plot type '{ptype}' does not exist (should be one of [{'|'.join(_PLOTS.keys())}])")
            return
        kw_plot = {k: kw.get(k, config[k]) for k in ['bbox_inches', 'dpi', 'format']}
        for img in (imgs if isinstance(imgs, (list, tuple, type(x for x in []))) else [imgs]):
            if img is None:
                continue
            img = figure_path(img, format=kw.get('format'))
            l.info("Saving to %s..." % img)
            matplotlib.pyplot.savefig(img, **kw_plot)
            l.debug("> saved to %s..." % img)
    return _wrapper

