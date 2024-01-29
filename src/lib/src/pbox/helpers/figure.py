# -*- coding: UTF-8 -*-
from tinyscript import functools
from tinyscript.helpers import Path


__all__ = ["figure_path", "mpl", "plt", "save_figure"]


def __init_mpl():
    import matplotlib as mpl
    return mpl


def __init_plt():
    import matplotlib.pyplot as plt
    plt.rcParams['figure.dpi'] = config['dpi']
    plt.rcParams['figure.titlesize'] = tfs = config['title_font_size']
    plt.rcParams['xtick.labelsize'] = plt.rcParams['ytick.labelsize'] = tfs - 2
    plt.rcParams['figure.titleweight'] = "bold"
    plt.rcParams['font.family'] = config['font_family']
    plt.rcParams['image.cmap'] = config['colormap']
    plt.set_cmap(config['colormap'])
    if config['dark_mode']:
        plt.style.use(['dark_background', 'presentation'])
    return plt


lazy_load_object("mpl", __init_mpl)
lazy_load_object("plt", __init_plt)


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
        import matplotlib.pyplot as plt
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
            plt.savefig(img, **kw_plot)
            l.debug("> saved to %s..." % img)
    return _wrapper

