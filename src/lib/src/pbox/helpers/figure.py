# -*- coding: UTF-8 -*-
from tinyscript import functools
from tinyscript.helpers import Path


__all__ = ["figure_path", "mpl", "plt", "save_figure"]


_RC_CONFIG = {
    'colormap':    "image.cmap",
    'font_family': "font.family",
    'style':       lambda v: plt.style.use(v),
    'font_size':   "font.size",
}


def __init_mpl():
    import matplotlib as mpl
    return mpl


def __config_plt(**kwargs):
    if len(kwargs) > 0:
        for k, v in kwargs.items():
            if k in _RC_CONFIG.keys():
                tgt = _RC_CONFIG[k]
                if isinstance(tgt, type(lambda: 0)):
                    tgt(v)
                else:
                    globals()['plt'].rcParams[_RC_CONFIG[k]] = v
            else:
                raise ValueError(f"Bad matplotlib parameter '{k}'")
    else:
        import matplotlib.pyplot as plt
        plt.rcParams['figure.dpi'] = config['dpi']
        plt.rcParams['font.family'] = config['font_family']
        plt.rcParams['font.size'] = tfs = config['font_size']
        plt.rcParams['xtick.labelsize'] = plt.rcParams['ytick.labelsize'] = tfs - 2
        plt.rcParams['figure.titleweight'] = "bold"
        plt.rcParams['image.cmap'] = config['colormap']
        plt.set_cmap(config['colormap'])
        plt.style.use(config['style'])
        plt.set_loglevel("critical")
        return plt


lazy_load_object("mpl", __init_mpl)
lazy_load_object("plt", __config_plt)


def figure_path(filename, format=None, **kw):
    # fix the extension if not specified
    p = Path(filename)
    if p.extension[1:] not in IMG_FORMATS:
        filename = f"{p.dirname.joinpath(p.stem)}.{format or config['format']}"
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
        l = getattr(a[0], "logger", null_logger)
        l.info("Preparing plot data...")
        values = {k: kw.get(k, config[k]) for k in config._defaults['visualization'].keys()}
        kw_plot = {k: v for k, v in values.items() if k not in _RC_CONFIG.keys()}
        __config_plt(**{k: v for k, v in values.items() if k in _RC_CONFIG.keys()})
        try:
            imgs = f(*a, **kw)
        except KeyError as e:
            l.error(f"Plot type '{e.args[0]}' does not exist (should be one of [{'|'.join(_PLOTS.keys())}])")
            return
        for img in (imgs if isinstance(imgs, (list, tuple, type(x for x in []))) else [imgs]):
            if img is None:
                continue
            img = figure_path(img, format=kw.get('format'))
            l.info(f"Saving to {img}...")
            plt.savefig(img, **kw_plot)
            l.debug(f"> saved to {img}...")
    return _wrapper

