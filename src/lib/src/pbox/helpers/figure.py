# -*- coding: UTF-8 -*-
from tinyscript import functools, logging, re
from tinyscript.helpers import Path


__all__ = ["configure_style", "figure_path", "mpl", "plt", "save_figure"]


def __init_mpl():
    import matplotlib as mpl
    return mpl


def __init_plt():
    import matplotlib.pyplot as plt
    return plt


lazy_load_object("mpl", __init_mpl)
lazy_load_object("plt", __init_plt)

logging.silentLogger("matplotlib.font_manager")


def configure_style(**kw):
    mpl.rc('font', **{k.split("_")[1]: kw.pop(k, config[k]) for k in ['font_family', 'font_size']})
    kw['title-font'] = {'fontfamily': kw.pop('title_font_family', config['font_family']),
                        'fontsize': kw.pop('title_font_size', int(config['font_size'] * 1.6)),
                        'fontweight': kw.pop('title_font_weight', "bold")}
    kw['suptitle-font'] = {'fontfamily': kw.pop('suptitle_font_family', config['font_family']),
                           'fontsize': kw.pop('suptitle_font_size', int(config['font_size'] * 1.2)),
                           'fontweight': kw.pop('suptitle_font_weight', "normal")}
    kw['legend-font'] = {'fontsize': kw.pop('legend_font_size', int(config['font_size'] * .8))}
    for p in "xy":
        kw[f'{p}label-font'] = {'fontfamily': kw.pop(f'{p}label_font_family', config['font_family']),
                                'fontsize': kw.pop(f'{p}label_font_size', config['font_size']),
                                'fontweight': kw.pop(f'{p}label_font_weight', "normal")}
    return kw


def figure_path(filename, img_format=None, **kw):
    # fix the extension if not specified
    p = Path(filename)
    if p.extension[1:] not in IMG_FORMATS:
        filename = f"{p.dirname.joinpath(p.stem)}.{img_format or config['img_format']}"
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
        try:
            imgs = f(*a, **configure_style(**kw))
        except KeyError as e:
            l.error(f"Plot type '{e.args[0]}' does not exist")
            return
        kw_plot = {{'img_format': "format"}.get(k, k): kw.get(k, config[k]) \
                   for k in ["bbox_inches", "dpi", "img_format"]}
        for img in (imgs if isinstance(imgs, (list, tuple, type(x for x in []))) else [imgs]):
            if img is None:
                continue
            img = figure_path(img, kw.get('img_format'))
            if kw.get('interactive_mode', False):
                from code import interact
                ns = {k: v for k, v in globals().items()}
                ns.update(locals())
                l.info(f"{img}: use 'plt.savefig(img, **kw_plot)' to save the figure")
                interact(local=ns)
            l.info(f"Saving to {img}...")
            plt.savefig(img, **kw_plot)
            l.debug(f"> saved to {img}...")
    return _wrapper

