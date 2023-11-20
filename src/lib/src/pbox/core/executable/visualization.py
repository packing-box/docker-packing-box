# -*- coding: UTF-8 -*-
from tinyscript.helpers import Path


__all__ = ["binary_diff_plot", "binary_diff_text"]

_MIN_ZONE_WIDTH = 0

# helper functions from Bintropy
__btype   = lambda b: str(type(b)).split(".")[2]
__secname = lambda s: s.strip("\x00") or s or "<empty>"


def _characteristics_no_entropy(executable):
    """Helper function to compute characteristcs of the file (like Bintropy) but avoiding entropy computations.
    
    :param executable: target executable for which the characteristics are to be computed
    :return:           dictionary of characteristics from the target binary
    """
    data = {'name': Path(executable).basename, 'sections': []}
    binary = executable.parse("lief")
    data['hash'] = executable.hash
    data['type'] = __btype(binary)
    chunksize = 1
    size = data['size'] = Path(str(executable)).size
    n_samples = size
    # entry point (EP)
    ep, ep_section = binary.entrypoint, binary.entrypoint_section.name
    # convert to 3-tuple (EP offset on plot, EP file offset, section name containing EP)
    data['entrypoint'] = None if ep is None else (int(ep // chunksize), ep, __secname(ep_section))
    # sections
    data['sections'] = [(0, int(max(_MIN_ZONE_WIDTH, binary.sections[0].offset // chunksize)), "Headers")] \
                       if len(binary.sections) > 0 else []
    for section in sorted(binary.sections, key=lambda x:x.offset):
        name = __secname(getattr(binary, "real_section_names", {}).get(section.name, section.name))
        start = max(data['sections'][-1][1] if len(data['sections']) > 0 else 0, int(section.offset // chunksize))
        max_end = min(max(start + _MIN_ZONE_WIDTH, int((section.offset + section.size) // chunksize)), n_samples)
        data['sections'].append((int(min(start, max_end - _MIN_ZONE_WIDTH)), int(max_end), name))
    # adjust the entry point (be sure that its position on the plot is within the EP section)
    if ep:
        ep_pos, _, ep_sec_name = data['entrypoint']
        for s, e, name in data['sections']:
            if name == ep_sec_name:
                data['entrypoint'] = (min(max(ep_pos, s), e), ep, ep_sec_name)
    # fill in undefined sections
    prev_end = None
    for i, t in enumerate(data['sections'][:]):
        start, end, name = t
        if prev_end and prev_end < start:
            data['sections'].insert(i, prev_end, start, "<undef>")
        prev_end = end
    if len(binary.sections) > 0:
        last = data['sections'][-1][1]
        if data['type'] == "ELF":
            # add section header table
            sh_size = binary.header.section_header_size * binary.header.numberof_sections
            data['sections'].append((int(last), int(last) + sh_size // chunksize, "Header"))
        elif data['type'] == "PE":
            # add overlay
            if last + 1 < n_samples:
                data['sections'].append((int(last), int(n_samples), "Overlay"))
    return data


def binary_diff_plot(file1, file2, img_name=None, img_format="png", legend1="", legend2="", dpi=400, title=None,
                     no_title=False, logger=null_logger, **kwargs):
    """ Plots the byte-wise difference between two exectables.
    
    :param file1:      first file's name
    :param file2:      second file's name
    :param img_name:   filename to save the image (without extension) ; if None, use file1
    :param img_format: image extension
    :param legend1:    first file's alias (file1 if None)
    :param legend2:    second file's alias (file2 if None)
    :param dpi:        dots per inch for the image
    :param title:      preferred plot title
    :return:           plot module object
    """
    import matplotlib.pyplot as plt
    from bintropy import COLORS
    from difflib import SequenceMatcher
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import ListedColormap
    lloc = kwargs.get('legend_location', "lower right")
    lloc_side = lloc.split()[1] in ["left", "right"]
    nf, N_TOP, N_TOP2, N_BOT, N_BOT2 = 2, 1.2, 1.6, -.15, -.37
    tl_offset = [1, 0][no_title]
    fig, objs = plt.subplots(nf+tl_offset, sharex=True)
    fig.set_size_inches(15, nf+tl_offset)
    fig.tight_layout(pad=2.5)
    objs[-1].axis("off")
    values = {'delete': 0, 'replace': 1, 'equal': 2, 'insert': 3}
    colors = ['red', 'gold', 'lightgrey', 'green']
    logger.debug("opening files...")
    with file1.open('rb') as f1, file2.open('rb') as f2:
        p1, p2 = f1.read(), f2.read()
    logger.debug("matching sequences...")
    cruncher = SequenceMatcher(a=p1, b=p2)
    tags, alo, ahi, blo, bhi = zip(*cruncher.get_opcodes())
    if not no_title:
        fig.suptitle("Byte-wise difference" if title is None else title, x=[.5, .55][legend1 is None], y=1,
                     ha="center", va="bottom", fontsize="xx-large", fontweight="bold")
    legend1, legend2 = legend1 or Path(file1).basename, legend2 or Path(file2).basename
    text_x = -0.012*max(len(p1)*(len(legend1)+3), len(p2)*(len(legend2)+3))
    logger.debug("plotting...")
    for i, d in enumerate([(p1, file1, zip(tags, alo, ahi), legend1), (p2, file2, zip(tags, blo, bhi), legend2)]):
        p, f, opcodes, label = d
        data = _characteristics_no_entropy(f)
        n = len(p)
        obj = objs[i]
        obj.axis("off")
        y_pos = ref_point = .65
        obj.text(s=label, x=text_x, y=y_pos, fontsize="large", ha="left", va="center")
        fh = "\n".join(data['hash'][i:i+32] for i in range(0, len(data['hash']), 32))
        obj.text(s=fh, x=text_x, y=y_pos-.65, fontsize="xx-small", color="lightgray", ha="left", va="center")
        # display the entry point
        if data['entrypoint']:
            obj.vlines(x=data['entrypoint'][0], ymin=0, ymax=1, color="r", zorder=11).set_label("Entry point")
            obj.text(data['entrypoint'][0], -.15, "______", color="r", ha="center", rotation=90, size=.8,
                     bbox={'boxstyle': "rarrow", 'fc': "r", 'ec': "r", 'lw': 1})
        color_cursor, last, j = 0, None, 0
        for start, end, name in data['sections']:
            x = range(start, min(n, end+1))
            # select the right color first
            try:
                c = COLORS[name.lower().lstrip("._").strip("\x00\n ")]
            except KeyError:
                co = COLORS[None]
                c = co[color_cursor % len(co)]
                color_cursor += 1
            # draw the section
            obj.fill_between(x, 0, 1, facecolor=c, alpha=.2)
            if name not in ["Headers", "Overlay"]:
                pos_y = [N_TOP2, N_TOP][j % 2]
                obj.text(s=name, x=start + (end - start) // 2, y=pos_y, zorder=12, color=c, ha="center", va="center")
                last = (start, end, last[0] if last else None, last[1] if last else None)
            j += 1
        # draw modifications
        for (tag, lo, hi) in opcodes:
            obj.fill_between((lo, hi), 0, 0.7, facecolor=colors[values[tag]], alpha=1)
        if len(data['sections']) == 0:
            obj.text(.5, ref_point, "Could not parse sections", fontsize=16, color="red", ha="center", va="center")
    cb = plt.colorbar(ScalarMappable(cmap=ListedColormap(colors, N=4)),
                      location='bottom', ax=objs[-1], fraction=0.3, aspect=50, ticks=[0.125, 0.375, 0.625, 0.875])
    cb.set_ticklabels(['removed', 'modified', 'untouched', 'added'])
    cb.ax.tick_params(length=0)
    cb.outline.set_visible(False)
    plt.subplots_adjust(left=[.15, .02][legend1 == "" and legend2 == ""], bottom=.5/max(1.75, nf))
    h, l = (objs[tl_offset] if nf+tl_offset > 1 else objs).get_legend_handles_labels()
    if len(h) > 0:
        plt.figlegend(h, l, loc=[.8, .135], ncol=1, prop={'size': 9})
    img_name = img_name or Path(file1).stem
    logger.debug("saving file...")
    # appending the extension to img_name is necessary for avoiding an error when the filename contains a ".[...]" ;
    #  e.g. "PortableWinCDEmu-4.0" => this fails with "ValueError: Format '0' is not suppored"
    try:
        plt.savefig(img_name + "." + img_format, img_format=img_format, dpi=dpi, bbox_inches="tight")
    except TypeError:  # format argument renamed in further versions of pyplot
        plt.savefig(img_name + "." + img_format, format=img_format, dpi=dpi, bbox_inches="tight")
    return plt


def binary_diff_text(file1, file2, legend1=None, legend2=None, n=0, logger=null_logger, **kwargs):
    """ Generates a text-based difference between two PE files. 
    
    :param file1:   first file's name
    :param file2:   second file's name
    :param legend1: first file's alias (file1 if None)
    :param legend2: second file's alias (file2 if None)
    :param n:       amount of carriage returns between the sequences
    :return:        difference between the files, in text format
    """
    from difflib import unified_diff as udiff
    from pefile import PE
    logger.debug("dumping files info...")
    dump1, dump2 = PE(file1).dump_info(), PE(file2).dump_info()
    return "\n".join(udiff(dump1.split('\n'), dump2.split('\n'), legend1 or str(file1), legend2 or str(file2), n=n))

