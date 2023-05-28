import difflib
import pefile
from matplotlib import pyplot
import seaborn
import numpy as np
import os
import lief
from matplotlib.colors import ListedColormap
from matplotlib.cm import ScalarMappable
from bintropy import COLORS


__all__ = ["binary_diff_readable", "binary_diff_plot"]


def binary_diff_readable(file1, file2, label1=None, label2=None, n=0):
    """Generates a text-based difference between two PE files. 

    Args:
        file1 (str): first file
        file2 (str): second file
        label1 (str, optional): Name to use for file1. If None, uses the filename. Defaults to None.
        label2 (str, optional): Name to use for file2. If None, uses the filename. Defaults to None.
        n (int, optional): Amount of carriage returns between the sequences. Defaults to 0.

    Returns:
        str: Difference between the files, in text format
    """
    if label1 == None:
        label1 = file1
    if label2 == None:
        label2 = file2

    dump1 = pefile.PE(file1).dump_info()
    dump2 = pefile.PE(file2).dump_info()

    return '\n'.join(difflib.unified_diff(dump1.split('\n'), dump2.split('\n'), label1, label2, n=n))

# Helper functions from Bintropy

def __btype(b): return str(type(b)).split(".")[2]
def __secname(s): return s.strip("\x00") or s or "<empty>"


def _get_ep_and_section(binary):
    """ Helper for computing the entry point and finding its section for each supported format.
    :param binary: LIEF-parsed binary object
    :return:       (ep_file_offset, name_of_ep_section)
    """
    btype = __btype(binary)
    try:
        if btype in ["ELF", "MachO"]:
            ep = binary.virtual_address_to_offset(binary.entrypoint)
            ep_section = binary.section_from_offset(ep)
        elif btype == "PE":
            ep = binary.rva_to_offset(
                binary.optional_header.addressof_entrypoint)
            ep_section = binary.section_from_rva(
                binary.optional_header.addressof_entrypoint)
        else:
            raise OSError("Unknown format")
        return ep, ep_section.name
    except (AttributeError, lief._lief.lief_errors.not_found, lief._lief.lief_errors.conversion_error):
        return None, None


def characteristics_no_entropy(executable):
    """Helper function to compute characteristcs of the file, like Bintropy, but avoid entropy computations.

    Args:
        executable (str or Path): The executable

    Raises:
        TypeError: Not an executable that can be parsed with LIEF

    Returns:
        dict: Characteristics of the binary
    """
    data = {'name': os.path.basename(executable), 'sections': []}
    binary = lief.parse(str(executable))
    data['type'] = __btype(binary)
    if binary is None:
        raise TypeError("Not an executable")

    chunksize = 1
    with open(str(executable), "rb") as f:
        size = data['size'] = os.fstat(f.fileno()).st_size
    n_samples = size
    MIN_ZONE_WIDTH = 0

    # entry point (EP)
    ep, ep_section = _get_ep_and_section(binary)
    # convert to 3-tuple (EP offset on plot, EP file offset, section name containing EP)
    data['entrypoint'] = None if ep is None else (
        int(ep // chunksize), ep, __secname(ep_section))
    # sections
    data['sections'] = [(0, int(max(MIN_ZONE_WIDTH, binary.sections[0].offset //
                         chunksize)), "Headers")] if len(binary.sections) > 0 else []

    for section in sorted(binary.sections, key=lambda x:x.offset):
        name = __secname(section.name)
        start = max(data['sections'][-1][1] if len(data['sections']) > 0 else 0,
                    int(section.offset // chunksize))
        max_end = min(max(start + MIN_ZONE_WIDTH, int((section.offset + section.size) // chunksize)),
                      n_samples)
        data['sections'].append(
            (int(min(start, max_end - MIN_ZONE_WIDTH)), int(max_end), name))
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
            data['sections'].append(
                (int(last), int(last) + sh_size // chunksize, "Header"))
        elif data['type'] == "PE":
            # add overlay
            if last + 1 < n_samples:
                data['sections'].append((int(last), int(n_samples), "Overlay"))
    return data


def binary_diff_plot(file1, file2, img_name=None, img_format="png", label1="", label2="", dpi=200, title=None, **kwargs):
    """Plots the byte-wise difference between two exectables

    Args:
        file1 (str): The first file
        file2 (str): The second file
        img_name (str, optional): Filename to save the image, without extension. If None, uses the name of file1. Defaults to None.
        img_format (str, optional): Image extension. Defaults to "png".
        label1 (str, optional): Name for file1. Defaults to None.
        label2 (str, optional): Name for file2. Defaults to None.
        dpi (int, optional): Dots per inch for the image. Defaults to 200.
        title (str, optional): Preferred plot title. Defaults to None.
    """
    import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = "serif"

    lloc, title_bool = kwargs.get(
        'legend_location', "lower right"), not kwargs.get('no_title', False)
    lloc_side = lloc.split()[1] in ["left", "right"]
    nf, N_TOP, N_TOP2, N_BOT, N_BOT2 = 2, 1.2, 1.6, -.15, -.37
    fig, objs = plt.subplots(nf+[0, 1][title_bool], sharex=True)
    fig.set_size_inches(15, nf+[0, 1][title_bool])
    fig.tight_layout(pad=2.5)
    objs[-1].axis("off")

    values = {'delete': 0, 'replace': 1, 'equal': 2, 'insert': 3}
    colors = ['red', 'gold', 'lightgrey', 'green']

    with open(file1, 'rb') as f1:
        p1 = f1.read()
    with open(file2, 'rb') as f2:
        p2 = f2.read()

    cruncher = difflib.SequenceMatcher(a=p1, b=p2)
    tags, alo, ahi, blo, bhi = zip(*cruncher.get_opcodes())
    opcodes_1 = zip(tags, alo, ahi)
    opcodes_2 = zip(tags, blo, bhi)

    if title_bool:
        fig.suptitle("Byte-wise difference" if title is None else title,
                     x=[.5, .55][label1 is None],
                     y=1,
                     ha="center", va="bottom",
                     fontsize="xx-large", fontweight="bold")
        
    label1 = os.path.basename(file1) if label1 == None else label1
    label2 = os.path.basename(file2) if label2 == None else label2
    
    text_x = -0.012*max(len(p1)*(len(label1)+3), len(p2)*(len(label2)+3))
    
    for i, d in enumerate([(p1, file1, opcodes_1, label1), (p2, file2, opcodes_2, label2)]):
        p, file, opcodes, label = d

        data = characteristics_no_entropy(file)
        n = len(p)
        obj = objs[i]

        obj.axis("off")

        ref_point = .65

        y_pos = ref_point
        obj.text(s=label, x=text_x, y=y_pos,
                 fontsize="large", ha="left", va="center")
        # display the entry point
        if data['entrypoint']:
            obj.vlines(x=data['entrypoint'][0], ymin=0, ymax=1,
                       color="r", zorder=11).set_label("Entry point")
            obj.text(data['entrypoint'][0], -.15, "______", color="r", ha="center", rotation=90, size=.8,
                     bbox={'boxstyle': "rarrow", 'fc': "r", 'ec': "r", 'lw': 1})
        color_cursor, last = 0, None
        j = 0
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
                obj.text(s=name, x=start + (end - start) // 2, y=pos_y,
                         zorder=12, color=c, ha="center", va="center")
                last = (
                    start, end, last[0] if last else None, last[1] if last else None)
            j += 1
        # draw modifications
        for (tag, lo, hi) in opcodes:
            obj.fill_between((lo, hi), 0, 0.7,
                             facecolor=colors[values[tag]], alpha=1)

        if len(data['sections']) == 0:
            obj.text(.5, ref_point, "Could not parse sections",
                     fontsize=16, color="red", ha="center", va="center")

    cb = plt.colorbar(ScalarMappable(cmap=ListedColormap(colors, N=4)),
                      location='bottom', ax=objs[-1], fraction=0.3, aspect=50, ticks=[0.125, 0.375, 0.625, 0.875])
    cb.set_ticklabels(['removed', 'modified', 'untouched', 'added'])
    cb.ax.tick_params(length=0)
    cb.outline.set_visible(False)

    plt.subplots_adjust(left=[.15, .02][label1 == "" and label2 == ""], right=[1.02, .82][lloc_side],
                        bottom=.5/max(1.75, nf))
    h, l = (objs[[0, 1][title_bool]] if nf+[0, 1][title_bool] >
            1 else objs).get_legend_handles_labels()
    if len(h) > 0:
        plt.figlegend(h, l, loc=[.8, .135], ncol=1 if lloc_side else 2,
                      prop={'size': 9})
    img_name = img_name or os.path.splitext(os.path.basename(file1))[0]
    # appending the extension to img_name is necessary for avoiding an error when the filename contains a ".[...]" ;
    #  e.g. "PortableWinCDEmu-4.0" => this fails with "ValueError: Format '0' is not supported"
    try:
        plt.savefig(img_name + "." + img_format,
                    img_format=img_format, dpi=dpi, bbox_inches="tight")
    except:  # format argument renamed in further versions of pyplot
        plt.savefig(img_name + "." + img_format, format=img_format,
                    dpi=dpi, bbox_inches="tight")
    return plt
