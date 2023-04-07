import difflib
import pefile
from matplotlib import pyplot
import seaborn
import numpy as np

__all__ = ["binary_diff_readable", "binary_diff_plot"]

def binary_diff_readable(file1, file2, label1="", label2="", n=0):
    
    dump1 = pefile.PE(file1).dump_info()
    dump2 = pefile.PE(file2).dump_info()
    
    return '\n'.join(difflib.context_diff(dump1.split('\n'), dump2.split('\n'), label1, label2, n=n))

def binary_diff_plot(file1, file2, img_name=None, label1="", label2=""):
    values = {'delete':-1, 'replace':0, 'equal': 1, 'insert':2}
    width = 4096
    colors = ['black', 'red', 'gold', 'white', 'green']
    
    with open(file1, 'rb') as f1:
        p1 = f1.read()
    with open(file2, 'rb') as f2:
        p2 = f2.read()
    
    arr1 = [0] * len(p1)
    arr2 = [0] * len(p2)

    cruncher = difflib.SequenceMatcher(a=p1, b=p2)
    for tag, alo, ahi, blo, bhi in cruncher.get_opcodes():
        v = values[tag]
        arr1[alo:ahi] = [v] * (ahi - alo)
        arr2[blo:bhi] = [v] * (bhi - blo)

    arr1 += [-2]*(width - len(arr1)%width)
    arr2 += [-2]*(width - len(arr2)%width)
    arr1 = np.reshape(arr1, (len(arr1)//width, width))
    arr2 = np.reshape(arr2, (len(arr2)//width, width))
    
    fig, axs = pyplot.subplots(1, 2, figsize=(15, 15), dpi=80, sharey=True)
    fig.suptitle("Bytes modified by the transformation")
    
    seaborn.heatmap(arr1,ax=axs[0], cmap=colors, vmin=-2.5, vmax=2.5,
                        cbar_kws={"ticks":list(values.values())})
    axs[0].collections[0].colorbar.set_ticklabels(['removed', 'modified', 'untouched', 'added'])
    axs[0].axhline(y=0, color='k',linewidth=1)
    axs[0].axhline(y=arr1.shape[0]*.999, color='k',linewidth=1)
    axs[0].axvline(x=0, color='k',linewidth=1)
    axs[0].axvline(x=arr1.shape[1]*.998, color='k',linewidth=1)
    axs[0].set_title(label1)
    axs[0].arrow(0, 0, 0, arr1.shape[0]*.999)
    axs[0].set_xlabel(f"{width} bytes")
    axs[0].set_yticks([0, arr1.shape[0]])
    axs[0].set_yticklabels(["Offset 0", "EOF"])
    axs[0].set_xticks([])
    axs[0].set_xticklabels([])
    
    seaborn.heatmap(arr2, ax=axs[1], cmap=colors, vmin=-2.5, vmax=2.5, #cbar=False,
                        cbar_kws={"ticks":list(values.values())})
    axs[1].collections[0].colorbar.set_ticklabels(['removed', 'modified', 'untouched', 'added'])
    axs[1].axhline(y=0, color='k',linewidth=1)
    axs[1].axhline(y=arr2.shape[0]*.999, color='k',linewidth=1)
    axs[1].axvline(x=0, color='k',linewidth=1)
    axs[1].axvline(x=arr2.shape[1]*.998, color='k',linewidth=1)
    axs[1].set_title(label2)
    axs[1].arrow(0, 0, 0, arr2.shape[0]*.999)
    axs[0].set_xlabel(f"{width} bytes")
    axs[1].set_yticks([0, arr2.shape[0]])
    axs[1].set_yticklabels(["Offset 0", "EOF"])
    axs[1].set_xticks([])
    axs[1].set_xticklabels([])
    
    fig.tight_layout()
    if img_name is not None:
        pyplot.savefig(img_name)
    else:
        pyplot.show()