# Static Detection

*Detectors* are programs that analyze executable samples. They aim to detect packing traces or algorithms. They can either work on binary classes only (True/False) or binary and multi-class (each class being a packer name).

## Tool

A [dedicated tool called `detector`](https://github.com/packing-box/docker-packing-box/blob/main/src/files/tools/detector) is provided with the [*Packing Box*](https://github.com/dhondta/docker-packing-box) to detect packers. Its help message tells everything the user needs to get started.

```console
┌──[user@packing-box]──[/mnt/share]────────
$ detector
usage: detector [-b] [-d [DETECTOR ...]] [-f | -m] [-s] [-t THRESHOLD] [-w] [-h] [--help] [-v] executable

┌──[user@packing-box]──[/mnt/share]────────
$ detector --help
[...]
This tool aims to detect the packer used on an input executable, folder of executables or Dataset.
[...]
positional arguments:
  executable  executable or folder containing executables or dataset

optional arguments:
  -b, --binary          only consider if packed or not (default: False)
  -d [DETECTOR [DETECTOR ...]], --detector [DETECTOR [DETECTOR ...]]
                        detector(s) to be used (default: None)
  -f, --failures-only   output failures only (default: False)
  -m, --metrics-only    output metrics only (default: False)
  -w, --weak            also consider weak assumptions (default: False)
[...]
```

From the optional arguments, we can see that it allows to force the detector to work with binary classes (`-b`/`--binary` -- packed or not packed) and also to consider weak assumptions (`-w`/`--weak` -- typically, when detectors output detections but suspicions as well, hence not formally deciding ; in this case, `detector` counts packer label occurrences from suspicions too and outputs a decision based on the highest number if the underlying tool did not formally decide). It can also compute metrics and output only entries that diverge from the expected result (`-f`/`--failures-only` -- this is only relevant when using the detector against a dataset, in which expected labels are included). It can display only the metrics (`-m`/`--metrics-only`) in raw text so that it can be piped to a file (i.e. in CSV foramt) when Bash-scripting the bulk-processing of many datasets.

## (Super)detection

This tool can be used directly on a dataset or a folder (or trivially on a single file). In this case, every sample from the target dataset is checked with the selected detector against its label and various performance metrics are determined, as shown in the example below.

```console
┌──[user@packing-box]──[/mnt/share]────────
$ detector test-upx -d manalyze -f

Detection results:
                                              
  Accuracy   Precision   Recall    F-Measure  
 ──────────────────────────────────────────── 
  100.00%    100.00%     100.00%   100.00%  

```

It can also be used without specifying a detector. In this case, every detector that is **allowed to vote** (this can be checked in the help message via the "`?`" tool) will be run on the target executable and the final decision is a simple plurality vote. In the following example, we use one level of verbosity ("`-v`") to see the results of each detector. Note that "`-vv`" would also display the output of each detector.

```console
┌──[user@packing-box]──[/mnt/share]────────
$ detector upx_calc.exe -v
12:34:56 [WARNING] Decisions:
die           : upx
manalyze      : upx
peframe       : upx
pepack        : -
peid          : upx
portex        : upx
pypackerdetect: upx
pypeid        : upx
22:21:22 [WARNING] upx_calc.exe: upx
upx
```

## Measurements

Different metrics are computed when applying detection to a dataset. When the target is:

- A dataset (respecting the [`Dataset` structure](datasets.html@structure)): Labels are retrieved from dataset's contained `data.csv` (if not set as "`?`" ; in this case, label is missing and can therefore not be used).
- A folder with samples: Labels are not available, hence no metric can be computed.

The following example shows execution on a very simple test dataset. Green and red colors (not present in the trace below) indicate successes or failures. If using the `-f`/`--failures-only` option, only failures will be displayed. When using the `-m`/`--metrics-only` option, only a single line will be output with the comma-separated metrics.

```console
┌──[user@packing-box]──[/mnt/share]────────
$ detector test-upx
00:00:00.205 [INFO] Superdetector: DIE, PEiD, PyPackerDetect, RetDec
00:00:03.437 [SUCCESS] /mnt/share/experiments/test/datasets/test-upx/files/1d8bf746ba84e321d1afd29c48a6f152e3195cb0c92d5559d75a455d5873eed9: not packed
00:00:06.965 [SUCCESS] /mnt/share/experiments/test/datasets/test-upx/files/21383926f0b5b3909f03f61f1df8b14c7d8f691136e5000c1612016638a9431f: not packed
00:00:09.489 [SUCCESS] /mnt/share/experiments/test/datasets/test-upx/files/3d2b2fd0d6f3fcb3a0107caa5679ace01f9d9cdb3c1ed37de66a8eb496428504: not packed
00:00:12.212 [SUCCESS] /mnt/share/experiments/test/datasets/test-upx/files/694b7a2075a9ec584346af7db65aacf55afce314e70753d0c88b5bb59d91ef27: not packed
00:00:15.112 [SUCCESS] /mnt/share/experiments/test/datasets/test-upx/files/695f311494155ba4890a2543d4398f97782f9a542b9ebf0978ff35633818200d: not packed
00:00:17.819 [SUCCESS] /mnt/share/experiments/test/datasets/test-upx/files/afa6815e81bd1da93d05465b21698da5d8decd177ac39d8fa2f724bbe4ab7711: upx
00:00:20.677 [SUCCESS] /mnt/share/experiments/test/datasets/test-upx/files/b73e7f86951d8f8a4b881cb69f6dbda28950b6015c558949f7a2d97781c71153: upx
00:00:23.584 [SUCCESS] /mnt/share/experiments/test/datasets/test-upx/files/c6e875d0be29bfefc9a5a517e108b395f1404cdb9e80cb5c1c3604f457eaa19d: not packed
00:00:26.155 [SUCCESS] /mnt/share/experiments/test/datasets/test-upx/files/e88f4aedd45410f2a44b94ff928529d9760bd1d35c09a818aa3039579068fe76: upx
00:00:28.726 [SUCCESS] /mnt/share/experiments/test/datasets/test-upx/files/ecd981b59b54c3e701079e478e908dd6b68f6ee8e8f7d319ba698c10a143ede0: upx

Detection results:
                                              
  Accuracy   Precision   Recall    F-Measure  
 ──────────────────────────────────────────── 
  100.00%    100.00%     100.00%   100.00%    
                                              
```

