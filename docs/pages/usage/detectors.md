# Detectors

*Detectors* are programs that analyze executable samples. They aim to detect packing traces or algorithms. They can either work on binary classes only (True/False) or binary and multi-class (each class being a packer name).

## Tool

A [dedicated tool](https://github.com/dhondta/docker-packing-box/blob/main/files/tools/detector) called `detector` is provided with the [*Packing Box*](https://github.com/dhondta/docker-packing-box) to detect packers. Its help message tells everything the user needs to get started.

```session
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

extra arguments:
  -h      show usage message and exit
  --help  show this help message and exit
  -v      verbose level (default: 0)
           NB: -vvv is the highest verbosity level
```

From the optional arguments, we can see that it allows to force the detector to work with binary classes (True/False, in other words packed or not packed) and also to consider weak assumptions (typically, when detectors output detections but suspicions as well). It can also compute metrics and output only entries that diverge from the expected result (this is only relevant when using the detector against a dataset, in which expected labels are included).

## Detection

This tool can be used directly on a dataset or a folder. In this case, every sample from the target dataset is checked with the selected detector against its label and various performance metrics are determined, as shown in the example below.

```session
# detector test-upx -d manalyze -f
100%|█████████████████████████████████████████████████████████████████████████████████████| 130/130 [00:29<00:00,  4.45executable/s]

 Detection results:
  ────────  ─────────  ───────  ─────────
  Accuracy  Precision  Recall   F-Measure
  100.00%   100.00%    100.00%  100.00%
  ────────  ─────────  ───────  ─────────

```

It can also be used on a single file without specifying a detector. In this case, every detector that is **allowed to vote** will be run on the target executable and the final decision is a simple majority vote. In the following example, we use one level of verbosity ("`-v`") to see the results of each detector. Note that "`-vv`" would also display the output of each detector.

```session
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


