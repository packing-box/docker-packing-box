# Use Cases

## Creating a dataset of packed executables from unlabelled samples

**Scenario**: You have a folder with samples and you need to ingest them into a new dataset while you don't have their labels. Moreover, you want to select only packed samples for your dataset, hence requiring detection.

**Solution**:

**Command** | **Description**
--- | ---
`dataset update my-dataset --source-dir path/to/samples --detect` | This will create `my-dataset` if it does not exist and `update` it with the samples from `path/to/samples`, applying [superdetection](/en/latest/usage/detectors.html#superdetection) based on the voting detectors configured (`vote: true`) in your [YAML configuration](https://github.com/packing-box/docker-packing-box/blob/main/src/conf/detectors.yml).
`dataset remove my-dataset --query "label == '-'"` | This will remove all the samples that have the label *not-packed* (represented as "`-`").

