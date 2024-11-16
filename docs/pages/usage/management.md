# Management

[*Packing Box*](https://github.com/dhondta/docker-packing-box) is designed to be used by researchers without the need to touch the code base. Therefore, it offers a tool, [`packing-box`](https://github.com/packing-box/docker-packing-box/blob/main/src/files/tools/packing-box), that allows to manage the most important parameters of the toolkit, including paths to YAML configurations for various assets like machine learning algorithms, packers, detectors, features and some others. It also allows to perform actions on the current workspace, like modifying data files (e.g. common API call names for PE).

## Configuration

The current configuration can be inspected using the `list` command as shown here below.

```console
┌──[user@packing-box]──[/mnt/share]────────
$ packing-box list config

Main

 • workspace     = /home/user/.packing-box
 • experiments   = /mnt/share/experiments
 • backup_copies = 3
 • exec_timeout  = 20
 • number_jobs   = 6

Cfg

 • angr_engine            = default
 • depth_max_iterations   = 1024
 • exclude_duplicate_sigs = True
 • extract_algorithm      = Emulated
 • extract_timeout        = 17
 • include_cut_edges      = True
 • only_opcodes           = True
 • opcode_mnemonics       = False
 • store_loop_cut_info    = True

Definitions

 • algorithms  = /home/user/.packing-box/conf/algorithms.yml
 • alterations = /home/user/.packing-box/conf/alterations.yml
 • analyzers   = /home/user/.packing-box/conf/analyzers.yml
 • detectors   = /home/user/.packing-box/conf/detectors.yml
 • features    = /home/user/.packing-box/conf/features.yml
 • packers     = /home/user/.packing-box/conf/packers.yml
 • scenarios   = /home/user/.packing-box/conf/scenarios.yml
 • unpackers   = /home/user/.packing-box/conf/unpackers.yml

Logging

 • lief_logging = False
 • wine_errors  = False

Others

 • autocommit     = False
 • data           = /home/user/.packing-box/data
 • hash_algorithm = sha256
 • vt_api_key     = <<REDACTED>>

Parsers

 • default_parser = lief
 • elf_parser     = lief
 • macho_parser   = lief
 • pe_parser      = lief

Visualization

 • bbox_inches    = tight
 • colormap_main  = RdYlGn_r
 • colormap_other = jet
 • dpi            = 300
 • font_family    = serif
 • font_size      = 10
 • img_format     = png
 • style          = default
 • transparent    = False

```

A setting can be adapted by stating its name as an option with the `config` command as shown here below.

```console
┌──[user@packing-box]──[/mnt/share]────────
$ packing-box config --vt-api-key [YOUR-API-KEY-HERE]
```

## Files

The status of the current workspace can be inspected with the `workspace view`.

```console
┌──[user@packing-box]──[/mnt/share]────────
$ packing-box workspace view
/home/user/.packing-box
├── conf
│   ├── algorithms.yml
│   ├── alterations.yml
│   ├── analyzers.yml
│   ├── detectors.yml
│   ├── features.yml
│   ├── packers.yml
│   ├── scenarios.yml
│   └── unpackers.yml
├── data
│   ├── elf
│   │   ├── common_packer_section_names.txt
│   │   └── standard_section_names.txt
│   ├── macho
│   │   └── standard_section_names.txt
│   └── pe
│       ├── common_api_imports.txt
│       ├── common_dll_imports.json
│       ├── common_malicious_apis.txt
│       ├── common_packer_section_names.txt
│       ├── dead_code.txt
│       ├── make-common-api-imports.py
│       └── standard_section_names.txt
├── datasets
└── models

8 directories, 18 files

``

To adapt something, the `workspace edit` command can simply be used, pointing to the target resource path relative to `/home/user/.packing-box`. Vim will then be opened on this file for edition.

```console
┌──[user@packing-box]──[/mnt/share]────────
$ packing-box workspace edit data/pe/common_api_imports.txt

```

