# Contributing to Packing-Box
We want to make contributing to this project as easy and transparent as possible.

## Our Development Process

We use GitHub to sync code to and from our internal repository. We'll use GitHub to track issues and feature requests, as well as accept pull requests.

## Pull Requests

We actively welcome your pull requests.

1. Fork the repo and create your branch from `master`.
2. If you want to **tune container's bash console**, you can add/edit files in this folder: `files/term`. Please test what you tune and be sure not to break other related features.
3. If you want to **add a packer**, please refer to the section for packers in the [Dockerfile](https://github.com/dhondta/docker-packing-box/blob/main/Dockerfile) and proceed accordingly for its install in the box.
4. If you want to **add a tool**, please refer to the section for tools in the [Dockerfile](https://github.com/dhondta/docker-packing-box/blob/main/Dockerfile) and proceed accordingly for its install in the box.
  Please beware that:
  - Already-built packers respectively go to the following project and container folders: `files/packers` and `/opt/packers/.bin`.
  - Tools respectively go to the following project and container folders: `files/tools` and `/opt/tools`.
  - The `help` tool relies on the `__description__` dunder of the new tool ; do not forget to fill it in.

Before submitting your pull requests, please follow the steps below to explain your contribution.

1. Copy the correct template for your contribution
  - 💻 Are you improving the console ? Copy the template from <PR_TEMPLATE_CONSOLE.md>
  - 📦 Are you adding a new packer ? Copy the template from <PR_TEMPLATE_PACKER.md>
  - 🛠️ Are you adding a new tool ? Copy the template from <PR_TEMPLATE_TOOL.md>
2. Replace this text with the contents of the template
3. Fill in all sections of the template
4. Click "Create pull request"

## Issues

We use GitHub issues to track public bugs. Please ensure your description is clear and has sufficient instructions to be able to reproduce the issue.
