# +--------------------------------------------------------------------------------------------------------------------+
# |                                         CREATE THE BOX BASED ON UBUNTU                                             |
# +--------------------------------------------------------------------------------------------------------------------+
# define global arguments
ARG USER=user
ARG HOME=/home/$USER
ARG UBIN=$HOME/.local/bin
ARG UOPT=$HOME/.opt
ARG PBOX=$UOPT/tools/packing-box
ARG FILES=src/files
# start creating the box
FROM ubuntu:22.04 AS base
MAINTAINER Alexandre DHondt <alexandre.dhondt@gmail.com>
LABEL version="2.0.0"
LABEL source="https://github.com/dhondta/packing-box"
ARG USER
ARG HOME
ARG UBIN
ENV DEBCONF_NOWARNINGS yes \
    DEBIAN_FRONTEND noninteractive \
    TERM xterm-256color \
    PIP_ROOT_USER_ACTION ignore
# configure locale
RUN (apt-get -qq update \
 && apt-get -qq -y install locales \
 && locale-gen en_US.UTF-8) >/dev/null 2>&1 \
 || echo -e "\033[1;31m LOCALE CONFIGURATION FAILED \033[0m"
# apply upgrade
RUN (echo "debconf debconf/frontend select Noninteractive" | debconf-set-selections \
 && apt-get -qq -y install dialog apt-utils \
 && apt-get -qq -y upgrade \
 && apt-get -qq -y autoremove \
 && apt-get -qq autoclean) >/dev/null 2>&1 \
 || echo -e "\033[1;31m SYSTEM UPGRADE FAILED \033[0m"
# add a non-privileged account
RUN (groupadd --gid 1000 $USER \
 && useradd --uid 1000 --gid 1000 -m $USER \
 && usermod -aG $USER $USER \
 && apt-get -qq update \
 && apt-get -qq install -y sudo \
 && echo $USER ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USER \
 && chmod 0440 /etc/sudoers.d/$USER) >/dev/null 2>&1 \
 || echo -e "\033[1;31m USER CREATION FAILED \033[0m"
# install common dependencies and libraries
RUN (apt-get -qq -y install apt-transport-https apt-utils \
 && apt-get -qq -y install add-apt-key bash-completion build-essential clang cmake software-properties-common \
 && apt-get -qq -y install libavcodec-dev libavformat-dev libavutil-dev libbsd-dev libboost-regex-dev libcapstone-dev \
                       libgirepository1.0-dev libelf-dev libffi-dev libfontconfig1-dev libgif-dev libjpeg-dev \
 && apt-get -qq -y install libboost-program-options-dev libboost-system-dev libboost-filesystem-dev libc6-dev-i386 \
                       libcairo2-dev libdbus-1-dev libegl1-mesa-dev libfreetype6-dev libfuse-dev libgl1-mesa-dev \
                       libglib2.0-dev libglu1-mesa-dev libpulse-dev libssl-dev libsvm-dev libsvm-java libtiff5-dev \
                       libudev-dev libxcursor-dev libxkbfile-dev libxml2-dev libxrandr-dev) >/dev/null 2>&1 \
# && apt-get -qq -y install libgtk2.0-0:i386 \
 || echo -e "\033[1;31m DEPENDENCIES INSTALL FAILED \033[0m"
# install useful tools
RUN (apt-get -qq -y install colordiff colortail cython3 dosbox git golang kmod less ltrace tree strace sudo tmate tmux \
 && apt-get -qq -y install iproute2 nftables nodejs npm python3-setuptools python3-pip swig vim weka x11-apps yarnpkg \
 && apt-get -qq -y install bc curl ffmpeg imagemagick pev psmisc tesseract-ocr unrar unzip wget zstd \
 && apt-get -qq -y install binwalk ent foremost jq visidata xdotool xterm xvfb \
 && wget -qO /tmp/bat.deb https://github.com/sharkdp/bat/releases/download/v0.18.2/bat-musl_0.18.2_amd64.deb \
 && dpkg -i /tmp/bat.deb \
 && rm -f /tmp/bat.deb) >/dev/null 2>&1 \
 || echo -e "\033[1;31m TOOLS INSTALL FAILED \033[0m"
# install wine (for running Windows software on Linux)
RUN (dpkg --add-architecture i386 \
 && wget -O /etc/apt/keyrings/winehq-archive.key https://dl.winehq.org/wine-builds/winehq.key \
 && wget -NP /etc/apt/sources.list.d/ https://dl.winehq.org/wine-builds/ubuntu/dists/jammy/winehq-jammy.sources \
 && apt-get -qq update \
 && apt-get -qq -y install --install-recommends winehq-stable wine32 winetricks \
 && WINEPREFIX="$HOME/.wine32" WINEARCH=win32 wineboot \
 &  wget https://dl.winehq.org/wine/wine-gecko/2.47.3/wine-gecko-2.47.3-x86.msi \
 && WINEPREFIX="$HOME/.wine32" WINEARCH=win32 wine msiexec /i wine-gecko-2.47.3-x86.msi) >/dev/null 2>&1 \
 || echo -e "\033[1;31m WINE INSTALL FAILED \033[0m"
# install mono (for running .NET apps on Linux)
RUN (add-apt-key --keyserver keyserver.ubuntu.com 3FA7E0328081BFF6A14DA29AA6A19B38D3D831EF \
 && apt-key export D3D831EF | gpg --dearmour -o /usr/share/keyrings/mono.gpg \
 && apt-key del D3D831EF \
 && echo "deb [arch=amd64 signed-by=/usr/share/keyrings/mono.gpg] https://download.mono-project.com/repo/ubuntu " \
         "stable-focal main" | tee /etc/apt/sources.list.d/mono.list \
 && apt-get -qq update \
 && apt-get -qq -y install mono-complete mono-vbnc) >/dev/null 2>&1 \
 || echo -e "\033[1;31m MONO INSTALL FAILED \033[0m"
# install MingW
RUN (apt-get -qq -y install --install-recommends clang mingw-w64 \
 && git clone https://github.com/tpoechtrager/wclang \
 && cd wclang \
 && cmake -DCMAKE_INSTALL_PREFIX=_prefix_ . \
 && make \
 && make install \
 && mv _prefix_/bin/* /usr/local/bin/ \
 && cd /tmp \
 && rm -rf wclang) >/dev/null 2>&1 \
 || echo -e "\033[1;31m MINGW INSTALL FAILED \033[0m"
# install darling (for running MacOS software on Linux)
#RUN (apt-get -qq -y install cmake clang bison flex pkg-config linux-headers-generic gcc-multilib \
# && cd /tmp/ \
# && git clone --recursive https://github.com/darlinghq/darling.git \
# && cd darling \
# && mkdir build \
# && cd build \
# && cmake .. \
# && make \
# && make install \
# && make lkm \
# && make lkm_install) >/dev/null 2>&1 \
# || echo -e "\033[1;31m DARLING INSTALL FAILED \033[0m"
# install .NET core
USER $USER
RUN (wget -qO /tmp/dotnet-install.sh https://dot.net/v1/dotnet-install.sh \
 && chmod +x /tmp/dotnet-install.sh \
 && /tmp/dotnet-install.sh -c Current \
 && rm -f /tmp/dotnet-install.sh \
 && chmod +x $HOME/.dotnet/dotnet \
 && mkdir -p $UBIN \
 && ln -s $HOME/.dotnet/dotnet $UBIN/dotnet) >/dev/null 2>&1 \
 || echo -e "\033[1;31m DOTNET INSTALL FAILED \033[0m"
# +--------------------------------------------------------------------------------------------------------------------+
# |                                     CUSTOMIZE THE BOX (refine the terminal)                                        |
# +--------------------------------------------------------------------------------------------------------------------+
FROM base AS customized
ARG USER
ARG UOPT
ENV TERM xterm-256color
# copy customized files for root
USER root
COPY src/term/[^profile]* /tmp/term/
RUN for f in `ls /tmp/term/`; do cp "/tmp/term/$f" "/root/.${f##*/}"; done \
 && rm -rf /tmp/term
# switch to the unprivileged account
USER $USER
# copy customized files
COPY --chown=$USER:$USER src/term /tmp/term
RUN for f in `ls /tmp/term/`; do cp "/tmp/term/$f" "/home/$USER/.${f##*/}"; done \
 && rm -rf /tmp/term
# install/update Python packages (install dl8.5 with root separately to avoid wheel's build failure)
RUN pip3 -qq install --user --no-warn-script-location --ignore-installed meson poetry scikit-learn tinyscript tldr \
 && pip3 -qq install --user --no-warn-script-location angr capstone dl8.5 pandas pefile pyelftools thefuck weka \
 || echo -e "\033[1;31m PIP PACKAGES UPDATE FAILED \033[0m"
# +--------------------------------------------------------------------------------------------------------------------+
# |                                              ADD FRAMEWORK ITEMS                                                   |
# +--------------------------------------------------------------------------------------------------------------------+
FROM customized AS framework
ARG USER
ARG HOME
ARG UOPT
ARG FILES
ARG PBOX
USER $USER
ENV TERM xterm-256color
RUN (go mod init pbox \
 & WINEPREFIX="$HOME/.wine32" WINEARCH=win32 wineboot \
 & WINEPREFIX="$HOME/.wine64" WINEARCH=win64 wineboot) >/dev/null 2>&1
# set the base files and folders for further setup (explicitly create ~/.cache/pip to avoid it not being owned by user)
COPY --chown=$USER:$USER src/conf/*.yml $UOPT/
RUN sudo mkdir -p /mnt/share \
 && sudo chown $USER:$USER /mnt/share \
 && mkdir -p $UOPT/bin $UOPT/tools $UOPT/analyzers $UOPT/detectors $UOPT/packers $UOPT/unpackers \
             /tmp/analyzers /tmp/detectors /tmp/packers /tmp/unpackers
# copy pre-built utils and tools
# note: libgtk is required for bytehist, even though it can be used in no-GUI mode
COPY --chown=$USER:$USER $FILES/utils/* $UOPT/utils/
RUN sudo $UOPT/utils/tridupdate
COPY --chown=$USER:$USER $FILES/tools/* $UOPT/tools/
RUN mv $UOPT/tools/help $UOPT/tools/?
# copy and install pbox (main library for tools) and pboxtools (lightweight library for items)
COPY --chown=$USER:$USER src/lib /tmp/lib
RUN pip3 -qq install --user --no-warn-script-location /tmp/lib/ \
 && rm -rf /tmp/lib
# install unpackers
COPY --chown=$USER:$USER $FILES/analyzers/* /tmp/analyzers/
RUN find /tmp/analyzers -type f -executable -exec mv {} $UOPT/bin/ \; \
 && $PBOX setup analyzer
# install detectors (including wrapper scripts)
COPY --chown=$USER:$USER $FILES/detectors/* /tmp/detectors/
RUN find /tmp/detectors -type f -executable -exec mv {} $UOPT/bin/ \; \
 && find /tmp/detectors -type f -iname '*.txt' -exec mv {} $UOPT/bin/ \; \
 && $PBOX setup detector
# install packers
COPY --chown=$USER:$USER $FILES/packers/* /tmp/packers/
RUN $PBOX setup packer
# install unpackers
#COPY ${FILES}/unpackers/* /tmp/unpackers/  # leave this commented as long as $FILES/unpackers has no file
RUN $PBOX setup unpacker
# ----------------------------------------------------------------------------------------------------------------------
RUN find $UOPT/bin -type f -exec chmod +x {} \;
ENV UOPT=$UOPT
ENTRYPOINT $UOPT/tools/startup
WORKDIR /mnt/share
