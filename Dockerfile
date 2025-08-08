# +--------------------------------------------------------------------------------------------------------------------+
# |                                         CREATE THE BOX BASED ON UBUNTU                                             |
# +--------------------------------------------------------------------------------------------------------------------+
# define global arguments
ARG USER=user
ARG HOME=/home/$USER
ARG UOPT=$HOME/.opt
ARG UBIN=$HOME/.local/bin \
    PBWS=$HOME/.packing-box \
    PBOX=$UOPT/tools/packing-box \
    FILES=src/files
# start creating the box
FROM ubuntu:24.04 AS base
LABEL org.opencontainers.image.authors="alexandre.dhondt@gmail.com"  \
      org.opencontainers.image.created="Feb 5, 2021" \
      org.opencontainers.image.licenses="GPL-3.0" \
      org.opencontainers.image.source="https://github.com/orgs/packing-box/repositories" \
      org.opencontainers.image.title="Packing-Box: Experimental toolkit for static detection of executable packing" \
      org.opencontainers.image.url="https://github.com/packing-box/docker-packing-box" \
      org.opencontainers.image.version="2.0.1"
ARG USER HOME UBIN
ENV DEBCONF_NOWARNINGS=yes \
    DEBIAN_FRONTEND=noninteractive \
    TERM=xterm-256color \
    PIP_ROOT_USER_ACTION=ignore
# configure locale
RUN apt-get update \
 && apt-get -y install locales \
 && locale-gen en_US.UTF-8
# apply upgrade
RUN echo "debconf debconf/frontend select Noninteractive" | debconf-set-selections \
 && apt-get -y install dialog apt-utils \
 && apt-get update \
 && apt-get -y upgrade \
 && apt-get -y autoremove \
 && apt-get autoclean
# add a non-privileged account
RUN usermod -l $USER ubuntu \
 && groupmod -n $USER ubuntu \
 && usermod -d /home/$USER -m $USER \
 && apt-get install -y sudo \
 && echo $USER ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USER \
 && chmod 0440 /etc/sudoers.d/$USER
# install common dependencies and libraries
RUN apt-get -y install apt-transport-https apt-utils \
 && apt-get -y install bash-completion build-essential clang cmake software-properties-common \
 && apt-get -y install libavcodec-dev libavformat-dev libavutil-dev libbsd-dev libboost-regex-dev libcapstone-dev \
                       libgirepository1.0-dev libelf-dev libffi-dev libfontconfig1-dev libgif-dev libjpeg-dev \
 && apt-get -y install libboost-program-options-dev libboost-system-dev libboost-filesystem-dev libc6-dev-i386 \
                       libdwarf-dev libcairo2-dev libdbus-1-dev libegl1-mesa-dev libfreetype6-dev libfuse-dev \
                       libgl1-mesa-dev libglib2.0-dev libglu1-mesa-dev libpulse-dev libssl-dev libsvm-dev libsvm-java \
                       libtiff5-dev libudev-dev libxcursor-dev libxkbfile-dev libxml2-dev libxrandr-dev libfuzzy-dev
# install useful tools
RUN apt-get update \
 && apt-get -y install colordiff colortail cython3 dos2unix dosbox git golang kmod less ltrace meson nasm tree strace \
 && apt-get -y install gcab genisoimage iproute2 jq nftables nodejs npm rubygems ssdeep swig vim visidata yarnpkg \
 && apt-get -y install python3-pip python3-pygraphviz python3-setuptools \
 && apt-get -y install bc curl ffmpeg imagemagick pev psmisc tesseract-ocr unrar unzip wget wimtools x11-apps zstd \
 && apt-get -y install bats binutils-dev binwalk dwarfdump ent foremost tmate tmux weka xdotool xterm xvfb \
 && wget -qO /tmp/b.deb https://github.com/sharkdp/bat/releases/download/v0.25.0/bat_0.25.0_amd64.deb \
 && dpkg -i /tmp/b.deb \
 && rm -f /tmp/b.deb
# install .NET runtime (necessary for ilspycmd)
RUN apt-get -y install dotnet-sdk-8.0
# install wine (for running Windows software on Linux)
RUN dpkg --add-architecture i386 \
 && wget -O /etc/apt/keyrings/winehq-archive.key https://dl.winehq.org/wine-builds/winehq.key \
 && wget -NP /etc/apt/sources.list.d/ https://dl.winehq.org/wine-builds/ubuntu/dists/noble/winehq-noble.sources \
 && apt-get update \
 && apt-get -y install --install-recommends winehq-stable wine32 winetricks \
 && mkdir /opt/wine-stable/share/wine/gecko \
 && wget -O /opt/wine-stable/share/wine/gecko/wine-gecko-2.47.1-x86.msi \
         https://dl.winehq.org/wine/wine-gecko/2.47.1/wine-gecko-2.47.1-x86.msi \
 && wget -O /opt/wine-stable/share/wine/gecko/wine-gecko-2.47.1-x86_64.msi \
         https://dl.winehq.org/wine/wine-gecko/2.47.1/wine-gecko-2.47.1-x86_64.msi
# install mono (for running .NET apps on Linux)
RUN apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys 3FA7E0328081BFF6A14DA29AA6A19B38D3D831EF \
 && apt-key export D3D831EF | gpg --dearmour -o /usr/share/keyrings/mono.gpg \
 && apt-key del D3D831EF \
 && echo "deb [arch=amd64 signed-by=/usr/share/keyrings/mono.gpg] https://download.mono-project.com/repo/ubuntu " \
         "stable-focal main" | tee /etc/apt/sources.list.d/mono.list \
 && apt-get update \
 && apt-get -y install mono-complete mono-vbnc
# install MingW
RUN apt-get -y install --install-recommends clang mingw-w64 \
 && git clone https://github.com/tpoechtrager/wclang \
 && cd wclang \
 && cmake -DCMAKE_INSTALL_PREFIX=_prefix_ . \
 && make \
 && make install \
 && mv _prefix_/bin/* /usr/local/bin/ \
 && cd /tmp \
 && rm -rf wclang
# install darling (for running MacOS software on Linux)
#RUN apt-get -y install cmake clang bison flex pkg-config linux-headers-generic gcc-multilib \
# && cd /tmp/ \
# && git clone --recursive https://github.com/darlinghq/darling.git \
# && cd darling \
# && mkdir build \
# && cd build \
# && cmake .. \
# && make \
# && make install \
# && make lkm \
# && make lkm_install
# install .NET core
USER $USER
RUN wget -qO /tmp/dotnet-install.sh https://dot.net/v1/dotnet-install.sh \
 && chmod +x /tmp/dotnet-install.sh \
 && /tmp/dotnet-install.sh -c Current \
 && rm -f /tmp/dotnet-install.sh \
 && chmod +x $HOME/.dotnet/dotnet \
 && mkdir -p $UBIN \
 && ln -s $HOME/.dotnet/dotnet $UBIN/dotnet
# install/update Python packages (install dl8.5 with root separately to avoid wheel's build failure)
RUN python3 -m pip install --user --upgrade --break-system-packages pip
RUN pip3 install --user --no-warn-script-location --ignore-installed --break-system-packages \
        capstone jinja2 meson poetry pythonnet thefuck tinyscript tldr vt-py \
 && pip3 install --user --no-warn-script-location --ignore-installed --break-system-packages \
        angr capa lightgbm pandas scikit-learn scikit-learn-extra weka \
 && rm -f /home/user/.local/lib/python3.*/site-packages/unicorn/lib \
 && pip3 uninstall -y --break-system-packages unicorn \
 && pip3 install --user --no-warn-script-location --ignore-installed --break-system-packages unicorn
#  FAILING PACKAGE: pydl8.5
# install ILSpyCmd
RUN dotnet tool install --global ilspycmd
# install Rust (user-level
RUN curl https://sh.rustup.rs -sSf | bash -s -- -y
# initialize Go
RUN go mod init pbox &
# install user-level tools
RUN go install github.com/antonmedv/fx@latest
# +--------------------------------------------------------------------------------------------------------------------+
# |                                     CUSTOMIZE THE BOX (refine the terminal)                                        |
# +--------------------------------------------------------------------------------------------------------------------+
FROM base AS customized
ARG USER UOPT
ENV TERM=xterm-256color
# copy customized files for root
USER root
COPY src/term/[^profile]* /tmp/term/
RUN for f in `ls /tmp/term/`; do cp -r "/tmp/term/$f" "/root/.${f##*/}"; done \
 && rm -rf /tmp/term
# switch to the unprivileged account
USER $USER
# copy customized files
COPY --chown=$USER src/term /tmp/term
RUN for f in `ls /tmp/term/`; do cp "/tmp/term/$f" "/home/$USER/.${f##*/}"; done \
 && rm -rf /tmp/term
# +--------------------------------------------------------------------------------------------------------------------+
# |                                              ADD FRAMEWORK ITEMS                                                   |
# +--------------------------------------------------------------------------------------------------------------------+
FROM customized AS framework
ARG USER HOME UOPT PBWS PBOX FILES
USER $USER
ENV TERM=xterm-256color
# set the base files and folders for further setup (explicitly create ~/.cache/pip to avoid it not being owned by user)
COPY --chown=$USER src/conf/*.yml $PBWS/conf/
RUN sudo mkdir -p /mnt/share \
 && sudo chown $USER /mnt/share \
 && mkdir -p $UOPT/bin $UOPT/tools $UOPT/analyzers $UOPT/detectors $UOPT/packers $UOPT/unpackers \
             /tmp/analyzers /tmp/detectors /tmp/packers /tmp/unpackers
# copy executable format related data
COPY --chown=$USER src/data $PBWS/data
# copy and install pbox (main library for tools) and pboxtools (lightweight library for items)
COPY --chown=$USER src/lib /tmp/lib
RUN pip3 install --user --no-warn-script-location --break-system-packages /tmp/lib/ \
 && rm -rf /tmp/lib
COPY --chown=$USER $FILES/tools/packing-box $PBOX
# install analyzers
COPY --chown=$USER $FILES/analyzers/* /tmp/analyzers/
RUN find /tmp/analyzers -type f -executable -exec mv {} $UOPT/bin/ \; \
 && $PBOX setup analyzer
# install detectors (including wrapper scripts)
COPY --chown=$USER $FILES/detectors/* /tmp/detectors/
RUN find /tmp/detectors -type f -executable -exec mv {} $UOPT/bin/ \; \
 && find /tmp/detectors -type f -iname '*.txt' -exec mv {} $UOPT/detectors/ \; \
 && $PBOX setup detector
# install packers
COPY --chown=$USER $FILES/packers/* /tmp/packers/
RUN $PBOX setup packer
# install unpackers
#COPY ${FILES}/unpackers/* /tmp/unpackers/  # leave this commented as long as $FILES/unpackers has no file
RUN $PBOX setup unpacker
# copy pre-built utils and tools
# note: libgtk is required for bytehist, even though it can be used in no-GUI mode
COPY --chown=$USER $FILES/utils/* $UOPT/utils/
COPY --chown=$USER $FILES/tools/* $UOPT/tools/
RUN mv $UOPT/tools/help $UOPT/tools/?
RUN wget https://github.com/packing-box/packer-masking-tool/raw/main/notpacked%2b%2b -O $UOPT/utils/notpacked++ \
 && chmod +x $UOPT/utils/notpacked++
# generate Bash completions
COPY --chown=$USER $FILES/utils/_pbox-compgen $UOPT/utils/
COPY --chown=$USER $FILES/utils/pbox-completions.json $UOPT/utils/
RUN $UOPT/utils/_pbox-compgen $UOPT/utils/pbox-completions.json -f $HOME/.bash_completion
# ----------------------------------------------------------------------------------------------------------------------
RUN find $UOPT/bin -type f -exec chmod +x {} \;
ENV UOPT=$UOPT
ENTRYPOINT $UOPT/tools/startup
WORKDIR /mnt/share
