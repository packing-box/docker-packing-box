# +--------------------------------------------------------------------------------------------------------------------+
# |                                         CREATE THE BOX BASED ON UBUNTU                                             |
# +--------------------------------------------------------------------------------------------------------------------+
FROM ubuntu:20.04 AS base
MAINTAINER Alexandre DHondt <alexandre.dhondt@gmail.com>
LABEL version="1.1.0"
LABEL source="https://github.com/dhondta/packing-box"
ENV DEBIAN_FRONTEND noninteractive
ENV TERM xterm-256color
# add a test user (for apps that require a non-privileged user)
RUN useradd test -p test
# apply upgrade
RUN (apt -qq update \
 && apt -qq -y upgrade \
 && apt -qq -y autoremove \
 && apt -qq autoclean) 2>&1 > /dev/null \
 || echo -e "\033[1;31m SYSTEM UPGRADE FAILED \033[0m"
# install common dependencies, libraries and tools
RUN (apt -qq -y install apt-transport-https apt-utils locales \
 && apt -qq -y install bash-completion build-essential clang cmake software-properties-common \
 && apt -qq -y install libavcodec-dev libavformat-dev libavresample-dev libavutil-dev libbsd-dev libboost-regex-dev \
                       libboost-program-options-dev libboost-system-dev libboost-filesystem-dev libc6-dev-i386 \
                       libcairo2-dev libdbus-1-dev libegl1-mesa-dev libelf-dev libffi-dev libfontconfig1-dev \
                       libfreetype6-dev libfuse-dev libgif-dev libgirepository1.0-dev libgl1-mesa-dev libglib2.0-dev \
                       libglu1-mesa-dev libjpeg-dev libpulse-dev libssl-dev libsvm-java libtiff5-dev libudev-dev \
                       libxcursor-dev libxkbfile-dev libxml2-dev libxrandr-dev  \
 && apt -qq -y install colordiff colortail cython3 dosbox git golang less ltrace tree strace sudo tmate tmux vim xterm \
 && apt -qq -y install iproute2 nodejs npm python3-setuptools python3-pip swig visidata weka x11-apps yarnpkg zstd \
 && apt -qq -y install curl ffmpeg imagemagick iptables jq psmisc tesseract-ocr unrar unzip wget xdotool xvfb \
 && apt -qq -y install binwalk foremost \
 && wget -qO /tmp/bat.deb https://github.com/sharkdp/bat/releases/download/v0.18.2/bat-musl_0.18.2_amd64.deb \
 && dpkg -i /tmp/bat.deb && rm -f /tmp/bat.deb) 2>&1 > /dev/null \
 || echo -e "\033[1;31m DEPENDENCIES INSTALL FAILED \033[0m"
# configure the locale
RUN sed -i '/en_US.UTF-8/s/^# //g' /etc/locale.gen && locale-gen
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8
# configure iptables
#RUN addgroup no-internet \
# && iptables -I OUTPUT 1 -m owner --gid-owner no-internet -j DROP \
# && ip6tables -I OUTPUT 1 -m owner --gid-owner no-internet -j DROP
# install wine (for running Windows software on Linux)
RUN (dpkg --add-architecture i386 \
 && wget -nc https://dl.winehq.org/wine-builds/winehq.key \
 && apt-key add winehq.key \
 && add-apt-repository 'deb https://dl.winehq.org/wine-builds/ubuntu/ focal main' \
 && apt -qq update \
 && apt -qq -y install --install-recommends winehq-stable wine32 winetricks \
 && wineboot) 2>&1 > /dev/null \
 || echo -e "\033[1;31m WINE INSTALL FAILED \033[0m"
# install dosemu (for emulating DOS programs on Linux)
RUN apt-key adv --recv-keys --keyserver keyserver.ubuntu.com 6D9CD73B401A130336ED0A56EBE1B5DED2AD45D6 \
 && add-apt-repository ppa:dosemu2/ppa -y \
 && apt -qq update \
 && apt -qq -y install dosemu2
# install mono (for running .NET apps on Linux)
RUN (apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys 3FA7E0328081BFF6A14DA29AA6A19B38D3D831EF \
 && apt-add-repository 'deb https://download.mono-project.com/repo/ubuntu stable-focal main' \
 && apt -qq update \
 && apt -qq -y install mono-complete mono-vbnc) 2>&1 > /dev/null \
 || echo -e "\033[1;31m MONO INSTALL FAILED \033[0m"
# install .NET core
RUN (wget -qO /tmp/dotnet-install.sh https://dot.net/v1/dotnet-install.sh \
 && chmod +x /tmp/dotnet-install.sh \
 && /tmp/dotnet-install.sh -c Current \
 && rm -f /tmp/dotnet-install.sh \
 && chmod +x /root/.dotnet/dotnet \
 && ln -s /root/.dotnet/dotnet /usr/bin/dotnet) 2>&1 > /dev/null \
 || echo -e "\033[1;31m DOTNET INSTALL FAILED \033[0m"
# install MingW
RUN (apt -qq -y install --install-recommends clang mingw-w64 \
 && git clone https://github.com/tpoechtrager/wclang \
 && cd wclang \
 && cmake -DCMAKE_INSTALL_PREFIX=_prefix_ . \
 && make && make install \
 && mv _prefix_/bin/* /usr/local/bin/ \
 && cd /tmp && rm -rf wclang) 2>&1 > /dev/null \
 || echo -e "\033[1;31m MINGW INSTALL FAILED \033[0m"
# install darling (for running MacOS software on Linux)
#RUN (apt -qq -y install cmake clang bison flex pkg-config linux-headers-generic gcc-multilib \
# && cd /tmp/ && git clone --recursive https://github.com/darlinghq/darling.git && cd darling \
# && mkdir build && cd build && cmake .. && make && make install \
# && make lkm && make lkm_install) 2>&1 > /dev/null \
# || echo -e "\033[1;31m DARLING INSTALL FAILED \033[0m"
# install/update Python packages
RUN (pip3 install poetry sklearn tinyscript tldr thefuck \
 && pip3 install angr capstone dl8.5 pandas pefile pyelftools weka \
 && pip3 freeze - local | grep -v "^\-e" | cut -d = -f 1 | xargs -n1 pip3 install -qU) 2>&1 > /dev/null \
 || echo -e "\033[1;31m PIP PACKAGES UPDATE FAILED \033[0m"
# +--------------------------------------------------------------------------------------------------------------------+
# |                     CUSTOMIZE THE BOX (refine the terminal, add folders to PATH and some aliases)                  |
# +--------------------------------------------------------------------------------------------------------------------+
FROM base AS customized
# copy customized files
COPY src/term /tmp/term
RUN for f in `ls /tmp/term/`; do cp "/tmp/term/$f" "/root/.${f##*/}"; done \
 && rm -rf /tmp/term
# set the base files and folders for further setup
COPY src/conf/*.yml /opt/
RUN mkdir -p /mnt/share /opt/bin /opt/detectors /opt/packers /opt/tools /opt/unpackers
# +--------------------------------------------------------------------------------------------------------------------+
# |                           ADD UTILITIES (that are not packers, unpackers or home-made tools)                       |
# +--------------------------------------------------------------------------------------------------------------------+
FROM customized AS utils
# copy pre-built utils
# note: libgtk is required for bytehist, even though it can be used in no-GUI mode
COPY src/files/utils/* /opt/utils/
RUN apt -qq -y install libgtk2.0-0:i386 \
 && /opt/utils/tridupdate
# +--------------------------------------------------------------------------------------------------------------------+
# |                                                    ADD TOOLS                                                       |
# +--------------------------------------------------------------------------------------------------------------------+
FROM utils AS tools
COPY src/files/tools/* /opt/tools/
COPY src/lib /tmp/lib
RUN pip3 install /tmp/lib/ 2>&1 > /dev/null \
 && mv /opt/tools/help /opt/tools/?
# +--------------------------------------------------------------------------------------------------------------------+
# |                                                  ADD DETECTORS                                                     |
# +--------------------------------------------------------------------------------------------------------------------+
FROM tools AS detectors
COPY src/files/detectors/* /opt/bin/
RUN mv /opt/bin/userdb.txt /opt/detectors/ \
 && mv /opt/bin/*.zip /tmp/ \
 && mv /opt/bin/*.tar.xz /tmp/ \
 && /opt/tools/packing-box setup detector
# +--------------------------------------------------------------------------------------------------------------------+
# |                                                  ADD UNPACKERS                                                     |
# +--------------------------------------------------------------------------------------------------------------------+
FROM detectors AS unpackers
#COPY src/files/unpackers/* /tmp/
RUN /opt/tools/packing-box setup unpacker
# +--------------------------------------------------------------------------------------------------------------------+
# |                                                   ADD PACKERS                                                      |
# +--------------------------------------------------------------------------------------------------------------------+
FROM unpackers AS packers
COPY src/files/packers/* /tmp/
RUN /opt/tools/packing-box setup packer
# ----------------------------------------------------------------------------------------------------------------------
RUN find /opt/bin -type f -exec chmod +x {} \;
ENTRYPOINT /opt/tools/startup
WORKDIR /mnt/share
