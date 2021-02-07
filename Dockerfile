FROM ubuntu:20.04 AS base
MAINTAINER Alexandre DHondt <alexandre.dhondt@gmail.com>
ENV DEBIAN_FRONTEND noninteractive
ENV TERM xterm-256color
# copy customized files
RUN mkdir -p /opt/tools /opt/packers/.bin
RUN touch /opt/packers/.aliases
ADD files/term/bash_aliases /root/.bash_aliases
ADD files/term/bash_colors /root/.bash_colors
ADD files/term/bash_gitprompt /root/.bash_gitprompt
ADD files/term/bashrc /root/.bashrc
ADD files/term/profile /root/.profile
ADD files/term/viminfo /root/.viminfo
# apply upgrade
RUN apt -qq update \
 && apt -qq -y upgrade \
 && apt -qq -y autoremove \
 && apt -qq autoclean
# install common dependencies, libraries and tools
RUN apt -qq -y install apt-transport-https apt-utils bash-completion build-essential curl software-properties-common \
 && apt -qq -y install libavcodec-dev libavformat-dev libavresample-dev libavutil-dev libbsd-dev libc6-dev-i386 \
                   libcairo2-dev libdbus-1-dev libegl1-mesa-dev libelf-dev libffi-dev libfontconfig1-dev \
                   libfreetype6-dev libfuse-dev libgif-dev libgirepository1.0-dev libgl1-mesa-dev libglib2.0-dev \
                   libglu1-mesa-dev libjpeg-dev libpulse-dev libssl-dev libtiff5-dev libudev-dev libxcursor-dev \
                   libxkbfile-dev libxml2-dev libxrandr-dev \
 && apt -qq -y install colordiff colortail dosbox git less ltrace strace sudo tmate tmux unzip vim wget yarnpkg xterm \
 && apt -qq -y install iproute2 lolcat nodejs npm python3-setuptools python3-pip unzip x11-apps xvfb wget zstd
# install wine (for running Windows software on Linux)
RUN dpkg --add-architecture i386 \
 && wget -nc https://dl.winehq.org/wine-builds/winehq.key \
 && apt-key add winehq.key \
 && add-apt-repository 'deb https://dl.winehq.org/wine-builds/ubuntu/ focal main' \
 && apt -qq update \
 && apt -qq -y install --install-recommends winehq-stable wine32
# install mono (for running .NET apps on Linux)
RUN apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys 3FA7E0328081BFF6A14DA29AA6A19B38D3D831EF \
 && apt-add-repository 'deb https://download.mono-project.com/repo/ubuntu stable-focal main' \
 && apt -qq update \
 && apt -qq -y install mono-complete
# install darling (for running MacOS software on Linux)
#RUN apt -qq -y install cmake clang bison flex pkg-config linux-headers-generic gcc-multilib \
# && cd /tmp/ && git clone --recursive https://github.com/darlinghq/darling.git && cd darling \
# && mkdir build && cd build && cmake .. && make && make install \
# && make lkm && make lkm_install
# update Python packages
RUN pip3 freeze - local | grep -v "^\-e" | cut -d = -f 1 | xargs -n1 pip3 install -U
# ----------------------------------------------------------------------------------------------------------------------
FROM base AS packers
RUN echo "" > /opt/packers/.aliases
# install ELF packers
ADD files/packers/ezuri /opt/packers/.bin/ezuri
RUN apt install -y upx
RUN (wget -q -O /tmp/m0dern_p4cker.zip https://github.com/n4sm/m0dern_p4cker/archive/master.zip \
 && unzip -qq /tmp/m0dern_p4cker.zip && rm -f /tmp/m0dern_p4cker.zip \
 && cd /tmp/m0dern_p4cker-master/packer\@git && chmod +x make.sh && ./make.sh \
 && cp main /opt/packers/.bin/ && cd /tmp && rm -rf m0dern_p4cker-master) \
 || echo -e "\033[1;31m INSTALL FAILED: m0dern_p4cker \033[0m"
# install PE packers
RUN (wget -q -O /tmp/amber.zip https://github.com/EgeBalci/amber/releases/download/v3.0/amber_linux_amd64_3.0.zip \
 && unzip -qq /tmp/amber.zip -d /opt/packers/ && rm -f /tmp/amber.zip \
 && echo "alias amber='/opt/packers/amber_linux_amd64_3.0/amber'" >> /opt/packers/.aliases) \
 || echo -e "\033[1;31m INSTALL FAILED: amber \033[0m"
RUN (wget -q -O /tmp/apack.zip https://www.ibsensoftware.com/files/apack-1.00.zip \
 && unzip -qq /tmp/apack.zip -d /opt/packers/apack && rm -f /tmp/apack.zip \
 && echo "alias apack='wine /opt/packers/apack/apackw.exe'" >> /opt/packers/.aliases) \
 || echo -e "\033[1;31m INSTALL FAILED: APack \033[0m"
RUN (wget -q -O /tmp/kkrunchy.zip http://www.farbrausch.de/~fg/kkrunchy/kkrunchy_023a2.zip \
 && unzip -qq /tmp/kkrunchy.zip -d /opt/packers/ && rm -f /tmp/kkrunchy.zip \
 && echo "alias kkrunchy='wine /opt/packers/kkrunchy_k7.exe\n" >> /opt/packers/.aliases) \
 || echo -e "\033[1;31m INSTALL FAILED: kkrunchy \033[0m"
RUN (wget -q -O /tmp/mew.zip https://ro.softpedia-secure-download.com/dl/3286613cc34b9167bf320ca0b7402156/601d21ca/100016531/software/SECURITY/CRIPTARE/mew11.zip \
 && unzip -qq /tmp/mew.zip -d /opt/packers/mew && rm -f /tmp/mew.zip \
 && echo "alias telock='wine /opt/packers/mew/mew11.exe\n" >> /opt/packers/.aliases) \
 || echo -e "\033[1;31m INSTALL FAILED: MEW \033[0m"
RUN (wget -q -O /tmp/petite.zip https://www.un4seen.com/files/petite24.zip \
 && unzip -qq /tmp/petite.zip -d /opt/packers/petite && rm -f /tmp/petite.zip \
 && echo "alias petite='wine /opt/packers/petite/petite.exe\n" >> /opt/packers/.aliases) \
 || echo -e "\033[1;31m INSTALL FAILED: PEtite \033[0m"
RUN (wget -q -O /tmp/telock.zip https://ro.softpedia-secure-download.com/dl/248b803a47ea6ac8098b879b0b9406e6/601d0d0b/100000023/software/UTILE/telock.zip \
 && unzip -qq /tmp/telock.zip -d /opt/packers/telock && rm -f /tmp/telock.zip \
 && echo "alias telock='wine /opt/packers/telock/telock.exe\n" >> /opt/packers/.aliases) \
 || echo -e "\033[1;31m INSTALL FAILED: tElock \033[0m"
#TODO: install Crinkler (http://www.crinkler.net/crinkler20.zip)
#TODO: istall FSG (http://in4k.untergrund.net/packers%20droppers%20etc/xt_fsg20.zip)
#FIXME: MEW only works as GUI
#FIXME: telock only works as GUI
# install .NET packers
RUN (wget -q -O /tmp/mpress.zip https://softpedia-secure-download.com/dl/edf6650a62b365cd2de4278f01664d0f/601d6308/100095719/software/programming/mpress.zip \
 && unzip -qq /tmp/mpress.zip -d /opt/packers/mpress && rm -f /tmp/mpress.zip \
 && echo "alias mpress='wine /opt/packers/mpress/mpress.exe\n" >> /opt/packers/.aliases) \
 || echo -e "\033[1;31m INSTALL FAILED: MPRESS \033[0m"
RUN (wget -q -O /tmp/netcrypt.zip https://github.com/friedkiwi/netcrypt/releases/download/v1.1/netcrypt.zip \
 && unzip -qq /tmp/netcrypt.zip -d /opt/packers/netcrypt && rm -f /tmp/netcrypt.zip \
 && echo "alias netcrypt='wine /opt/packers/netcrypt/SimplePacker.exe\n" >> /opt/packers/.aliases) \
 || echo -e "\033[1;31m INSTALL FAILED: NetCrypt \033[0m"
#FIXME: netcrypt only works as GUI
# install MacOS packers
#TODO: install muncho (http://www.pouet.net/prod.php?which=51324)
#TODO: install iPackk (http://www.pouet.net/prod.php?which=29185)
#TODO: install oneKpaq (http://www.pouet.net/prod.php?which=66926)
# ----------------------------------------------------------------------------------------------------------------------
FROM packers AS tools
# install & upgrade Python packages
ADD files/packers.csv /root/.packers.csv
ADD files/tools.csv /root/.tools.csv
ADD files/tools/requirements.txt /tmp/requirements.txt
ADD files/tools/dataset-maker.py /opt/tools/dataset-maker
ADD files/tools/help.py /root/help
RUN pip3 install -r /tmp/requirements.txt
