#!/bin/sh

apt-get update --fix-missing && \
ln -fs /usr/share/zoneinfo/Europe/Moscow /etc/localtime && \
apt-get install -y tzdata && \
dpkg-reconfigure --frontend noninteractive tzdata && \
apt-get install -y sudo wget \
                   libqt5x11extras5 \
                   net-tools netcat && \
apt-get install -y locales && \
apt-get install -y ondir less figlet psmisc && \
apt-get install -y atool && \
apt-get install -y rsync && \
apt-get install -y graphviz-dev && \
locale-gen "en_US.UTF-8"

