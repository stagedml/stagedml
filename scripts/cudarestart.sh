#!/bin/sh
set -e -x
rmmod nvidia_uvm
modprobe nvidia_uvm
