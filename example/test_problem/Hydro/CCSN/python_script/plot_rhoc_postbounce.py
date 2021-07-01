#! /usr/bin/env python
#
# -*- coding: utf-8 -*-
#
#  Purpose:
#    Compare the evolution of central density between FLASH and GAMER
#    for Post Bounce test problem
#

import matplotlib
matplotlib.use("Agg")

import re
import numpy as np
import matplotlib.pyplot as plt


### load data
# FLASH
fn = "../ReferenceSolution/PostBounce/flash_1d_s20GREP_LS220_none.dat"  # for data_s20GREP_LS220_15ms_none.data

# retrieve bounce time
with open(fn) as f:
    tb_flash = f.readline()

tb_flash = re.findall(r"\d\.\d+", tb_flash)
tb_flash = float(tb_flash[0])

t_flash, rhoc_flash = np.genfromtxt(fn, usecols = [0, 16], unpack = True)
t_flash -= tb_flash

# GAMER
fn = "../Record__CentralDens"
t_gamer, rhoc_gamer = np.genfromtxt(fn, usecols = [0, 2], unpack = True)
t_gamer += 0.015  # the IC is from the FLASH simulation at t_pb = 15 ms


### plot the evolution of central density
fig, ax = plt.subplots()
alpha = 0.8

ax.plot(t_gamer * 1e3, rhoc_gamer / 1e14, label = "GAMER", alpha = alpha)
ax.plot(t_flash * 1e3, rhoc_flash / 1e14, label = "FLASH", alpha = alpha)

ax.set_xlabel(r"$t_{pb}$ [ms]")
ax.set_ylabel(r"Central density [$10^{14}$ g cm$^{-3}$]")
ax.set_ylim(3.4, 3.7)
ax.legend(loc = "upper right")

fig.tight_layout()
plt.savefig("Rhoc_PostBounce.png", dpi = 200)
