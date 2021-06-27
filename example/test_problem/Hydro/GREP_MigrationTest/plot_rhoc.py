#! /usr/bin/env python
#
# -*- coding: utf-8 -*-
#
#  Purpose:
#    Compare the evolution of central density between FLASH and GAMER
#

import matplotlib
matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt


### load data
t_gamer, rhoc_gamer = np.genfromtxt("Record__CentralDens", usecols = [0, 2], unpack = 1)
t_flash, rhoc_flash = np.genfromtxt("ref_flash_1d.dat", unpack = 1)

### plot the evolution of central density
fig, ax = plt.subplots()
alpha = 0.8

ax.plot(t_gamer * 1e3, rhoc_gamer / 1e14, c = "k", label = "GAMER", alpha = alpha)
ax.plot(t_flash * 1e3, rhoc_flash / 1e14, c = "r", label = "FLASH", alpha = alpha)

ax.set_xlabel("Time (ms)")
ax.set_ylabel(r"Central density ($10^{14}$ g cm$^{-3}$)")

fig.tight_layout()
plt.savefig("Rhoc.png")
