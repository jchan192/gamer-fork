#! /usr/bin/env python
#
# -*- coding: utf-8 -*-
#
#  Purpose:
#    Compute and plot the averaged profile from GAMER HDF5 data
#    for comparison with FLASH (if FLASH data are given)
#
#    This script will plot 7 quantities in both linear and log scale:
#      Density, |Vr|, Ye, Temperature, Pressure, Entropy, Potential
#


import matplotlib
matplotlib.use("Agg")

import os
import re
import yt
import numpy as np
from matplotlib import pyplot as plt


### runtime parameter
# GAMER
path_gamer = ".."                 # path to the GAMER data (relative path here)
profile_rad_gamer = (3200, "km")  # maximum radius for GAMER profile (in km)
idx_gamer = 0,                    # output index for the target time
dt_pb     = 15                    # relative time to core bounce (in ms)

# FLASH
include_flash = False                                 # flag to include FLASH data
path_flash = "/data/ceag/gamer/210409_s20gamer/non"   # path to the FLASH data
idx_flash  = 435,                                     # output index crossponding to idx_gamer
tpb_flash  = 0.4097362362170411                       # physical time of core bounce in FLASH data

# matplotlib
plt.rcParams["font.size"] = 18


### retrieve the unit system in GAMER (assumed using length, mass, and time)
reg_pattern = r"\s*([-+]?\d+\.?\d*[eE]?[-+]?\d*)"

parfile = os.path.join(path_gamer, "Input__Parameter")
par     = open(parfile).read()

UNIT_L  = re.findall(r"UNIT_L" + reg_pattern, par)
UNIT_M  = re.findall(r"UNIT_M" + reg_pattern, par)
UNIT_T  = re.findall(r"UNIT_T" + reg_pattern, par)

UNIT_L  = float(UNIT_L[0])
UNIT_M  = float(UNIT_M[0])
UNIT_T  = float(UNIT_T[0])
const_G = 6.67428e-8

# compute the unit for potential and pressure
UNIT_Pot  = UNIT_L**2 / UNIT_T**2
UNIT_Pres = (UNIT_M / UNIT_L**3) * (UNIT_L / UNIT_T)**2


### derived field for GAMER data
def _ye(field, data):
    return data["Ye"] / data["Dens"]


### main routine
for idx, idx_out in enumerate(idx_gamer):
    ## load data
    # GAMER
    quantity_gamer = ["density", "radial_velocity", "ye", "Temp", "Pres", "Entr", "Pote"]

    fn = "Data_{:06d}".format(idx_out)
    fn = os.path.join(path_gamer, fn)

    ds_gamer = yt.load(fn)
    ds_gamer.add_field(("gas", "ye"), function = _ye, units = "code_length**3/code_mass")

    time = ds_gamer.parameters["Time"][0] + dt_pb
    print("Load GAMER data at t_pb = {:.4f} from file {}".format(time, fn))

    # compute averaged profile
    sp = ds_gamer.sphere("c", profile_rad_gamer)
    profile_gamer = yt.create_profile(sp, "radius", quantity_gamer,
                                      units = {"radius": "km"},
                                      weight_field = ("gas", "cell_mass") )

    cond_gamer = profile_gamer["density"] != 0.0  # for removing empty bins

    # FLASH
    if include_flash:
        quantity_flash = "density", "radial_velocity", "ye  ", "temp", "pres", "entr", "gpot"

        fn = "ccsn1d_hdf5_chk_{:04d}".format(idx_flash[idx])
        fn = os.path.join(path_flash, fn)

        ds_flash = yt.load(fn)

        # relative to core bounce
        print("load FLASH output at t_pb = {:.4f} from file {}".format(ds_flash.parameters["time"] - tpb_flash, fn))


    ## plot
    fig, axes  = plt.subplots(figsize = (20, 16), ncols = 3, nrows = 3)
    axes_ravel = axes.ravel()

    # GAMER
    color = "r"
    alpha = 0.5
    lw = 4
    ls = "solid"

    rad_km  = profile_gamer.x.value[cond_gamer]

    for q, ax in zip(quantity_gamer, axes_ravel):
        # prepare the required profile
        prof = profile_gamer[q]

        if q == "radial_velocity": prof  = np.abs(prof)
        if q == "Pres"           : prof *= UNIT_Pres
        if q == "Pote"           : prof *= UNIT_Pot

        # remove empty bins
        prof = prof[cond_gamer]

        # combine the radius and profile for sorting
        foo = np.vstack( [rad_km, prof] ).T
        foo = foo[ foo[:, 0].argsort() ]

        ax.semilogx(foo.T[0], foo.T[1], label = "GAMER", alpha = alpha, lw = lw, ls = ls, c = color)

    # FLASH
    if include_flash:
        color = "b"
        alpha = 0.5
        lw = 4
        ls = "solid"

        rad_km = ds_flash.r["r"] / 1e5

        for q, ax in zip(quantity_flash, axes_ravel):
            if q == "radial_velocity":
                prof = np.sqrt(ds_flash.r["velx"]**2 + ds_flash.r["vely"]**2 + ds_flash.r["velz"]**2)
            else:
                prof = ds_flash.r[q]

            ax.semilogx(rad_km, prof, label = "FLASH", alpha = alpha, lw = lw, ls = ls, c = color)


    ## decoration
    field_labels = "Density", "|Vr|", "Ye", "Temperature", "Pressure", "Entropy", "Potential"

    for ax, ylabel in zip(axes_ravel, field_labels):
        ax.set_xlabel("Radius (km)")
        ax.set_ylabel(ylabel)

    # set the yscale of |Vr| to symlog
    axes_ravel[1].set_yscale("symlog")

    # remove uncessary axes
    fig.delaxes(axes_ravel[-2])
    fig.delaxes(axes_ravel[-1])

    # add legend
    axes_ravel[-3].legend(loc = "lower right")

    ## savefig
    # linear scale
    fig.tight_layout()

    fig.subplots_adjust(top = 0.94)
    fig.suptitle(r"$t_\mathrm{pb} = " + "{:.1f}$ [ms]".format(time))

    fnout = "Data_{:06d}_Profile_linear.png".format(idx_out)
    fnout = os.path.join(path_gamer, fnout)
    plt.savefig(fnout, dpi = 200)

    # switch to log scale for density, temperature, and presssure
    axes_ravel[0].set_yscale("log")
    axes_ravel[3].set_yscale("log")
    axes_ravel[4].set_yscale("log")

    fnout = "Data_{:06d}_Profile_log.png".format(idx_out)
    fnout = os.path.join(path_gamer, fnout)
    plt.savefig(fnout, dpi = 200)

    plt.close(fig)
