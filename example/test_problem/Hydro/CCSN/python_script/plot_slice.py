#! /usr/bin/env python
#
# -*- coding: utf-8 -*-
#
#  Purpose:
#    Use yt to plot the slice of specified quantities with/without grid
#    and combined the figures into animations
#


import matplotlib
matplotlib.use("Agg")

import os
import yt
import re
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


### runtime parameters
path_HDF5   = ".."  # path to the HDF5 output (relative path here)
path_RefSol = "../ReferenceSolution/PostBounce"
RefSol      = {"s20GREP_LS220_15ms_none": "flash_1d_s20GREP_LS220_none.dat",
               "s20GREP_LS220_15ms_LB1" : "flash_1d_s20GREP_LS220_15ms_LB1.dat",
               "s20GREP_LS220_15ms_LB2" : "flash_1d_s20GREP_LS220_15ms_LB2.dat",
               "s20GREP_LS220_15ms_LB3" : "flash_1d_s20GREP_LS220_15ms_LB3.dat" }

# yt setup
quantity  = "density", "ye", "Entr"  # quantity to be plot
direction = "x"                      # along the yz plane
plot_rsh  = False                    # overplot the average shock radius (for Entropy only)

zoom       = 10                      # zoom factor
zlim       = {"density": (1.0e8, 5.0e14),
              "ye"     : (0.25, 0.5),
              "Entr"   : (2, 10),
              "pressure": (None, None),
              "magnetic_field_strength": (None, None),
              "plasma_beta": (None, None)              }  # range in colorbar for each quantity
zscale_log = {"density": True,
              "ye"     : False,
              "Entr"   : False,
              "pressure": True,
              "magnetic_field_strength": True,
              "plasma_beta": True              }  # flag for zscale
                                                  # True: log scale, False: linear scale

fmt_fnout_fig = "{}_SlicePlot_" + direction + "_{}"  # format of name of output files

# ffmpeg setup
flag_gene_animation = True                     # use ffmpeg to generate animation automatically
fmt_fnout_animation = "SlicePlot_{}_both.mp4"  # format of name of output animation
ffmpeg_opt   = " -vcodec h264 -pix_fmt yuv420p -filter_complex hstack "
ffmpeg_frame = 8


### retrieve all GAMER HDF5 output
fn_hdf5 = glob( os.path.join(path_HDF5, "Data_" + "[0-9]" * 6) )
fn_hdf5.sort()


### obtain averaged shock radius in reference solution
if plot_rsh:
    # retrieve the filename of input profile in Input__TestProb
    par_testprob   = open("../Input__TestProb").read()
    CCSN_Prof_File = re.findall(r"CCSN_Prof_File\s*(\S*)\s", par_testprob)
    CCSN_Prof_File = os.path.basename(CCSN_Prof_File[0])

    fn_refsol = RefSol[CCSN_Prof_File]
    fn_refsol = os.path.join(path_RefSol, fn_refsol)

    # retrieve bounce time
    with open(fn_refsol) as f:
        tb_refsol = f.readline()

    tb_refsol = re.findall(r"\d\.\d+", tb_refsol)
    tb_refsol = float(tb_refsol[0])

    # load data and prepare function for interpolation
    t_refsol, rsh_refsol = np.genfromtxt(fn_refsol, usecols = [0, 11], unpack = True)

    t_refsol = (t_refsol - tb_refsol) * 1e3  # to ms
    interp_func = interp1d(t_refsol, rsh_refsol)


### plot here
for fn in fn_hdf5:
    # load file
    ds = yt.load(fn)

    # add "ye" derived field if necessary
    if "ye" in quantity:
        def _ye(field, data):
            return data["Ye"] / data["Dens"]

        ds.add_field(("gas", "ye"), function = _ye, units = "code_length**3/code_mass")

    # retrieve the simulation time at lv = 0
    time = ds.parameters["Time"][0]
    time_str = "Time = {:6.2f} [ms]".format(time)

    # plot here
    for q in quantity:
        # slice plot without grid
        slc = yt.SlicePlot(ds, direction, q, center = "c")

        slc.annotate_title(time_str)
        slc.set_font_size(25)
        slc.zoom(zoom)
        slc.set_zlim(q, *zlim[q])

        if zscale_log[q] is False:
            slc.set_log(q, False)

        # add shock radius
        if plot_rsh and q == "Entr":
            # Note that PostBounce simluations begins at t_pb = 15 ms
            rsh = interp_func(time + 15)

            slc.annotate_sphere([0.5, 0.5],
                                radius = (rsh, "cm"),
                                coord_system = "axis",
                                circle_args = {"color": "k",
                                               "linewidth": 2,
                                               "linestyle": "dashed"})

        fn_out = fmt_fnout_fig.format(fn, q) + ".png"
        slc.save(fn_out)

        # slice plot with grid
        try:
            slc.annotate_grids()
        except:
            # if annotate_grids() fails, replot the figure again
            slc = yt.SlicePlot(ds, direction, q, center = "c")

            slc.annotate_grids()
            slc.annotate_title(time_str)
            slc.set_font_size(25)
            slc.zoom(zoom)
            slc.set_zlim(q, *zlim[q])

            if zscale_log[q] is False:
                slc.set_log(q, False)

        fn_out = fmt_fnout_fig.format(fn, q) + "_grid.png"
        slc.save(fn_out)

        # free memory
        del slc

    # free memory
    del ds


### generate animation
# combine slice plots with and without grid into one
if flag_gene_animation:
    for q in quantity:
        fn_wogrid = fmt_fnout_fig.format("Data_%6d", q) + ".png"
        fn_wgrid  = fmt_fnout_fig.format("Data_%6d", q) + "_grid.png"
        fn_ani    = fmt_fnout_animation.format(q)

        fn_wogrid = os.path.join(path_HDF5, fn_wogrid)
        fn_wgrid  = os.path.join(path_HDF5, fn_wgrid)
        fn_ani    = os.path.join(path_HDF5, fn_ani)

        cmd = "ffmpeg -r {} -i {} -r {} -i {} ".format(ffmpeg_frame, fn_wogrid, ffmpeg_frame, fn_wgrid) \
            + ffmpeg_opt + fn_ani

        # print command in case it fails
        print(cmd)
        os.system(cmd)
