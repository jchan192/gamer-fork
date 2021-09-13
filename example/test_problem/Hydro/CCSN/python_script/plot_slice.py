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
from glob import glob
import matplotlib.pyplot as plt


### runtime parameters
path_HDF5 = ".."  # path to the HDF5 output (relative path here)

# yt setup
quantity  = "density", "ye", "Entr"  # quantity to be plot
direction = "x"                      # along the yz plane

zoom       = 10                      # zoom factor
zlim       = {"density": (1.0e8, 5.0e14),
              "ye"     : (0.25, 0.5),
              "Entr"   : (2, 9),
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

        os.system(cmd)
