#! /usr/bin/env python
#
# -*- coding: utf-8 -*-
#
#  Purpose:
#    Converting the progenitor file format (Heager2005, WH07, Sukhbold2015, and MESA)
#    to the format that GAMER can use for CCSN test problems
#

import argparse
import sys
import numpy as np


def convert_WHS(file):
    # read file info
    f_info = file.readline()

    # read the header
    line_header = file.readline()
    headers = [word.strip() for word in line_header.strip().split("  ") if word]

    # find and read targeting variables (radius, density, temperature, radial velocity,
    #                                    electron fraction, pressure, and angular velocity)
    nvars = 7
    keys = ["cell outer radius", "cell density", "cell temperature", "cell outer velocity",
            "cell Y_e", "cell pressure", "cell angular velocity"]
    TargetCols = [headers.index(key) for key in keys]

    # GAMER-formatted array
    GAMER_DATA_FORM = np.empty((0, nvars), dtype=np.float)
    variables       = np.empty(nvars,      dtype=np.float)

    outer_radius = 0.0
    outer_vel    = 0.0
    for line in f:
        if 'wind:' in line:
            break

        line_split = line.split()
        for idx, icol in enumerate(TargetCols):
            variables[idx] = float(line_split[icol])
            # convert cell outer variables to cell centered values (radius, velocity)
            if idx == 0:
                variables[idx], outer_radius = 0.5 * (outer_radius + variables[idx]), variables[idx]
            if idx == 3:
                variables[idx], outer_vel = 0.5 * (outer_vel + variables[idx]), variables[idx]

        GAMER_DATA_FORM = np.vstack([GAMER_DATA_FORM, variables])

    # add a safer line for the innermost radius
    GAMER_DATA_FORM = np.vstack([GAMER_DATA_FORM[0], GAMER_DATA_FORM])
    GAMER_DATA_FORM[0, 0] = 1.0

    return f_info, GAMER_DATA_FORM


def convert_MESA(file):
    # physical constants
    R_sol = 6.957e10 # solar radius
    M2CM  = 1e2      # meter to centimeter

    # read the header
    f_info = ''
    for i in range(5):
        f.readline()
    headers = f.readline().split()

    # find and read targeting variables (radius, density, temperature, radial velocity,
    #                                    electron fraction, pressure, and angular velocity)
    nvars = 7
    TargetCols = np.empty(nvars, dtype=np.int)
    keys = ["logR", "density", "temperature", "velocity",
            "ye", "pressure", "omega"]
    TargetCols = [headers.index(key) for key in keys]

    # GAMER-formatted array
    GAMER_DATA_FORM = np.empty((0, nvars), dtype=np.float)
    variables       = np.empty(nvars,      dtype=np.float)

    for line in f:
        line_split = line.split()
        for idx, icol in enumerate(TargetCols):
            variables[idx] = float(line_split[icol])
            # unit conversion (radius, velocity)
            if idx == 0:
                variables[idx] = R_sol * np.power(10, variables[idx])
            if idx == 3:
                variables[idx] = M2CM * variables[idx]

        GAMER_DATA_FORM = np.vstack([GAMER_DATA_FORM, variables])

    GAMER_DATA_FORM = np.flipud(GAMER_DATA_FORM)

    # add a safer line for the innermost radius
    GAMER_DATA_FORM = np.vstack([GAMER_DATA_FORM[0], GAMER_DATA_FORM])
    GAMER_DATA_FORM[0, 0] = 1.0

    return f_info, GAMER_DATA_FORM


if __name__ == "__main__":
    # progenitor model format
    WHS  = '1'
    MESA = '2'

    # load the command-line parameters
    parser = argparse.ArgumentParser(description='Convert the progenitor model to GAMER-supported format\n')

    parser.add_argument('-f', action='store', required=True, type=str, dest='format',
                        help='original progenitor file format [1: Heager2005, WH07, Sukhbold2015 2: MESA] [%(default)s]')
    parser.add_argument('-i', action='store', required=True, type=str, dest='input_file',
                        help='input progenitor file path [%(default)s]')

    args=parser.parse_args()

    # take note
    print('\nCommand-line arguments:')
    print('-------------------------------------------------------------------')
    for t in range(len(sys.argv)):
        print(sys.argv[t])
    print('')
    print('-------------------------------------------------------------------\n')

    form = args.format
    fin  = args.input_file

    # check the validity of the format
    assert form in [WHS, MESA], 'unsupported format "{0}"'.format(form)

    # open file
    with open(fin, 'r') as f:

        # read and store the f_info, data
        if   form == WHS:
            f_info, GAMER_DATA_FORM = convert_WHS (f)
        elif form == MESA:
            f_info, GAMER_DATA_FORM = convert_MESA(f)

    # write GAMER-formatted header
    fmt = "{:>21s}" + "{:>25s}" * 6
    header_GAMER  = fmt.format("radius", "density",  "temperature", "radial velocity", "ye",  "pressure", "omega"  ) + "\n"
    header_GAMER += fmt.format("[cm]",   "[g/cm^3]", "[K]",         "[cm/s]",          "[1]", "[bar]",    "[rad/s]")
    header_GAMER = f_info + header_GAMER

    # write data to the text file
    fout = fin + '_GAMER'
    np.savetxt(fout, GAMER_DATA_FORM, fmt='%23.16e', header=header_GAMER, delimiter='  ')

    print('Progenitor model is converted to the GAMER-formatted output file\n"{file}"'.format(file=fout))
