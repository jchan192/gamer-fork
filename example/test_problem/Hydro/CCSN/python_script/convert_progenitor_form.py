#! /usr/bin/env python
#
# -*- coding: utf-8 -*-
#
#  Purpose:
#    Converting the progenitor file format (WH07, Heager2005, Sukhbold) to the
#    format that GAMER can be used for Core Collapse test problems
#

import argparse
import sys
import numpy as np


# progenitor model format
WHS  = '1'
MESA = '2'

#load the command-line parameters
parser = argparse.ArgumentParser( description='Convert the progenitor model to GAMER-supported format\n' )

parser.add_argument( '-f', action='store', required=True, type=str, dest='format',
                     help='original progenitor file format [1: WH07, Sukhbold, Heager2005 2: MESA] [%(default)s]' )
parser.add_argument( '-i', action='store', required=True, type=str, dest='input_file',
                     help='input progenitor file path [%(default)s]' )

args=parser.parse_args()

# take note
print( '\nCommand-line arguments:' )
print( '-------------------------------------------------------------------' )
for t in range( len(sys.argv) ):
   print ( str(sys.argv[t]) ),
print( '' )
print( '-------------------------------------------------------------------\n' )

form = args.format
fin  = args.input_file

# check the validity of the format
assert form == '1' or form == '2', 'unsupported format "{0}"'.format(form)


# open file
f = open( fin, 'r' )

# read file info
if   ( form == WHS  ):
    f_info = f.readline()
elif ( form == MESA ):
    f_info = ''

# read the header
headers = []
if   ( form == WHS  ):
    first_column_size = 6
    other_column_size = 25
    header  = f.readline()
    headers = []
    headers.append( header[0:first_column_size].strip() )
    leng_header = len(header)
    i_end = first_column_size
    while ( i_end < leng_header ):
        i_start = i_end
        i_end   = i_start + other_column_size
        header_append = header[i_start:i_end].strip()
        headers.append( header_append )
elif ( form == MESA ):
    for i in range(5):
       f.readline()
    headers = f.readline().split()


# find and read targeting variables (radius, density, temperature, radial velocity,
#                                    electron fraction, pressure, and angular velocity )
nvars = 7
TargetCols = np.empty( nvars, dtype=np.int )
if   ( form == WHS  ):
    for i in range( len(headers) ):
       if ( headers[i] == 'cell outer radius'     ): TargetCols[0] = i
       if ( headers[i] == 'cell density'          ): TargetCols[1] = i
       if ( headers[i] == 'cell temperature'      ): TargetCols[2] = i
       if ( headers[i] == 'cell outer velocity'   ): TargetCols[3] = i
       if ( headers[i] == 'cell Y_e'              ): TargetCols[4] = i
       if ( headers[i] == 'cell pressure'         ): TargetCols[5] = i
       if ( headers[i] == 'cell angular velocity' ): TargetCols[6] = i
elif ( form == MESA ):
   for i in range( len(headers) ):
       if ( headers[i] == 'logR'                  ): TargetCols[0] = i
       if ( headers[i] == 'density'               ): TargetCols[1] = i
       if ( headers[i] == 'temperature'           ): TargetCols[2] = i
       if ( headers[i] == 'velocity'              ): TargetCols[3] = i
       if ( headers[i] == 'ye'                    ): TargetCols[4] = i
       if ( headers[i] == 'pressure'              ): TargetCols[5] = i
       if ( headers[i] == 'omega'                 ): TargetCols[6] = i


# GAMER-formatted array
GAMER_DATA_FORM = np.empty( (0, nvars), dtype=np.float )
variables       = np.empty( nvars,      dtype=np.float )

outer_radius = 0.0
outer_vel    = 0.0
for line in f:
    if ( form == WHS  and  line[0:first_column_size].strip() == 'wind:' ): break
    for idx, icol in enumerate(TargetCols):
        variables[idx] = float( line.split()[icol] )
#       convert cell outer variables to cell centered values (radius, velocity)
        if ( form == WHS  ):
            if ( idx == 0 ):
                cell_centered_radius = ( outer_radius + variables[idx] ) / 2.0
                outer_radius         = variables[idx]
                variables[idx]       = cell_centered_radius
            if ( idx == 3 ):
                cell_centered_vel    = ( outer_vel + variables[idx] ) / 2.0 
                outer_vel            = variables[idx]
                variables[idx]       = cell_centered_vel
#       unit conversion
        if ( form == MESA ):
            if ( idx == 0 ):
                variables[idx] = 6.977e10 * np.power( 10, variables[idx] ) # unit conversion: radius
            if ( idx == 3 ):
                variables[idx] = 1e2 * variables[idx]                      # unit conversion: velocity

    GAMER_DATA_FORM = np.vstack( [GAMER_DATA_FORM, variables] )

if ( form == MESA ):
    GAMER_DATA_FORM = np.flipud( GAMER_DATA_FORM )

# add a safer line for the innermost radius
GAMER_DATA_FORM = np.vstack( [GAMER_DATA_FORM[0], GAMER_DATA_FORM] )
GAMER_DATA_FORM[0, 0] = 1.0


# close file
f.close()


# write GAMER-formatted header
header_GAMER = "radius".rjust(21) + "density".rjust(25) + "temperature".rjust(25) + "radial velocity".rjust(25) \
             + "ye".rjust(25) + "pressure".rjust(25) + "omega".rjust(25) + "\n" \
             + "[cm]".rjust(21) + "[g/cm^3]".rjust(25) + "[K]".rjust(25) + "[cm/s]".rjust(25) \
             + "[1]".rjust(25) + "[bar]".rjust(25) + "[rad/s]".rjust(25)
header_GAMER = f_info + header_GAMER

# write data to the text file
fout = fin + '_GAMER'
np.savetxt( fout, GAMER_DATA_FORM, fmt='% .16e', header=header_GAMER, delimiter='  ',
            newline='\n', footer='', comments='# ', encoding=None )

print( 'Progenitor model is converted to the GAMER-formatted output file\n"{file}"'.format(file=fout) )
