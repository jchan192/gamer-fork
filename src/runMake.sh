#!/bin/bash                                                                                                                                                                                                                                              
set -e 
make -j 
cp gamer ../../gamer-cudft/bin/Gravity
cd ../../gamer-cudft/bin/Gravity
./gamer > log  
cd ../../../gamer-fork/src    

