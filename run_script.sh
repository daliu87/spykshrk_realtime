#!/bin/bash

python setup.py install

echo ''

time mpiexec -np $1 -bind-to core python -m spykshrk.realtime.simulator --config $2

time python -m spykshrk.realtime.postprocessing.rec_merge_hdf5 --config $2
