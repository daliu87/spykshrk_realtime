#!/bin/bash

python setup.py install

echo ''

time mpiexec -np $1 -bind-to core python -m spykshrk.realtime.simulator --config $2
