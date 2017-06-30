#!/bin/bash

python setup.py install

time mpiexec -np $1 -bind-to core python -m spykshrk.realtime.simulator --config $2
