#!/bin/bash

mpiexec -np $1 -bind-to hwthread:2 python -m spykshrk.realtime.simulator --config $2
