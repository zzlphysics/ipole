#! /bin/bash

export OMP_NUM_THREADS=12

FILE_LIST=(
  "/home/zzl/kx-4t/kharma/output/20241009_cuda_sane_a09375_288-128-128/torus.out0.*0.phdf"
)

TRAT_LARGE_LIST=(
  1
  10
  20
  160
)

for FILE in ${FILE_LIST[@]}; do
  FILE_NAME=$(basename $FILE)
  echo $FILE_NAME
  /home/zzl/kx-4t/ipole/ipole -par /home/zzl/kx-4t/ipole/output/kharma/example.par --M_unit=8e27  --thetacam=163 --rotcam=180 --nx=160 --ny=160 --trat_large=20 --dump=$FILE --outfile=/home/zzl/kx-4t/ipole/output/kharma/20241009_cuda_sane_a09375_288-128-128/${FILE_NAME%.phdf}.h5
done
