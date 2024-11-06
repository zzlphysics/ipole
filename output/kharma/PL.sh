#! /bin/bash

FILE_DIR="/home/zzl/kx-4t/ipole/output/kharma/20241009_cuda_sane_a09375_288-128-128"

for FILE in $FILE_DIR/*.h5; do
  echo $FILE
  python /home/zzl/kx-4t/ipole/output/kharma/plot_pol_gauss.py $FILE
done

FPS=10
echo "Encoding $FILE_DIR"
# Encode frames to video
ffmpeg -hide_banner -loglevel error -y -r ${FPS} -f image2 -pattern_type glob -i "$FILE_DIR/*.png" -vcodec libx264 -crf 22 -pix_fmt yuv420p "$FILE_DIR.mp4"
