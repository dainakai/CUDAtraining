nvcc -O3 12far.cu -lcufft -o 12far
nvcc -O3 13far.cu -lcufft -o 13far
nvcc -O3 14far.cu -lcufft -o 14far
nvcc -O3 15far.cu -lcufft -o 15far
nvcc -O3 12close.cu -lcufft -o 12close
nvcc -O3 13close.cu -lcufft -o 13close
nvcc -O3 14close.cu -lcufft -o 14close
nvcc -O3 15close.cu -lcufft -o 15close
nvcc -O3 t12far.cu -lcufft -o t12far
nvcc -O3 t13far.cu -lcufft -o t13far
nvcc -O3 t14far.cu -lcufft -o t14far
nvcc -O3 t15far.cu -lcufft -o t15far
nvcc -O3 t12close.cu -lcufft -o t12close
nvcc -O3 t13close.cu -lcufft -o t13close
nvcc -O3 t14close.cu -lcufft -o t14close
nvcc -O3 t15close.cu -lcufft -o t15close
./12far
./13far
./14far
./15far
./12close
./13close
./14close
./15close
./t12far
./t13far
./t14far
./t15far
./t12close
./t13close
./t14close
./t15close
