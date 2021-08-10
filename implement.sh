nvcc -O3 close_datasetsgen_itr.cu -lcufft -o close_datasetsgen_itr
nvcc -O3 far_datasetsgen_itr.cu -lcufft -o far_datasetsgen_itr
./close_datasetsgen_itr
./far_datasetsgen_itr