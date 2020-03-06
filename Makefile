all:
	 nvcc main.cu -o depth_cu

pf:
	sudo /usr/local/cuda/bin/nvprof ./depth_cu
