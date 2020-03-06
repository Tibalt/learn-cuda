all:
	 nvcc main.cu -o depth_cu
pin:
	 nvcc main.cu -DPINNED -o pinnned_depth_cu

pf:
	sudo /usr/local/cuda/bin/nvprof ./depth_cu
