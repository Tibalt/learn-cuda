all:
	 nvcc main.cu -DSTREAM -DNULLSTREAM -o depth_cu
pin:
	 nvcc main.cu -DPINNED -DSTREAM -DNULLSTREAM  -o pinned_depth_cu

stream:
	 nvcc main.cu -DPINNED -DSTREAM -o stream_depth_cu
null:
	 nvcc main.cu -DPINNED -DNULLSTREAM -o null_depth_cu

pf: all pin stream null
	sudo /usr/local/cuda/bin/nvprof ./depth_cu
	sudo /usr/local/cuda/bin/nvprof ./pinned_depth_cu
	sudo /usr/local/cuda/bin/nvprof ./stream_depth_cu
	sudo /usr/local/cuda/bin/nvprof ./null_depth_cu
