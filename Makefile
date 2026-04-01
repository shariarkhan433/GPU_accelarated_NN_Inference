CXX = nvcc
FLAGS = -std=c++17 -I include
SRCS = src/tensor.cu src/linear.cu src/activations.cu src/network.cu src/kernels.cu src/npy.cu src/main.cu
LIBS = -lcublas

all:
	$(CXX) $(FLAGS) $(SRCS) $(LIBS) -o inference

clean:
	rm -f inference