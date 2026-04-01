CXX = nvcc
FLAGS = -std=c++17 -I include
SRCS = src/tensor.cu src/linear.cu src/activations.cu src/network.cu src/kernels.cu src/main.cu

all:
	$(CXX) $(FLAGS) $(SRCS) -o inference

clean:
	rm -f inference
