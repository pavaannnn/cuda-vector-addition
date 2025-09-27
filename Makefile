# CUDA Vector Addition Makefile

NVCC = nvcc
CFLAGS = -O2
TARGET = vector_add
SOURCE = basic.cu

all: $(TARGET)

$(TARGET): $(SOURCE)
	$(NVCC) $(CFLAGS) -o $(TARGET) $(SOURCE)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)

.PHONY: all run clean
