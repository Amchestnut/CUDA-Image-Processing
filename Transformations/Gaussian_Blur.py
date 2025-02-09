import numpy as np
import cv2
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule


# Smaller sigma results in a SHARPER kernel with less blur
def gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    """Generate a Gaussian kernel based on the given size and sigma."""
    kernel = np.fromfunction(
        lambda x, y: (1 / (2 * np.pi * sigma ** 2)) * np.exp(
            - ((x - (size - 1) / 2) ** 2 + (y - (size - 1) / 2) ** 2) / (2 * sigma ** 2)
        ),
        (size, size)
    )
    return kernel / np.sum(kernel)  # Normalize the kernel


def gaussian_blur_CPU(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Apply Gaussian blur to the image using the given kernel."""
    # Convert to float32 for processing
    image = image.astype(np.float32)
    height, width, channels = image.shape
    kernel_size = kernel.shape[0]
    half_kernel = kernel_size // 2

    # Prepare the output image
    output_image = np.zeros_like(image)

    # Apply the filter to each pixel
    for y in range(half_kernel, height - half_kernel):
        for x in range(half_kernel, width - half_kernel):
            for c in range(channels):  # Process each channel separately
                value = 0
                for ky in range(-half_kernel, half_kernel + 1):
                    for kx in range(-half_kernel, half_kernel + 1):
                        value += image[y + ky, x + kx, c] * kernel[ky + half_kernel, kx + half_kernel]
                output_image[y, x, c] = value

    return np.clip(output_image, 0, 255).astype(np.uint8)


def gaussian_blur_GPU_cuda(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Apply Gaussian blur to the image using CUDA."""
    height, width, channels = image.shape
    kernel_size = kernel.shape[0]

    # Flatten the kernel and image to pass to the CUDA function
    kernel_flat = kernel.astype(np.float32).flatten()
    image_flat = image.astype(np.float32)

    # Allocate memory on the GPU
    image_gpu = cuda.mem_alloc(image_flat.nbytes)
    kernel_gpu = cuda.mem_alloc(kernel_flat.nbytes)
    output_gpu = cuda.mem_alloc(image_flat.nbytes)

    # Copy data to GPU
    cuda.memcpy_htod(image_gpu, image_flat)

    # PUT KERNEL IN CONSTANT MEMORY
    cuda.memcpy_htod(kernel_gpu, kernel_flat)

    # Prepare CUDA kernel code
    mod = SourceModule("""

    #define MAX_KERNEL_SIZE 32
    __constant__ float kernel[MAX_KERNEL_SIZE * MAX_KERNEL_SIZE];

    __global__ void apply_gaussian_blur(
    float *image, 
    float *kernel, 
    float *output,
    int width, 
    int height, 
    int kernel_size) {
        extern __shared__ float shared_image[]; // Dynamically allocated shared memory (from "shared" parameter)

        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        int half_kernel = kernel_size / 2;

        int thread_x = threadIdx.x;
        int thread_y = threadIdx.y;

        int shared_width = blockDim.x + 2 * half_kernel;
        int shared_height = blockDim.y + 2 * half_kernel;

        int shared_x = thread_x + half_kernel;
        int shared_y = thread_y + half_kernel;

        // Load data into shared memory with halo padding
        for (int c = 0; c < 3; c++) { 
            if (x < width && y < height) {
                for (int i = -half_kernel; i <= half_kernel; i++) {
                    for (int j = -half_kernel; j <= half_kernel; j++) {
                        int global_x = x + j;
                        int global_y = y + i;
    
                        // Handle boundary conditions with clamping
                        global_x = max(0, min(global_x, width - 1));
                        global_y = max(0, min(global_y, height - 1));
    
                        int global_idx = (global_y * width + global_x) * 3 + c;
                        int shared_idx = ((shared_y + i) * shared_width + (shared_x + j)) * 3 + c;
    
                        shared_image[shared_idx] = image[global_idx];
                    }
                }
            }
        }

        __syncthreads();

        // Perform convolution
        if (threadIdx.x < blockDim.x && threadIdx.y < blockDim.y &&
            x >= half_kernel && x < (width - half_kernel) &&
            y >= half_kernel && y < (height - half_kernel)) {

            for (int channel = 0; channel < 3; channel++) {
                float result = 0.0;
                for (int i = -half_kernel; i <= half_kernel; i++) {
                    for (int j = -half_kernel; j <= half_kernel; j++) {
                        int kernel_idx = (i + half_kernel) * kernel_size + (j + half_kernel);
                        int shared_idx = ((shared_y + i) * shared_width + (shared_x + j)) * 3 + channel;

                        result += shared_image[shared_idx] * kernel[kernel_idx];
                    }
                }
                int global_idx = (y * width + x) * 3 + channel;
                output[global_idx] = result;
            }
        }
    }
    """)

    func = mod.get_function("apply_gaussian_blur")

    block_size = (32, 32, 1)
    grid_size = (int(np.ceil(width / block_size[0])), int(np.ceil(height / block_size[1])))

    shared_mem_size = (block_size[0] + 2 * (kernel_size // 2)) * \
                      (block_size[1] + 2 * (kernel_size // 2)) * 3 * 4  # 3 channels, 4 bytes (float size)

    # Launch the kernel
    func(image_gpu,
         kernel_gpu,
         output_gpu,
         np.int32(width),
         np.int32(height),
         np.int32(kernel_size),
         shared=shared_mem_size,
         block=block_size,
         grid=grid_size)

    # Copy result back to host
    output_flat = np.empty_like(image_flat)
    cuda.memcpy_dtoh(output_flat, output_gpu)

    image_gpu.free()
    kernel_gpu.free()
    output_gpu.free()

    # Reshape the output back to the image shape and return
    return np.clip(output_flat.reshape(height, width, channels), 0, 255).astype(np.uint8)


# If my image is large, I would use (32,32,1), if i want speed i will go with (16,16,3)
if __name__ == '__main__':
    # Parameters for the Gaussian kernel
    kernel_size = 5
    sigma = 1.0
    kernel = gaussian_kernel(kernel_size, sigma)

    image_path = '../ImageExamples/dog.jpg'
    image = cv2.imread(image_path)

    # blurred_image_cpu = gaussian_blur_CPU(image, kernel)
    # cv2.imwrite('ImageOutputs/gaussian_blur_output_CPU.png', blurred_image_cpu)

    blurred_image_cuda = gaussian_blur_GPU_cuda(image, kernel)
    cv2.imwrite('../ImageOutputs/gaussian_blur_output_GPU_cuda.png', blurred_image_cuda)

    print("Gaussian blur applied to image!")
