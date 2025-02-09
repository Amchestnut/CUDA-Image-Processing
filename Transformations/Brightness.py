import numpy as np
import cv2
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule


def adjust_brightness_gpu(image: np.ndarray, factor: float) -> np.ndarray:
    """Adjust the brightness of an image using CUDA."""
    height, width, channels = image.shape
    image_flat = image.astype(np.float32).flatten()

    # Allocate memory on the GPU
    image_gpu = cuda.mem_alloc(image_flat.nbytes)
    output_gpu = cuda.mem_alloc(image_flat.nbytes)
    sum_gpu = cuda.mem_alloc(np.float32(0).nbytes)

    # Copy data to GPU
    cuda.memcpy_htod(image_gpu, image_flat)

    # CUDA kernel code
    mod = SourceModule("""
    __global__ void compute_sum(const float *image, float *sum, int width, int height, int channels) {
        __shared__ float block_sum[256];
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int stride = blockDim.x * gridDim.x;
        int total_pixels = width * height * channels;

        float local_sum = 0;
        for (int i = idx; i < total_pixels; i += stride) {
            local_sum += image[i];
        }

        block_sum[threadIdx.x] = local_sum;
        __syncthreads();

        // Reduction within the block
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (threadIdx.x < s) {
                block_sum[threadIdx.x] += block_sum[threadIdx.x + s];
            }
            __syncthreads();
        }

        // Write block result to global sum
        if (threadIdx.x == 0) {
            atomicAdd(sum, block_sum[0]);
        }
    }

    __global__ void adjust_brightness(float *image, float *output, float avg_intensity, float factor, int width, int height, int channels) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int total_pixels = width * height * channels;

        if (idx < total_pixels) {
            float pixel_value = image[idx];
            float adjusted_value = avg_intensity + (pixel_value - avg_intensity) * factor;
            output[idx] = fmaxf(0.0f, fminf(255.0f, adjusted_value)); // Clamp to 0-255
        }
    }
    """)

    # Compile functions
    compute_sum = mod.get_function("compute_sum")
    adjust_brightness = mod.get_function("adjust_brightness")

    # Block and grid sizes
    block_size = 256
    grid_size = (height * width * channels + block_size - 1) // block_size

    # Calculate sum
    cuda.memcpy_htod(sum_gpu, np.float32(0))
    compute_sum(
        image_gpu,
        sum_gpu,
        np.int32(width),
        np.int32(height),
        np.int32(channels),
        block=(block_size, 1, 1),
        grid=(grid_size, 1, 1))

    # Copy sum back to host and calculate average
    sum_host = np.zeros(1, dtype=np.float32)
    cuda.memcpy_dtoh(sum_host, sum_gpu)
    avg_intensity = sum_host[0] / (width * height * channels)

    # Adjust brightness
    adjust_brightness(
        image_gpu,
        output_gpu,
        np.float32(avg_intensity),
        np.float32(factor),
        np.int32(width),
        np.int32(height),
        np.int32(channels),
        block=(block_size, 1, 1),
        grid=(grid_size, 1, 1))

    # Copy result back to host
    output_flat = np.empty_like(image_flat)
    cuda.memcpy_dtoh(output_flat, output_gpu)

    # Free GPU memory
    image_gpu.free()
    output_gpu.free()
    sum_gpu.free()

    # Reshape and return the adjusted image
    return output_flat.reshape(height, width, channels).astype(np.uint8)


if __name__ == '__main__':
    # Load an image
    image_path = '../ImageExamples/dog.jpg'
    image = cv2.imread(image_path)

    # Adjust brightness
    factor = 1.5  # Brightness scaling factor
    adjusted_image = adjust_brightness_gpu(image, factor)

    # Save the result
    cv2.imwrite('../ImageOutputs/brightness_GPU.png', adjusted_image)
    print("Brightness adjustment applied to image!")
