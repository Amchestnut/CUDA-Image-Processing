import cv2
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import math


def grayscale_image_CPU(image: np.ndarray) -> np.ndarray:
    """
    Processes an input image by splitting it into R, G, and B channels, applying the formula
    0.299*R + 0.587*G + 0.114*B to each channel, and merging them back into one image array.

    Args:
        image (np.ndarray): Input image as a NumPy array of shape (height, width, 3).

    Returns:
        np.ndarray: Processed image array of the same shape as the input.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Input image must be a 3D NumPy array with 3 channels (RGB).")

    # Split into R, G, B channels
    R, G, B = image[:, :, 0], image[:, :, 1], image[:, :, 2]

    # Apply the formula to each channel
    processed_channel = 0.299 * R + 0.587 * G + 0.114 * B
    # Stack the processed channel into a 3-channel image
    processed_image = np.stack([processed_channel] * 3, axis=-1).astype(np.float32)

    return processed_image


# Generally, I think we dont need SHARED MEMORY here because threads do not share or reuse pixel data
def grayscale_image_GPU_CUDA(image: np.ndarray) -> np.ndarray:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Image must be 3D np array, with 3 channels (RGB)")

    height, width, _ = image.shape
    image_flat = image.astype(np.uint8).flatten()
    output_flat = np.zeros((height * width), dtype=np.uint8)

    image_gpu = cuda.mem_alloc(image_flat.nbytes)
    cuda.memcpy_htod(image_gpu, image_flat)

    output_gpu = cuda.mem_alloc(output_flat.nbytes)

    mod = SourceModule("""
        __global__ void grayscale_operation(unsigned char *image, unsigned char *output, int width, int height){
            int index = threadIdx.x + blockIdx.x * blockDim.x;            // whatever i like, lets flatten this time, we will use 2d matrix next time for some other operation
            int total_pixels = width * height;
            
            if (index < total_pixels){
                int pixel_3_next_values = index * 3;
                float R = image[pixel_3_next_values];
                float G = image[pixel_3_next_values + 1];
                float B = image[pixel_3_next_values + 2];
                
                output[index] = (unsigned char)(0.299f * R + 0.587f * G + 0.114f * B);
            }
        }
    """)

    block_size = 1024
    total_pixels = width * height
    function = mod.get_function("grayscale_operation")

    function(
        image_gpu, output_gpu,
        np.int32(width), np.int32(height),
        block=(block_size, 1, 1),
        grid=(math.ceil(total_pixels / block_size), 1, 1)
    )

    cuda.memcpy_dtoh(output_flat, output_gpu)

    image_gpu.free()
    output_gpu.free()

    return output_flat.reshape(height, width)


if __name__ == '__main__':
    image_path = '../ImageExamples/meda.jpg'
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not read the input image.")
    else:
        image_CPU = grayscale_image_CPU(image)
        cv2.imwrite('../ImageOutputs/grayscale_output_CPU.png', image_CPU)

        image_GPU = grayscale_image_GPU_CUDA(image)
        cv2.imwrite('../ImageOutputs/grayscale_output_GPU_cuda.png', image_GPU)

    print("Grayscale applied to image!")
