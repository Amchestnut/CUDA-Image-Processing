from Transformations.Brightness import adjust_brightness_gpu
from Transformations.Gaussian_Blur import gaussian_kernel, gaussian_blur_GPU_cuda
import cv2

from Transformations.Grayscale import grayscale_image_CPU, grayscale_image_GPU_CUDA


def apply_gaussian_blur(image_path: str) -> None:
    """Apply Gaussian blur to an image."""
    kernel_size = 5
    sigma = 1.0
    kernel = gaussian_kernel(kernel_size, sigma)
    blurred_image = gaussian_blur_GPU_cuda(cv2.imread(image_path), kernel)
    cv2.imwrite('ImageOutputs/gaussian_blur_output_GPU_cuda.png', blurred_image)

def apply_grayscale(image_path: str) -> None:
    """Apply grayscale to an image."""
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not read the input image.")
    else:
        image_CPU = grayscale_image_CPU(image)
        cv2.imwrite('ImageOutputs/grayscale_output_CPU.png', image_CPU)

        image_GPU = grayscale_image_GPU_CUDA(image)
        cv2.imwrite('ImageOutputs/grayscale_output_GPU_cuda.png', image_GPU)

def adjust_brightness(image_path: str) -> None:
    """Adjust brightness of an image."""
    image = cv2.imread(image_path)
    factor = 1.5
    adjusted_image = adjust_brightness_gpu(image, factor)
    cv2.imwrite('ImageOutputs/brightness_GPU.png', adjusted_image)


if __name__ == '__main__':
    while True:
        print("Choose a test:")
        print("1. Gaussian blur")
        print("2. Grayscale")
        print("3. Brightness adjustment")
        choice = int(input("Enter your choice: "))
        if choice == 1:
            apply_gaussian_blur('ImageExamples/dog.jpg')
        elif choice == 2:
            apply_grayscale('ImageExamples/meda.jpg')
        else:
            adjust_brightness('ImageExamples/dog.jpg')
