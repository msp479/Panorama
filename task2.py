#!/usr/bin/env python
# coding: utf-8

# In[2]:


"""
 Grayscale Image Processing
(Due date: Nov. 26, 11:59 P.M., 2021)

The goal of this task is to experiment with two commonly used 
image processing techniques: image denoising and edge detection. 
Specifically, you are given a grayscale image with salt-and-pepper noise, 
which is named 'task2.png' for your code testing. 
Note that different image might be used when grading your code. 

You are required to write programs to: 
(i) denoise the image using 3x3 median filter;
(ii) detect edges in the denoised image along both x and y directions using Sobel operators (provided in line 30-32).
(iii) design two 3x3 kernels and detect edges in the denoised image along both 45° and 135° diagonal directions.
Hint: 
• Zero-padding is needed before filtering or convolution. 
• Normalization is needed before saving edge images. You can normalize image using the following equation:
    normalized_img = 255 * frac{img - min(img)}{max(img) - min(img)}

Do NOT modify the code provided to you.
You are NOT allowed to use OpenCV library except the functions we already been imported from cv2. 
You are allowed to use Numpy for basic matrix calculations EXCEPT any function/operation related to convolution or correlation. 
You should NOT use any other libraries, which provide APIs for convolution/correlation ormedian filtering. 
Please write the convolution code ON YOUR OWN. 
"""

from cv2 import imread, imwrite, imshow, IMREAD_GRAYSCALE, namedWindow, waitKey, destroyAllWindows
import numpy as np


# Sobel operators are given here, do NOT modify them.
sobel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]]).astype(int)
sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).astype(int)


def filter(img):
    """
    :param img: numpy.ndarray(int), image
    :return denoise_img: numpy.ndarray(int), image, same size as the input image

    Apply 3x3 Median Filter and reduce salt-and-pepper noises in the input noise image
    """
    
    # TO DO: implement your solution here    
    
    r,c=img.shape
    pad_img=np.pad(img, ((1,1),(1,1)), 'constant')
    
    
    for i in range(1, r-1):
        for j in range(1,c-1):
            k=[pad_img[i-1][j-1], pad_img[i-1][j], pad_img[i-1][j+1], pad_img[i][j-1], pad_img[i][j], pad_img[i][j+1], pad_img[i+1][j-1], pad_img[i+1][j], pad_img[i+1][j+1]]
            m=np.median(k)
            pad_img[i][j]=m

    
    pad_img=np.delete(pad_img, 0, 0)
    r-=1
    pad_img=np.delete(pad_img, r-1, 0)
    r-=1
    
    pad_img=np.delete(pad_img, 0, 1)
    c-=1
    pad_img=np.delete(pad_img, c-1, 1)
    c-=1
    
 
    denoise_img=pad_img
    return denoise_img

    raise NotImplementedError
    return denoise_img


def convolve2d(img, kernel):
    """
    :param img: numpy.ndarray, image
    :param kernel: numpy.ndarray, kernel
    :return conv_img: numpy.ndarray, image, same size as the input image

    Convolves a given image (or matrix) and a given kernel.
    """
    
    # TO DO: implement your solution here
    
    flipped_kernel= np.flip(kernel)

    pad_img=np.pad(img, ((1,1),(1,1)), 'constant')
    r,c=pad_img.shape
    
    t = np.zeros((img.shape[0], img.shape[1]))
    
    for i in range(1, r-1):
        for j in range(1,c-1):
            k=[pad_img[i-1][j-1], pad_img[i-1][j], pad_img[i-1][j+1], pad_img[i][j-1], pad_img[i][j], pad_img[i][j+1], pad_img[i+1][j-1], pad_img[i+1][j], pad_img[i+1][j+1]]
            k = np.reshape(k, (3,3))
            fk = flipped_kernel * k
            t[i-1][j-1]=np.sum(fk)

    
    pad_img=np.delete(pad_img, 0, 0)
    r-=1
    pad_img=np.delete(pad_img, r-1, 0)
    r-=1
    
    pad_img=np.delete(pad_img, 0, 1)
    c-=1
    pad_img=np.delete(pad_img, c-1, 1)
    c-=1
    
    conv_img=t
    return conv_img

    raise NotImplementedError
    return conv_img


def edge_detect(img):
    """
    :param img: numpy.ndarray(int), image
    :return edge_x: numpy.ndarray(int), image, same size as the input image, edges along x direction
    :return edge_y: numpy.ndarray(int), image, same size as the input image, edges along y direction
    :return edge_mag: numpy.ndarray(int), image, same size as the input image, 
                      magnitude of edges by combining edges along two orthogonal directions.

    Detect edges using Sobel kernel along x and y directions.
    Please use the Sobel operators provided in line 30-32.
    Calculate magnitude of edges by combining edges along two orthogonal directions.
    All returned images should be normalized to [0, 255].
    """
    
    # TO DO: implement your solution here
    
    conv_sobel_x = convolve2d(img, sobel_x)
    conv_sobel_y = convolve2d(img, sobel_y)
    conv_final = np.sqrt((conv_sobel_x ** 2) + (conv_sobel_y ** 2))



    img_norm_x = 255*((conv_sobel_x - np.min(conv_sobel_x))/(np.max(conv_sobel_x) - np.min(conv_sobel_x)))
    img_norm_y = 255*((conv_sobel_y - np.min(conv_sobel_y))/(np.max(conv_sobel_y) - np.min(conv_sobel_y)))
    img_norm = 255*((conv_final - np.min(conv_final))/(np.max(conv_final) - np.min(conv_final)))

    edge_x = img_norm_x.astype(np.uint8)
    edge_y = img_norm_y.astype(np.uint8)
    edge_mag = img_norm.astype(np.uint8)

    return edge_x, edge_y, edge_mag 
    
    
    raise NotImplementedError
    return edge_x, edge_y, edge_mag


def edge_diag(img):
    """
    :param img: numpy.ndarray(int), image
    :return edge_45: numpy.ndarray(int), image, same size as the input image, edges along x direction
    :return edge_135: numpy.ndarray(int), image, same size as the input image, edges along y direction

    Design two 3x3 kernels to detect the diagonal edges of input image. Please print out the kernels you designed.
    Detect diagonal edges along 45° and 135° diagonal directions using the kernels you designed.
    All returned images should be normalized to [0, 255].
    """

    # TO DO: implement your solution here
    
    kernel_45 = np.array([[0, -1, -2], [1, 0, -1], [2, 1, 0]]).astype(int)
    kernel_135 = np.array([[-2, -1, 0], [-1, 0, 1], [0, 1, 2]]).astype(int)
    
    print("Kernel to detect edges along 45 degrees direction: \n", kernel_45)
    print("\nKernel to detect edges along 135 degrees direction:\n", kernel_135)

    conv_sobel_45 = convolve2d(img, kernel_45)
    conv_sobel_135 = convolve2d(img, kernel_135)

    img_norm_45 = 255 * ((conv_sobel_45 - np.min(conv_sobel_45)) / (np.max(conv_sobel_45) - np.min(conv_sobel_45)))
    img_norm_135 = 255 * ((conv_sobel_135 - np.min(conv_sobel_135)) / (np.max(conv_sobel_135) - np.min(conv_sobel_135)))

    edge_45 = img_norm_45.astype(np.uint8)
    edge_135 = img_norm_135.astype(np.uint8)

    return edge_45, edge_135

    raise NotImplementedError
    print() # print the two kernels you designed here
    return edge_45, edge_135


if __name__ == "__main__":
    noise_img = imread('task2.png', IMREAD_GRAYSCALE)
    denoise_img = filter(noise_img)
    imwrite('results/task2_denoise.jpg', denoise_img)
    edge_x_img, edge_y_img, edge_mag_img = edge_detect(denoise_img)
    imwrite('results/task2_edge_x.jpg', edge_x_img)
    imwrite('results/task2_edge_y.jpg', edge_y_img)
    imwrite('results/task2_edge_mag.jpg', edge_mag_img)
    edge_45_img, edge_135_img = edge_diag(denoise_img)
    imwrite('results/task2_edge_diag1.jpg', edge_45_img)
    imwrite('results/task2_edge_diag2.jpg', edge_135_img)






# In[ ]:




