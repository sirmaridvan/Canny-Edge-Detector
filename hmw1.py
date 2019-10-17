import os
import glob
import cv2
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import scipy

# Rıdvan SIRMA
# 504181566

def read_all_image_files():
    current_working_directory = os.path.dirname(os.path.realpath(__file__))
    image_folder_name = "/images"
    images_directory = current_working_directory+image_folder_name

    all_image_files = glob.glob(images_directory + "/*")
    return all_image_files

def generate_output_image_file_path(image_file):
    seperator = '/'
    words = image_file.split(seperator)
    words[ len(words) - 1 ] = "edge_detected_" + words[ len(words) - 1 ]
    return seperator.join(words)

def get_grayscale_image(image_file):
    image = cv2.imread(image_file)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def convolve(image, kernel):
    kernel = np.flipud(np.fliplr(kernel))
    r,c = image.shape
    output = np.zeros((r-2,c-2))

    for x in range(1, r-1):
        for y in range(1, c-1):
            output[x-1,y-1]=np.sum(np.multiply(image[x-1:x+2,y-1:y+2], kernel))

    return output

def blur_image(image):
    gauss_blur_kernel = np.array([np.array([1/16, 1/8, 1/16]),
                                  np.array([1/8, 1/4, 1/8]),
                                  np.array([1/16, 1/8, 1/16])], np.float32)
    return convolve(image,gauss_blur_kernel)
def sobel_filter(image, horizontal = 0):
    h_kernel = np.ones((3,3))
    if horizontal == 1:
        h_kernel = np.array([np.array([1, 2, 1]),
                                      np.array([0, 0, 0]),
                                      np.array([-1, -2, -1])], np.float32)
    elif horizontal == 0:
        h_kernel = np.array([np.array([-1, 0, 1]),
                                      np.array([-2, 0, 2]),
                                      np.array([-1, 0, 1])], np.float32)
    return convolve(image,h_kernel)

def get_gradients(image):
    Gx = sobel_filter(image, 1)
    Gy = sobel_filter(image, 0)
    G = np.sqrt(np.square(Gx)+np.square(Gy))
    return G / G.max() * 255

def get_orientation(image):
    Gx = sobel_filter(image, 1)
    Gy = sobel_filter(image, 0)
    orientation = scipy.arctan2(Gy, Gx)
    orientation = np.rad2deg(orientation)
    return orientation

def non_max_suppression(gradient, orientation):
    r,c = orientation.shape
    suppressed_image = np.zeros((r,c))
    for x in range(1, r-1):
        for y in range(1, c-1):
            if (orientation[x][y] < 22.5 and orientation[x][y] >= 0) or \
                    (orientation[x][y] >= 157.5 and orientation[x][y] < 202.5) or \
                    (orientation[x][y] >= 337.5 and orientation[x][y] <= 360):
                if (gradient[x][y] >= gradient[x][y + 1]) or \
                        (gradient[x][y] >= gradient[x][y - 1]):
                    suppressed_image[x][y] = gradient[x,y]
            elif (orientation[x][y] >= 22.5 and orientation[x][y] < 67.5) or \
                    (orientation[x][y] >= 202.5 and orientation[x][y] < 247.5):
                if (gradient[x][y] >= gradient[x - 1][y + 1]) or \
                        (gradient[x][y] >= gradient[x + 1][y - 1]):
                    suppressed_image[x][y] = gradient[x,y]
            elif (orientation[x][y] >= 67.5 and orientation[x][y] < 112.5) or \
                    (orientation[x][y] >= 247.5 and orientation[x][y] < 292.5):
                if (gradient[x][y] >= gradient[x + 1][y]) or \
                        (gradient[x][y] >= gradient[x - 1][y]):
                    suppressed_image[x][y] = gradient[x,y]
            else:
                if (gradient[x][y] >= gradient[x + 1][y + 1]) or \
                        (gradient[x][y] >= gradient[x - 1][y - 1]):
                    suppressed_image[x][y] = gradient[x,y]
    return suppressed_image

def apply_threshold(image, low_ratio=0.15, high_ratio=0.8, w=100, s=255):
    high_thres = image.max() * high_ratio
    low_thres = image.max()  * low_ratio
    r,c = image.shape
    thres = np.zeros((r,c))
    strongs = []
    for x in range(0, r):
        for y in range(0, c):
            if image[x][y] > high_thres:
                thres[x][y] = s
                strongs.append((x,y))
            elif image[x][y] > low_thres:
                thres[x][y] = w
    return thres, strongs

def hys(image, strongs, w=100, s=255):
    r,c = image.shape
    res = np.zeros((r,c))
    vis = np.zeros((r,c), bool)
    dx = [1, 0, -1,  0, -1, -1, 1,  1]
    dy = [0, 1,  0, -1,  1, -1, 1, -1]

    while len(strongs) > 0:
        str = strongs.pop()
        if vis[str] == False:
            vis[str] = True
            res[str] = s
            for k in range(8):
                for col in range(1, 20):
                    nx, ny = str[0] + col* dx[k], str[1] + col* dy[k]
                    if (nx >= 0 and nx < r and ny >= 0 and ny < c)\
                            and (image[nx, ny] >= w) and (vis[nx, ny] == False):
                        strongs.append((nx, ny))
    pass

    return res


def apply_canny_edge_detector(image):
    blurred_image = blur_image(image)

    edge_gradiant = get_gradients(blurred_image)
    orientation = get_orientation(blurred_image)

    suppressed_image = non_max_suppression(edge_gradiant, orientation)

    thresholded_image, strongs = apply_threshold(suppressed_image)

    final = hys(thresholded_image, strongs)

    return final

for image_file in read_all_image_files():
    print("Edge detection for image: " + image_file)

    grayscale_image = get_grayscale_image(image_file)

    edge_detected_image = apply_canny_edge_detector(grayscale_image)
    #edge_detected_image = cv2.Canny(grayscale_image,100,200)

    destination_path = generate_output_image_file_path(image_file)
    cv2.imwrite(destination_path, edge_detected_image)
    print("Final image saved to: " + destination_path)

