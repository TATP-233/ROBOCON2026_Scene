import numpy as np

def camera2k(fovy, width, height):
    cx = width / 2
    cy = height / 2
    fovx = 2 * np.arctan(np.tan(fovy / 2.) * width / height)
    fx = cx / np.tan(fovx / 2)
    fy = cy / np.tan(fovy / 2)
    return np.array([[fx, 0, cx],
                     [0, fy, cy],
                     [0,  0,  1]])