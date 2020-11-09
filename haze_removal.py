import cv2
import numpy as np
from skimage.util import view_as_windows
import os
import shutil


def read_img(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float)
    return img


def show_img(img, win_name):
    img = img.copy()
    img = np.where(img > 255, 255, img)
    img = np.where(img < 0, 0, img)
    img = img.copy().astype(np.uint8)
    if img.shape[2] == 3:   # RGB
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imshow(win_name, img)
    cv2.waitKey()
    cv2.destroyAllWindows()


def save_results(folder_name, original_img, dark_channel, t, J):
    folder_name = os.path.join("results", folder_name)
    if os.path.exists(folder_name):
        shutil.rmtree(folder_name)
        print("All files under folder \"{}\" has been deleted. ".format(folder_name))
    os.makedirs(folder_name)

    original_img = cv2.cvtColor(original_img.astype(np.uint8), cv2.COLOR_RGB2BGR)
    dark_channel = dark_channel.astype(np.uint8)
    t = t.astype(np.uint8)
    J = np.where(J > 255, 255, J)
    J = np.where(J < 0, 0, J)
    J = cv2.cvtColor(J.astype(np.uint8), cv2.COLOR_RGB2BGR)

    cv2.imwrite(os.path.join(folder_name, "original image.jpg"), original_img)
    cv2.imwrite(os.path.join(folder_name, "dark_channel.jpg"), dark_channel)
    cv2.imwrite(os.path.join(folder_name, "t.jpg"), t)
    cv2.imwrite(os.path.join(folder_name, "J.jpg"), J)


def calculate_dark_channel(original_img, win_size):
    darkest_in_RGB = np.min(original_img, axis=2)
    padding = win_size // 2
    dark_channel = np.ones((original_img.shape[0] + 2 * padding, original_img.shape[1] + 2 * padding)) * 255
    dark_channel[padding:-padding, padding:-padding] = darkest_in_RGB
    dark_channel = view_as_windows(dark_channel, (win_size, win_size), (1, 1))
    dark_channel = dark_channel.reshape((original_img.shape[0], original_img.shape[1], win_size * win_size))
    dark_channel = np.min(dark_channel, axis=2, keepdims=True)
    return dark_channel


def calculate_A(original_img, dark_channel):
    sort = -np.sort(-dark_channel, axis=None)     # sort the flattened array, 2'-' for decreasing order
    thresh = sort[int(sort.shape[0] * 0.001)]
    show_img(np.where(dark_channel >= thresh, dark_channel, 0), "dark_channel")
    haze_opaque = np.where(dark_channel >= thresh, original_img, 0)
    show_img(haze_opaque, "haze_opaque")
    haze_opaque = np.moveaxis(haze_opaque, 2, 0)
    haze_opaque = haze_opaque.reshape((3, -1))
    A = np.max(haze_opaque, axis=1)
    return A


def estimate_t(original_img, A, win_size, w):
    dense_ratio = original_img / A
    dense_ratio = np.min(dense_ratio, axis=2)
    padding = win_size // 2
    t = np.ones((original_img.shape[0] + 2 * padding, original_img.shape[1] + 2 * padding))
    t[padding:-padding, padding:-padding] = dense_ratio
    t = view_as_windows(t, (win_size, win_size), (1, 1))
    t = t.reshape((original_img.shape[0], original_img.shape[1], win_size * win_size))
    t = np.min(t, axis=2, keepdims=True)
    return 1 - w * t


def calculate_t(original_img, t_estimate, win_size, eps):
    win_radius = win_size // 2
    U3 = np.identity(3)
    L = np.zeros((original_img.shape[0], original_img.shape[1]))
    for x in range(original_img.shape[0]):
        for y in range(original_img.shape[1]):
            # x and y specify the local region wk
            sum_ = 0
            win_left = max(0, x - win_radius)
            win_right = min(original_img.shape[0], x + win_radius + 1)
            win_up = max(0, y - win_radius)
            win_down = min(original_img.shape[1], y + win_radius + 1)
            wk_absolute = (win_right - win_left) * (win_down - win_up)
            all_pixels_in_wk = original_img[win_left: win_right, win_up: win_down]
            all_pixels_in_wk = np.moveaxis(all_pixels_in_wk, 2, 0)
            all_pixels_in_wk = all_pixels_in_wk.reshape((3, wk_absolute))
            mu_k = np.mean(all_pixels_in_wk, axis=1, keepdims=True)
            cov_k = np.cov(all_pixels_in_wk)
            for i in range(wk_absolute):
                for j in range(wk_absolute):
                    delta_ij = 1 if i == j else 0
                    component1 = np.transpose(all_pixels_in_wk[:, i] - mu_k)
                    component2 = np.linalg.inv(cov_k + eps / wk_absolute * U3)
                    component3 = all_pixels_in_wk[:, j] - mu_k
                    sum_ += delta_ij - (1 / wk_absolute) * (1 + np.dot(np.dot(component1, component2), component3))
            L[x, y] = sum_








def estimate_J(original_img, A, t, t0):
    J = (original_img - A) / np.where(t < t0, t0, t) + A
    return J