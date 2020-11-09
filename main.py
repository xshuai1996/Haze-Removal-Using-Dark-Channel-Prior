import os
from haze_removal import read_img, calculate_dark_channel, show_img, calculate_A, estimate_t, estimate_J, save_results


img_name = "haze10.jpg"
img_path = os.path.join("test_samples", img_name)
patch_size = 11
t0 = 0.1
w = 0.95

assert patch_size % 2 == 1, "Parameter \"patch_size\" is supposed to be an odd number."
img = read_img(img_path)
img = img[2:-1, 2:-1]
print("Image shape : {}".format(img.shape))
show_img(img, "original image")
dark_channel = calculate_dark_channel(img, patch_size)
show_img(dark_channel, "dark channel")
A = calculate_A(img, dark_channel.copy())
print("A: {}".format(A))
t = estimate_t(img, A, patch_size, w)
vis_t = t * 255
show_img(vis_t, "t")
J = estimate_J(img, A, t, t0)
show_img(J, "haze free")
save_results(img_name.split('.')[0], img, dark_channel, vis_t, J)


