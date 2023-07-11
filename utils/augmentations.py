import cv2 as cv
import numpy as np
from random import randint, uniform
# from perlin_noise import PerlinNoise


def scale_to_range(_array, _min, _max):
    # 1 - Scale noise so it's between -1 and
    _array *= 2 / (np.max(_array) - np.min(_array))
    _array -= (1 + np.min(_array))

    # 2 - Scale and offset noise so it's between noise_intensity_min and noise_intensity_max
    _array *= (_max - _min) / 2
    _array += (_min - np.min(_array))

    return _array


def blend_normal(_img_in, _overlay, _alpha_overlay):
    # Normal overlay method
    alpha_background_mask = np.ones_like(_overlay)

    alpha_tot = _alpha_overlay + alpha_background_mask * (1 - _alpha_overlay)
    img_out = ((_overlay * _alpha_overlay) + (_img_in * alpha_background_mask * (1 - _alpha_overlay))) / alpha_tot

    return img_out


def blend_screen(_img_in, _overlay, _opacity):
    # Screen overlay method
    img_in_inv = 1 - _img_in
    overlay_inv = 1 - (_opacity * _overlay)
    img_out = 1 - (img_in_inv * overlay_inv)

    return img_out


def desaturate(_img_in, _percent):
    mask_hsv = cv.cvtColor(np.array(_img_in * 255, dtype=np.uint8), cv.COLOR_BGR2HSV)
    mask_hsv[:, :, 1] = mask_hsv[:, :, 1] * (1 - _percent)
    output = cv.cvtColor(mask_hsv, cv.COLOR_HSV2BGR)
    output = np.array(output, dtype=np.float32) / 255

    return output


def lighter(_colour, _percent):
    # assumes _colour is rgb between (0, 0, 0) and (255, 255, 255)
    _colour = np.array(_colour)

    if _colour.all() == np.zeros_like(_colour).all():
        # Do not process if input colour is black
        output = _colour
    else:
        white = np.array([255, 255, 255])
        vector = white - _colour
        output = _colour + vector * _percent

    return output


def flat_array_to_3d(_input_array, colour_tuple):
    # Find height and width of array
    height, width = np.shape(_input_array)

    # Make 3d output array with 3 layers
    output_array = np.zeros((height, width, 3))

    # Put data from input array into appropriate layer of output array, and scale each layer to create desired colour
    for i in range(3):
        output_array[:, :, i] = _input_array * colour_tuple[i]

    return output_array


def show_img(_img_array):
    cv.imshow('Image Display', _img_array)
    cv.waitKey(0)

#
# def add_dust(img_raw):
#     PERLIN_NOISE_RED_FACTOR = 10  # Integer value representing fraction of image size to use for noise size. Larger number gives coarser noise but faster performance
#     CROP_SIZE = 100  # Half of window height/width in centre of image where crop colour is taken from
#     BLUR_SIZE_MAX = 41  # Max blur pixels
#
#     # Opacity ranges (actual opacity is a random value between these limits)
#     NORMAL_OPACITY_MIN = 0.5
#     NORMAL_OPACITY_MAX = 0.8
#     SCREEN_OPACITY_MIN = 0.1
#     SCREEN_OPACITY_MAX = 0.3
#
#     # Percentage range to desaturate colour (actual percentage is a random value between these limits)
#     DESATURATE_FRAC_MIN = 0.1
#     DESATURATE_FRAC_MAX = 0.5
#
#     # Octave range for  perlin noise. Higher octave gives higher frequency noise variations
#     OCTAVES_MIN = 1
#     OCTAVES_MAX = 4
#
#     # Load raw image
#     img_raw = np.array(img_raw, dtype=np.float32)
#     img_raw /= 255
#     height, width, _ = np.shape(img_raw)
#     height_noise = int(height / PERLIN_NOISE_RED_FACTOR)
#     width_noise = int((height_noise + width / PERLIN_NOISE_RED_FACTOR / 2) / 2)
#
#     # Cut out the centre of the image so we can find the average colour here
#     height_centre = int(height / 2)
#     width_centre = int(width / 2)
#     img_centre = img_raw[height_centre - CROP_SIZE:height_centre + CROP_SIZE,
#                  width_centre - CROP_SIZE:width_centre + CROP_SIZE]
#
#     # Find colour for dust by taking average colour in centre of image
#     colour_centre_avg = np.average(img_centre, 0)
#     colour_centre_avg = np.average(colour_centre_avg, 0)
#     colour_centre_avg /= np.max(colour_centre_avg)
#
#     # Create perlin noise
#     rand_seed = randint(0, 10000)
#     rand_octaves = randint(OCTAVES_MIN, OCTAVES_MAX)
#     noise = PerlinNoise(octaves=rand_octaves, seed=rand_seed)
#     noise_array_sm = np.array(
#         [[noise([i / width_noise, j / height_noise]) for j in range(width_noise)] for i in range(height_noise)])
#     noise_array = cv.resize(noise_array_sm, (width, height))
#
#     # Offset and scale intensity of noise so values are between 0 and noise_intensity_max
#     noise_array = scale_to_range(noise_array, 0, 1)
#
#     # Create 3D white overlay to use as a mask for filtering which areas are dusty and which aren't
#     dust_mask_white = flat_array_to_3d(noise_array, [1, 1, 1])
#
#     # Turn greyscale noise into coloured overlay
#     dust_mask_coloured = flat_array_to_3d(noise_array, colour_centre_avg)
#
#     ############
#     # OVERLAYS #
#     ############
#     # Use Normal overlay method to add non-translucent dust
#     normal_opacity = uniform(NORMAL_OPACITY_MIN, NORMAL_OPACITY_MAX)
#     alpha_overlay = scale_to_range(dust_mask_white, 0, normal_opacity)
#
#     desaturate_frac_norm = uniform(DESATURATE_FRAC_MIN, DESATURATE_FRAC_MAX)
#     img_dusty_norm = blend_normal(img_raw, desaturate(dust_mask_coloured, desaturate_frac_norm), alpha_overlay)
#
#     # Use 'screen' overlay method to create dusty image
#     desaturate_frac_screen = uniform(DESATURATE_FRAC_MIN, DESATURATE_FRAC_MAX)
#     screen_opacity = uniform(SCREEN_OPACITY_MIN, SCREEN_OPACITY_MAX)
#     img_dusty_screen = blend_screen(img_dusty_norm, desaturate(dust_mask_coloured, desaturate_frac_screen),
#                                     screen_opacity)
#
#     ############
#     # BLURRING #
#     ############
#     # Create random blur size
#     blur_size = randint(min(BLUR_SIZE_MAX, 3), BLUR_SIZE_MAX)
#     if blur_size % 2 == 0:
#         blur_size += 1
#
#     # Blur image, and combine with original image using dust_mask_white as a mask
#     blur_kernel = (blur_size, blur_size)
#     img_dusty_blurred_all = cv.GaussianBlur(img_dusty_screen, blur_kernel, 0)
#     img_dusty_blurred = (dust_mask_white * img_dusty_blurred_all) + ((1 - dust_mask_white) * img_dusty_screen)
#
#     return img_dusty_blurred * 255


if __name__ == '__main__':
    # Small script for testing function
    path_img_input = "./01_InputImages/00_sample.jpg"
    path_img_output = "./02_OutputImages/01_sample.jpg"
    input_file = cv.imread(path_img_input)
    # output_file = add_dust(input_file)
    cv.imwrite(path_img_output, output_file)