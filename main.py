import cv2
import os
from find_axis import *
from find_blob import *
from find_scale import *

# Define the parameters
p1 = {'thresh': 254,
      'debug_path': None
      }

p2 = {'thresh': 15,
      'preprocessing': True,
      'debug_path': None
      }

# Define input and output directories
input_folder = './particles'
output_folder = './results'

# Ensure the output directory exists
os.makedirs(output_folder, exist_ok=True)

# Iterate over all image files in the input folder
for i, filename in enumerate(os.listdir(input_folder), start=1):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
        print(f'{i}: Processing {filename} ...................................................')

        # Read the image
        img_path = os.path.join(input_folder, filename)
        img = cv2.imread(img_path)

        # Step 1: Scale calculation
        pixel_length, text = get_scale(img)
        scale = extract_float(text)
        s = scale / pixel_length
        print('Scale value', s)

        # Step 2: Thresholding methods
        resu1 = thresh_method(img, **p1)
        cv2.imwrite(os.path.join(output_folder, f'result{i}_1.png'), resu1)

        resu2 = sobel_method(img, **p2)
        cv2.imwrite(os.path.join(output_folder, f'result{i}_2.png'), resu2)
        print('Thresholding done!')

        # Step 3: Fitting methods
        midpoints1 = quad_fit(resu1, method='rect')
        midpoints2 = max_min_fit(resu1)

        img_to_save = draw_midpoints_fit(img, midpoints1, s)
        cv2.imwrite(os.path.join(output_folder, f'fit{i}_1.png'), img_to_save)

        img_to_save = draw_midpoints_fit(img, midpoints2, s)
        cv2.imwrite(os.path.join(output_folder, f'fit{i}_2.png'), img_to_save)
        print('Fitting done!')
