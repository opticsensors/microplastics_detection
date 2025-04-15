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

        # Read the image
        img_path = os.path.join(input_folder, filename)
        img = cv2.imread(img_path)

        scale, contours = get_contour_of_scale(img, scale_type='white')
        best_contour, best_rect = find_best_rect_by_ratio(contours)
        debug_img = img.copy()


        # Step 1: Scale calculation
        pixel_length, text = get_scale(img, scale_type='white')
        scale = extract_float(text)

        if scale is None:
            print(f'{i}: Processing {filename} ...................................................')
            print('Scale', scale)
            if best_rect is not None:
                 print('bestRectangle is NOT None!!!!')
                 x, y, w, h = best_rect
                 cv2.rectangle(debug_img, (x, y), (x+w, y+h), (255, 0, 255), -1)  # Magenta color
                 cv2.imshow("debugger", cv2.resize(debug_img, (0,0), fx=0.2, fy=0.2))
                 key=cv2.waitKey(1500)
                 if key==27:
                       break
            else:
                  print('bestRectangle is None!!!!')

        # # Step 2: Thresholding methods
        # resu = sobel_method(img, **p2)
        # cv2.imwrite(os.path.join(output_folder, f'result{i}_2.png'), resu)
        # print('Thresholding done!')
 
        # # Step 3: Fitting methods
        # midpoints = max_min_fit(resu)
 
        # img_to_save,_,_ = draw_midpoints_fit(img, midpoints, s)
        # cv2.imwrite(os.path.join(output_folder, f'fit{i}_2.png'), img_to_save)
        # print('Fitting done!')
