import cv2
from find_axis import *
from find_blob import *
from find_scale import *

p1 = {'thresh': 254,
        'debug_path': None 
    }

p2 = {'thresh': 15,
        'preprocessing': True, 
        'debug_path': None 
    }

for i in range(1,6):
    print(i, '...................................................')
    img = cv2.imread(f'./particles/Captured {i}.jpg')

    pixel_length, text = get_scale(img)
    scale = extract_float(text)
    s = scale/pixel_length
    s=1
    print('scale done!')

    resu1 = thresh_method(img, **p1)
    cv2.imwrite(f'./particles_results/result{i}_1.png', resu1)

    resu2 = sobel_method(img, **p2)
    cv2.imwrite(f'./particles_results/result{i}_2.png', resu2)
    print('thresh done!')

    midpoints1 = quad_fit(resu1, method='rect')
    midpoints2 = max_min_fit(resu1)

    img_to_save = draw_midpoints_fit(img, midpoints1, s)
    cv2.imwrite(f'./particles_results/fit{i}_1.png', img_to_save)

    img_to_save = draw_midpoints_fit(img, midpoints2, s)
    cv2.imwrite(f'./particles_results/fit{i}_2.png', img_to_save)
    print('fit done!')



