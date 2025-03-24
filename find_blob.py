import numpy as np
import cv2
import os

def find_largest_contour(thresh):
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        largest_contour_filled = np.zeros((thresh.shape[0], thresh.shape[1], 3), dtype=np.uint8)
        cv2.drawContours(largest_contour_filled, [largest_contour], -1, (255, 255, 255), -1)  # Green color
        largest_contour_filled = largest_contour_filled[...,0]

        return largest_contour_filled
    
    else:
        return np.zeros((thresh.shape[0], thresh.shape[1]), dtype=np.uint8)
    
def thresh_method(img, **kwargs):

    # Default parameter values
    p = {'thresh': 254,
        'debug_path': None 
    }
    
    # Update with any provided keyword arguments
    p.update(kwargs)

    # Function to save debug images
    def save_debug_image(name, image):
        if p['debug_path'] is not None:
            cv2.imwrite(os.path.join(p['debug_path'], f"{name}.png"), image)

    hsv=cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    v = hsv[...,2]
    save_debug_image("00_value", v)

    _, thresh = cv2.threshold(v, p['thresh'], 255, cv2.THRESH_BINARY)
    save_debug_image("01_thresholded", thresh)

    largest_contour_filled = find_largest_contour(thresh)
    save_debug_image("02_largest_blob", largest_contour_filled)

    thresh_inv = cv2.bitwise_not(thresh)
    particle_with_dirt = cv2.bitwise_and(thresh_inv, largest_contour_filled)
    save_debug_image("03_bit_operation", particle_with_dirt)

    particle = find_largest_contour(particle_with_dirt)

    return particle

def sobel_method(img, **kwargs):

    # Default parameter values
    p = {'thresh': 15,
         'preprocessing': True, 
         'debug_path': None 
    }
    
    # Update with any provided keyword arguments
    p.update(kwargs)

    # Function to save debug images
    def save_debug_image(name, image):
        if p['debug_path'] is not None:
            cv2.imwrite(os.path.join(p['debug_path'], f"{name}.png"), image)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    save_debug_image("00_gray", gray)

    if p['preprocessing']:
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
        
        # Apply bilateral filter to reduce noise while preserving edges
        gray = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
        save_debug_image("01_gray_preprocessed", gray)
    
    # Calculate Sobel gradients with larger kernel
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
    gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
    
    # Normalize gradient magnitude to 0-255
    gradient_magnitude = np.uint8(255 * gradient_magnitude / np.max(gradient_magnitude))
    save_debug_image("02_gradient_magnitude", gradient_magnitude)

    # Apply threshold to gradient magnitude
    _, gradient_thresh = cv2.threshold(gradient_magnitude, p['thresh'], 255, cv2.THRESH_BINARY)
    save_debug_image("03_gradient_thresh", gradient_thresh)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31,31))
    gradient_thresh = cv2.morphologyEx(gradient_thresh, cv2.MORPH_CLOSE, kernel, iterations=3)
    save_debug_image("04_gradient_morph", gradient_thresh)

    particle = find_largest_contour(gradient_thresh)

    return particle

def compute_particle_color(binary_mask, original_image):

    # Convert original image to HSV
    hsv_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)
    
    # Get the particle and background regions
    particle_pixels = hsv_image[binary_mask == 255]
    background_pixels = hsv_image[binary_mask == 0]
    
    # Check if particle has enough pixels
    if len(particle_pixels) < 10:
        return 'C', 'Opaque'  # Default if particle is too small
    
    # Compute mean HSV values for particle
    particle_mean_h = np.mean(particle_pixels[:, 0])
    particle_mean_s = np.mean(particle_pixels[:, 1])
    particle_mean_v = np.mean(particle_pixels[:, 2])
    particle_std_h = np.std(particle_pixels[:, 0])
    
    # Determine transparency
    # Compare value (brightness) between particle and background
    particle_mean_v = np.mean(particle_pixels[:, 2])
    background_mean_v = np.mean(background_pixels[:, 2])
    
    # If saturation is low and value is high, likely transparent
    if particle_mean_s < 30 and particle_mean_v > 150:
        transparency = 'Transparent'
    else:
        transparency = 'Opaque'
    
    # Determine color based on HSV
    # Low saturation indicates white, grey, black, or colorless
    if particle_mean_s < 20:
        if particle_mean_v < 60:
            color_code = 'K'  # Black
        elif particle_mean_v < 180:
            color_code = 'A'  # Grey
        else:
            color_code = 'W'  # White
    # For colored particles
    else:
        if particle_std_h > 30:  # High standard deviation in hue indicates multiple colors
            color_code = 'M'  # Multicolor
        else:
            # Determine color based on hue value
            if 0 <= particle_mean_h < 10 or 170 <= particle_mean_h <= 180:
                color_code = 'R'  # Red
            elif 10 <= particle_mean_h < 25:
                color_code = 'O'  # Orange
            elif 25 <= particle_mean_h < 35:
                color_code = 'Y'  # Yellow
            elif 35 <= particle_mean_h < 85:
                color_code = 'G'  # Green
            elif 85 <= particle_mean_h < 130:
                color_code = 'B'  # Blue
            elif 130 <= particle_mean_h < 155:
                color_code = 'V'  # Purple
            elif 155 <= particle_mean_h < 170:
                color_code = 'P'  # Pink
            else:
                color_code = 'N'  # Brown (fallback)
    
    # Special case for colorless (low saturation and special case)
    if particle_mean_s < 15 and 20 < particle_mean_v < 230:
        color_code = 'C'  # Colorless
    
    return color_code, transparency