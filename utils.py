
import cv2
import numpy as np

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