import os
import re
import numpy as np
import cv2

def get_scale(img, scale_type='dark'):
    # Create binary image based on scale type
    if scale_type == 'dark':
        scale = cv2.inRange(img, np.array([60, 0, 0]), np.array([130, 10, 10]))
    elif scale_type == 'white':
        scale = cv2.inRange(img, np.array([252, 252, 252]), np.array([255, 255, 255]))
    else:
        raise ValueError("scale_type must be 'dark' or 'white'")
    
    # Check if scale image is empty
    if cv2.countNonZero(scale) == 0:
        return None, None
    
    # Find all contours in the binary image
    contours, _ = cv2.findContours(scale, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # If no contours found, return None
    if not contours:
        return None, None
    
    # Find the contour with the highest width/height ratio
    best_ratio = 0
    best_contour = None
    best_rect = None
    
    for contour in contours:
        # Get bounding rectangle for this contour
        x, y, w, h = cv2.boundingRect(contour)
        
        # Skip very small contours that might be noise
        if w * h < 20:
            continue
        
        # Calculate width to height ratio (avoid division by zero)
        if h > 0:
            ratio = w / h
        else:
            continue
        
        # Update best if this ratio is higher
        if ratio > best_ratio:
            best_ratio = ratio
            best_contour = contour
            best_rect = (x, y, w, h)
    
    # If no suitable contour found
    if best_contour is None:
        return None, None
    
    # Unpack the best rectangle
    x, y, w, h = best_rect
    
    # The length of the scale bar in pixels is the width
    pixel_length = w
    
    try:
        # Extract text region above the scale bar
        offset1 = 60  # Distance above to start capturing text
        offset2 = 8   # Distance above to end capturing text
        
        # Make sure we don't go out of bounds
        y_start = max(0, y - offset1)
        y_end = max(0, y - offset2)
        
        # If text region is invalid, return only pixel length
        if y_start >= y_end or y_start >= scale.shape[0] or x >= scale.shape[1]:
            return pixel_length, None
        
        # Extract text region
        text_roi = scale[y_start:y_end, x:x+w]
        
        # Check if text_roi is empty
        if text_roi.size == 0:
            return pixel_length, None
        
        # Use Tesseract to extract text
        import pytesseract
        tesseract_path = os.path.join(
            os.environ['ProgramFiles'], "Tesseract-OCR", "tesseract.exe"
        )
        pytesseract.pytesseract.tesseract_cmd = tesseract_path
        text = pytesseract.image_to_string(text_roi)
        
        # Try to extract float value from text
        scale_value = extract_float(text)
        
        return pixel_length, scale_value
        
    except Exception:
        # If anything goes wrong with text processing, return just the pixel length
        return pixel_length, None

def extract_float(text):
    # If already a float or int, return it directly
    if isinstance(text, (float, int)):
        return float(text)
    
    # If None, return None
    if text is None:
        return None
    
    # If it's a string, try to extract a float using regex
    if isinstance(text, str):
        match = re.search(r'\d*\.?\d+', text)
        if match:
            try:
                return float(match.group())
            except ValueError:
                return None
    
    return None