import os
import re
import numpy as np
import cv2

def get_scale(img):

    scale = cv2.inRange(img, np.array([60, 0, 0]), np.array([130, 10, 10]))
    # cv2.imwrite('./particles_results/scale.png', scale)

    contours, _ = cv2.findContours(scale, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)

    # Get the bounding rectangle
    x, y, w, h = cv2.boundingRect(largest_contour)

    # The length of the scale bar in pixels will be the larger of width or height
    pixel_length = max(w, h)
    offset1 = 60
    offset2 = 8
    text_roi = scale[y-offset1:y-offset2, x:x+w]
    # cv2.imwrite('./particles_results/text.png', text_roi)

    import pytesseract
    tesseract_path = os.path.join(
        os.environ['ProgramFiles'], "Tesseract-OCR", "tesseract.exe"
    )
    pytesseract.pytesseract.tesseract_cmd = tesseract_path
    text = pytesseract.image_to_string(text_roi)
    return pixel_length, text

def extract_float(string):
    # Use regex to find float or integer numbers
    match = re.search(r'\d*\.?\d+', string)
    if match:
        return float(match.group())
    return None  # Return None if no match is found                                                                                            

