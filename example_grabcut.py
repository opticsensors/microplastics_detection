#!/usr/bin/env python
'''
===============================================================================
Interactive Image Segmentation using GrabCut algorithm with image scaling.

This sample shows interactive image segmentation using grabcut algorithm with
the ability to scale large images for faster processing and better display.

USAGE:
    python grabcut.py <filename> [display_scale] [processing_scale]

README FIRST:
    Two windows will show up, one for input and one for output.

    At first, in input window, draw a rectangle around the object using the
right mouse button. Then press 'n' to segment the object (once or a few times)
For any finer touch-ups, you can press any of the keys below and draw lines on
the areas you want. Then again press 'n' to update the output.

Key '0' - To select areas of sure background
Key '1' - To select areas of sure foreground
Key '2' - To select areas of probable background
Key '3' - To select areas of probable foreground

Key 'n' - To update the segmentation
Key 'r' - To reset the setup
Key 's' - To save the results
Key '+' - To increase display scaling factor
Key '-' - To decrease display scaling factor
===============================================================================
'''

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2 as cv
import sys

class App():
    BLUE = [255,0,0]        # rectangle color
    RED = [0,0,255]         # PR BG
    GREEN = [0,255,0]       # PR FG
    BLACK = [0,0,0]         # sure BG
    WHITE = [255,255,255]   # sure FG

    DRAW_BG = {'color' : BLACK, 'val' : 0}
    DRAW_FG = {'color' : WHITE, 'val' : 1}
    DRAW_PR_BG = {'color' : RED, 'val' : 2}
    DRAW_PR_FG = {'color' : GREEN, 'val' : 3}

    # setting up flags
    rect = (0,0,1,1)
    drawing = False         # flag for drawing curves
    rectangle = False       # flag for drawing rect
    rect_over = False       # flag to check if rect drawn
    rect_or_mask = 100      # flag for selecting rect or mask mode
    value = DRAW_FG         # drawing initialized to FG
    thickness = 3           # brush thickness
    display_scale = 0.5     # initial display scale factor
    processing_scale = 0.5  # scale for internal processing (affects speed)
    
    def __init__(self, display_scale=0.5, processing_scale=0.25):
        self.display_scale = display_scale
        self.processing_scale = processing_scale
        
    def scale_image(self, image, factor):
        """Scale the image by the given factor."""
        if factor == 1.0:
            return image.copy()
        h, w = image.shape[:2]
        new_h, new_w = int(h * factor), int(w * factor)
        return cv.resize(image, (new_w, new_h), interpolation=cv.INTER_AREA)
    
    def onmouse(self, event, x, y, flags, param):
        # Convert display coordinates to working image coordinates
        work_x, work_y = int(x / (self.display_scale / self.processing_scale)), int(y / (self.display_scale / self.processing_scale))
        
        # Draw Rectangle
        if event == cv.EVENT_RBUTTONDOWN:
            self.rectangle = True
            self.ix, self.iy = x, y
            self.work_ix, self.work_iy = work_x, work_y

        elif event == cv.EVENT_MOUSEMOVE:
            if self.rectangle == True:
                self.img = self.img_display.copy()
                cv.rectangle(self.img, (self.ix, self.iy), (x, y), self.BLUE, 2)
                self.rect = (min(self.work_ix, work_x), min(self.work_iy, work_y), 
                            abs(self.work_ix - work_x), abs(self.work_iy - work_y))
                self.rect_or_mask = 0

        elif event == cv.EVENT_RBUTTONUP:
            self.rectangle = False
            self.rect_over = True
            cv.rectangle(self.img, (self.ix, self.iy), (x, y), self.BLUE, 2)
            self.rect = (min(self.work_ix, work_x), min(self.work_iy, work_y), 
                        abs(self.work_ix - work_x), abs(self.work_iy - work_y))
            self.rect_or_mask = 0
            print(" Now press the key 'n' a few times until no further change \n")

        # draw touchup curves
        if event == cv.EVENT_LBUTTONDOWN:
            if self.rect_over == False:
                print("first draw rectangle \n")
            else:
                self.drawing = True
                cv.circle(self.img, (x, y), self.thickness, self.value['color'], -1)
                cv.circle(self.mask_display, (x, y), self.thickness, self.value['val'], -1)

        elif event == cv.EVENT_MOUSEMOVE:
            if self.drawing == True:
                cv.circle(self.img, (x, y), self.thickness, self.value['color'], -1)
                cv.circle(self.mask_display, (x, y), self.thickness, self.value['val'], -1)

        elif event == cv.EVENT_LBUTTONUP:
            if self.drawing == True:
                self.drawing = False
                cv.circle(self.img, (x, y), self.thickness, self.value['color'], -1)
                cv.circle(self.mask_display, (x, y), self.thickness, self.value['val'], -1)

    def create_binary_mask(self, mask):
        """Create a binary mask where foreground is 255 and background is 0."""
        return np.where((mask==1) + (mask==3), 255, 0).astype('uint8')
    
    def update_output(self):
        """Update the full-resolution output using the current working mask."""
        # Create binary mask from working mask
        binary_mask_work = self.create_binary_mask(self.mask_work)
        
        # Scale the binary mask to original image size
        self.binary_mask_orig = cv.resize(binary_mask_work, 
                                         (self.img_orig.shape[1], self.img_orig.shape[0]), 
                                         interpolation=cv.INTER_NEAREST)
        
        # Apply the mask to the original image
        self.output_orig = cv.bitwise_and(self.img_orig, self.img_orig, mask=self.binary_mask_orig)

    def update_display(self):
        """Update the display images based on current scale factor."""
        # Create display scaled version from working image
        self.img_display = self.scale_image(self.img_work, self.display_scale / self.processing_scale)
        self.img = self.img_display.copy()
        
        # Scale the mask for display
        h, w = self.img_display.shape[:2]
        self.mask_display = cv.resize(self.mask_work, (w, h), interpolation=cv.INTER_NEAREST)
        
        # Scale the full-resolution output for display
        if hasattr(self, 'output_orig'):
            self.output_display = self.scale_image(self.output_orig, self.display_scale)
        else:
            self.output_display = np.zeros((h, w, 3), np.uint8)
        
        # Update window sizes
        cv.namedWindow('output', cv.WINDOW_NORMAL)
        cv.namedWindow('input', cv.WINDOW_NORMAL)
        cv.resizeWindow('output', w, h)
        cv.resizeWindow('input', w, h)
        cv.moveWindow('input', w+10, 90)

    def run(self):
        # Loading images
        if len(sys.argv) > 1:
            filename = sys.argv[1]
        else:
            filename = './particles/P6-F2.jpg'
            
        # Get scale factors from command line if provided
        if len(sys.argv) > 2:
            try:
                self.display_scale = float(sys.argv[2])
            except ValueError:
                print("Invalid display scale factor. Using default 0.5")
                self.display_scale = 0.5
                
        if len(sys.argv) > 3:
            try:
                self.processing_scale = float(sys.argv[3])
            except ValueError:
                print("Invalid processing scale factor. Using default 0.25")
                self.processing_scale = 0.25

        # Load and check original image
        self.img_orig = cv.imread(cv.samples.findFile(filename))
        if self.img_orig is None:
            print('Failed to load image file:', filename)
            return
            
        # Create working image (downscaled for faster processing)
        self.img_work = self.scale_image(self.img_orig, self.processing_scale)
        self.img_work_copy = self.img_work.copy()
        
        # Initialize working mask
        self.mask_work = np.zeros(self.img_work.shape[:2], dtype=np.uint8)
        
        # Create the display versions
        self.update_display()

        # Set up windows and mouse callback
        cv.setMouseCallback('input', self.onmouse)

        print(" Instructions: \n")
        print(" Draw a rectangle around the object using right mouse button \n")
        print(" Press '+' to increase display scaling factor, '-' to decrease \n")
        print(f" Processing at {self.processing_scale:.2f}x scale for speed, displaying at {self.display_scale:.2f}x scale \n")
        print(f" Final output will be at original {self.img_orig.shape[1]}x{self.img_orig.shape[0]} resolution \n")

        while(1):
            cv.imshow('output', self.output_display)
            cv.imshow('input', self.img)
            k = cv.waitKey(1)

            # key bindings
            if k == 27:         # esc to exit
                break
            elif k == ord('0'): # BG drawing
                print(" mark background regions with left mouse button \n")
                self.value = self.DRAW_BG
            elif k == ord('1'): # FG drawing
                print(" mark foreground regions with left mouse button \n")
                self.value = self.DRAW_FG
            elif k == ord('2'): # PR_BG drawing
                self.value = self.DRAW_PR_BG
            elif k == ord('3'): # PR_FG drawing
                self.value = self.DRAW_PR_FG
            elif k == ord('s'): # save image
                # Make sure output is updated
                if not hasattr(self, 'output_orig'):
                    self.update_output()
                
                # Save results at original resolution
                bar = np.zeros((self.img_orig.shape[0], 5, 3), np.uint8)
                
                # Save both the original image, the binary mask, and the masked result
                cv.imwrite('grabcut_mask.png', self.binary_mask_orig)
                cv.imwrite('grabcut_output.png', self.output_orig)
                
                # Save a side-by-side comparison
                res = np.hstack((self.img_orig, bar, self.output_orig))
                cv.imwrite('grabcut_comparison.png', res)
                print(" Results saved: \n")
                print(" - grabcut_mask.png (binary mask at original resolution) \n")
                print(" - grabcut_output.png (masked image at original resolution) \n")
                print(" - grabcut_comparison.png (side-by-side comparison) \n")
                
            elif k == ord('r'): # reset everything
                print("resetting \n")
                self.rect = (0,0,1,1)
                self.drawing = False
                self.rectangle = False
                self.rect_or_mask = 100
                self.rect_over = False
                self.value = self.DRAW_FG
                
                # Reset the working images
                self.img_work = self.img_work_copy.copy()
                self.mask_work = np.zeros(self.img_work.shape[:2], dtype=np.uint8)
                
                # Remove output attributes
                if hasattr(self, 'output_orig'):
                    delattr(self, 'output_orig')
                if hasattr(self, 'binary_mask_orig'):
                    delattr(self, 'binary_mask_orig')
                
                # Update display
                self.update_display()
                
            elif k == ord('n'): # segment the image
                print(""" For finer touchups, mark foreground and background after pressing keys 0-3
                and again press 'n' \n""")
                
                # Update the working mask from the display mask
                if self.display_scale != self.processing_scale:
                    self.mask_work = cv.resize(self.mask_display, 
                                            (self.img_work.shape[1], self.img_work.shape[0]), 
                                            interpolation=cv.INTER_NEAREST)
                
                try:
                    bgdmodel = np.zeros((1, 65), np.float64)
                    fgdmodel = np.zeros((1, 65), np.float64)
                    
                    # Run GrabCut on the working-sized image for speed
                    if (self.rect_or_mask == 0):         # grabcut with rect
                        cv.grabCut(self.img_work, self.mask_work, self.rect, bgdmodel, fgdmodel, 1, cv.GC_INIT_WITH_RECT)
                        self.rect_or_mask = 1
                    elif (self.rect_or_mask == 1):       # grabcut with mask
                        cv.grabCut(self.img_work, self.mask_work, self.rect, bgdmodel, fgdmodel, 1, cv.GC_INIT_WITH_MASK)
                except:
                    import traceback
                    traceback.print_exc()

                # Update the full-resolution output
                self.update_output()
                
                # Update the display 
                self.update_display()
                
            elif k == ord('+'): # increase display scale factor
                self.display_scale = min(2.0, self.display_scale + 0.1)
                print(f"Display scale factor: {self.display_scale:.2f}")
                self.update_display()
                
            elif k == ord('-'): # decrease display scale factor
                self.display_scale = max(0.1, self.display_scale - 0.1)
                print(f"Display scale factor: {self.display_scale:.2f}")
                self.update_display()

        print('Done')


if __name__ == '__main__':
    print(__doc__)
    # Default scale factors can be set here
    # First parameter is display scale (for viewing)
    # Second parameter is processing scale (for speed)
    App(display_scale=0.35, processing_scale=0.1).run()
    cv.destroyAllWindows()