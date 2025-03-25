import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import os
from find_axis import *
from find_blob import *
from find_scale import *
import json
import re
import numpy as np
import torch
from ultralytics import FastSAM
from segment_anything import sam_model_registry, SamPredictor



class ImageApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Viewer")
        self.root.geometry("1400x960")

        self.scale_type = 'white'
        self.segmentation_algorithm = 'fastsam'  # Default, can be changed to 'sam'

        # Attributes to store folder paths
        self.input_images_folder = None
        self.save_folder = None
        
        # Image-related attributes
        self.image_files = []
        self.current_image_index = 0
        self.current_image = None
        
        # Color and transparency attributes
        self.current_color = 'C'  # Default colorless
        self.current_transparency = 'Opaque'  # Default opaque
        
        # Points for interactive segmentation
        self.points = []
        self.point_markers = []
        
        # Initialize device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        # Initial screen with buttons
        self.create_folder_selection_screen()
    
    def create_folder_selection_screen(self):
        # Clear existing widgets
        for widget in self.root.winfo_children():
            widget.destroy()
        
        # Create a frame to center the buttons
        frame = tk.Frame(self.root)
        frame.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
        
        # Algorithm selection
        algorithm_frame = tk.Frame(frame)
        algorithm_frame.pack(pady=10)
        
        algorithm_label = tk.Label(algorithm_frame, text="Select Segmentation Algorithm:")
        algorithm_label.pack(side=tk.LEFT, padx=(0, 10))
        
        self.algorithm_var = tk.StringVar(value=self.segmentation_algorithm)
        
        fastsam_rb = tk.Radiobutton(algorithm_frame, text="FastSAM", variable=self.algorithm_var, value="fastsam")
        fastsam_rb.pack(side=tk.LEFT)
        
        sam_rb = tk.Radiobutton(algorithm_frame, text="SAM", variable=self.algorithm_var, value="sam")
        sam_rb.pack(side=tk.LEFT)
        
        # Buttons for selecting folders
        tk.Button(frame, text="Select Input Images Folder", 
                  command=self.select_input_images_folder, 
                  width=40).pack(pady=10)
        
        tk.Button(frame, text="Select Output Folder", 
                  command=self.select_save_folder, 
                  width=40).pack(pady=10)
        
        # Start button
        tk.Button(frame, text="Start", 
                  command=self.initialize_algorithm, 
                  width=40).pack(pady=10)
    
    def select_input_images_folder(self):
        self.input_images_folder = filedialog.askdirectory(title="Select Original Images Folder")
    
    def select_save_folder(self):
        self.save_folder = filedialog.askdirectory(title="Select Save Folder")
    
    def initialize_algorithm(self):
        # Get the selected algorithm
        self.segmentation_algorithm = self.algorithm_var.get()
        
        # Check if folders are selected
        if not all([self.input_images_folder, self.save_folder]):
            messagebox.showerror("Error", "Please select both input and output folders.")
            return
        
        # Initialize models based on algorithm
        try:
            if self.segmentation_algorithm == 'fastsam':
                # Initialize FastSAM model
                self.model_weights = os.path.join("weights", "FastSAM.pt")
                self.model = FastSAM(self.model_weights)
                print(f"FastSAM model loaded from {self.model_weights}")
                
            elif self.segmentation_algorithm == 'sam':
                # Initialize SAM model
                self.model_weights = os.path.join("weights", "sam_vit_b_01ec64.pth")
                self.model_type = "vit_b"  # Could be vit_h, vit_l, vit_b depending on weights
                
                # Initialize SAM model
                self.sam = sam_model_registry[self.model_type](checkpoint=self.model_weights)
                self.sam.to(self.device)
                self.predictor = SamPredictor(self.sam)
                print(f"SAM model loaded from {self.model_weights}")
                
            # Continue with loading images
            self.load_images_with_scale()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to initialize model: {str(e)}")
            print(f"Error initializing model: {str(e)}")
    
    def load_images(self):
        # Get images from the images folder
        try:
            self.image_files = [f for f in os.listdir(self.input_images_folder) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
            if self.image_files:
                self.current_image_index = 0
                self.show_current_image()
            else:
                messagebox.showerror("Error", "No images found in the selected folder.")
        except Exception as e:
            messagebox.showerror("Error", f"Error loading images: {e}")

    def load_images_with_scale(self):
        # Initialize a dictionary to store scales
        self.image_scales = {}
        
        # Get images from the white background images folder
        try:
            self.image_files = [f for f in os.listdir(self.input_images_folder) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
            if self.image_files:
                # Compute scale for each image in the original images folder
                for image_file in self.image_files:
                    original_image_path = os.path.join(self.input_images_folder, image_file)
                    if os.path.exists(original_image_path):
                        # Load the image and compute scale
                        original_img = cv2.imread(original_image_path)
                        pixel_length, text = get_scale(original_img, scale_type=self.scale_type)
                        scale = extract_float(text) / pixel_length
                        
                        # Store the scale in the dictionary
                        self.image_scales[image_file] = scale
                    else:
                        # Handle missing original image
                        self.image_scales[image_file] = None
                
                # Set the current image index and display the first image
                self.current_image_index = 0
                self.show_current_image()
            else:
                messagebox.showerror("Error", "No images found in the selected folder.")
        except Exception as e:
            messagebox.showerror("Error", f"Error loading images: {e}")
    
    def process_and_display_images(self, input_image, scale):
        """Process the input image and display processed results"""
        # Process the image to compute new results
        self.midpoints1 = quad_fit(input_image, method='rect')
        self.midpoints2 = max_min_fit(input_image)
        self.midpoints3 = quad_fit(input_image, method='para')
        self.midpoints4 = quad_fit(input_image, method='quad')
        
        img_to_save1, _ , _ = draw_midpoints_fit(self.current_image, self.midpoints1, scale=scale)
        img_to_save2, _ , _ = draw_midpoints_fit(self.current_image, self.midpoints2, scale=scale)
        img_to_save3, _ , _ = draw_midpoints_fit(self.current_image, self.midpoints3, scale=scale)
        img_to_save4, _ , _ = draw_midpoints_fit(self.current_image, self.midpoints4, scale=scale)
        
        # Display processed images
        self.display_processed_images([img_to_save1, img_to_save2, img_to_save3, img_to_save4])

    def display_processed_images(self, processed_images):
        """Display the four processed images as buttons."""
        # Clear existing widgets
        for widget in self.root.winfo_children():
            widget.destroy()
        
        # Display image info
        image_file = self.image_files[self.current_image_index]
        image_name = os.path.splitext(image_file)[0]
        tk.Label(self.root, text=f"Image {self.current_image_index + 1} of {len(self.image_files)} - {image_name}").pack()
        
        # Resize and display processed images
        frame_top = tk.Frame(self.root)
        frame_top.pack(pady=10)
        
        frame_bottom = tk.Frame(self.root)
        frame_bottom.pack(pady=10)
        
        # Map each image to its respective midpoints
        midpoints_list = [self.midpoints1, self.midpoints2, self.midpoints3, self.midpoints4]
        
        for i, (img, midpoints) in enumerate(zip(processed_images, midpoints_list)):
            # Resize image for display
            h, w = img.shape[:2]
            max_width, max_height = 600, 400
            scale = min(max_width / w, max_height / h)
            new_size = (int(w * scale), int(h * scale))
            
            img_resized = cv2.resize(img, new_size)
            img_photo = ImageTk.PhotoImage(Image.fromarray(img_resized))
            
            # Alternate between top and bottom frames
            parent_frame = frame_top if i < 2 else frame_bottom
            
            # Create button for each image
            button = tk.Button(
                parent_frame, 
                image=img_photo, 
                command=lambda img=img, midpoints=midpoints: self.on_image_click(img, midpoints)
            )
            button.image = img_photo
            button.pack(side=tk.LEFT, padx=20)
        
        # Navigation buttons
        nav_frame = tk.Frame(self.root)
        nav_frame.pack(pady=10)
        
        # Back button (instead of Previous)
        back_button = tk.Button(nav_frame, text="Back", 
                               command=self.show_current_image)
        back_button.pack(side=tk.LEFT, padx=10)
        
        # Skip button (renamed from Next) - just navigate without saving
        skip_button = tk.Button(nav_frame, text="Skip", 
                               command=self.next_image_without_saving, 
                               state=tk.DISABLED if self.current_image_index == len(self.image_files) - 1 else tk.NORMAL)
        skip_button.pack(side=tk.RIGHT, padx=10)

    def on_image_click(self, img, midpoints):
        """Handle image click: zoom to full screen and set selected midpoints."""
        self.selected_midpoints = midpoints
        self.display_single_image(img)
    
    def add_color_text_to_image(self, img, size):
        """Add color and transparency text to an image."""
        # For display
        trans_code = '1' if self.current_transparency == 'Transparent' else '2'
        
        # Position the text in the middle right
        text_x = int(size[0] * 0.75)
        text_y_color = int(size[1] * 0.45)
        text_y_trans = int(size[1] * 0.5)
        
        # Draw text on the image
        color_text = f"Color: {self.current_color}"
        trans_text = f"Transparency: {trans_code}"
        
        cv2.putText(img, color_text, (text_x, text_y_color), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(img, trans_text, (text_x, text_y_trans), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        return img
    
    def handle_key_press(self, event):
        """Handle key presses to update color and transparency values."""
        key = event.char.upper()
        update_needed = False
        
        # Color keys
        if key in ['K', 'A', 'W', 'R', 'O', 'Y', 'G', 'B', 'V', 'P', 'N', 'M', 'C']:
            self.current_color = key
            update_needed = True
        
        # Transparency keys
        elif key == '1':
            self.current_transparency = 'Transparent'
            update_needed = True
        elif key == '2':
            self.current_transparency = 'Opaque'
            update_needed = True
        
        # If a valid key was pressed, update the display
        if update_needed:
            # Determine which mode we're in and update accordingly
            if hasattr(self, 'canvas') and self.canvas.winfo_exists():
                self.update_interactive_display()
            else:
                self.update_single_image_display()
    
    def update_single_image_display(self):
        """Update the single image display with new color and transparency."""
        if hasattr(self, 'color_trans_label') and self.color_trans_label.winfo_exists():
            # Update the text label
            trans_code = '1' if self.current_transparency == 'Transparent' else '2'
            self.color_trans_label.config(text=f"Color: {self.current_color}, Transparency: {trans_code}")
    
    def update_interactive_display(self):
        """Update the interactive display with new color and transparency."""
        if hasattr(self, 'color_trans_label') and self.color_trans_label.winfo_exists():
            # Update the text label
            trans_code = '1' if self.current_transparency == 'Transparent' else '2'
            self.color_trans_label.config(text=f"Color: {self.current_color}, Transparency: {trans_code}")

    def display_single_image(self, selected_image):
        """Display the selected image full-screen with interaction option."""
        # Clear existing widgets
        for widget in self.root.winfo_children():
            widget.destroy()
        
        # Unbind any existing key events
        self.root.unbind('<Key>')

        # Compute color and transparency
        color_code, transparency = compute_particle_color(self.current_binary_image, cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB))
        self.current_color = color_code
        self.current_transparency = transparency
        
        # Resize the selected image for full-screen display
        h, w = selected_image.shape[:2]
        max_width, max_height = 1200, 800
        scale = min(max_width / w, max_height / h)
        new_size = (int(w * scale), int(h * scale))
        
        # Store the original image and display size for later updates
        self.current_displayed_image = selected_image.copy()
        self.current_displayed_size = new_size
        
        img_resized = cv2.resize(selected_image, new_size)
        img_photo = ImageTk.PhotoImage(Image.fromarray(img_resized))

        # Display image info
        tk.Label(self.root, text=f"Image {self.current_image_index + 1} of {len(self.image_files)} - {self.image_name}").pack()
        
        # Display color and transparency info
        trans_code = '1' if self.current_transparency == 'Transparent' else '2'
        self.color_trans_label = tk.Label(self.root, text=f"Color: {self.current_color}, Transparency: {trans_code}")
        self.color_trans_label.pack()
        
        # Create a frame for the image
        image_frame = tk.Frame(self.root)
        image_frame.pack(pady=20)

        # Display the image
        self.image_label = tk.Label(image_frame, image=img_photo)
        self.image_label.image = img_photo
        self.image_label.pack()

        # Add interactive mode button
        interactive_button = tk.Button(
            self.root,
            text="Enable Interactive Mode",
            command=lambda: self.enable_interactive_mode(selected_image, self.selected_midpoints)
        )
        interactive_button.pack(pady=10)

        # Navigation and back buttons frame
        nav_frame = tk.Frame(self.root)
        nav_frame.pack(pady=10)

        # Back button to return to the four-image screen
        back_button = tk.Button(nav_frame, text="Back", 
                            command=lambda: self.process_and_display_images(self.current_binary_image, self.scale))
        back_button.pack(side=tk.LEFT, padx=10)

        # Next and Save button (or just Save for the last image)
        is_last_image = self.current_image_index == len(self.image_files) - 1
        next_save_button = tk.Button(
            nav_frame,
            text="Save" if is_last_image else "Next and Save",
            command=self.save_and_next_image,
            state=tk.NORMAL  # Always enabled
        )
        next_save_button.pack(side=tk.RIGHT, padx=10)
        
        # Add key bindings for color and transparency
        self.root.bind('<Key>', self.handle_key_press)

    def enable_interactive_mode(self, image, midpoints):
        """Enable the user to interact with the lines using the cursor."""
        # Clear existing widgets
        for widget in self.root.winfo_children():
            widget.destroy()
        
        # Unbind any existing key events
        self.root.unbind('<Key>')

        # Display image info
        tk.Label(self.root, text=f"Image {self.current_image_index + 1} of {len(self.image_files)} - {self.image_name}").pack()
        
        # Display color and transparency info
        trans_code = '1' if self.current_transparency == 'Transparent' else '2'
        self.color_trans_label = tk.Label(self.root, text=f"Color: {self.current_color}, Transparency: {trans_code}")
        self.color_trans_label.pack()
        
        # Resize the image for display
        h, w = image.shape[:2]
        max_width, max_height = 1200, 800
        scale = min(max_width / w, max_height / h)
        new_size = (int(w * scale), int(h * scale))
        resized_image = cv2.resize(image, new_size)
        
        # Create a frame to hold the canvas with padding to match previous screen
        canvas_frame = tk.Frame(self.root)
        canvas_frame.pack(pady=20)
        
        # Set up canvas with the exact image dimensions
        self.canvas = tk.Canvas(canvas_frame, width=new_size[0], height=new_size[1], bg="white", highlightthickness=0)
        self.canvas.pack()
        
        # Convert the resized image to a format tkinter can use
        display_image = ImageTk.PhotoImage(Image.fromarray(resized_image))
        
        # Place the image at the top-left (no need for centering since canvas is exactly image size)
        canvas_image = self.canvas.create_image(0, 0, anchor=tk.NW, image=display_image)
        self.canvas.image = display_image  # Keep reference to prevent garbage collection

        # Scale midpoints to the new image size
        scaled_midpoints = [(int(x * scale), int(y * scale)) for x, y in midpoints]

        # Draw two lines connecting specific points
        line1_id = self.canvas.create_line(
            *scaled_midpoints[0], *scaled_midpoints[2],
            fill="magenta",
            width=2
        )
        line2_id = self.canvas.create_line(
            *scaled_midpoints[1], *scaled_midpoints[3],
            fill="magenta",
            width=2
        )

        # Store references for points (to update their positions later)
        point_ids = []

        # Function to update the lines and points
        def update_lines_and_points():
            self.canvas.coords(line1_id, *scaled_midpoints[0], *scaled_midpoints[2])
            self.canvas.coords(line2_id, *scaled_midpoints[1], *scaled_midpoints[3])
            for idx, (x, y) in enumerate(scaled_midpoints):
                x1, y1, x2, y2 = x - 5, y - 5, x + 5, y + 5
                self.canvas.coords(point_ids[idx], x1, y1, x2, y2)

        # Function to handle dragging points
        def drag_point(event, point_index):
            x, y = event.x, event.y
            scaled_midpoints[point_index] = (x, y)
            update_lines_and_points()

        # Create draggable points for midpoints
        for idx, (x, y) in enumerate(scaled_midpoints):
            point_id = self.canvas.create_oval(x - 5, y - 5, x + 5, y + 5, fill="magenta", outline="magenta")
            self.canvas.tag_bind(
                point_id,
                "<B1-Motion>",
                lambda event, i=idx: drag_point(event, i)
            )
            point_ids.append(point_id)

        # Function to handle "Back" button click
        def on_back_click():
            # Return to the four-image view without saving
            self.root.unbind('<Key>')  # Unbind key events
            self.process_and_display_images(self.current_binary_image, self.scale)

        # Function to save and go to next image
        def on_save_next_click():
            # Save the updated midpoints (convert back to original scale)
            final_midpoints = np.array([(int(x / scale), int(y / scale)) for x, y in scaled_midpoints])
            self.selected_midpoints = final_midpoints  # Update the selected midpoints with the interactive changes
            self.root.unbind('<Key>')  # Unbind key events
            self.save_and_next_image()

        # Button frame
        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=10)

        # Back button to return to the four-image view
        back_button = tk.Button(button_frame, text="Back", command=on_back_click)
        back_button.pack(side=tk.LEFT, padx=10)

        # Next and Save button (or just Save for the last image)
        is_last_image = self.current_image_index == len(self.image_files) - 1
        next_save_button = tk.Button(
            button_frame, 
            text="Save" if is_last_image else "Next and Save", 
            command=on_save_next_click,
            state=tk.NORMAL  # Always enabled
        )
        next_save_button.pack(side=tk.RIGHT, padx=10)
        
        # Bind key events
        self.root.bind('<Key>', self.handle_key_press)


    def show_current_image(self):
        """Display the current image with interactive point selection."""
        # Clear existing widgets and points
        for widget in self.root.winfo_children():
            widget.destroy()
        
        self.points = []
        self.point_markers = []
        
        # Get current image
        image_file = self.image_files[self.current_image_index]
        image_path = os.path.join(self.input_images_folder, image_file)
        self.image_name = os.path.splitext(image_file)[0]
        
        # Load the image with OpenCV
        self.current_image = cv2.imread(image_path)
        # Convert BGR to RGB for processing
        self.current_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
        
        # For SAM, we need to set the image in the predictor
        if self.segmentation_algorithm == 'sam':
            self.predictor.set_image(self.current_image)
        
        # Create empty binary image with same dimensions
        self.current_binary_image = np.zeros((self.current_image.shape[0], self.current_image.shape[1]), dtype=np.uint8)
        
        # Resize image for display
        h, w = self.current_image.shape[:2]
        max_width, max_height = 800, 600
        scale = min(max_width / w, max_height / h)
        new_size = (int(w * scale), int(h * scale))
        
        self.display_scale = scale  # Store the display scale for point conversion
        image_resized = cv2.resize(self.current_image, new_size)
        
        # Display image info
        tk.Label(self.root, text=f"Image {self.current_image_index + 1} of {len(self.image_files)} - {self.image_name}").pack(pady=(10, 0))
        tk.Label(self.root, text=f"Using {self.segmentation_algorithm.upper()} - Click on the image to add points").pack(pady=(0, 10))
        
        # Create a frame for the image
        image_frame = tk.Frame(self.root)
        image_frame.pack(pady=10)
        
        # Create a canvas for the image to capture clicks
        self.canvas = tk.Canvas(image_frame, width=new_size[0], height=new_size[1], highlightthickness=1, highlightbackground="black")
        self.canvas.pack()
        
        # Convert to PIL Image and then to PhotoImage for tkinter
        pil_image = Image.fromarray(image_resized)
        self.image_photo = ImageTk.PhotoImage(pil_image)
        
        # Display the image on the canvas
        self.canvas_image = self.canvas.create_image(0, 0, anchor=tk.NW, image=self.image_photo)
        
        # Bind click event to the canvas for adding points
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        
        # Get the scale factor for the images
        self.scale = self.image_scales[image_file]
        
        # Add button frame
        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=10)
        
        # Clear button
        clear_button = tk.Button(button_frame, text="Clear", command=self.clear_points)
        clear_button.pack(side=tk.LEFT, padx=10)
        
        # Previous button
        prev_button = tk.Button(button_frame, text="Previous", 
                                command=self.previous_image, 
                                state=tk.DISABLED if self.current_image_index == 0 else tk.NORMAL)
        prev_button.pack(side=tk.LEFT, padx=10)
        
        # Next button - goes to 4 image screen - should always be enabled
        next_button = tk.Button(button_frame, text="Next", 
                            command=lambda: self.process_and_display_images(self.current_binary_image, self.scale), 
                            state=tk.NORMAL)  # Always enabled
        next_button.pack(side=tk.RIGHT, padx=10)

    def on_canvas_click(self, event):
        """Handle clicks on the image to add points."""
        # Get the coordinates of the click
        x, y = event.x, event.y
        self.points.append((x, y))
        
        # Draw a marker for the point
        point_radius = 5
        marker = self.canvas.create_oval(
            x - point_radius, y - point_radius, 
            x + point_radius, y + point_radius, 
            fill="magenta", outline="magenta", width=2
        )
        
        # Store the marker ID
        self.point_markers.append(marker)
        
        print(f"Added point at ({x}, {y}), total points: {len(self.points)}")
        
        # Automatically process segmentation with the new point
        self.process_with_points()

    def process_with_points(self):
        """Process the image using the selected algorithm with the selected points."""
        if not self.points:
            return
        
        # Convert points to the original image coordinates
        inv_scale = 1.0 / self.display_scale
        scaled_points = [(int(x * inv_scale), int(y * inv_scale)) for x, y in self.points]
        
        # Process image using the selected algorithm
        if self.segmentation_algorithm == 'fastsam':
            initial_binary_image = fastsam_method(self.model, self.device, self.current_image, points=scaled_points)
        elif self.segmentation_algorithm == 'sam':
            initial_binary_image = sam_method(self.predictor, self.device, self.current_image, points=scaled_points)
        
        # Find contours in binary image
        contours, _ = cv2.findContours(initial_binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        if not contours:
            return
                
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Create a blank image and draw only the largest contour filled in
        clean_binary = np.zeros(initial_binary_image.shape, dtype=np.uint8)
        cv2.drawContours(clean_binary, [largest_contour], -1, 255, -1)  # -1 thickness means filled
        
        # Set this as our current binary image
        self.current_binary_image = clean_binary
        
        # Create a copy of the original image to draw contours on (for display)
        image_with_contour = self.current_image.copy()
        
        # Draw the contours on the original image for visualization
        cv2.drawContours(image_with_contour, [largest_contour], -1, (0, 255, 0), 10)
        
        # Resize for display
        h, w = image_with_contour.shape[:2]
        max_width, max_height = 800, 600
        scale = min(max_width / w, max_height / h)
        new_size = (int(w * scale), int(h * scale))
        
        image_resized = cv2.resize(image_with_contour, new_size)
        
        # Update the canvas with the new image
        pil_image = Image.fromarray(image_resized)
        self.image_photo = ImageTk.PhotoImage(pil_image)
        
        # Update the canvas image
        self.canvas.itemconfig(self.canvas_image, image=self.image_photo)
        
        # Redraw the points on top of the new image
        for marker in self.point_markers:
            self.canvas.delete(marker)
        
        self.point_markers = []
        
        for x, y in self.points:
            point_radius = 5
            marker = self.canvas.create_oval(
                x - point_radius, y - point_radius, 
                x + point_radius, y + point_radius, 
                fill="magenta", outline="magenta", width=2
            )
            self.point_markers.append(marker)

    def clear_points(self):
        """Clear all points and reset the view."""
        # Remove all point markers
        for marker in self.point_markers:
            self.canvas.delete(marker)
        
        # Clear the points list
        self.points = []
        self.point_markers = []
        
        # Reset the binary image
        self.current_binary_image = np.zeros((self.current_image.shape[0], self.current_image.shape[1]), dtype=np.uint8)
        
        # Reset the image to the original
        h, w = self.current_image.shape[:2]
        max_width, max_height = 800, 600
        scale = min(max_width / w, max_height / h)
        new_size = (int(w * scale), int(h * scale))
        
        image_resized = cv2.resize(self.current_image, new_size)
        pil_image = Image.fromarray(image_resized)
        self.image_photo = ImageTk.PhotoImage(pil_image)
        self.canvas.itemconfig(self.canvas_image, image=self.image_photo)
        
        print("Cleared all points and reset view")
    
    def next_image_without_saving(self):
        """Move to the next image without saving anything."""
        if self.current_image_index < len(self.image_files) - 1:
            self.current_image_index += 1
            self.show_current_image()
    
    def next_image(self):
        """Move to the next image if available."""
        if self.current_image_index < len(self.image_files) - 1:
            self.current_image_index += 1
            self.show_current_image()
    
    def save_and_next_image(self):
        # Save the current image with axis lines using the selected midpoints.
        if hasattr(self, 'selected_midpoints'):
            img_to_save, axis_length1, axis_length2 = draw_midpoints_fit(self.current_image, self.selected_midpoints, scale=self.scale)
            # Convert RGB back to BGR for saving with OpenCV
            img_to_save_bgr = cv2.cvtColor(img_to_save, cv2.COLOR_RGB2BGR)
            image_name = f'{self.image_name}.png'
            save_path = os.path.join(self.save_folder, image_name)
            cv2.imwrite(save_path, img_to_save_bgr)

        if axis_length1 <= axis_length2:
            axis_x, axis_y = axis_length1, axis_length2
        else:
            axis_x, axis_y = axis_length2, axis_length1
        
        # Extract plate number and hole with error handling
        plate_match = re.search(r"P(\d+)", self.image_name)
        plate_number = plate_match.group(1) if plate_match else "unknown"
        
        hole_match = re.search(r"-([A-Za-z]\d+)", self.image_name)
        plate_hole = hole_match.group(1) if hole_match else "unknown"
        
        data = {
            "id": self.current_image_index + 1,
            "image_name": self.image_name,
            "surface": 'plate',
            "plate_number": plate_number,
            "plate_hole": plate_hole,
            "axis_x": axis_x,
            "axis_y": axis_y,
            "size": compute_size_given_axis_len(axis_y),
            "colour": self.current_color,
            "transparency": self.current_transparency
        }
        data_filename = os.path.join(self.save_folder, f'{self.image_name}.json')
        with open(data_filename, 'w') as f:
            json.dump(data, f, indent=4)

        # Only proceed to next image if it's not the last one
        if self.current_image_index < len(self.image_files) - 1:
            self.next_image()
        else:
            # For the last image, return to the folder selection screen
            self.create_folder_selection_screen()
        
    def previous_image(self):
        """Move to the previous image if available."""
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self.show_current_image()

# Run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = ImageApp(root)
    root.mainloop()