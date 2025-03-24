import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import os
import numpy as np
import json

def draw_curved_axis(img, axis1_points, axis2_points, scale=1):
    """
    Draw two curved axes based on the given points and calculate their lengths.
    
    Parameters:
        img: The image to draw on
        axis1_points: List of points for the first axis (left mouse button)
        axis2_points: List of points for the second axis (right mouse button)
        scale: Scale factor to convert pixel distances to mm
        
    Returns:
        img_to_save: The image with axes drawn
        len1: Length of the first axis in mm
        len2: Length of the second axis in mm
    """
    img_to_save = img.copy()
    
    # Calculate lengths and draw curves
    len1 = 0
    len2 = 0
    
    # Draw and calculate length for axis 1 (left button points)
    if len(axis1_points) > 1:
        # For drawing a smooth curve, we connect the points with lines
        for i in range(len(axis1_points)-1):
            pt1 = axis1_points[i]
            pt2 = axis1_points[i+1]
            cv2.line(img_to_save, (int(pt1[0]), int(pt1[1])), 
                    (int(pt2[0]), int(pt2[1])), (0, 0, 255), thickness=3)
            # Add to length
            segment_length = np.linalg.norm(np.array(pt2) - np.array(pt1)) * scale
            len1 += segment_length
            
        # Display the length near the last point
        last_point = axis1_points[-1]
        cv2.putText(img_to_save, f"{len1:.2f} mm", (int(last_point[0])+20, int(last_point[1])-3),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                   
    # Draw and calculate length for axis 2 (right button points)
    if len(axis2_points) > 1:
        for i in range(len(axis2_points)-1):
            pt1 = axis2_points[i]
            pt2 = axis2_points[i+1]
            cv2.line(img_to_save, (int(pt1[0]), int(pt1[1])), 
                    (int(pt2[0]), int(pt2[1])), (255, 0, 0), thickness=3)
            # Add to length
            segment_length = np.linalg.norm(np.array(pt2) - np.array(pt1)) * scale
            len2 += segment_length
            
        # Display the length near the last point
        last_point = axis2_points[-1]
        cv2.putText(img_to_save, f"{len2:.2f} mm", (int(last_point[0])+20, int(last_point[1])-3),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
    
    return img_to_save, len1, len2

class CurvedAxisMeasurement:
    def __init__(self, root):
        self.root = root
        self.root.title("Curved Axis Measurement")
        self.root.geometry("1400x900")
        
        # Initialize variables
        self.input_image_path = None
        self.output_folder = None
        self.image = None
        self.scale = 1.0  # Default scale (pixels to mm)
        
        # Points for each axis
        self.axis1_points = []  # Left mouse button points
        self.axis2_points = []  # Right mouse button points
        
        # Point markers
        self.axis1_markers = []
        self.axis2_markers = []
        
        # Create initial screen
        self.create_initial_screen()
    
    def create_initial_screen(self):
        # Clear existing widgets
        for widget in self.root.winfo_children():
            widget.destroy()
        
        # Create a frame to center the buttons
        frame = tk.Frame(self.root)
        frame.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
        
        # Buttons for selecting image and folder
        tk.Button(frame, text="Select Input Image", 
                  command=self.select_input_image, 
                  width=40).pack(pady=10)
        
        tk.Button(frame, text="Select Output Folder", 
                  command=self.select_output_folder, 
                  width=40).pack(pady=10)
        
        # Scale entry
        scale_frame = tk.Frame(frame)
        scale_frame.pack(pady=10)
        
        tk.Label(scale_frame, text="Scale (mm/pixel):").pack(side=tk.LEFT, padx=(0, 10))
        self.scale_entry = tk.Entry(scale_frame, width=10)
        self.scale_entry.insert(0, str(self.scale))
        self.scale_entry.pack(side=tk.LEFT)
        
        # Start button
        tk.Button(frame, text="Start", 
                  command=self.start_measurement, 
                  width=40).pack(pady=20)
    
    def select_input_image(self):
        self.input_image_path = filedialog.askopenfilename(
            title="Select Input Image",
            filetypes=[("Image files", "*.jpg;*.jpeg;*.png"), ("All files", "*.*")]
        )
    
    def select_output_folder(self):
        self.output_folder = filedialog.askdirectory(title="Select Output Folder")
    
    def start_measurement(self):
        # Check if both image and folder are selected
        if not self.input_image_path or not self.output_folder:
            messagebox.showerror("Error", "Please select both input image and output folder.")
            return
        
        # Try to parse the scale
        try:
            self.scale = float(self.scale_entry.get())
            if self.scale <= 0:
                raise ValueError("Scale must be positive")
        except ValueError:
            messagebox.showerror("Error", "Invalid scale value. Please enter a positive number.")
            return
        
        # Load the image
        try:
            self.image = cv2.imread(self.input_image_path)
            if self.image is None:
                raise ValueError("Failed to load image")
            
            # Convert BGR to RGB for display
            self.image_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            
            # Show the image for measurement
            self.show_measurement_screen()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {str(e)}")
    
    def show_measurement_screen(self):
        # Clear existing widgets
        for widget in self.root.winfo_children():
            widget.destroy()
        
        # Reset points
        self.axis1_points = []
        self.axis2_points = []
        self.axis1_markers = []
        self.axis2_markers = []
        
        # Image info
        self.image_name = os.path.splitext(os.path.basename(self.input_image_path))[0]
        
        # Display image info
        tk.Label(self.root, text=f"Image: {self.image_name}").pack(pady=(10, 0))
        tk.Label(self.root, text="Left click to add points for Axis 1 (red), Right click for Axis 2 (blue)").pack(pady=(0, 10))
        
        # Resize image for display
        h, w = self.image_rgb.shape[:2]
        max_width, max_height = 1200, 700
        scale = min(max_width / w, max_height / h)
        self.display_scale = scale
        new_size = (int(w * scale), int(h * scale))
        
        image_resized = cv2.resize(self.image_rgb, new_size)
        
        # Create a frame for the image
        image_frame = tk.Frame(self.root)
        image_frame.pack(pady=10)
        
        # Create a canvas for the image to capture clicks
        self.canvas = tk.Canvas(image_frame, width=new_size[0], height=new_size[1], 
                               highlightthickness=1, highlightbackground="black")
        self.canvas.pack()
        
        # Convert to PIL Image and then to PhotoImage for tkinter
        pil_image = Image.fromarray(image_resized)
        self.image_photo = ImageTk.PhotoImage(pil_image)
        
        # Display the image on the canvas
        self.canvas_image = self.canvas.create_image(0, 0, anchor=tk.NW, image=self.image_photo)
        
        # Bind click events
        self.canvas.bind("<Button-1>", self.on_left_click)  # Left click for axis 1
        self.canvas.bind("<Button-3>", self.on_right_click)  # Right click for axis 2
        
        # Button frame
        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=10)
        
        # Clear button
        clear_button = tk.Button(button_frame, text="Clear", command=self.clear_points)
        clear_button.pack(side=tk.LEFT, padx=10)
        
        # Save button
        save_button = tk.Button(button_frame, text="Save", command=self.save_results)
        save_button.pack(side=tk.RIGHT, padx=10)
    
    def on_left_click(self, event):
        """Handle left-click to add points for axis 1."""
        # Get click coordinates
        x, y = event.x, event.y
        
        # Add point to axis 1
        self.axis1_points.append((x, y))
        
        # Draw marker
        point_radius = 5
        marker = self.canvas.create_oval(
            x - point_radius, y - point_radius, 
            x + point_radius, y + point_radius, 
            fill="magenta", outline="magenta", width=2
        )
        self.axis1_markers.append(marker)
        
        # Update the display
        self.update_display()
    
    def on_right_click(self, event):
        """Handle right-click to add points for axis 2."""
        # Get click coordinates
        x, y = event.x, event.y
        
        # Add point to axis 2
        self.axis2_points.append((x, y))
        
        # Draw marker
        point_radius = 5
        marker = self.canvas.create_oval(
            x - point_radius, y - point_radius, 
            x + point_radius, y + point_radius, 
            fill="magenta", outline="magenta", width=2
        )
        self.axis2_markers.append(marker)
        
        # Update the display
        self.update_display()
    
    def update_display(self):
        """Update the display with current axis lines."""
        if not (self.axis1_points or self.axis2_points):
            return
        
        # Convert display points to original image coordinates
        inv_scale = 1.0 / self.display_scale
        
        axis1_orig = [(int(x * inv_scale), int(y * inv_scale)) for x, y in self.axis1_points]
        axis2_orig = [(int(x * inv_scale), int(y * inv_scale)) for x, y in self.axis2_points]
        
        # Generate image with axes
        img_with_axes, len1, len2 = draw_curved_axis(self.image_rgb, axis1_orig, axis2_orig, scale=self.scale)
        
        # Resize for display
        h, w = img_with_axes.shape[:2]
        new_size = (int(w * self.display_scale), int(h * self.display_scale))
        img_resized = cv2.resize(img_with_axes, new_size)
        
        # Update the canvas with the new image
        pil_image = Image.fromarray(img_resized)
        self.image_photo = ImageTk.PhotoImage(pil_image)
        self.canvas.itemconfig(self.canvas_image, image=self.image_photo)
        
        # Store the lengths
        self.axis1_length = len1
        self.axis2_length = len2
        
        # Redraw all markers on top of the updated image
        for marker in self.axis1_markers + self.axis2_markers:
            self.canvas.delete(marker)
        
        self.axis1_markers = []
        self.axis2_markers = []
        
        # Redraw axis 1 markers (red)
        for x, y in self.axis1_points:
            point_radius = 5
            marker = self.canvas.create_oval(
                x - point_radius, y - point_radius, 
                x + point_radius, y + point_radius, 
                fill="magenta", outline="magenta", width=2
            )
            self.axis1_markers.append(marker)
        
        # Redraw axis 2 markers (blue)
        for x, y in self.axis2_points:
            point_radius = 5
            marker = self.canvas.create_oval(
                x - point_radius, y - point_radius, 
                x + point_radius, y + point_radius, 
                fill="magenta", outline="magenta", width=2
            )
            self.axis2_markers.append(marker)
    
    def clear_points(self):
        """Clear all points and reset the display."""
        # Remove all markers
        for marker in self.axis1_markers + self.axis2_markers:
            self.canvas.delete(marker)
        
        # Clear point lists
        self.axis1_points = []
        self.axis2_points = []
        self.axis1_markers = []
        self.axis2_markers = []
        
        # Reset display to original image
        h, w = self.image_rgb.shape[:2]
        new_size = (int(w * self.display_scale), int(h * self.display_scale))
        img_resized = cv2.resize(self.image_rgb, new_size)
        
        pil_image = Image.fromarray(img_resized)
        self.image_photo = ImageTk.PhotoImage(pil_image)
        self.canvas.itemconfig(self.canvas_image, image=self.image_photo)
    
    def save_results(self):
        """Save the measured image and results."""
        if not (self.axis1_points or self.axis2_points):
            messagebox.showinfo("Info", "No measurements to save.")
            return
        
        try:
            # Convert display points to original image coordinates
            inv_scale = 1.0 / self.display_scale
            
            axis1_orig = [(int(x * inv_scale), int(y * inv_scale)) for x, y in self.axis1_points]
            axis2_orig = [(int(x * inv_scale), int(y * inv_scale)) for x, y in self.axis2_points]
            
            # Generate final image with axes
            img_with_axes, len1, len2 = draw_curved_axis(self.image_rgb, axis1_orig, axis2_orig, scale=self.scale)
            
            # Convert RGB back to BGR for saving with OpenCV
            img_bgr = cv2.cvtColor(img_with_axes, cv2.COLOR_RGB2BGR)
            
            # Save the image
            image_name = f'{self.image_name}.png'
            save_path = os.path.join(self.output_folder, image_name)
            cv2.imwrite(save_path, img_bgr)
            
            # Determine which axis is longer
            if len1 <= len2:
                axis_x, axis_y = len1, len2
            else:
                axis_x, axis_y = len2, len1
            
            # Create and save JSON data
            data = {
                "image_name": self.image_name,
                "axis_x": axis_x,
                "axis_y": axis_y,
                "axis1_points": axis1_orig,
                "axis2_points": axis2_orig,
                "scale": self.scale
            }
            
            json_filename = os.path.join(self.output_folder, f'{self.image_name}.json')
            with open(json_filename, 'w') as f:
                json.dump(data, f, indent=4)
            
            messagebox.showinfo("Success", f"Results saved to {self.output_folder}")
            
            # Return to initial screen
            self.create_initial_screen()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save results: {str(e)}")

# Run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = CurvedAxisMeasurement(root)
    root.mainloop()