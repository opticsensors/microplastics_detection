import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import os
from find_axis import *
from find_blob import *
from find_scale import *

class ImageApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Viewer")
        self.root.geometry("1400x960")
        
        # Attributes to store folder paths
        self.original_images_folder = None
        self.white_bg_images_folder = None
        self.save_folder = None
        
        # Image-related attributes
        self.image_files = []
        self.current_image_index = 0
        self.current_image = None
        
        # Initial screen with buttons
        self.create_folder_selection_screen()  # This line ensures buttons are shown
    
    def create_folder_selection_screen(self):
        # Clear existing widgets
        for widget in self.root.winfo_children():
            widget.destroy()
        
        # Create a frame to center the buttons
        frame = tk.Frame(self.root)
        frame.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
        
        # Buttons for selecting folders
        tk.Button(frame, text="Select Original Images Folder", 
                  command=self.select_original_images_folder, 
                  width=40).pack(pady=10)
        
        tk.Button(frame, text="Select White Background Images Folder", 
                  command=self.select_white_bg_images_folder, 
                  width=40).pack(pady=10)
        
        tk.Button(frame, text="Select Save Folder", 
                  command=self.select_save_folder, 
                  width=40).pack(pady=10)
    
    def create_folder_selection_screen(self):
        # Clear existing widgets
        for widget in self.root.winfo_children():
            widget.destroy()
        
        # Buttons for selecting folders
        tk.Button(self.root, text="Select Original Images Folder", command=self.select_original_images_folder, width=40).pack(pady=20)
        tk.Button(self.root, text="Select White Background Images Folder", command=self.select_white_bg_images_folder, width=40).pack(pady=20)
        tk.Button(self.root, text="Select Save Folder", command=self.select_save_folder, width=40).pack(pady=20)
    
    def select_original_images_folder(self):
        self.original_images_folder = filedialog.askdirectory(title="Select Original Images Folder")
        self.check_all_folders_selected()
    
    def select_white_bg_images_folder(self):
        self.white_bg_images_folder = filedialog.askdirectory(title="Select White Background Images Folder")
        self.check_all_folders_selected()
    
    def select_save_folder(self):
        self.save_folder = filedialog.askdirectory(title="Select Save Folder")
        self.check_all_folders_selected()
    
    def check_all_folders_selected(self):
        if all([self.original_images_folder, self.white_bg_images_folder, self.save_folder]):
            self.load_images()
    
    def load_images(self):
        # Get images from the white background images folder
        try:
            self.image_files = [f for f in os.listdir(self.white_bg_images_folder) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
            if self.image_files:
                self.current_image_index = 0
                self.show_current_image()
            else:
                messagebox.showerror("Error", "No images found in the selected folder.")
        except Exception as e:
            messagebox.showerror("Error", f"Error loading images: {e}")
    
    def process_and_display_images(self, input_image, ):
        """Process the input image and display processed results"""
        # Save the input image
        image_name = f"blob_{self.image_name}.png"
        save_path = os.path.join(self.save_folder, image_name)
        cv2.imwrite(save_path, input_image)
        
        # Process the image to compute new results
        self.midpoints1 = quad_fit(input_image, method='rect')
        self.midpoints2 = max_min_fit(input_image)
        self.midpoints3 = quad_fit(input_image, method='para')
        self.midpoints4 = quad_fit(input_image, method='quad')
        
        img_to_save1 = draw_midpoints_fit(self.current_image, self.midpoints1, scale=1)
        img_to_save2 = draw_midpoints_fit(self.current_image, self.midpoints2, scale=1)
        img_to_save3 = draw_midpoints_fit(self.current_image, self.midpoints3, scale=1)
        img_to_save4 = draw_midpoints_fit(self.current_image, self.midpoints4, scale=1)
        
        # Display processed images
        self.display_processed_images([img_to_save1, img_to_save2, img_to_save3, img_to_save4])

    def display_processed_images(self, processed_images):
        """Display the four processed images as buttons."""
        # Clear existing widgets
        for widget in self.root.winfo_children():
            widget.destroy()
        
        # Display image info
        tk.Label(self.root, text=f"Image {self.current_image_index + 1} of {len(self.image_files)}").pack()
        
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
            img_photo = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)))
            
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
        
        # Previous button
        prev_button = tk.Button(nav_frame, text="Previous", 
                                command=self.previous_image, 
                                state=tk.DISABLED if self.current_image_index == 0 else tk.NORMAL)
        prev_button.pack(side=tk.LEFT, padx=10)
        
        # Next button
        next_button = tk.Button(nav_frame, text="Next", 
                                command=self.next_image, 
                                state=tk.DISABLED if self.current_image_index == len(self.image_files) - 1 else tk.NORMAL)
        next_button.pack(side=tk.RIGHT, padx=10)

    def on_image_click(self, img, midpoints):
        """Handle image click: zoom to full screen and set selected midpoints."""
        self.selected_midpoints = midpoints
        self.display_single_image(img)


    def display_single_image(self, selected_image):
        """Display the selected image full-screen with interaction option."""
        # Clear existing widgets
        for widget in self.root.winfo_children():
            widget.destroy()

        # Resize the selected image for full-screen display
        h, w = selected_image.shape[:2]
        max_width, max_height = 1200, 800  # Reduced size to leave space for buttons
        scale = min(max_width / w, max_height / h)
        new_size = (int(w * scale), int(h * scale))

        img_resized = cv2.resize(selected_image, new_size)
        img_photo = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)))

        # Create a frame for the image to allow space for buttons
        image_frame = tk.Frame(self.root)
        image_frame.pack(pady=20)

        # Display the image
        label = tk.Label(image_frame, image=img_photo)
        label.image = img_photo
        label.pack()

        # Add interactive mode button
        interactive_button = tk.Button(
            self.root,
            text="Enable Interactive Mode",
            command=lambda: self.enable_interactive_mode(selected_image, self.selected_midpoints)  # Pass relevant midpoints
        )
        interactive_button.pack(pady=10)

        # Navigation and back buttons frame
        nav_frame = tk.Frame(self.root)
        nav_frame.pack(pady=10)

        # Previous button
        prev_button = tk.Button(
            nav_frame,
            text="Previous",
            command=self.previous_image,
            state=tk.DISABLED if self.current_image_index == 0 else tk.NORMAL
        )
        prev_button.pack(side=tk.LEFT, padx=10)

        # Back button to return to the four-image screen
        back_button = tk.Button(nav_frame, text="Back", command=self.show_current_image)
        back_button.pack(side=tk.LEFT, padx=10)

        # Next button
        next_button = tk.Button(
            nav_frame,
            text="Next",
            command=self.next_image,
            state=tk.DISABLED if self.current_image_index == len(self.image_files) - 1 else tk.NORMAL
        )
        next_button.pack(side=tk.RIGHT, padx=10)

    def enable_interactive_mode(self, image, midpoints):
        """Enable the user to interact with the lines using the cursor."""
        # Clear existing widgets
        for widget in self.root.winfo_children():
            widget.destroy()

        # Set up canvas for image and interactivity
        canvas = tk.Canvas(self.root, width=1200, height=800, bg="white")
        canvas.pack(fill=tk.BOTH, expand=True)
        
        # Resize the image for display
        h, w = image.shape[:2]
        max_width, max_height = 1200, 800
        scale = min(max_width / w, max_height / h)
        new_size = (int(w * scale), int(h * scale))
        resized_image = cv2.resize(image, new_size)
        
        # Convert the resized image to a format tkinter can use
        display_image = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)))
        canvas.create_image(0, 0, anchor=tk.NW, image=display_image)
        canvas.image = display_image  # Keep reference to prevent garbage collection

        # Scale midpoints to the new image size
        scaled_midpoints = [(int(x * scale), int(y * scale)) for x, y in midpoints]

        # Draw two lines connecting specific points
        line1_id = canvas.create_line(
            *scaled_midpoints[0], *scaled_midpoints[2],
            fill="magenta",
            width=2
        )
        line2_id = canvas.create_line(
            *scaled_midpoints[1], *scaled_midpoints[3],
            fill="magenta",
            width=2
        )

        # Store references for points (to update their positions later)
        point_ids = []

        # Function to update the lines and points
        def update_lines_and_points():
            canvas.coords(line1_id, *scaled_midpoints[0], *scaled_midpoints[2])
            canvas.coords(line2_id, *scaled_midpoints[1], *scaled_midpoints[3])
            for idx, (x, y) in enumerate(scaled_midpoints):
                x1, y1, x2, y2 = x - 5, y - 5, x + 5, y + 5
                canvas.coords(point_ids[idx], x1, y1, x2, y2)

        # Function to handle dragging points
        def drag_point(event, point_index):
            x, y = event.x, event.y
            scaled_midpoints[point_index] = (x, y)
            update_lines_and_points()

        # Create draggable points for midpoints
        for idx, (x, y) in enumerate(scaled_midpoints):
            point_id = canvas.create_oval(x - 5, y - 5, x + 5, y + 5, fill="magenta", outline="magenta")
            canvas.tag_bind(
                point_id,
                "<B1-Motion>",
                lambda event, i=idx: drag_point(event, i)
            )
            point_ids.append(point_id)

        # Function to handle "Back" button click
        def on_back_click():
            # Save the updated midpoints (convert back to original scale)
            self.final_midpoints = np.array([(int(x / scale), int(y / scale)) for x, y in scaled_midpoints])
            img_to_save = draw_midpoints_fit(self.current_image, self.final_midpoints, scale=1)
            image_name = f'axis_{self.image_name}.png'
            save_path = os.path.join(self.save_folder, image_name)
            cv2.imwrite(save_path, img_to_save)
            self.show_current_image()

        # Back button to return to the four-image view
        back_button = tk.Button(self.root, text="Back", command=on_back_click)
        back_button.pack(pady=10)


    def show_current_image(self):
        # Clear existing widgets
        for widget in self.root.winfo_children():
            widget.destroy()
        
        # Get current image
        image_file = self.image_files[self.current_image_index]
        image_path = os.path.join(self.white_bg_images_folder, image_file)
        self.image_name = os.path.splitext(image_file)[0]
        
        # Load the image with OpenCV
        self.current_image = cv2.imread(image_path)
        
        # Compute additional images
        resu1 = thresh_method(self.current_image)
        resu2 = sobel_method(self.current_image)
        
        # Resize images for display
        h, w = self.current_image.shape[:2]
        max_width, max_height = 600, 400
        scale = min(max_width / w, max_height / h)
        new_size = (int(w * scale), int(h * scale))
        
        image_resized = cv2.resize(self.current_image, new_size)
        resu1_resized = cv2.resize(resu1, new_size)
        resu2_resized = cv2.resize(resu2, new_size)
        
        # Convert to PIL Image for tkinter
        original_photo = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)))
        resu1_photo = ImageTk.PhotoImage(Image.fromarray(resu1_resized))
        resu2_photo = ImageTk.PhotoImage(Image.fromarray(resu2_resized))
        
        # Display image info
        tk.Label(self.root, text=f"Image {self.current_image_index + 1} of {len(self.image_files)}").pack()
        
        # Display original image
        original_label = tk.Button(
            self.root,
            image=original_photo,
        )
        original_label.image = original_photo
        original_label.pack(pady=10)
        
        # Display processed images
        frame = tk.Frame(self.root)
        frame.pack()
        
        resu1_label = tk.Button(
            frame,
            image=resu1_photo,
            command=lambda img=resu1: self.process_and_display_images(img, )
        )
        resu1_label.image = resu1_photo
        resu1_label.pack(side=tk.LEFT, padx=20)
        
        resu2_label = tk.Button(
            frame,
            image=resu2_photo,
            command=lambda img=resu2: self.process_and_display_images(img, )
        )
        resu2_label.image = resu2_photo
        resu2_label.pack(side=tk.RIGHT, padx=20)
        
        # Navigation buttons
        nav_frame = tk.Frame(self.root)
        nav_frame.pack(pady=10)
        
        # Previous button
        prev_button = tk.Button(nav_frame, text="Previous", 
                                command=self.previous_image, 
                                state=tk.DISABLED if self.current_image_index == 0 else tk.NORMAL)
        prev_button.pack(side=tk.LEFT, padx=10)
        
        # Next button
        next_button = tk.Button(nav_frame, text="Next", 
                                command=self.next_image, 
                                state=tk.DISABLED if self.current_image_index == len(self.image_files) - 1 else tk.NORMAL)
        next_button.pack(side=tk.RIGHT, padx=10)
    
    def next_image(self):
        """Move to the next image if available."""
        if self.current_image_index < len(self.image_files) - 1:
            self.current_image_index += 1
            self.show_current_image()
    
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