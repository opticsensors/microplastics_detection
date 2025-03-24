import os
import glob
import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk

import torch
# Make sure you've done: pip install git+https://github.com/facebookresearch/segment-anything.git
from segment_anything import sam_model_registry, SamPredictor

class SegmentAnythingApp:
    """
    Main application that:
      1) Shows an initial screen to pick input folder, output folder, checkpoint.
      2) On 'Start', switches to the segmentation GUI.
    """
    def __init__(self, root):
        self.root = root
        self.root.title("Segment Anything - Single Blob Segmentation (Resized GUI)")
        self.root.geometry("1000x700")

        # User selections
        self.input_folder = None
        self.output_folder = None
        self.checkpoint_path = None

        # Model + device config
        self.model_type = "vit_b"  # can be "vit_l" or "vit_b" if you have those
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Create initial screen
        self.create_initial_screen()

    def create_initial_screen(self):
        # Clear everything on the window
        for widget in self.root.winfo_children():
            widget.destroy()

        frame = tk.Frame(self.root)
        frame.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

        btn_input = tk.Button(frame, text="Select Input Folder", width=40, command=self.select_input_folder)
        btn_input.pack(pady=5)

        btn_output = tk.Button(frame, text="Select Output Folder", width=40, command=self.select_output_folder)
        btn_output.pack(pady=5)

        btn_checkpoint = tk.Button(frame, text="Select SAM Checkpoint (.pth)", width=40, command=self.select_checkpoint_file)
        btn_checkpoint.pack(pady=5)

        btn_start = tk.Button(frame, text="Start", width=40, command=self.start_segmentation_gui)
        btn_start.pack(pady=20)

    def select_input_folder(self):
        folder = filedialog.askdirectory(title="Select Folder Containing Images")
        if folder:
            self.input_folder = folder

    def select_output_folder(self):
        folder = filedialog.askdirectory(title="Select Output Folder for Masks")
        if folder:
            self.output_folder = folder

    def select_checkpoint_file(self):
        file = filedialog.askopenfilename(
            title="Select SAM Checkpoint (.pth)",
            filetypes=[("PyTorch Model", "*.pth"), ("All files", "*.*")]
        )
        if file:
            self.checkpoint_path = file

    def start_segmentation_gui(self):
        """Validate user selections and switch to the segmentation interface."""
        if not self.input_folder or not self.output_folder or not self.checkpoint_path:
            messagebox.showwarning("Missing Info", "Please select the input folder, output folder, and checkpoint file.")
            return

        for widget in self.root.winfo_children():
            widget.destroy()

        # Launch the segmentation interface
        self.seg_gui = SegmentAnythingGUI(
            parent=self.root,
            input_folder=self.input_folder,
            output_folder=self.output_folder,
            checkpoint_path=self.checkpoint_path,
            model_type=self.model_type,
            device=self.device
        )

class SegmentAnythingGUI:
    """
    Segmentation GUI that:
      - Resizes images to a max dimension so it fits the screen.
      - On left-click, add a positive point; on right-click, add a negative point.
      - Shows a semi-transparent mask preview (Segment Anything).
      - 'Next' button saves the mask (0/255) and moves on.
    """
    def __init__(self, parent, input_folder, output_folder, checkpoint_path, model_type="vit_h", device="cpu"):
        self.parent = parent

        # Create 'Next' button as before
        self.btn_next = tk.Button(self.parent, text="Next", command=self.save_and_next)
        self.btn_next.pack(side=tk.BOTTOM, pady=5)

        # Bind the Enter key to save_and_next
        # Now pressing 'Enter' will do the same thing as clicking the 'Next' button.
        self.parent.bind("<Return>", lambda event: self.save_and_next())

        self.input_folder = input_folder
        self.output_folder = output_folder
        self.checkpoint_path = checkpoint_path
        self.model_type = model_type
        self.device = device

        # Collect images
        self.image_paths = sorted(glob.glob(os.path.join(self.input_folder, "*.*")))
        if not self.image_paths:
            messagebox.showerror("No Images Found", f"No images in {self.input_folder}")
            self.parent.quit()
            return

        # Initialize SAM
        sam = sam_model_registry[self.model_type](checkpoint=self.checkpoint_path)
        sam.to(self.device)
        self.predictor = SamPredictor(sam)

        # GUI elements
        self.canvas = tk.Canvas(self.parent, cursor="cross", bg="gray")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.btn_next = tk.Button(self.parent, text="Next", command=self.save_and_next)
        self.btn_next.pack(side=tk.BOTTOM, pady=5)

        # States
        self.current_index = 0
        self.original_image = None       # (H, W, C) in original scale
        self.resized_image = None        # (H', W', C) possibly scaled
        self.display_pil = None         # PIL image for display
        self.tk_image_on_canvas = None

        self.mask = None                 # (H, W) mask in original scale
        self.points = []                 # list of ((x_orig, y_orig), label)
        self.radius = 4                  # circle radius

        # For resizing
        self.scale_x = 1.0
        self.scale_y = 1.0
        self.max_width = 1000
        self.max_height = 700

        # Bind
        self.canvas.bind("<Button-1>", self.on_left_click)   # positive
        self.canvas.bind("<Button-3>", self.on_right_click)  # negative (Windows)
        # (Mac might need <Button-2>, or control-click, etc.)

        self.load_image(0)

    def load_image(self, idx):
        """Load the image at index, reset points/mask, handle resizing, display on canvas."""
        if idx < 0 or idx >= len(self.image_paths):
            return

        self.current_index = idx
        self.points.clear()
        self.mask = None

        img_path = self.image_paths[idx]
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            messagebox.showerror("Error", f"Failed to load image: {img_path}")
            return

        # Convert BGR -> RGB
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        self.original_image = img_rgb

        # We set the image in the SAM predictor (original scale)
        self.predictor.set_image(self.original_image)

        # Resize to fit our max_width / max_height if needed
        h, w, _ = img_rgb.shape
        scale = min(self.max_width / w, self.max_height / h, 1.0)  # never enlarge, only shrink if necessary
        new_w = int(w * scale)
        new_h = int(h * scale)

        # Store scale factors for coordinate mapping
        self.scale_x = w / new_w
        self.scale_y = h / new_h

        if scale < 1.0:
            # Resize the image to display size
            resized = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            # No resizing needed
            resized = img_rgb

        self.resized_image = resized
        self.display_pil = Image.fromarray(self.resized_image)
        self.show_image_on_canvas(self.display_pil)

    def show_image_on_canvas(self, pil_image):
        """Displays the given PIL image on the canvas."""
        self.canvas.delete("all")
        self.canvas.config(width=pil_image.width, height=pil_image.height)
        self.tk_image_on_canvas = ImageTk.PhotoImage(pil_image)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image_on_canvas)

    def on_left_click(self, event):
        """Left-click = positive point (include)."""
        # (x, y) are in the resized image's coordinates.
        # Convert them back to original image coords for SAM.
        x_orig = int(event.x * self.scale_x)
        y_orig = int(event.y * self.scale_y)
        self.points.append(((x_orig, y_orig), 1))
        self.update_mask()

    def on_right_click(self, event):
        """Right-click = negative point (exclude)."""
        x_orig = int(event.x * self.scale_x)
        y_orig = int(event.y * self.scale_y)
        self.points.append(((x_orig, y_orig), 0))
        self.update_mask()

    def update_mask(self):
        """Run SAM with the current points (in original coords) to get a single mask. Then preview."""
        if not self.points:
            self.mask = None
            self.draw_points_and_mask()
            return

        # Prepare coords/labels for SAM
        input_points = np.array([p for (p, _) in self.points], dtype=np.float32)
        input_labels = np.array([l for (_, l) in self.points], dtype=np.int32)

        masks, scores, logits = self.predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            multimask_output=False  # single mask
        )
        self.mask = masks[0]  # shape (H, W)

        self.draw_points_and_mask()

    def draw_points_and_mask(self):
        """Create a display version with the overlay of the mask + clicked points."""
        # If no image loaded yet, do nothing
        if self.resized_image is None:
            return

        # Start from the resized image
        disp_pil = Image.fromarray(self.resized_image)

        if self.mask is not None:
            # We have a full-resolution mask. Need to scale it down to display size
            # so we can overlay with alpha.
            H, W = self.mask.shape
            display_w, display_h = disp_pil.size
            mask_255 = (self.mask * 255).astype(np.uint8)
            mask_resized = cv2.resize(mask_255, (display_w, display_h), interpolation=cv2.INTER_NEAREST)

            # Convert to RGBA, apply alpha
            overlay = Image.fromarray(mask_resized).convert("RGBA")
            alpha_level = 100  # 0=transparent, 255=opaque
            datas = overlay.getdata()
            newData = []
            for item in datas:
                # item is the grayscale value in "L" or the R in "RGBA"
                val = item if isinstance(item, int) else item[0]
                if val > 0:
                    # Make it red with alpha
                    newData.append((255, 0, 0, alpha_level))
                else:
                    newData.append((0, 0, 0, 0))
            overlay.putdata(newData)

            disp_pil = Image.alpha_composite(disp_pil.convert("RGBA"), overlay)

        # Draw the points (need to scale from original to display coords).
        disp_np = np.array(disp_pil.convert("RGB"), dtype=np.uint8)
        for ((x_orig, y_orig), label) in self.points:
            x_disp = int(x_orig / self.scale_x)
            y_disp = int(y_orig / self.scale_y)
            color = (0, 255, 0) if label == 1 else (255, 0, 0)
            cv2.circle(disp_np, (x_disp, y_disp), self.radius, color, -1)

        final_pil = Image.fromarray(disp_np)
        self.show_image_on_canvas(final_pil)

    def save_and_next(self):
        """Save the binary mask of the current image, move to the next image."""
        if self.mask is not None:
            # Save it in 0/255 form
            mask_255 = (self.mask * 255).astype(np.uint8)
            filename = os.path.basename(self.image_paths[self.current_index])
            base, _ = os.path.splitext(filename)
            out_path = os.path.join(self.output_folder, f"{base}_mask.png")
            cv2.imwrite(out_path, mask_255)
            print(f"Saved mask -> {out_path}")

        # Next image
        next_idx = self.current_index + 1
        if next_idx < len(self.image_paths):
            self.load_image(next_idx)
        else:
            messagebox.showinfo("Done", "All images processed!")
            self.parent.quit()

def main():
    root = tk.Tk()
    app = SegmentAnythingApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
