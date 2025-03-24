import cv2
import numpy as np
from ultralytics import FastSAM
import cv2
import numpy as np
from segment_anything import sam_model_registry, SamPredictor

def fastsam_method(model, device, input_image, points=None):
    """Process the input image using FastSAM to create a binary mask"""
    # Convert RGB to BGR for FastSAM (if image is in RGB format)
    if len(input_image.shape) == 3 and input_image.shape[2] == 3:
        input_bgr = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)
    else:
        input_bgr = input_image
        
    # Process with FastSAM
    try:
        # If no points provided, use the center point
        if points is None or len(points) == 0:
            h, w = input_image.shape[:2]
            points = [(w//2, h//2)]
        
        # Create labels (all 1 for positive points)
        labels = [1] * len(points)
        
        # Run FastSAM
        results = model(input_bgr, device=device, retina_masks=True, 
                        points=points, labels=labels, imgsz=512,) #conf=0.4, iou=0.9)
        
        # Get the mask - take only the first one if there are multiple
        if results[0].masks is not None and len(results[0].masks.data) > 0:
            # Get the first mask
            mask_tensor = results[0].masks.data[0]
            
            # Convert to numpy and reshape to match processing image dimensions
            mask_shape = (results[0].masks.shape[1], results[0].masks.shape[2])
            mask_np = mask_tensor.cpu().numpy().reshape(mask_shape)
            
            # Make sure the mask is binary and has the right size
            binary_mask = (mask_np * 255).astype(np.uint8)
            
            # Resize to match the input image dimensions
            if binary_mask.shape[:2] != input_image.shape[:2]:
                binary_mask = cv2.resize(binary_mask, (input_image.shape[1], input_image.shape[0]), 
                                         interpolation=cv2.INTER_NEAREST)
            
            return binary_mask
        else:
            print("No mask found with FastSAM")
            # Return an empty mask if no mask is found
            return np.zeros((input_image.shape[0], input_image.shape[1]), dtype=np.uint8)
            
    except Exception as e:
        print(f"Error in FastSAM processing: {str(e)}")
        # Return an empty mask on error
        return np.zeros((input_image.shape[0], input_image.shape[1]), dtype=np.uint8)
    

def sam_method(predictor, device, input_image, points=None):
    """Process the input image using SAM to create a binary mask"""
    # Convert RGB to BGR for processing if needed
    if len(input_image.shape) == 3 and input_image.shape[2] == 3:
        input_rgb = input_image  # Already in RGB format for SAM
    else:
        # Convert grayscale to RGB if needed
        input_rgb = cv2.cvtColor(input_image, cv2.COLOR_GRAY2RGB)
    
    # Process with SAM
    try:
        # If no points provided, use the center point
        if points is None or len(points) == 0:
            h, w = input_image.shape[:2]
            points = [(w//2, h//2)]
            labels = [1]  # Positive point
        else:
            # For user-clicked points, all are positive
            labels = [1] * len(points)
        
        # Convert points to numpy array
        points_array = np.array(points, dtype=np.float32)
        labels_array = np.array(labels, dtype=np.int32)
        
        # Run SAM prediction
        masks, _, _ = predictor.predict(
            point_coords=points_array,
            point_labels=labels_array,
            multimask_output=False  # Get a single mask
        )
        
        # Convert to binary mask (0/255)
        binary_mask = (masks[0] * 255).astype(np.uint8)
        
        return binary_mask
    
    except Exception as e:
        print(f"Error in SAM processing: {str(e)}")
        # Return an empty mask on error
        return np.zeros((input_image.shape[0], input_image.shape[1]), dtype=np.uint8)