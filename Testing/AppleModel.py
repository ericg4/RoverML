import cv2
import numpy as np
import torch
from PIL import Image
import depth_pro
from matplotlib import pyplot as plt

def get_torch_device():
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    return device

# Load model and preprocessing transform
model, transform = depth_pro.create_model_and_transforms(
    device=get_torch_device(),
    precision=torch.half,
)
model.eval()

# Initialize the camera
cap = cv2.VideoCapture(0)

# Initialize matplotlib figure
plt.ion()
fig = plt.figure(figsize=(12, 5))
ax_rgb = fig.add_subplot(121)
ax_depth = fig.add_subplot(122)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Display the live camera feed
    cv2.imshow('Press SPACE to capture', frame)

    # Wait for spacebar press (ASCII 32)
    key = cv2.waitKey(1) & 0xFF
    if key == 32:  # SPACE key
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(rgb_frame)
        
        # Process image through model
        image_tensor = transform(pil_image)
        
        # Run inference
        prediction = model.infer(image_tensor)
        depth = prediction["depth"].detach().cpu().numpy().squeeze()
        
        # Calculate inverse depth for visualization
        inverse_depth = 1 / depth
        max_invdepth_vizu = min(inverse_depth.max(), 1 / 0.1)
        min_invdepth_vizu = max(1 / 250, inverse_depth.min())
        inverse_depth_normalized = (inverse_depth - min_invdepth_vizu) / (
            max_invdepth_vizu - min_invdepth_vizu
        )

        # Display results
        ax_rgb.clear()
        ax_depth.clear()
        ax_rgb.imshow(rgb_frame)
        ax_depth.imshow(inverse_depth_normalized, cmap='turbo')
        ax_rgb.set_title('Original Image')
        ax_depth.set_title('Depth Map')
        plt.draw()
        plt.pause(0.001)
    
    elif key == 27:  # ESC key
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
plt.close()
