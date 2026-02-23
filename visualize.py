import os
import cv2
import torch
import numpy as np
from model.cnn_model import ForgeryCNN

# ---------------------------
# Load Model
# ---------------------------
model = ForgeryCNN()
model.load_state_dict(torch.load("model.pth", map_location="cpu"))
model.eval()

# ---------------------------
# Storage for hooks
# ---------------------------
feature_maps = None
gradients = None

# ---------------------------
# Hooks
# ---------------------------
def forward_hook(module, input, output):
    global feature_maps
    feature_maps = output

def backward_hook(module, grad_input, grad_output):
    global gradients
    gradients = grad_output[0]

# ---------------------------
# Register hooks
# ---------------------------
target_layer = model.conv3  # adjust if needed
target_layer.register_forward_hook(forward_hook)
target_layer.register_full_backward_hook(backward_hook)

# ---------------------------
# Forgery Detection and Visualization Function
# ---------------------------
def visualize_and_generate_gradcam(image_path):

    # ---- FIX: make relative path reliable ----
    base_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(base_dir, image_path)

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    image_resized = cv2.resize(image, (224, 224))
    image_norm = image_resized / 255.0

    image_tensor = torch.tensor(
        image_norm.transpose(2, 0, 1)
    ).unsqueeze(0).float()

    # Forward pass
    output = model(image_tensor)

    # Binary classification score
    score = output.squeeze()
    model.zero_grad()
    score.backward()

    # Feature maps & gradients
    fmap = feature_maps.detach().numpy()[0]   # [C, H, W]
    grads = gradients.detach().numpy()[0]     # [C, H, W]

    # Global Average Pooling on gradients
    weights = np.mean(grads, axis=(1, 2))

    # Build Grad-CAM
    cam = np.zeros(fmap.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * fmap[i]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224, 224))
    cam = cam / (cam.max() + 1e-8)

    # ---------------------------
    # Convert CAM → circles
    # ---------------------------
    cam_uint8 = np.uint8(255 * cam)

    _, binary_map = cv2.threshold(
        cam_uint8, 160, 255, cv2.THRESH_BINARY
    )

    kernel = np.ones((5, 5), np.uint8)
    binary_map = cv2.morphologyEx(
        binary_map, cv2.MORPH_OPEN, kernel
    )

    contours, _ = cv2.findContours(
        binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    output_img = image_resized.copy()

    # --------- FIND AND DRAW TAMPERED REGIONS ----------
    for cnt in contours:
        if cv2.contourArea(cnt) < 100:
            continue

        (x, y), radius = cv2.minEnclosingCircle(cnt)
        center = (int(x), int(y))
        radius = int(radius)

        cv2.circle(output_img, center, radius, (0, 0, 255), 2)

    # --------- DISPLAY FORGERY INFORMATION ----------
    # Assuming "Forgery Detected" if the model predicts the image is tampered
    forgery_type = "Forgery Detected" if score.item() > 0.5 else "Not Forged"
    
    # Here, you could have logic to determine the forgery type (e.g., splicing, copy-move)
    # For demonstration, let's mock it as "Image Splicing" if score > 0.5
    detailed_forgery_type = "Image Splicing" if score.item() > 0.5 else "None"
    
    confidence = score.item() * 100  # Confidence as percentage

    # Smaller text for forgery information
    font_scale = 0.5  # Smaller text
    font_thickness = 1  # Thin text

    # Add forgery details on the image (smaller text)
    cv2.putText(
        output_img,
        f"Status: {forgery_type}",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (0, 0, 255),
        font_thickness
    )

    cv2.putText(
        output_img,
        f"Forgery Type: {detailed_forgery_type}",
        (20, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (0, 0, 255),
        font_thickness
    )

    cv2.putText(
        output_img,
        f"Confidence: {confidence:.2f}%",
        (20, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (0, 0, 255),
        font_thickness
    )

    # Show result
    cv2.imshow("Forgery Localization", output_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ---------------------------
# Test with Image Path (same as before)
# ---------------------------
visualize_and_generate_gradcam(
    "dataset/tampered/Tp_D_CRN_M_N_nat10161_nat10157_12081.jpg"
)

