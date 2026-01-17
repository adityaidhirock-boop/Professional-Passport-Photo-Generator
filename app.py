import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
from PIL import Image, ImageDraw
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
import io
import base64
import gdown  # ADDED FOR GOOGLE DRIVE

# ========== EXACT MODEL ARCHITECTURE FROM YOUR TRAINING ==========

class ResNetEncoder(nn.Module):
    """Pretrained ResNet encoder for U2-Net"""
    def __init__(self, encoder_name='resnet34', pretrained=True):
        super().__init__()
        if encoder_name == 'resnet34':
            resnet = models.resnet34(pretrained=pretrained)
            self.channels = [64, 64, 128, 256, 512]
        elif encoder_name == 'resnet50':
            resnet = models.resnet50(pretrained=pretrained)
            self.channels = [64, 256, 512, 1024, 2048]
        else:
            raise ValueError(f"Unsupported encoder: {encoder_name}")
        
        # Extract ResNet layers
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
    
    def forward(self, x):
        features = []
        
        # Stage 0
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        features.append(x)  # 64 channels, H/2 x W/2
        
        # Stage 1
        x = self.maxpool(x)
        x = self.layer1(x)
        features.append(x)  # 64 channels, H/4 x W/4
        
        # Stage 2
        x = self.layer2(x)
        features.append(x)  # 128 channels, H/8 x W/8
        
        # Stage 3
        x = self.layer3(x)
        features.append(x)  # 256 channels, H/16 x W/16
        
        # Stage 4
        x = self.layer4(x)
        features.append(x)  # 512 channels, H/32 x W/32
        
        return features

class HybridU2Net(nn.Module):
    """Completely Fixed Hybrid U2-Net with proper channel flow"""
    def __init__(self, encoder_name='resnet34', pretrained=True, num_classes=1):
        super().__init__()
        
        # Pretrained ResNet encoder
        self.encoder = ResNetEncoder(encoder_name, pretrained)
        encoder_channels = self.encoder.channels  # [64, 64, 128, 256, 512]
        
        # Decoder layers - EXACT same as your training
        # Stage 1: 512 -> 256 (from deepest encoder)
        self.dec4 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 2, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # Stage 2: (256+256=512) -> 128
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(512, 128, 2, stride=2),  # 256+256=512 input
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        # Stage 3: (128+128=256) -> 64
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(256, 64, 2, stride=2),   # 128+128=256 input
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Stage 4: (64+64=128) -> 32
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(128, 32, 2, stride=2),   # 64+64=128 input
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        # Final output: (32+64=96) -> 1
        self.final_conv = nn.Sequential(
            nn.Conv2d(96, 32, 3, padding=1),  # 32+64=96 input
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, 1),
            nn.Sigmoid()
        )
        
        # Side outputs for deep supervision
        self.side4 = nn.Conv2d(256, num_classes, 1)
        self.side3 = nn.Conv2d(128, num_classes, 1)
        self.side2 = nn.Conv2d(64, num_classes, 1)
        self.side1 = nn.Conv2d(32, num_classes, 1)
    
    def forward(self, x):
        input_size = x.shape[2:]
        
        # Encoder
        enc_features = self.encoder(x)
        
        # Decoder with skip connections
        # Stage 1: Decode deepest features
        dec4 = self.dec4(enc_features[4])  # 512 -> 256
        dec4_skip = torch.cat((dec4, enc_features[3]), dim=1)  # 256+256=512 channels
        
        # Stage 2
        dec3 = self.dec3(dec4_skip)  # 512 -> 128
        dec3_skip = torch.cat((dec3, enc_features[2]), dim=1)  # 128+128=256 channels
        
        # Stage 3
        dec2 = self.dec2(dec3_skip)  # 256 -> 64
        dec2_skip = torch.cat((dec2, enc_features[1]), dim=1)  # 64+64=128 channels
        
        # Stage 4
        dec1 = self.dec1(dec2_skip)  # 128 -> 32
        # Upsample to match first encoder feature size
        dec1_up = F.interpolate(dec1, size=enc_features[0].shape[2:], mode='bilinear', align_corners=False)
        dec1_skip = torch.cat((dec1_up, enc_features[0]), dim=1)  # 32+64=96 channels
        
        # Final output
        main_output = self.final_conv(dec1_skip)
        main_output = F.interpolate(main_output, size=input_size, mode='bilinear', align_corners=False)
        
        # Side outputs (for training, not needed for inference but must be present for loading)
        side4 = torch.sigmoid(self.side4(dec4))
        side4 = F.interpolate(side4, size=input_size, mode='bilinear', align_corners=False)
        
        side3 = torch.sigmoid(self.side3(dec3))
        side3 = F.interpolate(side3, size=input_size, mode='bilinear', align_corners=False)
        
        side2 = torch.sigmoid(self.side2(dec2))
        side2 = F.interpolate(side2, size=input_size, mode='bilinear', align_corners=False)
        
        side1 = torch.sigmoid(self.side1(dec1))
        side1 = F.interpolate(side1, size=input_size, mode='bilinear', align_corners=False)
        
        side_outputs = (side4, side3, side2, side1)
        
        return main_output, side_outputs

# ========== PASSPORT PHOTO DIMENSIONS ==========

PASSPORT_SIZES = {
    "US Passport (2x2 inch)": {"width": 600, "height": 600, "dpi": 300},
    "EU Passport (35x45mm)": {"width": 413, "height": 531, "dpi": 300},
    "India Passport (35x45mm)": {"width": 413, "height": 531, "dpi": 300},
    "UK Passport (45x35mm)": {"width": 531, "height": 413, "dpi": 300},
    "Canada Passport (50x70mm)": {"width": 591, "height": 827, "dpi": 300},
    "Australia Passport (45x35mm)": {"width": 531, "height": 413, "dpi": 300},
    "Custom Size": {"width": 600, "height": 600, "dpi": 300}
}

# ========== BACKGROUND COLORS ==========

BACKGROUND_COLORS = {
    "white": {"name": "White", "color": (255, 255, 255), "description": "Standard passport photos"},
    "blue": {"name": "Blue", "color": (70, 130, 180), "description": "Visa/ID photos"},
    "light_blue": {"name": "Light Blue", "color": (173, 216, 230), "description": "Alternative ID photos"},
    "grey": {"name": "Grey", "color": (128, 128, 128), "description": "Professional photos"},
    "light_grey": {"name": "Light Grey", "color": (211, 211, 211), "description": "Soft professional look"},
    "red": {"name": "Red", "color": (220, 20, 60), "description": "Special documents"},
    "cream": {"name": "Cream", "color": (245, 245, 220), "description": "Warm professional look"},
    "transparent": {"name": "Transparent", "color": None, "description": "For custom backgrounds"}
}

# ========== MODEL LOADING FROM GOOGLE DRIVE ==========
@st.cache_resource
def load_model():
    """Load the trained HybridU2Net model from Google Drive (280MB)"""
    try:
        model_path = "best_model.pth"
        
        # Download from YOUR Google Drive link if not exists
        if not os.path.exists(model_path):
            st.info("📥 Downloading AI model from Google Drive... (~280MB, 30-60s only once)")
            gdown.download("https://drive.google.com/uc?id=1-mvaOvD7Hs_yM36ZCv52OvOBFy3gdnU1", 
                          model_path, quiet=False)
            st.success("✅ Model downloaded successfully!")
        
        # Create model with EXACT same configuration as training
        model = HybridU2Net(encoder_name='resnet34', pretrained=False, num_classes=1)
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        # Load state dict
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        model.eval()
        
        st.sidebar.success("✅ AI Model loaded from Google Drive!")
        return model, "success"
        
    except Exception as e:
        return None, f"❌ Error loading model: {str(e)}"

# ========== IMAGE PROCESSING FUNCTIONS ==========

def preprocess_image(image):
    """Preprocess image for model input (same as training)"""
    # Same transforms as your training code
    transform = A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    # Resize to training input size (320x320)
    image_resized = cv2.resize(image, (320, 320))
    
    # Apply transforms
    transformed = transform(image=image_resized)
    image_tensor = transformed['image'].unsqueeze(0)
    
    return image_tensor

def detect_face_region(image):
    """Detect face region for better cropping"""
    try:
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Load face cascade (built-in OpenCV)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) > 0:
            # Get the largest face
            largest_face = max(faces, key=lambda x: x[2] * x[3])
            x, y, w, h = largest_face
            
            # Expand the face region for passport photo
            expand_factor = 1.8
            center_x, center_y = x + w//2, y + h//2
            new_w, new_h = int(w * expand_factor), int(h * expand_factor)
            
            # Calculate new boundaries
            x1 = max(0, center_x - new_w//2)
            y1 = max(0, center_y - new_h//2)
            x2 = min(image.shape[1], center_x + new_w//2)
            y2 = min(image.shape[0], center_y + new_h//2)
            
            return (x1, y1, x2, y2)
    except:
        pass
    
    # If face detection fails, return center crop
    h, w = image.shape[:2]
    size = min(h, w)
    start_x = (w - size) // 2
    start_y = (h - size) // 2
    return (start_x, start_y, start_x + size, start_y + size)

def crop_to_passport_size(image, mask, passport_size):
    """Crop and resize image to passport dimensions with smart centering"""
    
    # Get face region
    face_region = detect_face_region(image)
    x1, y1, x2, y2 = face_region
    
    # Crop image and mask to face region
    cropped_image = image[y1:y2, x1:x2]
    cropped_mask = mask[y1:y2, x1:x2]
    
    # Get passport dimensions
    target_width = passport_size["width"]
    target_height = passport_size["height"]
    
    # Resize to passport size
    resized_image = cv2.resize(cropped_image, (target_width, target_height))
    resized_mask = cv2.resize(cropped_mask, (target_width, target_height))
    
    return resized_image, resized_mask

def apply_background(image, mask, bg_color):
    """Apply background color using mask"""
    if bg_color is None:  # Transparent
        # Create RGBA image
        rgba_image = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
        rgba_image[:, :, :3] = image
        rgba_image[:, :, 3] = (mask * 255).astype(np.uint8)
        return rgba_image
    else:
        # Solid background
        mask_3ch = np.stack([mask] * 3, axis=2)
        background = np.full_like(image, bg_color, dtype=np.uint8)
        result = (image * mask_3ch + background * (1 - mask_3ch)).astype(np.uint8)
        return result

# ========== STREAMLIT APP ==========

def main():
    st.set_page_config(
        page_title="Professional Passport Photo Generator",
        page_icon="📸",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better mobile experience
    st.markdown("""
    <style>
    .main > div {
        max-width: 100%;
        padding-top: 2rem;
    }
    .stButton > button {
        width: 100%;
        height: 3rem;
        font-size: 1.1rem;
    }
    .upload-section {
        border: 2px dashed #ccc;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("📸 Professional Passport Photo Generator")
    st.markdown("**AI-Powered | Real-time Processing | Multiple Formats | Perfect Dimensions**")
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    # Load model from Google Drive (REMOVED TEXT INPUT - AUTOMATIC)
    with st.spinner("🚀 Loading AI model from Google Drive..."):
        model, load_status = load_model()
    
    if model is None:
        st.error(f"❌ Model loading failed: {load_status}")
        st.stop()
    
    # Passport size selection
    st.sidebar.subheader("📐 Passport Size")
    selected_size_name = st.sidebar.selectbox(
        "Choose passport format:",
        list(PASSPORT_SIZES.keys()),
        help="Select the standard passport photo size for your country"
    )
    
    if selected_size_name == "Custom Size":
        custom_width = st.sidebar.number_input("Width (pixels):", min_value=200, max_value=1000, value=600)
        custom_height = st.sidebar.number_input("Height (pixels):", min_value=200, max_value=1000, value=600)
        passport_size = {"width": custom_width, "height": custom_height, "dpi": 300}
    else:
        passport_size = PASSPORT_SIZES[selected_size_name]
    
    st.sidebar.info(f"📏 Output: {passport_size['width']}×{passport_size['height']} px @ {passport_size['dpi']} DPI")
    
    # Background selection
    st.sidebar.subheader("🎨 Background Options")
    bg_options = {v['name']: k for k, v in BACKGROUND_COLORS.items()}
    selected_bg_name = st.sidebar.selectbox(
        "Choose background:",
        list(bg_options.keys()),
        help="Select background color for your passport photo"
    )
    selected_bg_key = bg_options[selected_bg_name]
    bg_info = BACKGROUND_COLORS[selected_bg_key]
    
    st.sidebar.info(f"🎯 {bg_info['description']}")
    
    # Main interface
    tab1, tab2, tab3 = st.tabs(["📤 Upload Photo", "📷 Camera Capture", "📋 Instructions"])
    
    with tab1:
        st.header("📤 Upload Your Photo")
        
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['jpg', 'jpeg', 'png', 'bmp', 'webp'],
            help="Upload a clear photo with good lighting"
        )
        
        if uploaded_file is not None:
            process_image(uploaded_file, model, passport_size, bg_info, selected_bg_key, source="upload")
    
    with tab2:
        st.header("📷 Camera Capture")
        st.markdown("📱 **Take a photo directly with your device camera**")
        
        # Built-in Streamlit camera input
        camera_photo = st.camera_input("Take a photo for passport", key="camera")
        
        if camera_photo is not None:
            st.success("📸 Photo captured successfully!")
            process_image(camera_photo, model, passport_size, bg_info, selected_bg_key, source="camera")
    
    with tab3:
        st.header("📋 How to Get Perfect Passport Photos")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📷 Photo Guidelines")
            st.markdown("""
            **Lighting:**
            - Use natural, even lighting
            - Avoid shadows on face or background
            - Face should be clearly lit
            
            **Pose:**
            - Look directly at camera
            - Keep head straight and centered
            - Maintain neutral expression
            - Keep both eyes open and visible
            
            **Background:**
            - Any background works (AI removes it)
            - Avoid busy or distracting backgrounds
            """)
        
        with col2:
            st.subheader("📐 Technical Requirements")
            st.markdown("""
            **Image Quality:**
            - Minimum 600×600 pixels
            - Clear, sharp, and in focus
            - Good contrast and color
            
            **Clothing:**
            - Wear normal street clothes
            - Avoid uniforms (unless required)
            - Remove hats and sunglasses
            
            **Processing:**
            - AI automatically crops to correct size
            - Background replaced with chosen color
            - Professional quality output
            """)
    
    # Footer
    with st.expander("🔧 Technical Information"):
        col_tech1, col_tech2, col_tech3 = st.columns(3)
        
        with col_tech1:
            st.markdown("### AI Model")
            st.write("- **Architecture**: HybridU2Net")
            st.write("- **Encoder**: ResNet34")
            st.write("- **Accuracy**: 94%+ IoU")
            st.write("- **Speed**: Real-time")
        
        with col_tech2:
            st.markdown("### Formats Supported")
            st.write("- **Input**: JPG, PNG, BMP, WebP")
            st.write("- **Output**: JPG, PNG")
            st.write("- **DPI**: 300 (print quality)")
            st.write("- **Sizes**: 6 standard formats")
        
        with col_tech3:
            st.markdown("### Features")
            st.write("- **Face Detection**: Auto-crop")
            st.write("- **Smart Centering**: Perfect positioning")
            st.write("- **8 Backgrounds**: Professional options")
            st.write("- **Mobile Friendly**: Responsive design")

def process_image(uploaded_file, model, passport_size, bg_info, selected_bg_key, source="upload"):
    """Process uploaded image through the complete pipeline"""
    
    # Display original image
    image = Image.open(uploaded_file).convert('RGB')
    image_np = np.array(image)
    
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.subheader("📷 Original Photo")
        st.image(image, use_column_width=True)
        st.info(f"📏 Size: {image_np.shape[1]}×{image_np.shape[0]} pixels")
        
        # Processing button with unique key
        if st.button("🚀 Generate Professional Passport Photo", 
                    type="primary", 
                    use_container_width=True, 
                    key=f"process_btn_{source}"):
            
            with st.spinner("🔄 AI Processing Pipeline..."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    # Step 1: Preprocessing
                    status_text.text("🔄 Step 1/5: Preprocessing image...")
                    progress_bar.progress(20)
                    input_tensor = preprocess_image(image_np)
                    
                    # Step 2: AI Segmentation
                    status_text.text("🤖 Step 2/5: AI segmentation...")
                    progress_bar.progress(40)
                    with torch.no_grad():
                        main_output, side_outputs = model(input_tensor)
                        mask_np = main_output.cpu().numpy()[0, 0]
                    
                    # Step 3: Face detection and cropping
                    status_text.text("👤 Step 3/5: Smart cropping...")
                    progress_bar.progress(60)
                    mask_resized = cv2.resize(mask_np, (image_np.shape[1], image_np.shape[0]))
                    mask_binary = (mask_resized > 0.5).astype(np.float32)
                    
                    # Crop to passport size
                    passport_image, passport_mask = crop_to_passport_size(image_np, mask_binary, passport_size)
                    
                    # Step 4: Background replacement
                    status_text.text("🎨 Step 4/5: Background replacement...")
                    progress_bar.progress(80)
                    result_image = apply_background(passport_image, passport_mask, bg_info['color'])
                    
                    # Step 5: Finalize
                    status_text.text("✅ Step 5/5: Finalizing...")
                    progress_bar.progress(100)
                    
                    # Store results
                    st.session_state.result_image = result_image
                    st.session_state.passport_mask = passport_mask
                    st.session_state.bg_info = bg_info
                    st.session_state.passport_size = passport_size
                    st.session_state.original_image = image_np
                    st.session_state.selected_bg_key = selected_bg_key
                    
                    status_text.text("🎉 Professional passport photo ready!")
                    
                except Exception as e:
                    st.error(f"❌ Processing error: {str(e)}")
                    return
                
                finally:
                    progress_bar.empty()
                    status_text.empty()
    
    with col2:
        st.subheader("📸 Professional Result")
        
        if hasattr(st.session_state, 'result_image') and st.session_state.result_image is not None:
            # Display result
            if st.session_state.selected_bg_key == "transparent":
                result_pil = Image.fromarray(st.session_state.result_image, 'RGBA')
                file_format = "PNG"
                mime_type = "image/png"
            else:
                result_pil = Image.fromarray(st.session_state.result_image)
                file_format = "JPEG"
                mime_type = "image/jpeg"
            
            st.image(result_pil, use_column_width=True)
            
            # Image info
            dimensions = st.session_state.passport_size
            st.success(f"✅ Perfect passport photo: {dimensions['width']}×{dimensions['height']} px")
            st.info(f"🎨 Background: {st.session_state.bg_info['name']} - {st.session_state.bg_info['description']}")
            
            # Download section
            st.subheader("💾 Download Options")
            
            # Create download buffer
            buf = io.BytesIO()
            if file_format == "PNG":
                result_pil.save(buf, format="PNG", dpi=(300, 300))
                file_ext = "png"
            else:
                result_pil.save(buf, format="JPEG", quality=95, dpi=(300, 300))
                file_ext = "jpg"
            
            # Download button
            st.download_button(
                label=f"📥 Download {file_format} (Print Quality)",
                data=buf.getvalue(),
                file_name=f"passport_photo_{st.session_state.bg_info['name'].lower().replace(' ', '_')}.{file_ext}",
                mime=mime_type,
                use_container_width=True
            )
            
            # Additional formats
            col_jpg, col_png = st.columns(2)
            
            with col_jpg:
                if st.button("📄 Download JPG", use_container_width=True, key=f"download_jpg_{source}"):
                    jpg_buf = io.BytesIO()
                    if st.session_state.selected_bg_key == "transparent":
                        # Convert RGBA to RGB for JPG
                        rgb_image = Image.new('RGB', result_pil.size, (255, 255, 255))
                        rgb_image.paste(result_pil, mask=result_pil.split()[-1])
                        rgb_image.save(jpg_buf, format="JPEG", quality=95, dpi=(300, 300))
                    else:
                        result_pil.save(jpg_buf, format="JPEG", quality=95, dpi=(300, 300))
                    
                    st.download_button(
                        label="💾 Save JPG",
                        data=jpg_buf.getvalue(),
                        file_name=f"passport_photo_{st.session_state.bg_info['name'].lower().replace(' ', '_')}.jpg",
                        mime="image/jpeg",
                        key=f"save_jpg_{source}"
                    )
            
            with col_png:
                if st.button("🖼️ Download PNG", use_container_width=True, key=f"download_png_{source}"):
                    png_buf = io.BytesIO()
                    if st.session_state.selected_bg_key != "transparent":
                        # Convert RGB to RGBA for PNG
                        rgba_image = Image.new('RGBA', result_pil.size)
                        rgba_image.paste(result_pil)
                        rgba_image.save(png_buf, format="PNG", dpi=(300, 300))
                    else:
                        result_pil.save(png_buf, format="PNG", dpi=(300, 300))
                    
                    st.download_button(
                        label="💾 Save PNG",
                        data=png_buf.getvalue(),
                        file_name=f"passport_photo_{st.session_state.bg_info['name'].lower().replace(' ', '_')}.png",
                        mime="image/png",
                        key=f"save_png_{source}"
                    )
            
            # Additional options
            with st.expander("🔍 Advanced Options"):
                col_mask, col_specs = st.columns(2)
                
                with col_mask:
                    if st.checkbox("Show AI Mask", key=f"show_mask_{source}"):
                        mask_display = (st.session_state.passport_mask * 255).astype(np.uint8)
                        st.image(mask_display, caption="AI Segmentation Mask", use_column_width=True)
                
                with col_specs:
                    st.write("**Photo Specifications:**")
                    st.write(f"- Dimensions: {dimensions['width']}×{dimensions['height']} px")
                    st.write(f"- DPI: {dimensions['dpi']} (print quality)")
                    st.write(f"- Background: {st.session_state.bg_info['name']}")
                    st.write(f"- Format: {file_format}")
                    st.write(f"- File size: ~{len(buf.getvalue())//1024} KB")
        
        else:
            st.info("👆 Upload an image and click 'Generate Professional Passport Photo'")
            
            # Preview of passport sizes
            st.subheader("📐 Supported Passport Sizes")
            for size_name, specs in PASSPORT_SIZES.items():
                if size_name != "Custom Size":
                    st.write(f"• **{size_name}**: {specs['width']}×{specs['height']} px")

if __name__ == "__main__":
    main()
