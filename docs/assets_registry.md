# Codeveritus Binary Assets Registry

This document lists and registers all binary files tracked in the repository via Git LFS. It acts as an integration and auditing reference for the weights, media files, and branding assets used across the model, front-end, and deployment subsystems.

## 1. Machine Learning Model Weights

### Asset: `model/training/primary_model.safetensors`
- **Path**: `model/training/primary_model.safetensors`
- **Size**: 476 MB
- **Format**: SafeTensors (float32 float weights)
- **Purpose**: Holds the trained parameters of the CodeBERT sequence classifier. This weights file is loaded directly by our FastAPI server, Streamlit prototype, and alternative web backends.
- **Load command**:
  ```python
  from safetensors.torch import load_model
  load_model(model, 'primary_model.safetensors')
  ```

### Asset: `model/deployment/hf_space/model.safetensors`
- **Path**: `model/deployment/hf_space/model.safetensors`
- **Size**: 476 MB
- **Format**: SafeTensors
- **Purpose**: Duplicate weights file utilized by the Hugging Face Space Docker image for inference execution in the cloud.

## 2. Frontend Loop Videos

### Asset: `front-end/public/3130182-uhd_3840_2160_30fps.mp4`
- **Path**: `front-end/public/3130182-uhd_3840_2160_30fps.mp4`
- **Size**: 20.8 MB
- **Format**: H.264 encoded MP4 (no audio track)
- **Purpose**: Looping background video for the React landing page hero section. Visualizes flowcharts, code syntax parsing animations, and analytics overlays.
- **React integration sample**:
  ```jsx
  <video autoPlay loop muted playsInline className='bg-video'>
      <source src='/3130182-uhd_3840_2160_30fps.mp4' type='video/mp4' />
  </video>
  ```

### Asset: `front-end/public/184815-874271897_medium.mp4`
- **Path**: `front-end/public/184815-874271897_medium.mp4`
- **Size**: 6.5 MB
- **Format**: H.264 encoded MP4
- **Purpose**: Background video loop for the features section, displaying abstract digital connectivity.

## 3. Frontend Branding and Graphics Assets

### Asset: `front-end/public/favicon.ico`
- **Path**: `front-end/public/favicon.ico`
- **Format**: ICO File (Multi-resolution icon)
- **Purpose**: Browser tab icon for Codeveritus.

### Asset: `front-end/public/logo.png`
- **Path**: `front-end/public/logo.png`
- **Format**: PNG Image (Lossless RGB)
- **Purpose**: Standard app logo with text used in headers and login page.

### Asset: `front-end/public/logot.png`
- **Path**: `front-end/public/logot.png`
- **Format**: PNG Image (Transparent alpha channel)
- **Purpose**: Clean transparent logo mark used on overlapping dark backgrounds.

### Asset: `front-end/public/logo192.png` & `logo512.png`
- **Path**: `front-end/public/logo192.png`, `front-end/public/logo512.png`
- **Format**: PNG Image
- **Purpose**: Manifest assets used by Progressive Web App (PWA) frameworks to render launcher icons on mobile platforms.

### Asset: `front-end/public/llogo.jpg`
- **Path**: `front-end/public/llogo.jpg`
- **Format**: JPEG Image
- **Purpose**: Secondary logo variant.
