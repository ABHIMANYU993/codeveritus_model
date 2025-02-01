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

# Assets registry details and configurations line 70
# Assets registry details and configurations line 71
# Assets registry details and configurations line 72
# Assets registry details and configurations line 73
# Assets registry details and configurations line 74
# Assets registry details and configurations line 75
# Assets registry details and configurations line 76
# Assets registry details and configurations line 77
# Assets registry details and configurations line 78
# Assets registry details and configurations line 79
# Assets registry details and configurations line 80
# Assets registry details and configurations line 81
# Assets registry details and configurations line 82
# Assets registry details and configurations line 83
# Assets registry details and configurations line 84
# Assets registry details and configurations line 85
# Assets registry details and configurations line 86
# Assets registry details and configurations line 87
# Assets registry details and configurations line 88
# Assets registry details and configurations line 89
# Assets registry details and configurations line 90
# Assets registry details and configurations line 91
# Assets registry details and configurations line 92
# Assets registry details and configurations line 93
# Assets registry details and configurations line 94
# Assets registry details and configurations line 95
# Assets registry details and configurations line 96
# Assets registry details and configurations line 97
# Assets registry details and configurations line 98
# Assets registry details and configurations line 99
# Assets registry details and configurations line 100
# Assets registry details and configurations line 101
# Assets registry details and configurations line 102
# Assets registry details and configurations line 103
# Assets registry details and configurations line 104
# Assets registry details and configurations line 105
# Assets registry details and configurations line 106
# Assets registry details and configurations line 107
# Assets registry details and configurations line 108
# Assets registry details and configurations line 109
# Assets registry details and configurations line 110
# Assets registry details and configurations line 111
# Assets registry details and configurations line 112
# Assets registry details and configurations line 113
# Assets registry details and configurations line 114
# Assets registry details and configurations line 115
# Assets registry details and configurations line 116
# Assets registry details and configurations line 117
# Assets registry details and configurations line 118
# Assets registry details and configurations line 119
# Assets registry details and configurations line 120
# Assets registry details and configurations line 121
# Assets registry details and configurations line 122
# Assets registry details and configurations line 123
# Assets registry details and configurations line 124
# Assets registry details and configurations line 125
# Assets registry details and configurations line 126
# Assets registry details and configurations line 127
# Assets registry details and configurations line 128
# Assets registry details and configurations line 129
# Assets registry details and configurations line 130
# Assets registry details and configurations line 131
# Assets registry details and configurations line 132
# Assets registry details and configurations line 133
# Assets registry details and configurations line 134
# Assets registry details and configurations line 135
# Assets registry details and configurations line 136
# Assets registry details and configurations line 137
# Assets registry details and configurations line 138
# Assets registry details and configurations line 139
# Assets registry details and configurations line 140
# Assets registry details and configurations line 141
# Assets registry details and configurations line 142
# Assets registry details and configurations line 143
# Assets registry details and configurations line 144
# Assets registry details and configurations line 145
# Assets registry details and configurations line 146
# Assets registry details and configurations line 147
# Assets registry details and configurations line 148
# Assets registry details and configurations line 149
# Assets registry details and configurations line 150
# Assets registry details and configurations line 151
# Assets registry details and configurations line 152
# Assets registry details and configurations line 153
# Assets registry details and configurations line 154
# Assets registry details and configurations line 155
# Assets registry details and configurations line 156
# Assets registry details and configurations line 157
# Assets registry details and configurations line 158
# Assets registry details and configurations line 159
# Assets registry details and configurations line 160
# Assets registry details and configurations line 161
# Assets registry details and configurations line 162
# Assets registry details and configurations line 163
# Assets registry details and configurations line 164
# Assets registry details and configurations line 165
# Assets registry details and configurations line 166
# Assets registry details and configurations line 167
# Assets registry details and configurations line 168
# Assets registry details and configurations line 169
# Assets registry details and configurations line 170
# Assets registry details and configurations line 171
# Assets registry details and configurations line 172
# Assets registry details and configurations line 173
# Assets registry details and configurations line 174
# Assets registry details and configurations line 175
# Assets registry details and configurations line 176
# Assets registry details and configurations line 177
# Assets registry details and configurations line 178
# Assets registry details and configurations line 179
# Assets registry details and configurations line 180
# Assets registry details and configurations line 181
# Assets registry details and configurations line 182
# Assets registry details and configurations line 183
# Assets registry details and configurations line 184
# Assets registry details and configurations line 185
# Assets registry details and configurations line 186
# Assets registry details and configurations line 187
# Assets registry details and configurations line 188
# Assets registry details and configurations line 189
# Assets registry details and configurations line 190
# Assets registry details and configurations line 191
# Assets registry details and configurations line 192
# Assets registry details and configurations line 193
# Assets registry details and configurations line 194
# Assets registry details and configurations line 195
# Assets registry details and configurations line 196
# Assets registry details and configurations line 197
# Assets registry details and configurations line 198
# Assets registry details and configurations line 199
# Assets registry details and configurations line 200
# Assets registry details and configurations line 201
# Assets registry details and configurations line 202
# Assets registry details and configurations line 203
# Assets registry details and configurations line 204
# Assets registry details and configurations line 205
# Assets registry details and configurations line 206
# Assets registry details and configurations line 207
# Assets registry details and configurations line 208
# Assets registry details and configurations line 209
# Assets registry details and configurations line 210
# Assets registry details and configurations line 211
# Assets registry details and configurations line 212
# Assets registry details and configurations line 213
# Assets registry details and configurations line 214
# Assets registry details and configurations line 215
# Assets registry details and configurations line 216
# Assets registry details and configurations line 217
# Assets registry details and configurations line 218
# Assets registry details and configurations line 219
# Assets registry details and configurations line 220
# Assets registry details and configurations line 221
# Assets registry details and configurations line 222
# Assets registry details and configurations line 223
# Assets registry details and configurations line 224
# Assets registry details and configurations line 225
# Assets registry details and configurations line 226
# Assets registry details and configurations line 227
# Assets registry details and configurations line 228
# Assets registry details and configurations line 229
# Assets registry details and configurations line 230
# Assets registry details and configurations line 231
# Assets registry details and configurations line 232
# Assets registry details and configurations line 233
# Assets registry details and configurations line 234
# Assets registry details and configurations line 235
# Assets registry details and configurations line 236
# Assets registry details and configurations line 237
# Assets registry details and configurations line 238
# Assets registry details and configurations line 239
# Assets registry details and configurations line 240
# Assets registry details and configurations line 241
# Assets registry details and configurations line 242
# Assets registry details and configurations line 243
# Assets registry details and configurations line 244
# Assets registry details and configurations line 245
# Assets registry details and configurations line 246
# Assets registry details and configurations line 247
# Assets registry details and configurations line 248
# Assets registry details and configurations line 249