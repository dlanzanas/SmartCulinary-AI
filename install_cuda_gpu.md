# GPU Setup Guide for TensorFlow 2.10.0

## Current Status
- **GPU**: NVIDIA GeForce RTX 4060 ✅
- **Driver Version**: 576.80 ✅
- **Current CUDA**: 12.9 (incompatible with TensorFlow 2.10.0)
- **Required CUDA**: 11.2
- **Required cuDNN**: 8.1

## Installation Steps

### Step 1: Download CUDA 11.2 Toolkit

1. Visit: https://developer.nvidia.com/cuda-11.2.0-download-archive
2. Select:
   - Operating System: **Windows**
   - Architecture: **x86_64**
   - Version: **10** (or 11)
   - Installer Type: **exe (local)**
3. Download the installer (~3 GB)

### Step 2: Install CUDA 11.2

1. Run the downloaded installer
2. Choose **Custom Installation**
3. Select these components:
   - ✅ CUDA Toolkit 11.2
   - ✅ CUDA Runtime 11.2
   - ✅ CUDA Development
   - ❌ Uncheck "GeForce Experience" (not needed)
   - ❌ Uncheck "Driver" (you already have newer drivers)
4. Install to default location: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2`

### Step 3: Download cuDNN 8.1 for CUDA 11.2

1. Visit: https://developer.nvidia.com/cudnn
2. Click "Download cuDNN"
3. Create/login to NVIDIA Developer account (free)
4. Download: **cuDNN v8.1.1 for CUDA 11.2** (Windows)
5. Extract the ZIP file

### Step 4: Install cuDNN

1. Extract the downloaded ZIP file
2. Copy files from extracted folder to CUDA installation:
   ```
   From: cudnn-11.2-windows-x64-v8.1.1.33\cuda\bin\*.dll
   To:   C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\bin\
   
   From: cudnn-11.2-windows-x64-v8.1.1.33\cuda\include\*.h
   To:   C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\include\
   
   From: cudnn-11.2-windows-x64-v8.1.1.33\cuda\lib\x64\*.lib
   To:   C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\lib\x64\
   ```

### Step 5: Update Environment Variables

1. Open PowerShell as Administrator
2. Run these commands:

```powershell
# Add CUDA 11.2 to PATH (prepend to use it first)
[Environment]::SetEnvironmentVariable(
    "Path",
    "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\bin;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\libnvvp;" + [Environment]::GetEnvironmentVariable("Path", "Machine"),
    "Machine"
)

# Set CUDA_PATH
[Environment]::SetEnvironmentVariable("CUDA_PATH", "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2", "Machine")
```

3. **Restart your terminal** (or reboot)

### Step 6: Verify Installation

Open a new PowerShell terminal and run:

```powershell
# Check CUDA version
nvcc --version

# Should show: Cuda compilation tools, release 11.2

# Test TensorFlow GPU detection
cd C:\Users\Administrator\Documents\Project
.\smartculinary_env\Scripts\activate
python -c "import tensorflow as tf; print('GPUs:', tf.config.list_physical_devices('GPU'))"
```

Expected output:
```
GPUs: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

## Troubleshooting

### If GPU still not detected:

1. **Restart computer** after installation
2. Verify PATH contains CUDA 11.2 **before** CUDA 12.9:
   ```powershell
   $env:Path -split ';' | Select-String -Pattern 'CUDA'
   ```
3. Reinstall TensorFlow:
   ```powershell
   pip uninstall tensorflow
   pip install tensorflow==2.10.0
   ```

### Alternative: Upgrade TensorFlow (Easier but requires testing)

If CUDA installation is too complex, upgrade to TensorFlow 2.15:

```powershell
.\smartculinary_env\Scripts\activate
pip install --upgrade tensorflow==2.15.0
```

This works with CUDA 12.x but may require minor code adjustments.

## Expected Performance Improvement

After GPU setup:
- **Current (CPU)**: ~120 seconds/epoch
- **With GPU**: ~5-10 seconds/epoch
- **Speedup**: 12-24x faster! ⚡

Total training time: ~6-9 minutes (vs 2+ hours on CPU)
