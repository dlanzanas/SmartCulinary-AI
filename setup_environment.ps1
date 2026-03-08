# SmartCulinary AI - Environment Setup Script
# This script automates the complete ML training environment setup on Windows

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "SmartCulinary AI - Environment Setup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Configuration
$PYTHON_VERSION = "3.9.13"
$PYTHON_INSTALLER_URL = "https://www.python.org/ftp/python/$PYTHON_VERSION/python-$PYTHON_VERSION-amd64.exe"
$PYTHON_INSTALLER = "$env:TEMP\python-$PYTHON_VERSION-amd64.exe"
$VENV_NAME = "smartculinary_env"

# Fix for running in ISE or direct copy-paste where PSScriptRoot might be null
if ($PSScriptRoot) {
    $PROJECT_ROOT = $PSScriptRoot
}
else {
    $PROJECT_ROOT = Get-Location
}

# Step 1: Check if Python 3.9.x is already installed
Write-Host "[1/6] Checking Python installation..." -ForegroundColor Yellow

$pythonInstalled = $false
$pythonPath = $null

try {
    $pythonVersion = & python --version 2>&1
    if ($pythonVersion -like "Python 3.9.*") {
        Write-Host "✓ Python 3.9.x already installed: $pythonVersion" -ForegroundColor Green
        $pythonInstalled = $true
        $pythonPath = "python"
    }
}
catch {
    Write-Host "Python 3.9.x not found in PATH" -ForegroundColor Gray
}

# Check py launcher for Python 3.9
if (-not $pythonInstalled) {
    try {
        $pyVersion = & py -3.9 --version 2>&1
        if ($pyVersion -like "Python 3.9.*") {
            Write-Host "✓ Python 3.9.x found via py launcher: $pyVersion" -ForegroundColor Green
            $pythonInstalled = $true
            $pythonPath = "py -3.9"
        }
    }
    catch {
        Write-Host "Python 3.9.x not found via py launcher" -ForegroundColor Gray
    }
}

# Step 2: Install Python if not found
if (-not $pythonInstalled) {
    Write-Host "Python 3.9.x not found. Installing Python $PYTHON_VERSION..." -ForegroundColor Yellow
    
    # Download Python installer
    Write-Host "Downloading Python installer..." -ForegroundColor Gray
    try {
        Invoke-WebRequest -Uri $PYTHON_INSTALLER_URL -OutFile $PYTHON_INSTALLER -UseBasicParsing
        Write-Host "✓ Download complete" -ForegroundColor Green
    }
    catch {
        Write-Host "✗ Failed to download Python installer" -ForegroundColor Red
        Write-Host "Please download manually from: $PYTHON_INSTALLER_URL" -ForegroundColor Yellow
        exit 1
    }
    
    # Install Python silently
    Write-Host "Installing Python (this may take a few minutes)..." -ForegroundColor Gray
    $installArgs = "/quiet InstallAllUsers=0 PrependPath=1 Include_test=0 Include_pip=1"
    Start-Process -FilePath $PYTHON_INSTALLER -ArgumentList $installArgs -Wait -NoNewWindow
    
    # Refresh environment variables
    $machinePath = [System.Environment]::GetEnvironmentVariable("Path", "Machine")
    $userPath = [System.Environment]::GetEnvironmentVariable("Path", "User")
    $env:Path = $machinePath + ";" + $userPath
    
    # Verify installation
    Start-Sleep -Seconds 3
    try {
        $pythonVersion = & python --version 2>&1
        if ($pythonVersion -like "Python 3.9.*") {
            Write-Host "✓ Python installed successfully: $pythonVersion" -ForegroundColor Green
            $pythonPath = "python"
        }
        else {
            throw "Python version mismatch"
        }
    }
    catch {
        Write-Host "✗ Python installation verification failed" -ForegroundColor Red
        Write-Host "Please restart your terminal and run this script again" -ForegroundColor Yellow
        exit 1
    }
    
    # Cleanup installer
    Remove-Item $PYTHON_INSTALLER -ErrorAction SilentlyContinue
}

Write-Host ""

# Step 3: Create virtual environment
Write-Host "[2/6] Creating virtual environment '$VENV_NAME'..." -ForegroundColor Yellow

$venvPath = Join-Path $PROJECT_ROOT $VENV_NAME

if (Test-Path $venvPath) {
    Write-Host "Virtual environment already exists. Removing old environment..." -ForegroundColor Gray
    Remove-Item -Recurse -Force $venvPath
}

try {
    if ($pythonPath -eq "python") {
        & python -m venv $venvPath
    }
    else {
        # Split the py launcher command if needed
        & py -3.9 -m venv $venvPath
    }
    Write-Host "✓ Virtual environment created" -ForegroundColor Green
}
catch {
    Write-Host "✗ Failed to create virtual environment" -ForegroundColor Red
    exit 1
}

Write-Host ""

# Step 4: Activate virtual environment and upgrade pip
Write-Host "[3/6] Activating virtual environment and upgrading pip..." -ForegroundColor Yellow

$activateScript = Join-Path $venvPath "Scripts\Activate.ps1"

# CRITICAL FIX: Ensure execution policy allows the activation script
if ((Get-ExecutionPolicy) -eq 'Restricted') {
    Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
}

# CRITICAL FIX: Use Dot-Sourcing (.) instead of Call (&) to keep environment active
try {
    . $activateScript
}
catch {
    Write-Host "✗ Failed to activate virtual environment" -ForegroundColor Red
    exit 1
}

if ($env:VIRTUAL_ENV) {
    try {
        & python -m pip install --upgrade pip setuptools wheel --quiet
        Write-Host "✓ Pip upgraded successfully inside venv" -ForegroundColor Green
    }
    catch {
        Write-Host "✗ Failed to upgrade pip" -ForegroundColor Red
        exit 1
    }
}
else {
    Write-Host "✗ Virtual environment was not activated correctly" -ForegroundColor Red
    exit 1
}

Write-Host ""

# Step 5: Install required packages
Write-Host "[4/6] Installing ML packages (this may take 5-10 minutes)..." -ForegroundColor Yellow

$requirementsFile = Join-Path $PROJECT_ROOT "requirements.txt"

if (-not (Test-Path $requirementsFile)) {
    Write-Host "⚠ requirements.txt not found at: $requirementsFile" -ForegroundColor Yellow
    Write-Host "  Skipping package installation." -ForegroundColor Gray
}
else {
    try {
        Write-Host "Installing packages from requirements.txt..." -ForegroundColor Gray
        & python -m pip install -r $requirementsFile --quiet
        Write-Host "✓ All packages installed successfully" -ForegroundColor Green
    }
    catch {
        Write-Host "✗ Failed to install packages" -ForegroundColor Red
        Write-Host "Try running manually: pip install -r requirements.txt" -ForegroundColor Yellow
        exit 1
    }
}

Write-Host ""

# Step 6: Verify installation and check GPU
Write-Host "[5/6] Verifying installation..." -ForegroundColor Yellow

# Test TensorFlow import
try {
    $tfVersion = & python -c "import tensorflow as tf; print(tf.__version__)" 2>&1
    Write-Host "✓ TensorFlow $tfVersion installed" -ForegroundColor Green
    
    # Check GPU availability
    $gpuCheckCmd = "import tensorflow as tf; gpus = tf.config.list_physical_devices('GPU'); print(f'GPUs: {len(gpus)}')"
    $gpuCheck = & python -c $gpuCheckCmd 2>&1
    if ($gpuCheck -like "*GPUs: 0*") {
        Write-Host "⚠ No GPU detected - training will use CPU (slower)" -ForegroundColor Yellow
        Write-Host "  For GPU support, install CUDA Toolkit and cuDNN separately" -ForegroundColor Gray
    }
    else {
        Write-Host "✓ GPU detected ($gpuCheck) - Acceleration enabled!" -ForegroundColor Green
    }
}
catch {
    Write-Host "⚠ TensorFlow import failed or not installed" -ForegroundColor Yellow
}

Write-Host ""

# Step 7: Create project directory structure
Write-Host "[6/6] Creating project directory structure..." -ForegroundColor Yellow

$directories = @(
    "config",
    "data",
    "models",
    "utils",
    "datasets",
    "checkpoints",
    "logs",
    "exports"
)

foreach ($dir in $directories) {
    $dirPath = Join-Path $PROJECT_ROOT $dir
    if (-not (Test-Path $dirPath)) {
        New-Item -ItemType Directory -Path $dirPath -Force | Out-Null
        Write-Host "  Created: $dir\" -ForegroundColor Gray
    }
}

Write-Host "✓ Project structure created" -ForegroundColor Green
Write-Host ""

# Final summary
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Setup Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan