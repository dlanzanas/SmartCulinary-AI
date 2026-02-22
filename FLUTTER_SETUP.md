# Flutter Installation Guide for SmartCulinary AI

## Prerequisites

1. **Git** - Required for Flutter SDK
2. **Android Studio** (for Android development)
3. **Chrome** (for Web development)

## Installation Steps

### 1. Download Flutter SDK

```powershell
# Create a directory for Flutter
New-Item -ItemType Directory -Path "C:\src" -Force

# Download Flutter SDK (latest stable)
# Visit: https://docs.flutter.dev/get-started/install/windows
# Download the ZIP file and extract to C:\src\flutter
```

### 2. Add Flutter to PATH

```powershell
# Add to System Environment Variables
$env:Path += ";C:\src\flutter\bin"

# Verify installation
flutter --version
flutter doctor
```

### 3. Install Required Dependencies

```powershell
# Run Flutter doctor to check dependencies
flutter doctor

# Install Android toolchain (if needed)
# Download Android Studio: https://developer.android.com/studio

# Enable web support
flutter config --enable-web
```

### 4. Create SmartCulinary Flutter Project

```powershell
cd C:\Users\Administrator\Documents\Project

# Create Flutter app
flutter create smartculinary_app --platforms=android,web

cd smartculinary_app

# Run on web (for testing)
flutter run -d chrome

# Build for Android
flutter build apk
```

## Quick Start (Without Flutter Installed)

Since Flutter isn't installed yet, I'll create the project structure manually with all necessary files. You can install Flutter later and run the app.

## Next Steps

1. Install Flutter SDK
2. Run `flutter doctor` to verify setup
3. Navigate to `smartculinary_app` folder
4. Run `flutter pub get` to install dependencies
5. Run `flutter run -d chrome` to test on web
