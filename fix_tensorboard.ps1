# Fix TensorBoard Installation Script
# This script fixes the pkg_resources issue by reinstalling the environment

Write-Host "Fixing TensorBoard installation..." -ForegroundColor Cyan

# Activate environment
.\smartculinary_env\Scripts\Activate.ps1

# Reinstall setuptools with pkg_resources
Write-Host "`nReinstalling setuptools..." -ForegroundColor Yellow
pip uninstall -y setuptools
pip install setuptools==65.5.0

# Reinstall tensorboard
Write-Host "`nReinstalling tensorboard..." -ForegroundColor Yellow  
pip uninstall -y tensorboard tensorboard-data-server
pip install tensorboard==2.10.1

Write-Host "`nTesting TensorBoard..." -ForegroundColor Yellow
python -c "import pkg_resources; print('pkg_resources OK')"
python -c "from tensorboard import default; print('TensorBoard OK')"

Write-Host "`n✓ TensorBoard fixed! You can now run:" -ForegroundColor Green
Write-Host "  tensorboard --logdir logs" -ForegroundColor White
