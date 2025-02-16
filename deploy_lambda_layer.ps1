# Create lambda layer directory structure
New-Item -ItemType Directory -Force -Path lambda_layer
New-Item -ItemType Directory -Force -Path lambda_layer/python/lib/python3.9/site-packages

# Install dependencies into the layer
pip install --platform manylinux2014_x86_64 --implementation cp --python-version 3.9 --only-binary=:all: --upgrade -r lambda_requirements.txt -t lambda_layer/python/lib/python3.9/site-packages/

# Create ZIP file
Compress-Archive -Path lambda_layer/python -DestinationPath lambda_layer.zip -Force

# Clean up
Remove-Item -Path lambda_layer -Recurse

Write-Host "Lambda layer package created: lambda_layer.zip" 