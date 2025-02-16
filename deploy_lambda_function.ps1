# Create deployment package directory
New-Item -ItemType Directory -Force -Path lambda_package

# Copy source files
Copy-Item -Path "src" -Destination "lambda_package/" -Recurse
Copy-Item -Path "lambda_handler.py" -Destination "lambda_package/"

# Create ZIP file
Compress-Archive -Path lambda_package/* -DestinationPath lambda_deployment.zip -Force

# Clean up
Remove-Item -Path lambda_package -Recurse

Write-Host "Lambda deployment package created: lambda_deployment.zip" 