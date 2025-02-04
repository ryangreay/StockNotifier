#!/bin/bash

# Create a temporary directory for the deployment package
echo "Creating deployment package..."
mkdir -p lambda_package

# Install dependencies
echo "Installing dependencies..."
pip install -r lambda_requirements.txt -t lambda_package/

# Copy the lambda handler and related files
echo "Copying source files..."
cp -r src/* lambda_package/

# Create the ZIP file
echo "Creating ZIP file..."
cd lambda_package
zip -r ../lambda_deployment.zip .
cd ..

# Clean up
echo "Cleaning up..."
rm -rf lambda_package

echo "Deployment package created: lambda_deployment.zip"
echo "You can now upload this file to AWS Lambda through the console or using AWS CLI:"
echo "aws lambda update-function-code --function-name YOUR_FUNCTION_NAME --zip-file fileb://lambda_deployment.zip" 