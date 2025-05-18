#!/bin/bash
# setup_terraform.sh - Initialize Terraform configuration for BTC Trading Bot
#
# This script initializes the Terraform environment and prepares it for deployment.
# It should be run once before the first deployment.

set -e  # Exit immediately if a command exits with a non-zero status

echo "🚀 Setting up Terraform for BTC Trading Bot infrastructure..."
echo "------------------------------------------------"

# Change to the terraform directory
cd "$(dirname "$0")/../terraform"
echo "📁 Switched to directory: $(pwd)"

# Check if terraform is installed
if ! command -v terraform &> /dev/null; then
    echo "❌ Error: Terraform is not installed. Please install Terraform."
    echo "   Visit https://learn.hashicorp.com/tutorials/terraform/install-cli for instructions."
    exit 1
fi

# Get GCP project ID
if [ -z "$PROJECT_ID" ]; then
    echo "📋 Enter your GCP project ID:"
    read -r PROJECT_ID
    
    if [ -z "$PROJECT_ID" ]; then
        echo "❌ Error: Project ID cannot be empty."
        exit 1
    fi
fi

# Get GCP project number
if [ -z "$PROJECT_NUMBER" ]; then
    echo "📋 Enter your GCP project number:"
    read -r PROJECT_NUMBER
    
    if [ -z "$PROJECT_NUMBER" ]; then
        echo "❌ Error: Project number cannot be empty."
        exit 1
    fi
fi

# Get GCP region
if [ -z "$REGION" ]; then
    REGION="us-central1"
    echo "📋 Using default region: $REGION"
    echo "   To use a different region, press Ctrl+C and set the REGION environment variable."
fi

# Create terraform.tfvars file from example if it doesn't exist
if [ ! -f terraform.tfvars ]; then
    echo "📝 Creating terraform.tfvars from example file..."
    cp terraform.tfvars.example terraform.tfvars
    
    # Replace placeholders with actual values
    sed -i '' "s/your-gcp-project-id/$PROJECT_ID/g" terraform.tfvars
    sed -i '' "s/your-gcp-project-number/$PROJECT_NUMBER/g" terraform.tfvars
    sed -i '' "s/us-central1/$REGION/g" terraform.tfvars
    
    echo "✅ Created terraform.tfvars with your GCP settings."
else
    echo "ℹ️ terraform.tfvars already exists. Skipping creation."
fi

# Create GCS bucket for Terraform state if it doesn't exist
TERRAFORM_STATE_BUCKET="${PROJECT_ID}-terraform-state"
echo "🪣 Creating bucket for Terraform state: gs://$TERRAFORM_STATE_BUCKET"

if ! gsutil ls -b "gs://$TERRAFORM_STATE_BUCKET" &> /dev/null; then
    gsutil mb -l "$REGION" "gs://$TERRAFORM_STATE_BUCKET"
    gsutil versioning set on "gs://$TERRAFORM_STATE_BUCKET"
    echo "✅ Created Terraform state bucket."
else
    echo "ℹ️ Terraform state bucket already exists. Skipping creation."
fi

# Uncomment backend configuration in versions.tf
echo "🔧 Configuring Terraform backend..."
sed -i '' 's/# backend "gcs" {/backend "gcs" {/g' versions.tf
sed -i '' 's/#   bucket  = "your-terraform-state-bucket"/  bucket  = "'"$TERRAFORM_STATE_BUCKET"'"/g' versions.tf
sed -i '' 's/#   prefix  = "btc-trading-bot\/terraform\/state"/  prefix  = "btc-trading-bot\/terraform\/state"/g' versions.tf
sed -i '' 's/# }/}/g' versions.tf

echo "✅ Configured Terraform backend."

# Initialize Terraform
echo "🔄 Initializing Terraform..."
terraform init

echo "------------------------------------------------"
echo "✅ Terraform setup complete! You can now run ./deploy_infrastructure.sh to deploy the infrastructure."