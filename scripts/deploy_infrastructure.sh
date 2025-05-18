#!/bin/bash
# deploy_infrastructure.sh - Deploy infrastructure for BTC Trading Bot
#
# This script applies the Terraform configuration to deploy the infrastructure on GCP.
# Make sure to run setup_terraform.sh first.

set -e  # Exit immediately if a command exits with a non-zero status

echo "🚀 Deploying infrastructure for BTC Trading Bot..."
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

# Check if setup_terraform.sh has been run
if [ ! -f .terraform/terraform.tfstate ]; then
    echo "❌ Error: Terraform has not been initialized. Please run ./setup_terraform.sh first."
    exit 1
fi

# Get Binance API credentials (optional - only if not already in terraform.tfvars)
if ! grep -q "binance_api_key" terraform.tfvars || ! grep -q "binance_api_secret" terraform.tfvars; then
    echo "📋 Would you like to set up Binance API credentials now? (y/n)"
    read -r SETUP_BINANCE
    
    if [ "$SETUP_BINANCE" = "y" ] || [ "$SETUP_BINANCE" = "Y" ]; then
        echo "📋 Enter your Binance API key:"
        read -r BINANCE_API_KEY
        
        echo "📋 Enter your Binance API secret:"
        read -rs BINANCE_API_SECRET
        
        # Add the credentials to terraform.tfvars
        echo "" >> terraform.tfvars
        echo "# Binance API credentials" >> terraform.tfvars
        echo "binance_api_key    = \"$BINANCE_API_KEY\"" >> terraform.tfvars
        echo "binance_api_secret = \"$BINANCE_API_SECRET\"" >> terraform.tfvars
        
        echo "✅ Added Binance API credentials to terraform.tfvars."
    else
        echo "ℹ️ Skipping Binance API credentials setup."
        echo "   You can add them later by editing terraform.tfvars."
    fi
fi

# Run terraform plan
echo "🔍 Planning Terraform deployment..."
terraform plan -out=tfplan || {
    echo "❌ Terraform plan failed. This might be due to existing resources."
    echo "📝 Attempting to import existing Secret Manager secrets if they exist..."
    
    # Check if the secrets already exist in GCP and import them
    # Get the project ID from terraform.tfvars if not already set
    if [ -z "$PROJECT_ID" ]; then
        PROJECT_ID=$(grep 'project_id' terraform.tfvars | cut -d'"' -f2)
    fi
    
    if gcloud secrets describe binance-api-key --project="$PROJECT_ID" &>/dev/null; then
        echo "🔄 Importing existing binance-api-key secret..."
        terraform import google_secret_manager_secret.binance_api_key projects/$PROJECT_ID/secrets/binance-api-key || true
    fi
    
    if gcloud secrets describe binance-api-secret --project="$PROJECT_ID" &>/dev/null; then
        echo "🔄 Importing existing binance-api-secret secret..."
        terraform import google_secret_manager_secret.binance_api_secret projects/$PROJECT_ID/secrets/binance-api-secret || true
    fi
    
    # Try planning again
    echo "🔍 Re-planning Terraform deployment..."
    terraform plan -out=tfplan
}

# Ask for confirmation
echo ""
echo "📋 Do you want to apply these changes? (y/n)"
read -r APPLY_CHANGES

if [ "$APPLY_CHANGES" = "y" ] || [ "$APPLY_CHANGES" = "Y" ]; then
    # Apply terraform
    echo "🔧 Applying Terraform configuration..."
    terraform apply tfplan
    
    # Capture and display outputs
    echo ""
    echo "✅ Infrastructure deployment complete!"
    echo ""
    echo "📊 Infrastructure Information:"
    echo "------------------------------------------------"
    terraform output
    
    # Set up Cloud Build trigger
    echo ""
    echo "📋 Would you like to set up a Cloud Build trigger now? (y/n)"
    read -r SETUP_TRIGGER
    
    if [ "$SETUP_TRIGGER" = "y" ] || [ "$SETUP_TRIGGER" = "Y" ]; then
        echo "📋 Enter the GitHub repository owner/name (e.g., 'username/repo'):"
        read -r GITHUB_REPO
        
        echo "📋 Enter the branch to trigger builds on (e.g., 'main'):"
        read -r BRANCH
        
        # Get service account email from terraform output
        SERVICE_ACCOUNT_EMAIL=$(terraform output -raw service_account_email)
        
        # Get Artifact Registry repository URL from terraform output
        ARTIFACT_REGISTRY_URL=$(terraform output -raw artifact_registry_repository_url)
        
        # Get project ID from terraform.tfvars
        PROJECT_ID=$(grep 'project_id' terraform.tfvars | cut -d'"' -f2)
        
        # Get region from terraform.tfvars
        REGION=$(grep 'region' terraform.tfvars | cut -d'"' -f2)
        
        # Get bucket names from terraform output
        ARTIFACTS_BUCKET=$(terraform output -raw artifacts_bucket)
        
        # Create Cloud Build trigger
        echo "🔧 Creating Cloud Build trigger..."
        gcloud beta builds triggers create github \
            --name="btc-trading-bot-trigger" \
            --repo="https://github.com/$GITHUB_REPO" \
            --branch-pattern="$BRANCH" \
            --build-config="cloudbuild.yaml" \
            --service-account="projects/$PROJECT_ID/serviceAccounts/$SERVICE_ACCOUNT_EMAIL" \
            --substitutions="_REGION=$REGION,_REPO_NAME=btc-trading-bot,_ARTIFACTS_BUCKET=$ARTIFACTS_BUCKET,_SERVICE_ACCOUNT_EMAIL=$SERVICE_ACCOUNT_EMAIL,_RUN_SMOKE_TEST=false"
        
        echo "✅ Cloud Build trigger created!"
    else
        echo "ℹ️ Skipping Cloud Build trigger setup."
        echo "   You can create it later using Google Cloud Console."
    fi
else
    echo "ℹ️ Deployment cancelled."
    rm tfplan
fi

echo "------------------------------------------------"
echo "⚙️ To deploy changes in the future, run this script again."
echo "🗑️ To destroy the infrastructure, run: terraform destroy"