#!/bin/bash
# update_secrets.sh - Update Secret Manager secrets for BTC Trading Bot
#
# This script updates the Secret Manager secrets with new values
# without relying on Terraform to handle the secret values.

set -e  # Exit immediately if a command exits with a non-zero status

echo "🔒 Updating Secret Manager secrets for BTC Trading Bot..."
echo "------------------------------------------------"

# Change to the terraform directory to get project ID
cd "$(dirname "$0")/../terraform"
echo "📁 Using directory: $(pwd)"

# Get project ID from terraform.tfvars
if [ -f terraform.tfvars ]; then
    PROJECT_ID=$(grep 'project_id' terraform.tfvars | cut -d'"' -f2)
else
    echo "❌ Error: terraform.tfvars not found. Run setup_terraform.sh first."
    exit 1
fi

# Prompt for Binance API credentials
echo "📋 Enter your Binance API key:"
read -r BINANCE_API_KEY

echo "📋 Enter your Binance API secret:"
read -rs BINANCE_API_SECRET

# Check if secrets exist
echo "🔍 Checking if secrets already exist..."

# Create or update API key secret
if ! gcloud secrets describe binance-api-key --project="$PROJECT_ID" &>/dev/null; then
    echo "🔧 Creating binance-api-key secret..."
    gcloud secrets create binance-api-key --project="$PROJECT_ID" --replication-policy="automatic"
fi

echo "🔄 Adding new version to binance-api-key secret..."
echo -n "$BINANCE_API_KEY" | gcloud secrets versions add binance-api-key --project="$PROJECT_ID" --data-file=-

# Create or update API secret
if ! gcloud secrets describe binance-api-secret --project="$PROJECT_ID" &>/dev/null; then
    echo "🔧 Creating binance-api-secret secret..."
    gcloud secrets create binance-api-secret --project="$PROJECT_ID" --replication-policy="automatic"
fi

echo "🔄 Adding new version to binance-api-secret secret..."
echo -n "$BINANCE_API_SECRET" | gcloud secrets versions add binance-api-secret --project="$PROJECT_ID" --data-file=-

echo "✅ Secrets updated successfully!"
echo "------------------------------------------------"
echo "⚙️ You can now run deploy_infrastructure.sh safely."
