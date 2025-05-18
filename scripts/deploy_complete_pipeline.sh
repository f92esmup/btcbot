#!/bin/bash
# deploy_complete_pipeline.sh - Complete deployment of BTC Trading Bot to GCP
#
# This script orchestrates the entire deployment process for the BTC Trading Bot:
# 1. Sets up Terraform
# 2. Updates secrets in Secret Manager
# 3. Deploys infrastructure with Terraform
# 4. Sets up Cloud Build trigger
# 5. Deploys the pipeline to Vertex AI

set -e  # Exit immediately if a command exits with a non-zero status

SCRIPT_DIR="$(dirname "$0")"
cd "$SCRIPT_DIR"

echo "🚀 Starting complete deployment of BTC Trading Bot to GCP..."
echo "=================================================="

# Step 1: Set up Terraform
echo "Step 1/5: Setting up Terraform..."
echo "--------------------------------------------------"
./setup_terraform.sh
echo "✅ Terraform setup complete."
echo ""

# Step 2: Update secrets
echo "Step 2/5: Updating secrets in Secret Manager..."
echo "--------------------------------------------------"
./update_secrets.sh
echo "✅ Secrets updated."
echo ""

# Step 3: Deploy infrastructure
echo "Step 3/5: Deploying infrastructure with Terraform..."
echo "--------------------------------------------------"
./deploy_infrastructure.sh
echo "✅ Infrastructure deployment complete."
echo ""

# Step 4: Set up Cloud Build trigger
echo "Step 4/5: Setting up Cloud Build trigger..."
echo "--------------------------------------------------"
./setup_cloud_build_trigger.sh
echo "✅ Cloud Build trigger setup complete."
echo ""

# Step 5: Deploy to Vertex AI
echo "Step 5/5: Deploying pipeline to Vertex AI..."
echo "--------------------------------------------------"
./deploy_to_vertex_ai.sh
echo "✅ Vertex AI pipeline deployment complete."
echo ""

echo "🎉 BTC Trading Bot deployment complete!"
echo "=================================================="
echo "You can now access your resources in the GCP Console:"
echo "- Vertex AI Pipelines: https://console.cloud.google.com/vertex-ai/pipelines/runs"
echo "- Cloud Build: https://console.cloud.google.com/cloud-build/triggers"
echo "- Storage buckets: https://console.cloud.google.com/storage/browser"
echo "- Secret Manager: https://console.cloud.google.com/security/secret-manager"
