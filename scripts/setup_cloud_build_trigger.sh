#!/bin/bash
# setup_cloud_build_trigger.sh - Configure Cloud Build trigger for BTC Trading Bot
#
# This script sets up a Cloud Build trigger to automatically build and
# deploy the BTC Trading Bot when changes are pushed to the repository.

set -e  # Exit immediately if a command exits with a non-zero status

echo "🚀 Setting up Cloud Build trigger for BTC Trading Bot..."
echo "------------------------------------------------"

# Get configuration from terraform.tfvars
cd "$(dirname "$0")/../terraform"
PROJECT_ID=$(grep 'project_id' terraform.tfvars | cut -d'"' -f2)
REGION=$(grep 'region' terraform.tfvars | cut -d'"' -f2)

# Prompt for GitHub repository information
echo "📋 Enter your GitHub repository in the format 'owner/repo':"
read -r GITHUB_REPO

# Extract owner and repo name
OWNER=$(echo "$GITHUB_REPO" | cut -d'/' -f1)
REPO=$(echo "$GITHUB_REPO" | cut -d'/' -f2)

# Check if the variables are not empty
if [ -z "$PROJECT_ID" ] || [ -z "$REGION" ] || [ -z "$OWNER" ] || [ -z "$REPO" ]; then
    echo "❌ Error: Missing required information."
    echo "   Project ID: $PROJECT_ID"
    echo "   Region: $REGION"
    echo "   GitHub Owner: $OWNER"
    echo "   GitHub Repo: $REPO"
    exit 1
fi

# Check if GitHub connection is already set up
if ! gcloud builds connections describe github-connection --region="$REGION" &>/dev/null; then
    echo "📋 You need to first connect your GitHub account to Cloud Build."
    echo "   Please run: gcloud beta builds connections create github --region=$REGION"
    echo "   Then follow the instructions to authorize Cloud Build to access your GitHub repositories."
    echo "   After that, please run this script again."
    exit 1
fi

# Create Cloud Build trigger
echo "🔧 Creating Cloud Build trigger..."
gcloud builds triggers create github \
    --name="btc-trading-bot-trigger" \
    --region="$REGION" \
    --repo="$OWNER/$REPO" \
    --branch-pattern="^main$" \
    --build-config="cloudbuild.yaml" \
    --include-logs-with-status \
    --require-approval \
    --substitutions="_REGION=$REGION,_REPO_NAME=btc-trading-bot,_ARTIFACTS_BUCKET=${PROJECT_ID}-btc-artifacts,_RAW_DATA_BUCKET=${PROJECT_ID}-btc-raw-data,_SERVICE_ACCOUNT_EMAIL=btc-trading-bot-sa@${PROJECT_ID}.iam.gserviceaccount.com,_RUN_SMOKE_TEST=false"

echo "✅ Cloud Build trigger created successfully!"
echo "------------------------------------------------"
echo "🔍 You can view your triggers at: https://console.cloud.google.com/cloud-build/triggers?project=$PROJECT_ID"