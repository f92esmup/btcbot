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
echo "📋 Enter the GitHub repository owner/name (e.g., 'username/repo'):"
read -r GITHUB_REPO

# Extract owner and repo name
OWNER=$(echo "$GITHUB_REPO" | cut -d'/' -f1)
REPO=$(echo "$GITHUB_REPO" | cut -d'/' -f2)

# Prompt for branch to trigger builds on
echo "📋 Enter the branch to trigger builds on (e.g., 'main'):"
read -r BRANCH

# Check if the variables are not empty
if [ -z "$PROJECT_ID" ] || [ -z "$REGION" ] || [ -z "$OWNER" ] || [ -z "$REPO" ] || [ -z "$BRANCH" ]; then
    echo "❌ Error: Missing required information."
    echo "   Project ID: $PROJECT_ID"
    echo "   Region: $REGION"
    echo "   GitHub Owner: $OWNER"
    echo "   GitHub Repo: $REPO"
    echo "   Branch: $BRANCH"
    exit 1
fi

# Check if GitHub connection is already set up
GITHUB_CONNECTION_NAME="github-connection"
CONNECTION_STATUS=$(gcloud builds connections describe "$GITHUB_CONNECTION_NAME" --region="$REGION" --format="value(installationState)" 2>/dev/null || echo "NOT_FOUND")

if [ "$CONNECTION_STATUS" = "NOT_FOUND" ]; then
    echo "📋 You need to first connect your GitHub account to Cloud Build."
    echo "   Please run: gcloud builds connections create github $GITHUB_CONNECTION_NAME --region=$REGION"
    echo "   Then follow the instructions to authorize Cloud Build to access your GitHub repositories."
    echo "   After that, please run this script again."
    
    # Ask if the user wants to create the connection now
    echo "📋 Would you like to create the GitHub connection now? (y/n)"
    read -r CREATE_CONNECTION
    
    if [ "$CREATE_CONNECTION" = "y" ] || [ "$CREATE_CONNECTION" = "Y" ]; then
        echo "🔧 Creating GitHub connection..."
        gcloud builds connections create github "$GITHUB_CONNECTION_NAME" \
            --region="$REGION"
        
        echo "⚠️ Important: Please complete the GitHub authorization process in your browser."
        echo "   After authorization is complete, run this script again."
        exit 0
    else
        echo "ℹ️ Please create the GitHub connection manually and run this script again."
        exit 1
    fi
elif [ "$CONNECTION_STATUS" != "COMPLETE" ]; then
    echo "⚠️ GitHub connection exists but its status is: $CONNECTION_STATUS"
    echo "   Please complete the GitHub authorization process by visiting:"
    echo "   https://console.cloud.google.com/cloud-build/triggers/connect?project=$PROJECT_ID"
    echo "   After authorization is complete, run this script again."
    exit 1
fi

# Create Cloud Build trigger
echo "🔧 Creating Cloud Build trigger..."

# Get the repository ID for the connection
echo "🔍 Fetching repository ID..."
REPO_ID=$(gcloud builds repositories list \
    --connection="$GITHUB_CONNECTION_NAME" \
    --region="$REGION" \
    --format="value(name)" \
    --filter="remote_uri ~ $OWNER/$REPO" 2>/dev/null) || true

if [ -z "$REPO_ID" ]; then
    echo "⚠️  Repository not found. Adding repository to connection..."
    gcloud builds repositories create \
        --connection="$GITHUB_CONNECTION_NAME" \
        --remote-uri="https://github.com/$OWNER/$REPO" \
        --region="$REGION"
        
    REPO_ID=$(gcloud builds repositories list \
        --connection="$GITHUB_CONNECTION_NAME" \
        --region="$REGION" \
        --format="value(name)" \
        --filter="remote_uri ~ $OWNER/$REPO")
fi

echo "📝 Using repository ID: $REPO_ID"

# Create the trigger
gcloud beta builds triggers create github \
    --name="btc-trading-bot-trigger" \
    --region="$REGION" \
    --description="Trigger for BTC Trading Bot on $BRANCH branch" \
    --repository="$REPO_ID" \
    --branch-pattern="^$BRANCH$" \
    --build-config="cloudbuild.yaml" \
    --include-logs-with-status \
    --require-approval \
    --substitutions="_REGION=$REGION,_REPO_NAME=btc-trading-bot,_ARTIFACTS_BUCKET=${PROJECT_ID}-btc-artifacts,_RAW_DATA_BUCKET=${PROJECT_ID}-btc-raw-data,_SERVICE_ACCOUNT_EMAIL=btc-trading-bot-sa@${PROJECT_ID}.iam.gserviceaccount.com,_RUN_SMOKE_TEST=false"

echo "✅ Cloud Build trigger created successfully!"
echo "------------------------------------------------"
echo "🔍 You can view your triggers at: https://console.cloud.google.com/cloud-build/triggers?project=$PROJECT_ID"