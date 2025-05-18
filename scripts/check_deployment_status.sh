#!/bin/bash
# check_deployment_status.sh - Check the status of BTC Trading Bot deployment
#
# This script checks the status of the various components of the BTC Trading Bot
# deployment in GCP.

set -e  # Exit immediately if a command exits with a non-zero status

echo "🔍 Checking BTC Trading Bot deployment status..."
echo "=================================================="

# Get configuration from terraform.tfvars
cd "$(dirname "$0")/../terraform"
PROJECT_ID=$(grep 'project_id' terraform.tfvars | cut -d'"' -f2)
REGION=$(grep 'region' terraform.tfvars | cut -d'"' -f2)
cd ..

# Check if required APIs are enabled
echo "📊 Checking enabled APIs..."
echo "--------------------------------------------------"
for api in artifactregistry.googleapis.com aiplatform.googleapis.com cloudbuild.googleapis.com cloudresourcemanager.googleapis.com containerregistry.googleapis.com iam.googleapis.com secretmanager.googleapis.com storage.googleapis.com; do
    if gcloud services list --enabled --filter="name:$api" --project="$PROJECT_ID" 2>/dev/null | grep -q "$api"; then
        echo "✅ $api is enabled"
    else
        echo "❌ $api is NOT enabled"
    fi
done
echo ""

# Check if service account exists
echo "📊 Checking service account..."
echo "--------------------------------------------------"
if gcloud iam service-accounts describe "btc-trading-bot-sa@${PROJECT_ID}.iam.gserviceaccount.com" --project="$PROJECT_ID" &>/dev/null; then
    echo "✅ Service account btc-trading-bot-sa exists"
else
    echo "❌ Service account btc-trading-bot-sa does NOT exist"
fi
echo ""

# Check if secrets exist
echo "📊 Checking secrets..."
echo "--------------------------------------------------"
for secret in binance-api-key binance-api-secret; do
    if gcloud secrets describe "$secret" --project="$PROJECT_ID" &>/dev/null; then
        latest_version=$(gcloud secrets versions list "$secret" --project="$PROJECT_ID" --format="value(name)" | sort -rn | head -1)
        if [ -n "$latest_version" ]; then
            echo "✅ Secret $secret exists (latest version: $latest_version)"
        else
            echo "⚠️ Secret $secret exists but has no versions"
        fi
    else
        echo "❌ Secret $secret does NOT exist"
    fi
done
echo ""

# Check if storage buckets exist
echo "📊 Checking storage buckets..."
echo "--------------------------------------------------"
for bucket in "${PROJECT_ID}-btc-raw-data" "${PROJECT_ID}-btc-processed-data" "${PROJECT_ID}-btc-models" "${PROJECT_ID}-btc-artifacts" "${PROJECT_ID}-btc-tensorboard"; do
    if gsutil ls -b "gs://$bucket" &>/dev/null; then
        echo "✅ Bucket $bucket exists"
    else
        echo "❌ Bucket $bucket does NOT exist"
    fi
done
echo ""

# Check if Artifact Registry repository exists
echo "📊 Checking Artifact Registry repository..."
echo "--------------------------------------------------"
if gcloud artifacts repositories describe "btc-trading-bot" --project="$PROJECT_ID" --location="$REGION" &>/dev/null; then
    echo "✅ Artifact Registry repository btc-trading-bot exists"
else
    echo "❌ Artifact Registry repository btc-trading-bot does NOT exist"
fi
echo ""

# Check if Vertex AI Tensorboard exists
echo "📊 Checking Vertex AI Tensorboard..."
echo "--------------------------------------------------"
if gcloud ai tensorboards list --region="$REGION" --project="$PROJECT_ID" --filter="display_name:btc-trading-bot-tensorboard" --format="value(name)" | grep -q "tensorboards"; then
    echo "✅ Vertex AI Tensorboard exists"
else
    echo "❌ Vertex AI Tensorboard does NOT exist"
fi
echo ""

# Check if Cloud Build trigger exists
echo "📊 Checking Cloud Build trigger..."
echo "--------------------------------------------------"
if gcloud builds triggers list --region="$REGION" --project="$PROJECT_ID" --filter="name:btc-trading-bot-trigger" --format="value(name)" | grep -q "btc-trading-bot-trigger"; then
    echo "✅ Cloud Build trigger btc-trading-bot-trigger exists"
else
    echo "❌ Cloud Build trigger btc-trading-bot-trigger does NOT exist"
fi
echo ""

# Check if Vertex AI pipelines have been run
echo "📊 Checking Vertex AI pipeline runs..."
echo "--------------------------------------------------"
pipeline_count=$(gcloud ai pipeline-jobs list --region="$REGION" --project="$PROJECT_ID" --filter="display_name~btc-trading-bot" --format="value(name)" | wc -l)
if [ "$pipeline_count" -gt 0 ]; then
    echo "✅ $pipeline_count Vertex AI pipeline runs found"
else
    echo "❌ No Vertex AI pipeline runs found"
fi
echo ""

echo "✅ Deployment status check complete!"
echo "=================================================="
