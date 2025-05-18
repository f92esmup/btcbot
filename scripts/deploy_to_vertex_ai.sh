#!/bin/bash
# deploy_to_vertex_ai.sh - Deploy BTC Trading Bot to Vertex AI
#
# This script automates the deployment of the BTC Trading Bot to Vertex AI
# by compiling the pipeline definition and creating or updating a Vertex AI pipeline.

set -e  # Exit immediately if a command exits with a non-zero status

echo "🚀 Deploying BTC Trading Bot to Vertex AI..."
echo "------------------------------------------------"

# Change to project root directory
cd "$(dirname "$0")/.."
echo "📁 Working directory: $(pwd)"

# Get configuration from terraform.tfvars
PROJECT_ID=$(grep 'project_id' terraform/terraform.tfvars | cut -d'"' -f2)
REGION=$(grep 'region' terraform/terraform.tfvars | cut -d'"' -f2)

# Check if the variables are not empty
if [ -z "$PROJECT_ID" ] || [ -z "$REGION" ]; then
    echo "❌ Error: Missing required information."
    echo "   Project ID: $PROJECT_ID"
    echo "   Region: $REGION"
    exit 1
fi

# Set environment variables
export PROJECT_ID=$PROJECT_ID
export REGION=$REGION
export ARTIFACTS_BUCKET="${PROJECT_ID}-btc-artifacts"
export IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/btc-trading-bot/btc-trading-bot:latest"
export SERVICE_ACCOUNT="btc-trading-bot-sa@${PROJECT_ID}.iam.gserviceaccount.com"

# Build Docker image and push it to Artifact Registry if requested
echo "📋 Do you want to build and push the Docker image? (y/n)"
read -r BUILD_IMAGE

if [ "$BUILD_IMAGE" = "y" ] || [ "$BUILD_IMAGE" = "Y" ]; then
    echo "🔧 Building Docker image..."
    docker build -t "$IMAGE_URI" .
    
    echo "📤 Pushing Docker image to Artifact Registry..."
    gcloud auth configure-docker ${REGION}-docker.pkg.dev --quiet
    docker push "$IMAGE_URI"
    
    echo "✅ Docker image built and pushed successfully!"
else
    echo "ℹ️ Skipping Docker image build and push."
fi

# Compile the pipeline
echo "🔧 Compiling pipeline definition..."
python pipeline_definition.py --output-file=btc_trading_pipeline.json --image-uri="$IMAGE_URI"

# Upload the compiled pipeline to GCS
echo "📤 Uploading pipeline to GCS..."
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
PIPELINE_GCS_PATH="gs://${ARTIFACTS_BUCKET}/pipelines/btc-trading-bot-pipeline-${TIMESTAMP}.json"
gsutil cp btc_trading_pipeline.json "$PIPELINE_GCS_PATH"

# Ask for pipeline parameters
echo "📋 Enter a start date for data download (YYYY-MM-DD, default: 2021-01-01):"
read -r DOWNLOAD_START_DATE
DOWNLOAD_START_DATE=${DOWNLOAD_START_DATE:-"2021-01-01"}

echo "📋 Enter an end date for data download (YYYY-MM-DD, default: 2022-12-31):"
read -r DOWNLOAD_END_DATE
DOWNLOAD_END_DATE=${DOWNLOAD_END_DATE:-"2022-12-31"}

echo "📋 Enter the number of training steps (default: 100000):"
read -r TRAINING_STEPS
TRAINING_STEPS=${TRAINING_STEPS:-"100000"}

echo "📋 Enter the number of backtest episodes (default: 5):"
read -r BACKTEST_EPISODES
BACKTEST_EPISODES=${BACKTEST_EPISODES:-"5"}

# Create or update the pipeline in Vertex AI
echo "🔧 Creating pipeline in Vertex AI..."
python - << EOF
from google.cloud import aiplatform

# Initialize the Vertex AI SDK
aiplatform.init(project='$PROJECT_ID', location='$REGION')

# Create a pipeline job
pipeline_job = aiplatform.PipelineJob(
    display_name='btc-trading-bot-pipeline',
    template_path='$PIPELINE_GCS_PATH',
    parameter_values={
        'project_id': '$PROJECT_ID',
        'region': '$REGION',
        'gcs_bucket': '$ARTIFACTS_BUCKET',
        'download_start_date': '$DOWNLOAD_START_DATE',
        'download_end_date': '$DOWNLOAD_END_DATE',
        'training_steps': $TRAINING_STEPS,
        'n_backtest_episodes': $BACKTEST_EPISODES
    },
    enable_caching=True
)

# Submit the pipeline job
pipeline_job.submit(service_account='$SERVICE_ACCOUNT')
print(f'Pipeline job submitted: {pipeline_job.name}')
EOF

echo "✅ Pipeline deployed to Vertex AI successfully!"
echo "------------------------------------------------"
echo "🔍 You can view your pipeline at: https://console.cloud.google.com/vertex-ai/pipelines?project=$PROJECT_ID"
