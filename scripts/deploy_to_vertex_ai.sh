#!/bin/bash
# deploy_to_vertex_ai.sh - Deploy BTC Trading Bot pipeline to Vertex AI
#
# This script compiles the pipeline definition and creates/updates the
# pipeline in Vertex AI Pipelines.

set -e  # Exit immediately if a command exits with a non-zero status

echo "🚀 Deploying BTC Trading Bot pipeline to Vertex AI..."
echo "------------------------------------------------"

# Get configuration from terraform.tfvars
cd "$(dirname "$0")/../terraform"
PROJECT_ID=$(grep 'project_id' terraform.tfvars | cut -d'"' -f2)
REGION=$(grep 'region' terraform.tfvars | cut -d'"' -f2)
cd ..

# Make sure needed packages are installed
echo "📦 Installing required packages..."
pip install -q kfp==1.8.22 google-cloud-aiplatform google-cloud-storage

# Compile the pipeline
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_FILE="tmp/compiled_pipeline_${TIMESTAMP}.json"

echo "🔧 Compiling pipeline definition..."
mkdir -p tmp
python pipeline_definition.py --output-file "$OUTPUT_FILE"

# Upload the pipeline to GCS
ARTIFACTS_BUCKET="${PROJECT_ID}-btc-artifacts"
GCS_PATH="gs://${ARTIFACTS_BUCKET}/pipelines/btc-trading-bot-pipeline-${TIMESTAMP}.json"

echo "📤 Uploading pipeline to GCS: $GCS_PATH"
gsutil cp "$OUTPUT_FILE" "$GCS_PATH"

# Get Binance API credentials from Secret Manager
echo "🔑 Retrieving Binance API credentials from Secret Manager..."
BINANCE_API_KEY=$(gcloud secrets versions access latest --secret="binance-api-key" --project="$PROJECT_ID")
BINANCE_API_SECRET=$(gcloud secrets versions access latest --secret="binance-api-secret" --project="$PROJECT_ID")

# Configure pipeline parameters
SERVICE_ACCOUNT="btc-trading-bot-sa@${PROJECT_ID}.iam.gserviceaccount.com"

echo "📋 Enter start date for data download (YYYY-MM-DD, default: 2023-01-01):"
read -r DOWNLOAD_START_DATE
DOWNLOAD_START_DATE=${DOWNLOAD_START_DATE:-"2023-01-01"}

echo "📋 Enter end date for data download (YYYY-MM-DD, default: $(date +%Y-%m-%d)):"
read -r DOWNLOAD_END_DATE
DOWNLOAD_END_DATE=${DOWNLOAD_END_DATE:-$(date +%Y-%m-%d)}

echo "📋 Enter training steps (default: 100000):"
read -r TRAINING_STEPS
TRAINING_STEPS=${TRAINING_STEPS:-100000}

echo "📋 Enter number of backtest episodes (default: 5):"
read -r N_BACKTEST_EPISODES
N_BACKTEST_EPISODES=${N_BACKTEST_EPISODES:-5}

# Deploy the pipeline to Vertex AI
echo "🚀 Deploying pipeline to Vertex AI..."
python - << EOF
from google.cloud import aiplatform

aiplatform.init(project='${PROJECT_ID}', location='${REGION}')

# Create pipeline job
pipeline_job = aiplatform.PipelineJob(
    display_name='btc-trading-bot-pipeline-${TIMESTAMP}',
    template_path='${GCS_PATH}',
    parameter_values={
        'project_id': '${PROJECT_ID}',
        'region': '${REGION}',
        'gcs_bucket': '${ARTIFACTS_BUCKET}',
        'binance_api_key': '${BINANCE_API_KEY}',
        'binance_api_secret': '${BINANCE_API_SECRET}',
        'download_start_date': '${DOWNLOAD_START_DATE}',
        'download_end_date': '${DOWNLOAD_END_DATE}',
        'symbol': 'BTCUSDT',
        'timeframe': '1h',
        'training_steps': ${TRAINING_STEPS},
        'n_backtest_episodes': ${N_BACKTEST_EPISODES}
    },
    enable_caching=True
)

# Submit the pipeline job
pipeline_job.submit(service_account='${SERVICE_ACCOUNT}')
print(f'Pipeline job submitted: {pipeline_job.resource_name}')
EOF

echo "✅ Pipeline deployment complete!"
echo "------------------------------------------------"
echo "🔍 You can view your pipeline runs at: https://console.cloud.google.com/vertex-ai/pipelines/runs?project=$PROJECT_ID"