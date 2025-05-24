# btcbot Cloud Build + GKE Integration - Deployment Guide

## 🚀 Deployment Overview

This guide provides step-by-step instructions for deploying the btcbot trading system using Google Cloud Build with GKE Autopilot cluster management and Kubernetes Jobs for ML pipeline orchestration.

## 📋 Prerequisites

### Required Google Cloud Services
- Google Cloud Build
- Google Kubernetes Engine (GKE)
- Artifact Registry
- Google Cloud Storage
- Google Cloud Secrets Manager
- BigQuery (for logging)

### Required Tools
```bash
# Install Google Cloud SDK
curl https://sdk.cloud.google.com | bash
exec -l $SHELL

# Install kubectl
gcloud components install kubectl

# Verify installations
gcloud version
kubectl version --client
docker --version
```

## 🔧 Pre-Deployment Setup

### 1. Configure Google Cloud Project
```bash
# Set your project ID
export PROJECT_ID="your-project-id"
gcloud config set project $PROJECT_ID

# Enable required APIs
gcloud services enable cloudbuild.googleapis.com
gcloud services enable container.googleapis.com
gcloud services enable artifactregistry.googleapis.com
gcloud services enable storage.googleapis.com
gcloud services enable secretmanager.googleapis.com
gcloud services enable bigquery.googleapis.com
```

### 2. Create Artifact Registry Repository
```bash
gcloud artifacts repositories create btcbot-images \
    --repository-format=docker \
    --location=europe-south1 \
    --description="btcbot Docker images"
```

### 3. Configure Docker Authentication
```bash
gcloud auth configure-docker europe-south1-docker.pkg.dev
```

### 4. Create Google Cloud Storage Bucket
```bash
gsutil mb -l europe-south1 gs://your-btcbot-bucket
```

### 5. Set up Secret Manager (Optional but Recommended)
```bash
# Create secrets for sensitive data
echo -n "your-binance-api-key" | gcloud secrets create binance-api-key --data-file=-
echo -n "your-binance-secret-key" | gcloud secrets create binance-secret-key --data-file=-
```

## 🚢 Deployment Steps

### 1. Prepare Environment Variables
Create a `.env` file or set environment variables for Cloud Build:
```bash
export SECRET_GCS_BUCKET_NAME="your-btcbot-bucket"
export SECRET_BIGQUERY_LOG_DATASET_ID="btcbot_logs"
export SECRET_USE_TESTNET="true"  # Set to "false" for production
export SECRET_LOG_LEVEL="INFO"
```

### 2. Deploy via Cloud Build
```bash
# Submit build with environment variables
gcloud builds submit . \
    --config=cloudbuild.yaml \
    --substitutions=\
_SECRET_GCS_BUCKET_NAME="$SECRET_GCS_BUCKET_NAME",\
_SECRET_BIGQUERY_LOG_DATASET_ID="$SECRET_BIGQUERY_LOG_DATASET_ID",\
_SECRET_USE_TESTNET="$SECRET_USE_TESTNET",\
_SECRET_LOG_LEVEL="$SECRET_LOG_LEVEL"
```

### 3. Monitor Deployment
```bash
# Watch build progress
gcloud builds list --limit=1 --ongoing

# Get build logs
BUILD_ID=$(gcloud builds list --limit=1 --format="value(id)")
gcloud builds log $BUILD_ID --stream
```

## 🔄 ML Pipeline Orchestration

### Automatic Execution
- **Data Acquisition**: Runs automatically every Sunday at 2:00 AM UTC via CronJob
- **Preprocessing & Training**: Can be triggered manually or automatically after data acquisition

### Manual Pipeline Execution
```bash
# Get GKE credentials
gcloud container clusters get-credentials btcbot-autopilot-cluster \
    --region europe-south1

# Run complete ML pipeline
kubectl apply -f k8s/data-preprocessing-job-processed.yaml -n btcbot
kubectl apply -f k8s/model-training-job-processed.yaml -n btcbot

# Or use the orchestration script
k8s/orchestrate-pipeline.sh --run-acquisition
```

### Monitor ML Jobs
```bash
# Check job status
kubectl get jobs -n btcbot
kubectl get cronjobs -n btcbot

# View job logs
kubectl logs -l app=btcbot,component=data-acquisition -n btcbot
kubectl logs -l app=btcbot,component=data-preprocessing -n btcbot
kubectl logs -l app=btcbot,component=model-training -n btcbot
```

## 📊 Monitoring & Troubleshooting

### Check Live Trader Status
```bash
# Check deployment
kubectl get deployment btcbot-live-trader -n btcbot

# Check pods
kubectl get pods -l app=btcbot,component=live-trader -n btcbot

# View logs
kubectl logs -l app=btcbot,component=live-trader -n btcbot --tail=100
```

### View Secrets and ConfigMaps
```bash
# Check secrets
kubectl get secrets -n btcbot
kubectl describe secret btcbot-env-vars -n btcbot

# Verify environment variables (without showing values)
kubectl get secret btcbot-env-vars -n btcbot -o jsonpath='{.data}' | jq 'keys'
```

### Debug Failed Jobs
```bash
# Get failed job details
kubectl describe job <job-name> -n btcbot

# Check pod events
kubectl get events --sort-by=.metadata.creationTimestamp -n btcbot

# Access job logs
kubectl logs job/<job-name> -n btcbot
```

## 🔒 Security Best Practices

### 1. Use Google Cloud Secrets Manager
Instead of storing sensitive data in Kubernetes secrets, integrate with Secret Manager:
```yaml
# Example: Using Secret Manager CSI driver
apiVersion: v1
kind: SecretProviderClass
metadata:
  name: btcbot-secrets
spec:
  provider: gcp
  parameters:
    secrets: |
      - resourceName: "projects/PROJECT_ID/secrets/binance-api-key/versions/latest"
        path: "binance-api-key"
```

### 2. Use Workload Identity
```bash
# Create Kubernetes service account
kubectl create serviceaccount btcbot-sa -n btcbot

# Bind to Google Service Account
gcloud iam service-accounts add-iam-policy-binding \
    btcbot-gsa@PROJECT_ID.iam.gserviceaccount.com \
    --role roles/iam.workloadIdentityUser \
    --member "serviceAccount:PROJECT_ID.svc.id.goog[btcbot/btcbot-sa]"
```

### 3. Network Policies
```yaml
# Restrict pod-to-pod communication
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: btcbot-network-policy
  namespace: btcbot
spec:
  podSelector:
    matchLabels:
      app: btcbot
  policyTypes:
  - Ingress
  - Egress
  egress:
  - to: []
    ports:
    - protocol: TCP
      port: 443
```

## 📈 Scaling and Performance

### Auto-scaling Live Trader
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: btcbot-live-trader-hpa
  namespace: btcbot
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: btcbot-live-trader
  minReplicas: 1
  maxReplicas: 3
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

### GPU Node Pool for Training
```bash
# Create GPU node pool (if needed)
gcloud container node-pools create gpu-pool \
    --cluster=btcbot-autopilot-cluster \
    --zone=europe-south1-a \
    --accelerator=type=nvidia-tesla-t4,count=1 \
    --num-nodes=0 \
    --enable-autoscaling \
    --min-nodes=0 \
    --max-nodes=2
```

## 🔄 Continuous Integration/Deployment

### GitHub Actions Integration
```yaml
# .github/workflows/deploy.yml
name: Deploy btcbot
on:
  push:
    branches: [main]
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - uses: google-github-actions/setup-gcloud@v0
      with:
        service_account_key: ${{ secrets.GCP_SA_KEY }}
        project_id: ${{ secrets.GCP_PROJECT_ID }}
    - run: |
        gcloud builds submit . --config=cloudbuild.yaml
```

### Rollback Strategy
```bash
# Rollback live trader deployment
kubectl rollout undo deployment/btcbot-live-trader -n btcbot

# Check rollout status
kubectl rollout status deployment/btcbot-live-trader -n btcbot
```

## 🎯 Production Checklist

- [ ] Environment variables configured
- [ ] Secrets properly managed
- [ ] Resource limits set appropriately
- [ ] Monitoring and alerting configured
- [ ] Backup strategy implemented
- [ ] Network policies in place
- [ ] RBAC configured
- [ ] Disaster recovery plan
- [ ] Performance testing completed
- [ ] Security scanning performed

## 🆘 Emergency Procedures

### Stop All Trading
```bash
# Scale down live trader
kubectl scale deployment btcbot-live-trader --replicas=0 -n btcbot

# Stop all ML jobs
kubectl delete jobs -l app=btcbot -n btcbot
```

### Quick Recovery
```bash
# Restart live trader with latest image
kubectl rollout restart deployment/btcbot-live-trader -n btcbot

# Force re-deploy
gcloud builds submit . --config=cloudbuild.yaml
```

## 📞 Support and Maintenance

### Regular Maintenance Tasks
1. **Weekly**: Review ML pipeline logs and performance
2. **Monthly**: Update dependencies and security patches
3. **Quarterly**: Review and optimize resource allocation
4. **Annually**: Disaster recovery testing

### Log Analysis
```bash
# Export logs to BigQuery for analysis
bq query --use_legacy_sql=false '
SELECT *
FROM `PROJECT_ID.btcbot_logs.trading_logs`
WHERE timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 24 HOUR)
ORDER BY timestamp DESC
'
```

## 📚 Additional Resources

- [Google Cloud Build Documentation](https://cloud.google.com/build/docs)
- [GKE Autopilot Documentation](https://cloud.google.com/kubernetes-engine/docs/concepts/autopilot-overview)
- [Kubernetes Jobs Documentation](https://kubernetes.io/docs/concepts/workloads/controllers/job/)
- [Vertex AI Pipelines](https://cloud.google.com/vertex-ai/docs/pipelines)

---

**🎉 Congratulations! Your btcbot is now running on a robust, scalable cloud infrastructure!**
