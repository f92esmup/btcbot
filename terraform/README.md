# Terraform Configuration for BTC Trading Bot

This directory contains Terraform configuration files for provisioning the Google Cloud Platform (GCP) infrastructure required for the Bitcoin Trading Bot.

## Infrastructure Components

The Terraform configuration creates the following resources:

1. **Service Accounts**:
   - BTC Trading Bot Service Account with appropriate IAM permissions
   - Permissions for Cloud Build Service Account

2. **Storage**:
   - GCS buckets for raw data, processed data, models, artifacts, and TensorBoard logs
   - Artifact Registry repository for Docker images

3. **Secrets**:
   - Secret Manager secrets for Binance API credentials

4. **Machine Learning**:
   - Vertex AI TensorBoard instance

## Getting Started

### Prerequisites

- [Terraform](https://www.terraform.io/downloads.html) >= 1.0.0
- [Google Cloud SDK](https://cloud.google.com/sdk/docs/install)
- Access to a Google Cloud project with billing enabled

### Configuration

1. Copy the example variables file:
   ```
   cp terraform.tfvars.example terraform.tfvars
   ```

2. Edit `terraform.tfvars` with your specific configuration:
   - Set your GCP project ID and project number
   - Set your preferred region and zone
   - Set your Binance API credentials (optional at this stage)

3. Initialize Terraform:
   ```
   terraform init
   ```

4. Review the planned changes:
   ```
   terraform plan
   ```

5. Apply the configuration:
   ```
   terraform apply
   ```

6. To destroy the infrastructure (when needed):
   ```
   terraform destroy
   ```

### Backend Configuration

For production use, it's recommended to store the Terraform state in a GCS bucket instead of locally. To enable this:

1. Create a GCS bucket for the Terraform state:
   ```
   gcloud storage buckets create gs://your-terraform-state-bucket --location=us-central1
   ```

2. Uncomment and modify the backend configuration in `versions.tf`:
   ```hcl
   backend "gcs" {
     bucket  = "your-terraform-state-bucket"
     prefix  = "btc-trading-bot/terraform/state"
   }
   ```

3. Reinitialize Terraform:
   ```
   terraform init
   ```

## Notes

- Secret values for Binance API credentials should be managed securely and not committed to version control.
- Some resources may incur costs in your GCP project.
- The service account permissions follow the principle of least privilege but might need adjustments based on your specific requirements.
- API enablement may take some time during the first apply operation.
