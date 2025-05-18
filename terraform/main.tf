/**
 * main.tf - Main Terraform configuration file for BTC Trading Bot infrastructure
 * 
 * This file defines the main infrastructure components required for the 
 * Bitcoin trading bot to operate in Google Cloud Platform.
 */

# Configure the Google Cloud provider
provider "google" {
  project = var.project_id
  region  = var.region
  zone    = var.zone
}

# Enable required Google Cloud APIs
resource "google_project_service" "required_apis" {
  for_each = toset([
    "artifactregistry.googleapis.com",  # For Docker container storage
    "aiplatform.googleapis.com",        # Vertex AI for ML pipelines and training
    "cloudbuild.googleapis.com",        # For CI/CD pipeline
    "cloudresourcemanager.googleapis.com", # For resource management
    "containerregistry.googleapis.com", # Legacy container registry (may be needed)
    "iam.googleapis.com",               # For identity and access management
    "secretmanager.googleapis.com",     # For storing API keys securely
    "storage.googleapis.com",           # For GCS buckets
  ])

  project            = var.project_id
  service            = each.key
  disable_on_destroy = false
}

# Create service account for the BTC Trading Bot
resource "google_service_account" "btc_trading_bot_sa" {
  account_id   = "btc-trading-bot-sa"
  display_name = "BTC Trading Bot Service Account"
  description  = "Service account for BTC trading bot pipelines"
  project      = var.project_id
  depends_on   = [google_project_service.required_apis]
}

# Grant necessary IAM roles to the service account
resource "google_project_iam_member" "storage_admin" {
  project    = var.project_id
  role       = "roles/storage.admin"
  member     = "serviceAccount:${google_service_account.btc_trading_bot_sa.email}"
  depends_on = [google_service_account.btc_trading_bot_sa]
}

resource "google_project_iam_member" "artifact_admin" {
  project    = var.project_id
  role       = "roles/artifactregistry.admin"
  member     = "serviceAccount:${google_service_account.btc_trading_bot_sa.email}"
  depends_on = [google_service_account.btc_trading_bot_sa]
}

resource "google_project_iam_member" "vertex_user" {
  project    = var.project_id
  role       = "roles/aiplatform.user"
  member     = "serviceAccount:${google_service_account.btc_trading_bot_sa.email}"
  depends_on = [google_service_account.btc_trading_bot_sa]
}

resource "google_project_iam_member" "secret_accessor" {
  project    = var.project_id
  role       = "roles/secretmanager.secretAccessor"
  member     = "serviceAccount:${google_service_account.btc_trading_bot_sa.email}"
  depends_on = [google_service_account.btc_trading_bot_sa]
}

# Grant roles to Cloud Build service account
resource "google_project_iam_member" "cloudbuild_storage_admin" {
  project    = var.project_id
  role       = "roles/storage.admin"
  member     = "serviceAccount:${var.project_number}@cloudbuild.gserviceaccount.com"
  depends_on = [google_project_service.required_apis]
}

resource "google_project_iam_member" "cloudbuild_artifact_admin" {
  project    = var.project_id
  role       = "roles/artifactregistry.admin"
  member     = "serviceAccount:${var.project_number}@cloudbuild.gserviceaccount.com"
  depends_on = [google_project_service.required_apis]
}

resource "google_project_iam_member" "cloudbuild_vertex_admin" {
  project    = var.project_id
  role       = "roles/aiplatform.admin"
  member     = "serviceAccount:${var.project_number}@cloudbuild.gserviceaccount.com"
  depends_on = [google_project_service.required_apis]
}

# Create GCS buckets for various purposes
resource "google_storage_bucket" "raw_data_bucket" {
  name          = "${var.project_id}-btc-raw-data"
  location      = var.region
  storage_class = "STANDARD"
  
  versioning {
    enabled = true
  }
  
  uniform_bucket_level_access = true
  depends_on                  = [google_project_service.required_apis]
}

resource "google_storage_bucket" "processed_data_bucket" {
  name          = "${var.project_id}-btc-processed-data"
  location      = var.region
  storage_class = "STANDARD"
  
  versioning {
    enabled = true
  }
  
  uniform_bucket_level_access = true
  depends_on                  = [google_project_service.required_apis]
}

resource "google_storage_bucket" "models_bucket" {
  name          = "${var.project_id}-btc-models"
  location      = var.region
  storage_class = "STANDARD"
  
  versioning {
    enabled = true
  }
  
  uniform_bucket_level_access = true
  depends_on                  = [google_project_service.required_apis]
}

resource "google_storage_bucket" "artifacts_bucket" {
  name          = "${var.project_id}-btc-artifacts"
  location      = var.region
  storage_class = "STANDARD"
  
  versioning {
    enabled = true
  }
  
  uniform_bucket_level_access = true
  depends_on                  = [google_project_service.required_apis]
}

resource "google_storage_bucket" "tensorboard_bucket" {
  name          = "${var.project_id}-btc-tensorboard"
  location      = var.region
  storage_class = "STANDARD"
  
  uniform_bucket_level_access = true
  depends_on                  = [google_project_service.required_apis]
}

# Create Artifact Registry repository for Docker images
resource "google_artifact_registry_repository" "btc_trading_bot_repo" {
  provider      = google
  location      = var.region
  repository_id = "btc-trading-bot"
  description   = "Docker repository for BTC trading bot images"
  format        = "DOCKER"
  depends_on    = [google_project_service.required_apis]
}

# Create Secret Manager secrets for Binance API credentials
resource "google_secret_manager_secret" "binance_api_key" {
  secret_id = "binance-api-key"
  
  replication {
    automatic = true
  }
  
  depends_on = [google_project_service.required_apis]
  
  lifecycle {
    prevent_destroy = true
    ignore_changes = all
  }
}

resource "google_secret_manager_secret" "binance_api_secret" {
  secret_id = "binance-api-secret"
  
  replication {
    automatic = true
  }
  
  depends_on = [google_project_service.required_apis]
  
  lifecycle {
    prevent_destroy = true
    ignore_changes = all
  }
}

# Create Vertex AI TensorBoard instance
resource "google_vertex_ai_tensorboard" "btc_trading_bot_tensorboard" {
  display_name = "btc-trading-bot-tensorboard"
  project      = var.project_id
  region       = var.region
  
  depends_on = [google_project_service.required_apis]
}
