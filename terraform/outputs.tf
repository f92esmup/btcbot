/**
 * outputs.tf - Terraform outputs for BTC Trading Bot infrastructure
 *
 * This file defines the outputs that will be displayed after the Terraform
 * configuration is applied, providing useful information about the created resources.
 */

output "service_account_email" {
  description = "The email address of the service account created for the BTC trading bot."
  value       = google_service_account.btc_trading_bot_sa.email
}

output "raw_data_bucket" {
  description = "The name of the GCS bucket for storing raw data."
  value       = google_storage_bucket.raw_data_bucket.name
}

output "processed_data_bucket" {
  description = "The name of the GCS bucket for storing processed data."
  value       = google_storage_bucket.processed_data_bucket.name
}

output "models_bucket" {
  description = "The name of the GCS bucket for storing trained models."
  value       = google_storage_bucket.models_bucket.name
}

output "artifacts_bucket" {
  description = "The name of the GCS bucket for storing pipeline artifacts."
  value       = google_storage_bucket.artifacts_bucket.name
}

output "tensorboard_bucket" {
  description = "The name of the GCS bucket for storing TensorBoard logs."
  value       = google_storage_bucket.tensorboard_bucket.name
}

output "artifact_registry_repository" {
  description = "The name of the Artifact Registry repository for Docker images."
  value       = google_artifact_registry_repository.btc_trading_bot_repo.name
}

output "artifact_registry_repository_url" {
  description = "The URL of the Artifact Registry repository for Docker images."
  value       = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.btc_trading_bot_repo.repository_id}"
}

output "tensorboard_instance_id" {
  description = "The ID of the Vertex AI TensorBoard instance."
  value       = google_vertex_ai_tensorboard.btc_trading_bot_tensorboard.id
}

output "api_services_enabled" {
  description = "The list of GCP API services enabled."
  value       = [for api in google_project_service.required_apis : api.service]
}
