/**
 * variables.tf - Terraform variables for BTC Trading Bot infrastructure
 *
 * This file defines all the variables used in the Terraform configuration
 * for the Bitcoin trading bot infrastructure.
 */

variable "project_id" {
  description = "The GCP project ID where resources will be created."
  type        = string
}

variable "project_number" {
  description = "The GCP project number. Required for setting up Cloud Build service account permissions."
  type        = string
}

variable "region" {
  description = "The GCP region where resources will be created."
  type        = string
  default     = "us-central1"
}

variable "zone" {
  description = "The GCP zone where resources will be created."
  type        = string
  default     = "us-central1-a"
}

variable "environment" {
  description = "The environment (dev, test, prod)."
  type        = string
  default     = "dev"
}

variable "tensorboard_instance_id" {
  description = "The ID for the Vertex AI TensorBoard instance."
  type        = string
  default     = "btc-trading-bot-tensorboard"
}

variable "service_account_id" {
  description = "The ID for the service account used by the BTC trading bot."
  type        = string
  default     = "btc-trading-bot-sa"
}

variable "repository_id" {
  description = "The ID for the Artifact Registry repository."
  type        = string
  default     = "btc-trading-bot"
}

variable "binance_api_key" {
  description = "The Binance API key. This should be provided securely, not stored in version control."
  type        = string
  sensitive   = true
  default     = ""
}

variable "binance_api_secret" {
  description = "The Binance API secret. This should be provided securely, not stored in version control."
  type        = string
  sensitive   = true
  default     = ""
}
