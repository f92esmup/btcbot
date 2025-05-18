/**
 * versions.tf - Terraform and provider version constraints
 *
 * This file defines the required versions for Terraform and the Google provider.
 */

terraform {
  required_version = ">= 1.0.0"
  
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = ">= 4.0.0, < 5.0.0"
    }
  }
  
  # Uncomment this block to configure a GCS backend for Terraform state
  # backend "gcs" {
  #   bucket  = "your-terraform-state-bucket"
  #   prefix  = "btc-trading-bot/terraform/state"
  # }
}
