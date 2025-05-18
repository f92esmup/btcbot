# Example Terraform variable values - COPY this file to terraform.tfvars and modify
# DO NOT commit terraform.tfvars to version control

# Required variables
project_id      = "btcbot276299"
project_number  = "119524710024" # Find this in GCP Console > IAM & Admin > Settings

# Optional variables with defaults
region          = "europe-southwest1"
zone            = "europe-west3-a"
environment     = "dev"

# Binance API credentials - Keep these secure!
# These variables will be used to set secret values in Secret Manager
# binance_api_key    = "cPSCcYghXSNToeRCVqaiWkPIvZJMb62yNWAKD9em08NGMSKGWOF2WgxKYHuMPDIm"
# binance_api_secret = "rHyyrIaoqhMuoz5G1nIfhZvM8CObXGqipAp103zvMnArsp5MVXl5pV49Qhutt90l"
