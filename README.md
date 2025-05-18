# BTC Trading Bot with Reinforcement Learning and Transformers

A cutting-edge Bitcoin trading bot that uses Reinforcement Learning with Transformer architecture for autonomous trading on Binance futures. This project is designed to be fully cloud-native, running on Google Cloud Platform (GCP) with a robust MLOps pipeline.

## Overview

The BTC Trading Bot is an advanced algorithmic trading system for Bitcoin futures (BTC/USDT) on Binance. It leverages:

- **Reinforcement Learning (RL)**: A Soft Actor-Critic (SAC) agent that learns to make trading decisions autonomously
- **Transformer Architecture**: For processing market data sequences and portfolio state
- **Google Cloud Platform (GCP)**: For infrastructure, training, and deployment
- **MLOps Pipeline**: End-to-end automation from data acquisition to model deployment

## Project Structure

```
├── cloudbuild.yaml       # CI/CD configuration for Cloud Build
├── Dockerfile            # Container definition for all components
├── pipeline_definition.py # Vertex AI Pipeline definition
├── requirements.txt      # Python dependencies
├── doc/                  # Documentation
├── scripts/              # Deployment and utility scripts
├── src/                  # Source code
│   ├── agent/            # RL agent with Transformer
│   ├── backtesting/      # Backtesting framework
│   ├── components/       # Vertex AI Pipeline components
│   ├── data_acquisition/ # Data download from Binance
│   ├── environments/     # Trading environment (gym compatible)
│   ├── preprocessing/    # Data preprocessing
│   └── utils/            # Utilities
├── terraform/            # Infrastructure as Code (IaC)
└── tests/                # Unit tests
```

## Features

- **Autonomous Trading**: The agent learns to make trading decisions (open long/short, close, hold) based on market data and portfolio state
- **Infrastructure as Code**: All GCP resources defined with Terraform
- **CI/CD Pipeline**: Automated testing, building, and deployment with Cloud Build
- **Vertex AI Integration**: End-to-end ML pipeline with data acquisition, preprocessing, training, and backtesting
- **Cloud-Native**: Designed to run entirely on GCP with scalable compute resources
- **Modular Design**: Components can be improved or replaced independently

## Setup and Deployment

### Prerequisites

- Google Cloud Platform (GCP) account with billing enabled
- Terraform installed locally
- Google Cloud SDK (gcloud) installed and configured
- Binance API key and secret (for data acquisition)

### Infrastructure Setup

1. Configure and initialize Terraform:
   ```bash
   cd scripts
   ./setup_terraform.sh
   ```

2. Deploy GCP infrastructure:
   ```bash
   ./deploy_infrastructure.sh
   ```

This creates all necessary GCP resources:
- GCS buckets for data and artifacts
- Artifact Registry for Docker images
- Secret Manager for Binance API credentials
- Vertex AI TensorBoard instance
- Service accounts and IAM permissions

### CI/CD Pipeline

The CI/CD pipeline is configured in `cloudbuild.yaml` and automatically:
1. Runs linting and unit tests
2. Builds and pushes the Docker image
3. Compiles the Vertex AI Pipeline
4. Deploys the pipeline to Vertex AI

The pipeline can be triggered manually or automatically on GitHub commits.

### ML Pipeline

The Vertex AI Pipeline includes the following stages:
1. **Data Acquisition**: Downloads historical OHLCV data from Binance
2. **Preprocessing**: Feature engineering, normalization, and sequence creation
3. **Training**: RL agent training with GPU acceleration
4. **Backtesting**: Evaluation of the trained agent

## Development

### Local Development

1. Clone the repository
2. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
3. Run tests:
   ```bash
   pytest tests/
   ```

### Adding Features

1. Create a feature branch
2. Implement changes
3. Add tests
4. Submit a pull request

## License

[Specify your license here]

## Acknowledgments

- Design developed with assistance from Gemini AI