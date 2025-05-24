# pipelines/trading_pipeline.py
import kfp
from kfp import dsl
from kfp.dsl import Output, Artifact # If needed for passing data between components
# from google.cloud import aiplatform # Potentially used by the compiler or for submission logic later

# --- Component Definitions ---

# For components, we often define them by creating a component YAML or directly using @component
# For simplicity here, we'll define Python functions and then use kfp.components.create_component_from_func
# if needed, or directly define them if they are simple command executions.
# A more robust way is to define component YAMLs separately or use custom container components.

# Placeholder for where the compiled pipeline will be stored by the compiler script
PIPELINE_JSON_PKG_PATH = "trading_pipeline.json"

@dsl.component(
    base_image="python:3.9", # A general python image, actual script runs in its own container
    packages_to_install=["google-cloud-aiplatform", "google-cloud-pipeline-components"] # For KFP execution
)
def merge_json_outputs(base_json_str: str, new_json_str: str) -> str:
    import json
    base_dict = json.loads(base_json_str) if base_json_str else {}
    new_dict = json.loads(new_json_str) if new_json_str else {}
    base_dict.update(new_dict)
    return json.dumps(base_dict)


@dsl.container_component
def data_acquisition_op(
    gcp_project_id: str,
    gcs_bucket_name: str,
    gcp_region: str,
    bigquery_log_dataset_id: str,
    use_testnet: str,
    log_level: str,
    cpu_image_uri: str,
    # KFP specific outputs for metadata or small data
    execution_details: Output[Artifact]
):
    """Defines the data acquisition step of the KFP pipeline."""
    return dsl.ContainerSpec(
        image=cpu_image_uri,
        command=["python", "scripts/download_data.py"],
        args=[
            # No specific args for download_data.py in its current form,
            # it reads from config.yaml which uses env vars.
        ],
        # Environment variables are set in the GKE nodes/pods via k8s secret specified
        # in the Vertex AI Pipeline's service account or pod spec if customized.
        # Here, we assume they are available in the execution environment of the container.
        # If direct pass-through is needed, one would use env_vars in KFP component spec.
        # This component outputs a dummy execution_details artifact.
    ).add_node_selector_constraint(
        label_name='cloud.google.com/gke-accelerator',
        # Ensure this step runs on a CPU node if specific node pools are used.
        # For Autopilot, this is less about node pools and more about workload separation.
        # This is more relevant if you have mixed node pools.
        # value='cpu-pool' # Example, not strictly needed for Autopilot unless optimizing
    ).set_display_name("Data Acquisition")


@dsl.container_component
def data_preprocessing_op(
    gcp_project_id: str,
    gcs_bucket_name: str,
    gcp_region: str,
    bigquery_log_dataset_id: str,
    use_testnet: str,
    log_level: str,
    cpu_image_uri: str,
    # KFP specific outputs
    execution_details: Output[Artifact],
    # Depends on output of data_acquisition_op.
    # KFP handles dependency via input parameters of type Artifact.
    # For simple sequencing, just using .after() is also an option.
    upstream_artifact: Artifact # Dummy input to ensure sequencing
):
    """Defines the data preprocessing step of the KFP pipeline."""
    return dsl.ContainerSpec(
        image=cpu_image_uri,
        command=["python", "scripts/preprocess_data.py"],
        args=[], # preprocess_data.py also uses config.yaml
    ).set_display_name("Data Preprocessing")


@dsl.container_component
def training_op(
    gcp_project_id: str,
    gcs_bucket_name: str,
    gcp_region: str,
    bigquery_log_dataset_id: str,
    use_testnet: str,
    log_level: str,
    gpu_image_uri: str,
    # KFP specific outputs
    execution_details: Output[Artifact],
    # Depends on output of data_preprocessing_op.
    upstream_artifact: Artifact # Dummy input
):
    """Defines the training step of the KFP pipeline, requesting GPU."""
    return dsl.ContainerSpec(
        image=gpu_image_uri,
        command=["python", "scripts/train_rl_agent.py"],
        args=[
            "--config", "src/config.yaml",
            # Timesteps could be a pipeline parameter if you want to vary it per run
            # "--timesteps", "1000000" # Example, or read from config
        ],
    ).set_display_name("Model Training").set_gpu_limit(1).add_node_selector_constraint(
         # For GKE Autopilot, specifying GPU directly in resources is preferred.
         # This constraint is more for GKE Standard or if specific node selectors are needed.
         # For Autopilot, the 'machine_type' or 'accelerator' is part of resource requests.
         # However, KFP's .set_gpu_limit() is the standard way to request GPUs for Vertex AI Pipelines.
        'cloud.google.com/gke-accelerator', 'NVIDIA_TESLA_T4'
    )


# --- Pipeline Definition ---
@dsl.pipeline(
    name="btcbot-mlops-pipeline",
    description="Orchestrates data acquisition, preprocessing, and model training for BTCBot.",
    # pipeline_root: A GCS path where KFP stores pipeline artifacts.
    # This should be parameterized or configured globally for Vertex AI Pipelines.
    # Example: "gs://YOUR_GCS_BUCKET_NAME/kfp_pipeline_root/btcbot"
    # This will be passed during pipeline submission by Cloud Build.
)
def btc_trading_pipeline(
    # Parameters to be passed to the pipeline, potentially by Cloud Build
    gcp_project_id: str,
    gcs_bucket_name: str,
    gcp_region: str,
    bigquery_log_dataset_id: str,
    use_testnet: str = "true", # Default value
    log_level: str = "INFO",   # Default value
    cpu_image_uri: str = "gcr.io/your-project/btcbot-cpu:latest", # Placeholder, Cloud Build provides this
    gpu_image_uri: str = "gcr.io/your-project/btcbot-gpu:latest", # Placeholder, Cloud Build provides this
    # Default values are good for local testing/compilation if needed
):
    """Defines the MLOps pipeline for the BTC Trading Bot."""

    # --- Instantiate Components ---
    # Data Acquisition
    data_acquisition_task = data_acquisition_op(
        gcp_project_id=gcp_project_id,
        gcs_bucket_name=gcs_bucket_name,
        gcp_region=gcp_region,
        bigquery_log_dataset_id=bigquery_log_dataset_id,
        use_testnet=use_testnet,
        log_level=log_level,
        cpu_image_uri=cpu_image_uri
    )

    # Data Preprocessing - runs after data acquisition
    data_preprocessing_task = data_preprocessing_op(
        gcp_project_id=gcp_project_id,
        gcs_bucket_name=gcs_bucket_name,
        gcp_region=gcp_region,
        bigquery_log_dataset_id=bigquery_log_dataset_id,
        use_testnet=use_testnet,
        log_level=log_level,
        cpu_image_uri=cpu_image_uri,
        upstream_artifact=data_acquisition_task.outputs["execution_details"]
    ).after(data_acquisition_task)

    # Model Training - runs after data preprocessing
    model_training_task = training_op(
        gcp_project_id=gcp_project_id,
        gcs_bucket_name=gcs_bucket_name,
        gcp_region=gcp_region,
        bigquery_log_dataset_id=bigquery_log_dataset_id,
        use_testnet=use_testnet,
        log_level=log_level,
        gpu_image_uri=gpu_image_uri,
        upstream_artifact=data_preprocessing_task.outputs["execution_details"]
    ).after(data_preprocessing_task)

# Note: The actual environment variables (GCP_PROJECT_ID, etc.) for the scripts
# inside the containers will be sourced from the Kubernetes Secret ('btcbot-env-vars')
# attached to the service account used by Vertex AI Pipelines, or by specifying
# 'env_vars' or 'secrets' directly in the dsl.ContainerSpec if KFP/Vertex AI allows
# direct pass-through from the k8s Secret to these container_ops in a secure way.
# For Vertex AI custom jobs (which KFP components become), you typically set env vars
# or use service accounts that can access secrets.
# The pipeline parameters are for configuring the pipeline itself and its components.
# The KFP components defined here assume the environment inside the container
# will have these variables set, typically through the GKE node's service account
# having access to the specified k8s Secret, or by Vertex AI injecting them.
# Cloud Build will set up the k8s Secret 'btcbot-env-vars'.
# The Pods run by Vertex AI Pipelines would need to be configured to use this Secret,
# often by configuring the default service account in the namespace or specifying one.
```
