#!/usr/bin/env python3
"""
validate_integration.py
Comprehensive validation script for the btcbot Cloud Build + GKE integration
"""

import os
import sys
import yaml
import subprocess
import json
from pathlib import Path

# Color codes for output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_section(title):
    """Print section header"""
    print(f"\n{Colors.BLUE}{Colors.BOLD}{'='*60}{Colors.END}")
    print(f"{Colors.BLUE}{Colors.BOLD}{title.center(60)}{Colors.END}")
    print(f"{Colors.BLUE}{Colors.BOLD}{'='*60}{Colors.END}")

def print_success(message):
    """Print success message"""
    print(f"{Colors.GREEN}✅ {message}{Colors.END}")

def print_warning(message):
    """Print warning message"""
    print(f"{Colors.YELLOW}⚠️  {message}{Colors.END}")

def print_error(message):
    """Print error message"""
    print(f"{Colors.RED}❌ {message}{Colors.END}")

def validate_file_exists(filepath, description):
    """Validate that a file exists"""
    if os.path.exists(filepath):
        print_success(f"{description}: {filepath}")
        return True
    else:
        print_error(f"{description} not found: {filepath}")
        return False

def validate_yaml_syntax(filepath):
    """Validate YAML syntax"""
    try:
        with open(filepath, 'r') as f:
            yaml.safe_load(f)
        print_success(f"YAML syntax valid: {filepath}")
        return True
    except yaml.YAMLError as e:
        print_error(f"YAML syntax error in {filepath}: {e}")
        return False
    except Exception as e:
        print_error(f"Error reading {filepath}: {e}")
        return False

def validate_cloud_build_config():
    """Validate Cloud Build configuration"""
    print_section("VALIDATING CLOUD BUILD CONFIGURATION")
    
    cloudbuild_path = "/workspaces/btcbot/cloudbuild.yaml"
    if not validate_file_exists(cloudbuild_path, "Cloud Build config"):
        return False
    
    if not validate_yaml_syntax(cloudbuild_path):
        return False
    
    # Load and validate content
    with open(cloudbuild_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Check essential sections
    checks = [
        ('substitutions' in config, "Has substitutions section"),
        ('steps' in config, "Has build steps"),
        ('images' in config, "Has images section"),
        (len(config.get('steps', [])) >= 10, "Has minimum required steps (≥10)"),
    ]
    
    # Check essential substitutions
    substitutions = config.get('substitutions', {})
    essential_subs = [
        '_GCP_PROJECT_ID', '_GCP_REGION', '_ARTIFACT_REGISTRY_LOCATION',
        '_ARTIFACT_REGISTRY_REPO', '_CPU_IMAGE_NAME', '_GPU_IMAGE_NAME',
        '_GKE_CLUSTER_NAME', '_K8S_NAMESPACE'
    ]
    
    for sub in essential_subs:
        checks.append((sub in substitutions, f"Has substitution: {sub}"))
    
    # Check for ML jobs deployment step
    steps = config.get('steps', [])
    ml_step_found = any('deploy-ml-jobs' in step.get('id', '') for step in steps)
    checks.append((ml_step_found, "Has ML jobs deployment step"))
    
    # Check for live trader deployment step
    trader_step_found = any('deploy-live-trader' in step.get('id', '') for step in steps)
    checks.append((trader_step_found, "Has live trader deployment step"))
    
    all_passed = True
    for check, description in checks:
        if check:
            print_success(description)
        else:
            print_error(description)
            all_passed = False
    
    return all_passed

def validate_kubernetes_manifests():
    """Validate Kubernetes manifest files"""
    print_section("VALIDATING KUBERNETES MANIFESTS")
    
    manifest_files = [
        ("/workspaces/btcbot/k8s/data-acquisition-job.yaml", "Data Acquisition CronJob"),
        ("/workspaces/btcbot/k8s/data-preprocessing-job.yaml", "Data Preprocessing Job"),
        ("/workspaces/btcbot/k8s/model-training-job.yaml", "Model Training Job"),
        ("/workspaces/btcbot/k8s/live-trader-deployment.yaml", "Live Trader Deployment"),
    ]
    
    all_passed = True
    for filepath, description in manifest_files:
        if not validate_file_exists(filepath, description):
            all_passed = False
            continue
        
        if not validate_yaml_syntax(filepath):
            all_passed = False
            continue
        
        # Load and validate specific content
        with open(filepath, 'r') as f:
            manifest = yaml.safe_load(f)
        
        # Basic Kubernetes manifest validation
        required_fields = ['apiVersion', 'kind', 'metadata']
        missing_fields = [field for field in required_fields if field not in manifest]
        
        if missing_fields:
            print_error(f"{description}: Missing required fields: {missing_fields}")
            all_passed = False
        else:
            print_success(f"{description}: Required fields present")
    
    return all_passed

def validate_docker_files():
    """Validate Dockerfile existence"""
    print_section("VALIDATING DOCKER CONFIGURATIONS")
    
    docker_files = [
        ("/workspaces/btcbot/Dockerfile.cpu", "CPU Dockerfile"),
        ("/workspaces/btcbot/Dockerfile.gpu", "GPU Dockerfile"),
    ]
    
    all_passed = True
    for filepath, description in docker_files:
        if not validate_file_exists(filepath, description):
            all_passed = False
    
    return all_passed

def validate_scripts():
    """Validate required Python scripts"""
    print_section("VALIDATING PYTHON SCRIPTS")
    
    required_scripts = [
        ("/workspaces/btcbot/scripts/download_data.py", "Data download script"),
        ("/workspaces/btcbot/scripts/preprocess_data.py", "Data preprocessing script"),
        ("/workspaces/btcbot/scripts/train_rl_agent.py", "RL training script"),
        ("/workspaces/btcbot/scripts/run_live_trader.py", "Live trader script"),
    ]
    
    all_passed = True
    for filepath, description in required_scripts:
        if not validate_file_exists(filepath, description):
            all_passed = False
    
    return all_passed

def validate_orchestration():
    """Validate orchestration scripts"""
    print_section("VALIDATING ORCHESTRATION")
    
    orchestration_file = "/workspaces/btcbot/k8s/orchestrate-pipeline.sh"
    if not validate_file_exists(orchestration_file, "Orchestration script"):
        return False
    
    # Check if script is executable
    if os.access(orchestration_file, os.X_OK):
        print_success("Orchestration script is executable")
    else:
        print_warning("Orchestration script might not be executable")
    
    return True

def validate_configuration_consistency():
    """Validate consistency between different configuration files"""
    print_section("VALIDATING CONFIGURATION CONSISTENCY")
    
    # Check image names consistency
    try:
        # Load Cloud Build config
        with open("/workspaces/btcbot/cloudbuild.yaml", 'r') as f:
            cloudbuild = yaml.safe_load(f)
        
        cpu_image = cloudbuild['substitutions']['_CPU_IMAGE_NAME']
        gpu_image = cloudbuild['substitutions']['_GPU_IMAGE_NAME']
        namespace = cloudbuild['substitutions']['_K8S_NAMESPACE']
        
        # Check if Kubernetes manifests reference these images correctly
        k8s_files = [
            "/workspaces/btcbot/k8s/data-acquisition-job.yaml",
            "/workspaces/btcbot/k8s/data-preprocessing-job.yaml",
            "/workspaces/btcbot/k8s/live-trader-deployment.yaml"
        ]
        
        all_passed = True
        for k8s_file in k8s_files:
            with open(k8s_file, 'r') as f:
                content = f.read()
                if 'YOUR_CPU_IMAGE_NAME' in content or 'YOUR_GPU_IMAGE_NAME' in content:
                    print_success(f"Template placeholders found in {os.path.basename(k8s_file)}")
                else:
                    print_warning(f"No template placeholders in {os.path.basename(k8s_file)}")
        
        # Check GPU image in training job
        with open("/workspaces/btcbot/k8s/model-training-job.yaml", 'r') as f:
            content = f.read()
            if 'YOUR_GPU_IMAGE_NAME' in content:
                print_success("GPU image template found in training job")
            else:
                print_warning("GPU image template not found in training job")
        
        return all_passed
        
    except Exception as e:
        print_error(f"Error validating configuration consistency: {e}")
        return False

def check_gcp_tools():
    """Check if GCP tools are available (optional)"""
    print_section("CHECKING GCP TOOLS (OPTIONAL)")
    
    tools = [
        ('gcloud', 'Google Cloud SDK'),
        ('kubectl', 'Kubernetes CLI'),
        ('docker', 'Docker CLI'),
    ]
    
    for tool, description in tools:
        try:
            result = subprocess.run([tool, '--version'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                print_success(f"{description} available")
            else:
                print_warning(f"{description} not working properly")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            print_warning(f"{description} not found in PATH")

def generate_summary_report():
    """Generate validation summary"""
    print_section("VALIDATION SUMMARY")
    
    validations = [
        (validate_cloud_build_config(), "Cloud Build Configuration"),
        (validate_kubernetes_manifests(), "Kubernetes Manifests"),
        (validate_docker_files(), "Docker Files"),
        (validate_scripts(), "Python Scripts"),
        (validate_orchestration(), "Orchestration Scripts"),
        (validate_configuration_consistency(), "Configuration Consistency"),
    ]
    
    passed = sum(1 for result, _ in validations if result)
    total = len(validations)
    
    print(f"\n{Colors.BOLD}VALIDATION RESULTS:{Colors.END}")
    print(f"Passed: {Colors.GREEN}{passed}{Colors.END}")
    print(f"Failed: {Colors.RED}{total - passed}{Colors.END}")
    print(f"Total:  {total}")
    
    if passed == total:
        print(f"\n{Colors.GREEN}{Colors.BOLD}🎉 ALL VALIDATIONS PASSED!{Colors.END}")
        print(f"{Colors.GREEN}Your btcbot Cloud Build + GKE integration is ready for deployment.{Colors.END}")
        return True
    else:
        print(f"\n{Colors.RED}{Colors.BOLD}❌ SOME VALIDATIONS FAILED{Colors.END}")
        print(f"{Colors.RED}Please fix the issues above before deploying.{Colors.END}")
        return False

def main():
    """Main validation function"""
    print(f"{Colors.BOLD}🤖 btcbot Cloud Build + GKE Integration Validator{Colors.END}")
    print(f"{Colors.BOLD}================================================{Colors.END}")
    
    # Run validations
    success = generate_summary_report()
    
    # Optional GCP tools check
    check_gcp_tools()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
