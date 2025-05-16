"""
Script para construir y subir la imagen Docker a Artifact Registry usando Google Cloud Build.
"""
import argparse
import os
import subprocess
from common import config, clients

def create_artifact_repo():
    """
    Crea un repositorio en Artifact Registry si no existe.
    """
    # Esto se ejecuta directamente con gcloud ya que la API de Artifact Registry
    # es más compleja de usar directamente
    repo_format = "docker"
    repo_location = config.REGION
    repo_name = config.ARTIFACT_REPO
    project = config.PROJECT_ID
    
    # Verificar si el repositorio ya existe
    list_command = [
        "gcloud", "artifacts", "repositories", "list",
        f"--project={project}",
        f"--location={repo_location}",
        f"--filter=name:repositories/{repo_name}",
        "--format=value(name)"
    ]
    
    result = subprocess.run(list_command, capture_output=True, text=True)
    if repo_name in result.stdout:
        print(f"El repositorio {repo_name} ya existe.")
        return
    
    # Crear el repositorio
    create_command = [
        "gcloud", "artifacts", "repositories", "create", repo_name,
        f"--project={project}",
        f"--location={repo_location}",
        f"--repository-format={repo_format}",
        "--description=Repositorio para imágenes Docker de BTCBot"
    ]
    
    subprocess.run(create_command, check=True)
    print(f"Repositorio {repo_name} creado exitosamente.")

def build_and_push_image(tag, use_gpu=False):
    """
    Construye y sube la imagen Docker a Artifact Registry usando Google Cloud Build.
    
    Args:
        tag: Etiqueta para la imagen (ej: "latest", "v1", etc).
        use_gpu: Indica si se debe construir la imagen con soporte para GPU.
    """
    # Obtener la ruta del proyecto
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Seleccionar el Dockerfile adecuado
    dockerfile_name = "Dockerfile.gpu" if use_gpu else "Dockerfile"
    image_suffix = "-gpu" if use_gpu else ""
    training_image_tag = f"{config.TRAINING_IMAGE_NAME}{image_suffix}:{tag}"
    
    # Construir la imagen con Cloud Build
    print(f"Construyendo imagen{'GPU' if use_gpu else ''} con Cloud Build: {training_image_tag}")
    
    # Crear un cloudbuild.yaml temporal
    cloudbuild_content = f"""
steps:
- name: 'gcr.io/cloud-builders/docker'
  args: ['build', 
         '-t', '{training_image_tag}', 
         '-f', '{dockerfile_name}', 
         '--build-arg', 'BUILDKIT_INLINE_CACHE=1',
         '--progress=plain',
         '.']
  timeout: '3600s'
images: ['{training_image_tag}']
timeout: '3600s'
options:
  machineType: 'E2_HIGHCPU_8'
  diskSizeGb: '100'
  env:
    - 'DOCKER_BUILDKIT=1'
    """
    
    cloudbuild_file = os.path.join(project_dir, "cloudbuild.yaml")
    try:
        # Guardar el archivo temporalmente
        with open(cloudbuild_file, "w") as f:
            f.write(cloudbuild_content)
        
        # Comando con el archivo de configuración
        build_command = [
            "gcloud", "builds", "submit",
            "--project", config.PROJECT_ID,
            "--region", config.REGION,
            "--verbosity", "info",
            "--log-http",
            "--config", "cloudbuild.yaml",
            project_dir
        ]
        
        print(f"Ejecutando: {' '.join(build_command)}")
        try:
            subprocess.run(build_command, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error al construir la imagen: {e}")
            print(f"Comando: {' '.join(build_command)}")
            raise
        
    finally:
        # Eliminar el archivo temporal si existe
        if os.path.exists(cloudbuild_file):
            os.remove(cloudbuild_file)
    
    print(f"Imagen {training_image_tag} construida y subida exitosamente con Cloud Build.")
    return training_image_tag

def main(tag, use_gpu=False):
    """
    Función principal para construir y subir la imagen Docker usando Cloud Build.
    
    Args:
        tag: Etiqueta para la imagen.
        use_gpu: Indica si se debe construir la imagen con soporte para GPU.
    """
    # Crear repositorio si no existe
    create_artifact_repo()
    
    # Construir y subir imagen
    image_uri = build_and_push_image(tag, use_gpu)
    
    print(f"Proceso completado. Imagen disponible en: {image_uri}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Construir y subir imagen Docker a Artifact Registry usando Cloud Build")
    parser.add_argument("--tag", default="latest", help="Etiqueta para la imagen Docker (default: latest)")
    parser.add_argument("--gpu", action="store_true", help="Construir imagen con soporte para GPU")
    
    args = parser.parse_args()
    main(args.tag, args.gpu)
