"""
Script para construir y subir la imagen Docker a Artifact Registry.
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
    Construye y sube la imagen Docker a Artifact Registry.
    
    Args:
        tag: Etiqueta para la imagen (ej: "latest", "v1", etc).
        use_gpu: Indica si se debe construir la imagen con soporte para GPU.
    """
    # Obtener la ruta del proyecto
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Seleccionar el Dockerfile adecuado y crear la etiqueta correspondiente
    dockerfile = os.path.join(project_dir, "Dockerfile.gpu" if use_gpu else "Dockerfile")
    image_suffix = "-gpu" if use_gpu else ""
    training_image_tag = f"{config.TRAINING_IMAGE_NAME}{image_suffix}:{tag}"
    
    # Configurar el acceso a Artifact Registry
    subprocess.run(["gcloud", "auth", "configure-docker", f"{config.REGION}-docker.pkg.dev"], check=True)
    
    # Construir la imagen Docker
    build_command = [
        "docker", "build",
        "-t", training_image_tag,
        "-f", dockerfile,
        project_dir
    ]
    
    print(f"Construyendo imagen{'GPU' if use_gpu else ''}: {training_image_tag}")
    subprocess.run(build_command, check=True)
    
    # Subir la imagen
    push_command = ["docker", "push", training_image_tag]
    
    print(f"Subiendo imagen a Artifact Registry: {training_image_tag}")
    subprocess.run(push_command, check=True)
    
    print(f"Imagen {training_image_tag} construida y subida exitosamente.")
    return training_image_tag

def main(tag, use_gpu=False):
    """
    Función principal para construir y subir la imagen Docker.
    
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
    parser = argparse.ArgumentParser(description="Construir y subir imagen Docker a Artifact Registry")
    parser.add_argument("--tag", default="latest", help="Etiqueta para la imagen Docker (default: latest)")
    parser.add_argument("--gpu", action="store_true", help="Construir imagen con soporte para GPU")
    
    args = parser.parse_args()
    main(args.tag, args.gpu)
    main(args.tag)
