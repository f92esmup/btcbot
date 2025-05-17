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

def build_and_push_image(tag, use_gpu=False, build_region=None, use_global=False):
    """
    Construye y sube la imagen Docker a Artifact Registry usando Google Cloud Build.
    
    Args:
        tag: Etiqueta para la imagen (ej: "latest", "v1", etc).
        use_gpu: Indica si se debe construir la imagen con soporte para GPU.
        build_region: Región en la que ejecutar Cloud Build. Si es None, usa la región por defecto.
        use_global: Si es True, usa Cloud Build global sin especificar región.
    """
    # Obtener la ruta del proyecto
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Seleccionar el Dockerfile adecuado
    dockerfile_name = "Dockerfile.gpu" if use_gpu else "Dockerfile"
    image_suffix = "-gpu" if use_gpu else ""
    training_image_tag = f"{config.TRAINING_IMAGE_NAME}{image_suffix}:{tag}"
    
    # Determinar la región de construcción
    region_to_use = build_region if build_region else config.REGION
    
    # Construir la imagen con Cloud Build
    print(f"Construyendo imagen{' GPU' if use_gpu else ''} con Cloud Build: {training_image_tag}")
    print(f"Usando {'Cloud Build global' if use_global else f'región: {region_to_use}'}")
    
    # Crear un cloudbuild.yaml temporal
    cloudbuild_content = f"""
steps:
- name: 'gcr.io/cloud-builders/docker'
  args: ['build', 
         '-t', '{training_image_tag}', 
         '-f', '{dockerfile_name}', 
         '--build-arg', 'BUILDKIT_INLINE_CACHE=1',
         '--progress=plain',  # Esta opción ayuda a tener una salida más estándar
         '.']
  timeout: '3600s'
images: ['{training_image_tag}']
timeout: '3600s'
options:
  machineType: 'E2_HIGHCPU_8'
  diskSizeGb: '100'
  # logging: CLOUD_LOGGING_ONLY # Opcional: para enviar logs solo a Cloud Logging y no a GCS
  env:
    - 'DOCKER_BUILDKIT=1'
    """
    
    cloudbuild_file = os.path.join(project_dir, "cloudbuild_temp.yaml") # Usar un nombre diferente para evitar conflictos
    try:
        # Guardar el archivo temporalmente
        with open(cloudbuild_file, "w") as f:
            f.write(cloudbuild_content)
        
        # Opcional: Define un bucket para los logs de construcción
        gcs_log_dir = f"gs://{config.MODELS_STAGING_BUCKET}/cloud_build_logs/" # Asegúrate que este bucket exista y tenga permisos

        # Comando con el archivo de configuración
        build_command = [
            "gcloud", "builds", "submit",
            "--project", config.PROJECT_ID,
            "--verbosity", "info", # Puedes probar con 'warning' si 'info' es muy verboso
            # "--log-http", # Eliminamos esta línea para evitar los caracteres extraños
            "--config", cloudbuild_file, # Referencia al archivo temporal
            # "--gcs-log-dir", gcs_log_dir, # Descomenta para guardar logs en GCS
            project_dir
        ]
        
        # Añadir la región solo si no estamos usando la configuración global
        if not use_global:
            build_command.extend(["--region", region_to_use])
        
        print(f"Ejecutando: {' '.join(build_command)}")
        try:
            # Usar subprocess.Popen para capturar stdout y stderr
            # y luego imprimirlo de forma controlada
            process = subprocess.Popen(build_command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, universal_newlines=True)
            
            # Imprimir la salida línea por línea
            if process.stdout:
                for line in iter(process.stdout.readline, ''):
                    print(line, end='') # Imprime la salida de Cloud Build
                process.stdout.close()

            return_code = process.wait()
            if return_code:
                # Si hubo un error, el stderr ya se imprimió (si se redirigió a stdout)
                raise subprocess.CalledProcessError(return_code, build_command)

        except subprocess.CalledProcessError as e:
            print(f"Error al construir la imagen: {e}")
            if not use_global and region_to_use == config.REGION:
                print("Error con la región por defecto. Intentando con Cloud Build global...")
                # Intentar sin especificar región (global)
                return build_and_push_image(tag, use_gpu, None, True)
            else:
                print(f"Comando: {' '.join(build_command)}")
                raise
        
    finally:
        # Eliminar el archivo temporal si existe
        if os.path.exists(cloudbuild_file):
            os.remove(cloudbuild_file)
    
    print(f"Imagen {training_image_tag} construida y subida exitosamente con Cloud Build.")
    return training_image_tag

def main(tag, use_gpu=False, build_region=None, use_global=False):
    """
    Función principal para construir y subir la imagen Docker usando Cloud Build.
    
    Args:
        tag: Etiqueta para la imagen.
        use_gpu: Indica si se debe construir la imagen con soporte para GPU.
        build_region: Región específica para Cloud Build.
        use_global: Si es True, usa Cloud Build global sin especificar región.
    """
    # Crear repositorio si no existe
    create_artifact_repo()
    
    # Construir y subir imagen
    image_uri = build_and_push_image(tag, use_gpu, build_region, use_global)
    
    print(f"Proceso completado. Imagen disponible en: {image_uri}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Construir y subir imagen Docker a Artifact Registry usando Cloud Build")
    parser.add_argument("--tag", default="latest", help="Etiqueta para la imagen Docker (default: latest)")
    parser.add_argument("--gpu", action="store_true", help="Construir imagen con soporte para GPU")
    parser.add_argument("--region", help=f"Región específica para Cloud Build (default: {config.REGION})")
    parser.add_argument("--global", dest="use_global", action="store_true", 
                        help="Usar Cloud Build global sin especificar región (útil para evitar restricciones de cuota)")
    
    args = parser.parse_args()
    main(args.tag, args.gpu, args.region, args.use_global)
