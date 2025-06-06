# Refactorización del Sistema de Almacenamiento de Modelos

## Resumen de Cambios

Se ha refactorizado completamente la lógica de guardado y carga de modelos para resolver los problemas de organización por `run_id` y evitar sobrescrituras entre diferentes ejecuciones.

## Problemas Solucionados

### 1. **Modelos sobrescritos en GCS**
- **Antes**: Los modelos se guardaban en `gs://<bucket>/models/` sin organización por `run_id`
- **Después**: Cada ejecución tiene su propia estructura: `gs://<bucket>/<run_id>/`

### 2. **Checkpoints con paths incorrectos**
- **Antes**: `agent.save_models()` recibía paths de GCS directamente, causando errores
- **Después**: Se usan archivos temporales locales que se suben individualmente a GCS

## Nueva Estructura de Directorios

### En GCS (storage_mode = "gcp")
```
gs://<bucket>/
├── <run_id>/
│   ├── best_model/
│   │   ├── agent_model_components_actor.pth
│   │   ├── agent_model_components_critic_1.pth
│   │   ├── agent_model_components_critic_2.pth
│   │   ├── agent_model_components_critic_target_1.pth
│   │   ├── agent_model_components_critic_target_2.pth
│   │   ├── agent_model_components_actor_optimizer.pth
│   │   ├── agent_model_components_critic_1_optimizer.pth
│   │   ├── agent_model_components_critic_2_optimizer.pth
│   │   ├── agent_model_components_log_alpha.pth
│   │   ├── agent_model_components_alpha_optimizer.pth
│   │   └── agent_model_components_metadata.pkl
│   ├── final_model/
│   │   └── [mismos componentes]
│   └── checkpoints/
│       ├── checkpoint_episode_100/
│       │   ├── ckpt_ep_100_actor.pth
│       │   ├── ckpt_ep_100_critic_1.pth
│       │   └── [otros componentes]
│       └── checkpoint_episode_200/
│           └── [componentes del episodio 200]
```

### En Local (storage_mode = "local")
```
Entrenamientos/
├── <run_id>/
│   ├── best_model/
│   │   ├── model_actor.pth
│   │   ├── model_critic_1.pth
│   │   └── [otros componentes]
│   ├── final_model/
│   │   └── [mismos componentes]
│   └── checkpoints/
│       ├── checkpoint_episode_100_actor.pth
│       ├── checkpoint_episode_100_critic_1.pth
│       └── [otros componentes]
```

## Cambios Técnicos Implementados

### 1. **src/agente/agent.py - Método `save()`**

#### Antes:
```python
# Construía: gs://bucket/models/{base_name}_actor.pth
base_name = Path(filepath).stem
gcs_prefix = f"models/{base_name}"
```

#### Después:
```python
# Respeta la estructura: gs://bucket/{run_id}/best_model/agent_model_components_actor.pth
gcs_target_directory = Path(filepath).parent
local_temp_file_prefix = os.path.join(temp_dir, "agent_model_components")
```

### 2. **src/agente/agent.py - Método `load()`**

#### Antes:
```python
# Buscaba: gs://bucket/models/{base_name}_actor.pth
gcs_prefix = f"models/{base_name}"
model_files = [("_actor.pth", f"{temp_prefix}_actor.pth"), ...]
```

#### Después:
```python
# Busca: gs://bucket/{run_id}/best_model/agent_model_components_actor.pth
gcs_source_directory = Path(filepath).parent
component_files = ["agent_model_components_actor.pth", ...]
```

### 3. **train.py - Guardado de Checkpoints**

#### Antes:
```python
if config.storage_mode == "gcp":
    checkpoint_path = f"{checkpoint_save_prefix}/checkpoint_episode_{episode + 1}"
    agent.save_models(checkpoint_path)  # ❌ Path de GCS usado directamente
```

#### Después:
```python
if config.storage_mode == "gcp":
    # Usar tempfile y subir individualmente
    with tempfile.TemporaryDirectory() as temp_dir:
        local_temp_ckpt_prefix = os.path.join(temp_dir, f"ckpt_ep_{episode + 1}")
        agent.save_models(local_temp_ckpt_prefix)
        
        # Subir cada archivo a GCS
        for local_file_path in Path(temp_dir).glob(f"ckpt_ep_{episode + 1}_*"):
            gcs_blob_name = f"{gcs_checkpoint_directory_prefix}/{local_file_path.name}"
            gcs_utils.upload_file_to_gcs(str(local_file_path), gcs_blob_name)
```

## Llamadas desde train.py

### Mejor Modelo:
```python
# Se llama con:
best_model_path = f"{run_id}/best_model/model.pth"
agent.save(best_model_path)

# Resultado en GCS:
# gs://bucket/run_id/best_model/agent_model_components_*.pth
```

### Modelo Final:
```python
# Se llama con:
final_model_path = f"{run_id}/final_model/model.pth"
agent.save(final_model_path)

# Resultado en GCS:
# gs://bucket/run_id/final_model/agent_model_components_*.pth
```

### Checkpoints:
```python
# Se maneja internamente en train.py:
gcs_checkpoint_directory_prefix = f"{run_id}/checkpoints/checkpoint_episode_{episode + 1}"

# Resultado en GCS:
# gs://bucket/run_id/checkpoints/checkpoint_episode_X/ckpt_ep_X_*.pth
```

## Consistencia entre Guardado y Carga

- **Guardado**: Usa `agent_model_components` como prefijo para componentes principales
- **Carga**: Busca exactamente los mismos nombres de archivo
- **Checkpoints**: Usa `ckpt_ep_{episode}` como prefijo y mantiene consistencia

## Ventajas de la Nueva Implementación

1. **Organización por Run ID**: Cada ejecución tiene su propio directorio
2. **No hay sobrescrituras**: Los modelos de diferentes ejecuciones están separados
3. **Estructura clara**: Fácil navegación y gestión en GCS
4. **Consistencia**: Los nombres de archivo son predecibles y consistentes
5. **Compatibilidad**: El modo local sigue funcionando sin cambios

## Testing

Se han creado scripts de prueba que verifican:

- ✅ `test_model_saving.py`: Verificación de lógica de paths
- ✅ `test_complete_flow.py`: Simulación completa del flujo de guardado/carga

## Uso

La nueva implementación es transparente para el usuario. Los scripts de entrenamiento funcionan igual, pero ahora organizan los modelos correctamente por `run_id`.

```bash
# El entrenamiento ahora organiza automáticamente los modelos:
python train.py --symbol BTCUSDT --interval 1h --start-date 2024-01-01

# Resultado en GCS:
# gs://bucket/BTCUSDT_1h_20241201_120000/best_model/
# gs://bucket/BTCUSDT_1h_20241201_120000/final_model/
# gs://bucket/BTCUSDT_1h_20241201_120000/checkpoints/
```
