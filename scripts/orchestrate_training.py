#!/usr/bin/env python3
"""
Orquestador de entrenamiento para BTCBot
Este script coordina toda la pipeline de entrenamiento de manera secuencial.
Siempre ejecuta todas las fases sin verificar si los datos existen previamente.
"""

import argparse
import logging
import sys
import subprocess
import os
from datetime import datetime, timedelta
from pathlib import Path

# Configurar logging solo para terminal
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class TrainingOrchestrator:
    """Orquestador para la pipeline completa de entrenamiento"""
    
    def __init__(self):
        self.scripts_dir = Path(__file__).parent
        self.base_dir = self.scripts_dir.parent
        
    def run_script(self, script_name, args=None):
        """Ejecuta un script Python y maneja errores"""
        script_path = self.scripts_dir / script_name
        if not script_path.exists():
            raise FileNotFoundError(f"Script no encontrado: {script_path}")
        
        cmd = [sys.executable, str(script_path)]
        if args:
            cmd.extend(args)
        
        logger.info(f"Ejecutando: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                cwd=self.base_dir
            )
            logger.info(f"✓ {script_name} completado exitosamente")
            if result.stdout:
                logger.info(f"Salida: {result.stdout}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"✗ Error ejecutando {script_name}")
            logger.error(f"Código de error: {e.returncode}")
            logger.error(f"Stderr: {e.stderr}")
            if e.stdout:
                logger.error(f"Stdout: {e.stdout}")
            return False
    
    def phase_1_download_data(self):
        """Fase 1: Descarga de datos de mercado"""
        logger.info("=" * 50)
        logger.info("FASE 1: DESCARGA DE DATOS")
        logger.info("=" * 50)
        
        # El script download_data.py usa configuración desde config.yaml
        logger.info("Descargando datos según configuración en src/config.yaml")
        
        return self.run_script('download_data.py')
    
    def phase_2_preprocess_data(self):
        """Fase 2: Preprocesamiento de datos"""
        logger.info("=" * 50)
        logger.info("FASE 2: PREPROCESAMIENTO DE DATOS")
        logger.info("=" * 50)
        
        # El script preprocess_data.py procesa todos los archivos disponibles
        logger.info("Preprocesando todos los archivos de datos disponibles")
        
        return self.run_script('preprocess_data.py')
    
    def phase_3_train_model(self, timesteps=None):
        """Fase 3: Entrenamiento del modelo RL"""
        logger.info("=" * 50)
        logger.info("FASE 3: ENTRENAMIENTO DEL MODELO")
        logger.info("=" * 50)
        
        args = []
        if timesteps is not None:
            args.extend(['--timesteps', str(timesteps)])
            logger.info(f"Entrenando modelo por {timesteps} timesteps")
        else:
            logger.info("Entrenando modelo con timesteps configurados en config.yaml")
        
        return self.run_script('train_rl_agent.py', args if args else None)
    
    def phase_4_evaluate_model(self):
        """Fase 4: Evaluación del modelo entrenado"""
        logger.info("=" * 50)
        logger.info("FASE 4: EVALUACIÓN DEL MODELO")
        logger.info("=" * 50)
        
        # evaluate_rl_agent.py evaluará con configuración por defecto
        args = ['--episodes', '3']
        
        logger.info("Evaluando modelo con 3 episodios")
        
        return self.run_script('evaluate_rl_agent.py', args)
    
    def run_full_pipeline(self, timesteps=None):
        """Ejecuta la pipeline completa de entrenamiento"""
        start_time = datetime.now()
        logger.info("🚀 INICIANDO PIPELINE COMPLETA DE ENTRENAMIENTO")
        logger.info("La configuración de símbolos, timeframes y datos se lee desde src/config.yaml")
        if timesteps:
            logger.info(f"Timesteps de entrenamiento: {timesteps}")
        logger.info(f"Hora de inicio: {start_time}")
        
        phases = [
            ("Descarga de datos", lambda: self.phase_1_download_data()),
            ("Preprocesamiento", lambda: self.phase_2_preprocess_data()),
            ("Entrenamiento", lambda: self.phase_3_train_model(timesteps)),
            ("Evaluación", lambda: self.phase_4_evaluate_model())
        ]
        
        completed_phases = 0
        failed_phases = []
        
        for phase_name, phase_func in phases:
            logger.info(f"\n🔄 Iniciando fase: {phase_name}")
            try:
                if phase_func():
                    completed_phases += 1
                    logger.info(f"✅ Fase '{phase_name}' completada exitosamente")
                else:
                    failed_phases.append(phase_name)
                    logger.error(f"❌ Fase '{phase_name}' falló")
                    # Continuar con las siguientes fases incluso si una falla
                    
            except Exception as e:
                failed_phases.append(phase_name)
                logger.error(f"❌ Excepción en fase '{phase_name}': {str(e)}")
                # Continuar con las siguientes fases
        
        # Resumen final
        end_time = datetime.now()
        duration = end_time - start_time
        
        logger.info("=" * 60)
        logger.info("📊 RESUMEN DE EJECUCIÓN")
        logger.info("=" * 60)
        logger.info(f"Hora de inicio: {start_time}")
        logger.info(f"Hora de fin: {end_time}")
        logger.info(f"Duración total: {duration}")
        logger.info(f"Fases completadas: {completed_phases}/{len(phases)}")
        
        if failed_phases:
            logger.warning(f"Fases fallidas: {', '.join(failed_phases)}")
            return False
        else:
            logger.info("🎉 ¡PIPELINE COMPLETADA EXITOSAMENTE!")
            return True

def main():
    parser = argparse.ArgumentParser(
        description="Orquestador de entrenamiento BTCBot - Ejecuta pipeline completa",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  python orchestrate_training.py
  python orchestrate_training.py --timesteps 10000
  python orchestrate_training.py --phase train --timesteps 5000
  python orchestrate_training.py --phase evaluate
        """
    )
    
    parser.add_argument(
        '--timesteps',
        type=int,
        default=None,
        help='Número de timesteps para entrenamiento (si se omite, usa valor de config.yaml)'
    )
    
    parser.add_argument(
        '--phase',
        choices=['download', 'preprocess', 'train', 'evaluate'],
        help='Ejecutar solo una fase específica (omitir para pipeline completa)'
    )
    
    args = parser.parse_args()
    
    # Validar argumentos
    if args.timesteps is not None and args.timesteps <= 0:
        logger.error("El número de timesteps debe ser positivo")
        sys.exit(1)
    
    # Crear orquestador
    orchestrator = TrainingOrchestrator()
    
    try:
        if args.phase:
            # Ejecutar solo una fase específica
            logger.info(f"Ejecutando solo la fase: {args.phase}")
            
            if args.phase == 'download':
                success = orchestrator.phase_1_download_data()
            elif args.phase == 'preprocess':
                success = orchestrator.phase_2_preprocess_data()
            elif args.phase == 'train':
                success = orchestrator.phase_3_train_model(args.timesteps)
            elif args.phase == 'evaluate':
                success = orchestrator.phase_4_evaluate_model()
            
            if success:
                logger.info(f"✅ Fase '{args.phase}' completada exitosamente")
                sys.exit(0)
            else:
                logger.error(f"❌ Fase '{args.phase}' falló")
                sys.exit(1)
        else:
            # Ejecutar pipeline completa
            success = orchestrator.run_full_pipeline(timesteps=args.timesteps)
            
            sys.exit(0 if success else 1)
            
    except KeyboardInterrupt:
        logger.info("🛑 Ejecución interrumpida por el usuario")
        sys.exit(1)
    except Exception as e:
        logger.error(f"💥 Error inesperado: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()