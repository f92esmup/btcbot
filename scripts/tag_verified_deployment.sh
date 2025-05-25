#!/usr/bin/env bash

# Script para formalizar el despliegue verificado
# Crea un commit y tag específicos para la verificación del modelo de producción

# Colores para output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Verificación de Modelo de Producción ===${NC}"

# Verificar que estamos en el directorio correcto
if [ ! -f "src/config.yaml" ]; then
  echo -e "${YELLOW}Error: Ejecute este script desde el directorio raíz del proyecto${NC}"
  exit 1
fi

# Verificar cambios pendientes
if [ -n "$(git status --porcelain)" ]; then
  echo -e "${YELLOW}Hay cambios pendientes en el repositorio. Estos cambios serán incluidos en el commit.${NC}"
fi

# Añadir los archivos actualizados
echo -e "${BLUE}Añadiendo archivos actualizados...${NC}"
git add src/config.yaml
git add README.md
git add DEPLOYMENT.md
git add MODEL_MANAGEMENT.md
git add DEPLOYMENT_VERIFICATION.md

# Crear commit
echo -e "${BLUE}Creando commit de verificación...${NC}"
git commit -m "chore: Verificado modelo de producción para live trading"

# Crear tag con fecha
FECHA=$(date +"%Y%m%d")
echo -e "${BLUE}Creando tag de despliegue...${NC}"
git tag -a "v1.0.0-production-$FECHA" -m "Modelo verificado para producción: sac_transformer_trading_agent_final_1000_steps.zip"

echo -e "${GREEN}✅ Commit y tag creados exitosamente${NC}"
echo -e "${YELLOW}Para completar, ejecute:${NC} git push && git push --tags"
