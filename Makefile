# Makefile para el Bitcoin Trading Bot
# Simplifica los comandos comunes para desarrollo y despliegue

.PHONY: setup deploy update-secrets run-local test backtest check status list clean help

# Variables
PROJECT_ROOT := $(shell pwd)
SCRIPTS_DIR := $(PROJECT_ROOT)/scripts
MODEL_DIR := $(PROJECT_ROOT)/tmp/models
RESULTS_DIR := $(PROJECT_ROOT)/tmp/results

help:
	@echo "🤖 Bitcoin Trading Bot - Comandos disponibles"
	@echo "----------------------------------------"
	@echo "setup              : Configura Terraform para el despliegue"
	@echo "deploy             : Despliega toda la infraestructura en GCP"
	@echo "update-secrets     : Actualiza las credenciales de Binance en Secret Manager"
	@echo "run-local          : Ejecuta un entrenamiento local básico"
	@echo "test               : Ejecuta las pruebas unitarias"
	@echo "backtest           : Ejecuta un backtest con el modelo entrenado"
	@echo "check              : Verifica el estado del despliegue"
	@echo "status             : Muestra el estado del despliegue (alias de check)"
	@echo "list               : Lista los recursos desplegados en GCP"
	@echo "clean              : Limpia archivos temporales"
	@echo ""
	@echo "Ejemplo: make deploy"

setup:
	@echo "🔧 Configurando Terraform..."
	@bash $(SCRIPTS_DIR)/setup_terraform.sh

deploy:
	@echo "🚀 Desplegando infraestructura completa..."
	@bash $(SCRIPTS_DIR)/deploy_complete_pipeline.sh

update-secrets:
	@echo "🔒 Actualizando secretos..."
	@bash $(SCRIPTS_DIR)/update_secrets.sh

run-local:
	@echo "🏃 Ejecutando entrenamiento local..."
	@mkdir -p $(MODEL_DIR) $(RESULTS_DIR)
	@python $(SCRIPTS_DIR)/simple_train_test_local.py

test:
	@echo "🧪 Ejecutando pruebas..."
	@python -m pytest tests/

backtest:
	@echo "📊 Ejecutando backtest..."
	@mkdir -p $(RESULTS_DIR)
	@python $(SCRIPTS_DIR)/simple_backtest_local.py

check status:
	@echo "🔍 Verificando estado del despliegue..."
	@bash $(SCRIPTS_DIR)/check_deployment_status.sh

list:
	@echo "📋 Listando recursos en GCP..."
	@bash $(SCRIPTS_DIR)/list_gcp_resources.sh

clean:
	@echo "🧹 Limpiando archivos temporales..."
	@rm -rf tmp/__pycache__
	@find . -name "*.pyc" -delete
	@find . -name "__pycache__" -exec rm -rf {} +;
	@echo "✅ Limpieza completada"
