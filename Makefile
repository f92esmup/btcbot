# Makefile para el Bitcoin Trading Bot

# Variables
PROJECT_ROOT := $(shell pwd)
SCRIPTS_DIR := $(PROJECT_ROOT)/scripts
TERRAFORM_DIR := $(PROJECT_ROOT)/terraform
TMP_DIR := $(PROJECT_ROOT)/tmp
RESULTS_DIR := $(TMP_DIR)/results
MODELS_DIR := $(TMP_DIR)/models

# Asegurarse de que existen los directorios temporales
$(TMP_DIR):
	mkdir -p $(TMP_DIR)

$(RESULTS_DIR): $(TMP_DIR)
	mkdir -p $(RESULTS_DIR)

$(MODELS_DIR): $(TMP_DIR)
	mkdir -p $(MODELS_DIR)

# Comandos de entrenamiento y pruebas locales
.PHONY: train-local
train-local: $(MODELS_DIR)
	@echo "🏋️ Ejecutando entrenamiento local..."
	python3 $(SCRIPTS_DIR)/simple_train_test_local.py

.PHONY: backtest-local
backtest-local: $(RESULTS_DIR)
	@echo "🔍 Ejecutando backtesting local..."
	python3 $(SCRIPTS_DIR)/simple_backtest_local.py

.PHONY: test-model
test-model: $(RESULTS_DIR)
	@echo "🧪 Probando modelo entrenado..."
	bash $(SCRIPTS_DIR)/run_trained_model.sh

.PHONY: test-improvements
test-improvements:
	@echo "🔍 Verificando mejoras de implementación..."
	python3 $(SCRIPTS_DIR)/test_implementation_improvements.py

# Comandos de despliegue en GCP
.PHONY: setup-terraform
setup-terraform:
	@echo "🔧 Configurando Terraform..."
	bash $(SCRIPTS_DIR)/setup_terraform.sh

.PHONY: update-secrets
update-secrets:
	@echo "🔒 Actualizando secretos en Secret Manager..."
	bash $(SCRIPTS_DIR)/update_secrets.sh

.PHONY: deploy-infrastructure
deploy-infrastructure:
	@echo "🚀 Desplegando infraestructura en GCP..."
	bash $(SCRIPTS_DIR)/deploy_infrastructure.sh

.PHONY: setup-trigger
setup-trigger:
	@echo "🔄 Configurando trigger de Cloud Build..."
	bash $(SCRIPTS_DIR)/setup_cloud_build_trigger.sh

.PHONY: deploy-pipeline
deploy-pipeline:
	@echo "⚙️ Desplegando pipeline en Vertex AI..."
	bash $(SCRIPTS_DIR)/deploy_to_vertex_ai.sh

.PHONY: deploy-all
deploy-all:
	@echo "🚀 Desplegando todo el proyecto en GCP..."
	bash $(SCRIPTS_DIR)/deploy_complete_pipeline.sh

# Comandos de monitoreo y verificación
.PHONY: check-status
check-status:
	@echo "🔍 Verificando estado del despliegue..."
	bash $(SCRIPTS_DIR)/check_deployment_status.sh

.PHONY: list-resources
list-resources:
	@echo "📋 Listando recursos en GCP..."
	bash $(SCRIPTS_DIR)/list_gcp_resources.sh

.PHONY: monitor
monitor:
	@echo "👀 Monitorizando despliegue..."
	bash $(SCRIPTS_DIR)/monitor_deployment.sh

# Comandos para ejecución en producción
.PHONY: scheduled-trading
scheduled-trading:
	@echo "⏱️ Configurando trading programado..."
	bash $(SCRIPTS_DIR)/scheduled_trading.sh

.PHONY: alerts
alerts:
	@echo "🔔 Configurando alertas de trading..."
	bash $(SCRIPTS_DIR)/send_trading_alerts.sh

# Limpieza
.PHONY: clean
clean:
	@echo "🧹 Limpiando archivos temporales..."
	rm -rf $(TMP_DIR)

# Comandos específicos para Google Cloud
.PHONY: upload-model
upload-model: $(MODELS_DIR)
	@echo "📤 Subiendo modelo a Google Cloud Storage..."
	gsutil -m cp -r $(MODELS_DIR)/simple_sac_model.zip gs://$(shell grep 'project_id' $(TERRAFORM_DIR)/terraform.tfvars | cut -d'"' -f2)-btc-models/

.PHONY: upload-results
upload-results: $(RESULTS_DIR)
	@echo "📤 Subiendo resultados a Google Cloud Storage..."
	gsutil -m cp -r $(RESULTS_DIR)/* gs://$(shell grep 'project_id' $(TERRAFORM_DIR)/terraform.tfvars | cut -d'"' -f2)-btc-evaluation-results/local_backtest/

.PHONY: help
help:
	@echo "Bitcoin Trading Bot - Comandos disponibles:"
	@echo ""
	@echo "== Entrenamiento y pruebas locales =="
	@echo "  make train-local        - Entrenar modelo localmente"
	@echo "  make backtest-local     - Ejecutar backtesting local"
	@echo "  make test-model         - Probar modelo entrenado"
	@echo "  make test-improvements  - Verificar mejoras de implementación"
	@echo ""
	@echo "== Despliegue en GCP =="
	@echo "  make setup-terraform    - Configurar Terraform"
	@echo "  make update-secrets     - Actualizar secretos en Secret Manager"
	@echo "  make deploy-infrastructure - Desplegar infraestructura con Terraform"
	@echo "  make setup-trigger      - Configurar trigger de Cloud Build"
	@echo "  make deploy-pipeline    - Desplegar pipeline en Vertex AI"
	@echo "  make deploy-all         - Desplegar todo el proyecto"
	@echo ""
	@echo "== Monitoreo y verificación =="
	@echo "  make check-status       - Verificar estado del despliegue"
	@echo "  make list-resources     - Listar recursos en GCP"
	@echo "  make monitor            - Monitorizar despliegue en tiempo real"
	@echo ""
	@echo "== Producción =="
	@echo "  make scheduled-trading  - Configurar trading programado"
	@echo "  make alerts             - Configurar alertas de trading"
	@echo "  make upload-model       - Subir modelo a GCS"
	@echo "  make upload-results     - Subir resultados a GCS"
	@echo ""
	@echo "== Otros =="
	@echo "  make clean              - Limpiar archivos temporales"
	@echo "  make help               - Mostrar esta ayuda"

# Comando por defecto
.DEFAULT_GOAL := help