# Plan de Desarrollo y Tareas Pendientes (BTCBot)

Este documento organiza las tareas de desarrollo para el proyecto btcbot, priorizando la implementación de nuevas funcionalidades y mejoras arquitectónicas.

## Merge-1: Mejoras Fundamentales y Diseño de Meta-Agente

### Épica 1: Mejoras en la Lógica de Trading del Agente Actual

#### 1.1. Gestión de Posiciones Múltiples (Scaling-in)
- [ ] **Problema:** El agente no tiene un comportamiento definido si recibe una señal de compra cuando ya tiene una posición larga abierta (o venta en corto).
- [ ] **Tarea:** Modificar `Portfolio.execute_order` para permitir incrementar una posición existente.
- [ ] **Decisión de Diseño:** Definir cómo se promedia el precio de entrada y cómo se re-calcula el margen.

#### 1.2. Implementación de Stop-Loss Intra-Vela
- [ ] **Problema:** El agente es vulnerable a caídas de precio drásticas que ocurren entre los intervalos de tiempo definidos (ej. entre una vela de 1h y la siguiente).
- [ ] **Tarea:** Diseñar e implementar una estrategia de stop-loss.
- [ ] **Decisión de Diseño:** Investigar la mejor forma de simular un stop-loss intra-vela en el entorno de backtesting.
- [ ] **Tarea:** Integrar la lógica en `LivePortfolio` y `RiskManager` para la operativa en vivo.

### Épica 2: Arquitectura de Agente Gestor (Meta-Agente)

#### 2.1. Diseño y Planificación de la Arquitectura
- [ ] **Objetivo:** Evolucionar de un agente monolítico a un sistema jerárquico con un agente "gestor" que coordine a múltiples agentes "especialistas".
- [ ] **Tarea:** Definir el concepto de "agentes especialistas" (ej. especialista en mercados alcistas, bajistas, volátiles).
- [ ] **Tarea:** Diseñar la estructura y responsabilidades del "agente gestor".
- [ ] **Entregable:** Crear un plano de la nueva arquitectura en un archivo `ARCHITECTURE.md`.

#### 2.2. Reestructuración de Directorios
- [ ] **Tarea:** Proponer y aplicar una nueva estructura de directorios que soporte la arquitectura de meta-agente (ej. `src/agents/manager`, `src/agents/specialists`).

#### 2.3. Adaptación del Pipeline de MLOps
- [ ] **Tarea:** Analizar y planificar las modificaciones necesarias en `train.py`, `evaluate.py` y `walk_forward_pipeline.py` para que sean compatibles con la nueva arquitectura.
