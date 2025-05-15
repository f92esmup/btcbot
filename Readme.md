# BTC-Transformer-RL-Trader: Bot Avanzado de Trading de Bitcoin con Reinforcement Learning y Transformers

**Versión:** 0.1.0 (MVP - En Fase de Implementación)
**Autor:** Pedro Escudero Murcia
**Fecha:** 15 de mayo de 2025

## 1. Visión General

**BTC-Transformer-RL-Trader** es un proyecto para desarrollar un bot de trading de Bitcoin (BTC) avanzado y autónomo, operando inicialmente contra la API de Binance para futuros BTCUSDT. El núcleo del bot se basa en el aprendizaje por refuerzo (Reinforcement Learning - RL), específicamente el algoritmo Soft Actor-Critic (SAC), y utiliza una arquitectura de red neuronal basada en Transformers para la toma de decisiones.

El objetivo principal del MVP (Producto Mínimo Viable) es desarrollar y validar rigurosamente un agente de trading mediante un robusto framework de **Walk-Forward Optimization (WFO)** antes de considerar la operativa en tiempo real.

## 2. Características Clave del MVP

* **Agente de RL:** Implementación de Soft Actor-Critic (SAC).
* **Modelo de Política:** Red neuronal con un codificador Transformer para procesar secuencias de datos de mercado.
* **Activo Objetivo:** Futuros de Bitcoin (BTCUSDT) en Binance.
* **Framework de Backtesting:** Optimización Walk-Forward (WFO) para una evaluación robusta del rendimiento histórico del agente.
* **Diseño Modular:** El sistema está desglosado en módulos para adquisición de datos, preprocesamiento, entorno de trading, agente, backtesting y configuración.
* **Entorno Dockerizado:** Todo el sistema (aplicación y servicios como Redis) se ejecuta dentro de contenedores Docker orquestados por Docker Compose para reproducibilidad y facilidad de despliegue.

## 3. Stack Tecnológico Principal

* **Lenguaje:** Python 3.x
* **Deep Learning:** PyTorch
* **Reinforcement Learning:** Stable Baselines3 (SB3)
* **Entorno de RL:** Gymnasium
* **Manipulación de Datos:** Pandas, NumPy
* **Indicadores Técnicos:** TA-Lib
* **Interacción con Exchange (Datos Históricos):** `python-binance`
* **Broker de Mensajes (Flujo Inicial de Datos):** Redis
* **Contenerización:** Docker, Docker Compose
* **Configuración:** Archivos YAML y variables de entorno (`.env`)
* **Análisis de Backtesting:** QuantStats, Matplotlib, Seaborn

## 4. Estructura del Proyecto (Simplificada)# btcbot
