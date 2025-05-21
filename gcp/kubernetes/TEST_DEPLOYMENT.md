# Prueba de Despliegue Completo de BTCBot en GKE

Este documento contiene los comandos necesarios para probar el despliegue completo de BTCBot (inferencia + trading) en GKE.

## 1. Verificar estado actual del clúster

```bash
# Verificar nodos disponibles
kubectl get nodes

# Verificar pods existentes en el namespace btcbot
kubectl get pods -n btcbot

# Verificar servicios existentes
kubectl get services -n btcbot

# Verificar deployments existentes
kubectl get deployments -n btcbot
```

## 2. Desplegar servidor de inferencia

```bash
# Aplicar el despliegue del servidor de inferencia
kubectl apply -f btcbot-inference-deployment.yaml

# Aplicar el servicio de inferencia
kubectl apply -f btcbot-inference-service.yaml

# Verificar el estado del despliegue
kubectl rollout status deployment/btcbot-inference-server -n btcbot

# Verificar que el pod esté ejecutándose
kubectl get pods -n btcbot -l app=btcbot-inference-server

# Verificar logs del servidor de inferencia
kubectl logs -n btcbot -l app=btcbot-inference-server
```

## 3. Desplegar trader en vivo

```bash
# Aplicar el despliegue del trader
kubectl apply -f btcbot-live-trader-deployment.yaml

# Verificar el estado del despliegue
kubectl rollout status deployment/btcbot-live-trader -n btcbot

# Verificar que el pod esté ejecutándose
kubectl get pods -n btcbot -l app=btcbot-trader

# Verificar logs del trader
kubectl logs -n btcbot -l app=btcbot-trader
```

## 4. Verificar comunicación entre componentes

```bash
# Verificar que el trader pueda conectarse al servidor de inferencia
# Buscar en los logs mensajes que indiquen comunicación exitosa
kubectl logs -n btcbot -l app=btcbot-trader | grep -i "inferencia"
kubectl logs -n btcbot -l app=btcbot-trader | grep -i "predicción"

# Verificar que el servidor de inferencia reciba solicitudes
kubectl logs -n btcbot -l app=btcbot-inference-server | grep -i "predict"
```

## 5. Verificar funcionamiento del sistema completo

```bash
# Ver todos los componentes del sistema
kubectl get all -n btcbot

# Monitorear los logs del trader continuamente
kubectl logs -f -n btcbot deployment/btcbot-live-trader

# Verificar que se esté conectando a Binance
kubectl logs -n btcbot -l app=btcbot-trader | grep -i "binance"

# Verificar que se estén ejecutando operaciones
kubectl logs -n btcbot -l app=btcbot-trader | grep -i "orden"
```

## 6. Solucionar problemas comunes

### Si el servidor de inferencia no puede cargar el modelo:

```bash
# Verificar los permisos de la cuenta de servicio
kubectl describe pod -n btcbot -l app=btcbot-inference-server

# Verificar los logs de error detallados
kubectl logs -n btcbot -l app=btcbot-inference-server
```

### Si el trader no puede conectarse al servidor de inferencia:

```bash
# Verificar que el servicio esté funcionando
kubectl describe service btcbot-inference-service -n btcbot

# Verificar la conectividad dentro del clúster
kubectl exec -it $(kubectl get pod -n btcbot -l app=btcbot-trader -o jsonpath='{.items[0].metadata.name}') -n btcbot -- curl http://btcbot-inference-service:8080/health
```

### Si el trader no puede conectarse a Binance:

```bash
# Verificar que los secretos estén configurados correctamente
kubectl describe pod -n btcbot -l app=btcbot-trader

# Verificar los logs de error detallados
kubectl logs -n btcbot -l app=btcbot-trader | grep -i "error"
```
