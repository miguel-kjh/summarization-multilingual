#!/bin/bash

# Ruta donde están los scripts que quieres ejecutar
chmod +x scripts_1/*.sh

SCRIPTS_DIR="./scripts_1"
LOG_DIR="./logs_1"

# Crear carpeta de logs si no existe
mkdir -p "$LOG_DIR"

for SCRIPT in "$SCRIPTS_DIR"/*.sh; do
    if [ -f "$SCRIPT" ]; then
        # Obtener el nombre del script sin la ruta
        SCRIPT_NAME=$(basename "$SCRIPT")
        LOG_FILE="$LOG_DIR/$SCRIPT_NAME.log"

        echo "Ejecutando $SCRIPT_NAME..."
        
        # Ejecutar el script y guardar el log
        bash "$SCRIPT" > "$LOG_FILE" 2>&1
        EXIT_CODE=$?

        # Eliminar el script tras ejecutarlo
        rm -f "$SCRIPT"
    fi
# Esperar antes de volver a revisar si hay nuevos scripts (opcional)
# sleep 5
done