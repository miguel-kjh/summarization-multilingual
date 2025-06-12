#!/bin/bash

# Directorio donde están los runs offline
WANDB_DIR="wandb"

# Recorre todos los directorios que empiezan por offline-run-
for run_dir in "$WANDB_DIR"/offline-run-*; do
  if [ -d "$run_dir" ]; then
    echo "Sincronizando $run_dir..."
    wandb sync "$run_dir"
  fi
done

echo "Sincronización completa."
