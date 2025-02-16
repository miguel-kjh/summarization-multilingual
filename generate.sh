#!/bin/bash

DIR="models/Qwen"
MAXDEPTH=4

find "$DIR" -mindepth "$MAXDEPTH" -maxdepth "$MAXDEPTH" -type d | while read -r subdir; do
    echo "Procesando: $subdir"
done
