#!/bin/bash

# Caminho até o novo projeto EN
cd "/Users/matheus_mafs10/Documents/PC/Profissional/Google Drive_Matheus Freitas/Projetos_Python/EasySpectra_EN"

# Ativa o ambiente virtual desse projeto
source .venv/bin/activate

# Executa a interface via launcher (mesma lógica do PyInstaller)
python3 launcher.py
