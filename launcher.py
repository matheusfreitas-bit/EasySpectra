# launcher.py
#
# Script de entrada para o PyInstaller.
# Ele garante que a pasta src/ esteja no sys.path
# e chama a função main() da interface do EasySpectra.

import sys
from pathlib import Path

# Caminho base = pasta onde está este arquivo (raiz do projeto)
BASE_DIR = Path(__file__).resolve().parent

# Pasta src/ onde está o pacote easyspectra
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from easyspectra.interface import main  # importa a função main() da interface


if __name__ == "__main__":
    main()
