# Este script contiene todas las importaciones necesarias.
# En este script se hallan las importaciones necesarias de librerías y módulos comunes a todos los cuadernos.

import os
import subprocess

# Obtiene el directorio actual del script de Python
current_directory = os.path.dirname(os.path.abspath(__file__))

# Ruta completa al archivo por lotes
batch_file = os.path.join(current_directory, "setup.bat")

# Ejecuta el archivo por lotes
subprocess.call([batch_file], shell=True)
