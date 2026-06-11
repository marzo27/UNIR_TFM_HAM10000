@echo off
:: ============================================================
:: subir_a_github.bat
:: Configura y sube el repositorio UNIR_TFM_HAM10000 a GitHub
:: 
:: ANTES DE EJECUTAR:
::   1. Crea el repositorio vacío en:
::      https://github.com/marzo27/UNIR_TFM_HAM10000
::      (sin README, sin .gitignore, sin licencia - lo ponemos nosotros)
::   2. Asegúrate de tener Git instalado: git --version
::   3. Ejecuta este archivo desde D:\claude\UNIR_TFM_HAM10000\
:: ============================================================

echo.
echo  =====================================================
echo   SUBIDA DEL REPOSITORIO TFM A GITHUB
echo  =====================================================
echo.

cd /d "D:\claude\UNIR_TFM_HAM10000"

:: Verificar Git
git --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Git no encontrado. Descarga desde: https://git-scm.com/download/win
    pause & exit /b 1
)

:: Configurar identidad Git (solo si no está configurada)
git config user.name >nul 2>&1 || (
    set /p GIT_NAME="Tu nombre completo para Git: "
    git config --global user.name "%GIT_NAME%"
)
git config user.email >nul 2>&1 || (
    set /p GIT_EMAIL="Tu email de GitHub: "
    git config --global user.email "%GIT_EMAIL%"
)

:: Inicializar repositorio local
echo [1/5] Inicializando repositorio local...
git init
git branch -M main

:: Copiar figuras generadas al repositorio
echo [2/5] Copiando figuras de resultados...
if exist "D:\claude\Resultados\fig01_distribucion_clases.png" (
    copy "D:\claude\Resultados\fig*.png" "outputs\figures\" >nul 2>&1
    echo       Figuras copiadas a outputs/figures/
) else (
    echo       [AVISO] No se encontraron figuras en D:\claude\Resultados\
    echo       Ejecuta primero: python src/eda_ham10000_v1.py (con datos disponibles)
)

:: Copiar Excel si existe
if exist "D:\claude\Resultados\eda_resultados_HAM10000.xlsx" (
    copy "D:\claude\Resultados\eda_resultados_HAM10000.xlsx" "outputs\excel\" >nul 2>&1
    echo       Excel copiado a outputs/excel/
)

:: Añadir todos los archivos
echo [3/5] Añadiendo archivos al staging area...
git add .
git status

:: Primer commit
echo [4/5] Creando commit inicial...
git commit -m "feat: EDA completo HAM10000 - limpieza, imputacion, visualizaciones y correlaciones

- Script EDA: src/eda_ham10000_v1.py (rutas relativas, reproducible)
- 10 figuras PNG en outputs/figures/
- Resultados Excel en outputs/excel/ (12 hojas)
- README con instrucciones de reproduccion del entorno
- requirements.txt con dependencias Python
- Instrucciones de descarga del dataset Kaggle en data/README_data.md
- .gitignore configurado (excluye datos, venv, __pycache__)
- Licencia MIT para el codigo fuente"

:: Conectar con GitHub y subir
echo [5/5] Conectando con GitHub y subiendo...
echo.
echo  IMPORTANTE: Asegurate de que el repositorio
echo  https://github.com/marzo27/UNIR_TFM_HAM10000
echo  existe y esta VACIO en GitHub antes de continuar.
echo.
pause

git remote add origin https://github.com/marzo27/UNIR_TFM_HAM10000.git
git push -u origin main

if errorlevel 1 (
    echo.
    echo [ERROR] Fallo en el push. Posibles causas:
    echo   - El repositorio no existe en GitHub (crealo primero)
    echo   - Problemas de autenticacion (configura token de acceso personal)
    echo   Ver instrucciones en README.md
) else (
    echo.
    echo =====================================================
    echo  Repositorio subido correctamente a:
    echo  https://github.com/marzo27/UNIR_TFM_HAM10000
    echo =====================================================
)
echo.
pause
