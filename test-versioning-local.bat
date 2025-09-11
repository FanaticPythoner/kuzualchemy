@echo off
setlocal enabledelayedexpansion

REM Local Versioning Logic Test Script for Windows
REM This script tests the versioning logic locally without GitHub Actions

echo Testing KuzuAlchemy Automatic Versioning Logic Locally
echo ========================================================

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is required but not installed
    exit /b 1
)

REM Check if required libraries are available
python -c "import toml" >nul 2>&1
if errorlevel 1 (
    echo [WARNING] toml library not found, installing...
    pip install toml
)

python -c "import packaging" >nul 2>&1
if errorlevel 1 (
    echo [WARNING] packaging library not found, installing...
    pip install packaging
)

REM Check if git is available
git --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Git is required but not installed
    exit /b 1
)

echo [SUCCESS] All dependencies available

REM Extract current version from pyproject.toml
echo [INFO] Extracting current version from pyproject.toml...

if not exist "pyproject.toml" (
    echo [ERROR] pyproject.toml not found in current directory
    exit /b 1
)

for /f "delims=" %%i in ('python -c "import toml; print(toml.load('pyproject.toml')['project']['version'])"') do set CURRENT_VERSION=%%i
echo [SUCCESS] Current version: %CURRENT_VERSION%

REM Check if version changed in current commit
echo [INFO] Checking if version changed in current commit...

REM Check if we're in a git repository
git rev-parse --git-dir >nul 2>&1
if errorlevel 1 (
    echo [WARNING] Not in a git repository, assuming version not changed
    set VERSION_CHANGED=false
    goto calculate_version
)

REM Check if pyproject.toml was modified in current commit
git show HEAD --name-only 2>nul | findstr "pyproject.toml" >nul
if errorlevel 1 (
    set VERSION_CHANGED=false
    echo [INFO] pyproject.toml not modified in current commit
) else (
    REM pyproject.toml was modified, check if version line specifically changed
    git show HEAD -- pyproject.toml 2>nul | findstr /R "^[+-].*version.*=" >nul
    if errorlevel 1 (
        set VERSION_CHANGED=false
        echo [INFO] pyproject.toml changed but version line unchanged
    ) else (
        set VERSION_CHANGED=true
        echo [SUCCESS] Version changed in current commit
    )
)

:calculate_version
echo [INFO] Calculating final version...

if "%VERSION_CHANGED%"=="true" (
    REM Use the version from pyproject.toml as-is (real release)
    set FINAL_VERSION=%CURRENT_VERSION%
    set RELEASE_TYPE=release
    echo [SUCCESS] Using real release version: !FINAL_VERSION!
) else (
    REM Generate dev version
    echo [INFO] Generating dev version...

    REM CORRECTED LOGIC: Find commits since LAST version change, not all commits
    REM For batch simplicity, we'll use a more conservative approach
    set DEV_NUMBER=1

    REM Try to get actual commit count since last version change (simplified)
    git rev-parse --git-dir >nul 2>&1
    if not errorlevel 1 (
        REM In a real implementation, this would find the last commit where version was set
        REM For now, use a simple count but modulo to keep numbers reasonable
        for /f "delims=" %%i in ('git rev-list --count HEAD 2^>nul ^| findstr /R "[0-9]"') do (
            set /a DEV_NUMBER=%%i %% 10
            if !DEV_NUMBER! equ 0 set DEV_NUMBER=1
        )
        echo [INFO] Using dev number: !DEV_NUMBER! ^(simplified batch logic^)
        echo [NOTE] Real workflow uses commits since last version change
    ) else (
        echo [WARNING] Not in git repository, defaulting to dev1
    )
    
    set FINAL_VERSION=%CURRENT_VERSION%.dev!DEV_NUMBER!
    set RELEASE_TYPE=prerelease
    echo [SUCCESS] Generated dev version: !FINAL_VERSION!
)

REM Test version update
echo [INFO] Testing version update (dry run)...

REM Create Python script to test version update
(
echo import toml
echo import sys
echo import os
echo.
echo final_version = '%FINAL_VERSION%'
echo.
echo try:
echo     # Read current pyproject.toml
echo     with open('pyproject.toml', 'r'^) as f:
echo         data = toml.load(f^)
echo.
echo     original_version = data['project']['version']
echo.
echo     # Update version (in memory only^)
echo     data['project']['version'] = final_version
echo.
echo     print(f'[OK] Would update pyproject.toml version: {original_version} -^> {final_version}'^)
echo     print(f'[OK] Version update test successful'^)
echo.
echo except Exception as e:
echo     print(f'[ERROR] Version update test failed: {e}'^)
echo     sys.exit(1^)
) > test_update_version.py

python test_update_version.py
if errorlevel 1 (
    echo [ERROR] Version update test failed
    del test_update_version.py
    exit /b 1
)

del test_update_version.py
echo [SUCCESS] Version update test completed

REM Test package building capability
echo [INFO] Testing package building capability...

python -c "import build" >nul 2>&1
if errorlevel 1 (
    echo [WARNING] build module not available (would be installed in CI)
) else (
    echo [SUCCESS] build module available
)

python -c "import twine" >nul 2>&1
if errorlevel 1 (
    echo [WARNING] twine module not available (would be installed in CI)
) else (
    echo [SUCCESS] twine module available
)

echo [SUCCESS] Package building test completed

REM Test README.md update
echo [INFO] Testing README.md update (dry run)...

python test_readme_markers.py %FINAL_VERSION%
if errorlevel 1 (
    echo [ERROR] README.md update test failed
    exit /b 1
)
echo [SUCCESS] README.md update test completed

REM Print summary
echo.
echo Test Results Summary
echo =======================
echo Current Version:    %CURRENT_VERSION%
echo Version Changed:    %VERSION_CHANGED%
echo Final Version:      %FINAL_VERSION%
echo Release Type:       %RELEASE_TYPE%

git rev-parse --git-dir >nul 2>&1
if errorlevel 1 (
    echo Git Repository:     No
    echo Current Commit:     N/A
    echo Current Branch:     N/A
) else (
    echo Git Repository:     Yes
    for /f "delims=" %%i in ('git rev-parse HEAD 2^>nul') do echo Current Commit:     %%i
    for /f "delims=" %%i in ('git branch --show-current 2^>nul') do echo Current Branch:     %%i
)

echo.
if "%RELEASE_TYPE%"=="release" (
    echo [SUCCESS] Would create a REAL RELEASE: %FINAL_VERSION%
) else (
    echo [SUCCESS] Would create a DEV RELEASE: %FINAL_VERSION%
)

echo.
echo Next steps to test with act:
echo   act push -W .github/workflows/test-auto-release.yml
echo.
echo To test the full workflow on GitHub:
echo   git checkout -b test-auto-release
echo   git push origin test-auto-release

pause
