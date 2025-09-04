#!/bin/bash

# Local Versioning Logic Test Script
# This script tests the versioning logic locally without GitHub Actions
# Works on both Linux and Windows (Git Bash/WSL)

set -e

echo "Testing KuzuAlchemy Automatic Versioning Logic Locally"
echo "========================================================"

# Colors for output (disable on Windows Git Bash if not supported)
if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
    # Windows Git Bash - use simple output
    RED=''
    GREEN=''
    YELLOW=''
    BLUE=''
    NC=''
else
    # Linux/macOS - use colors
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[1;33m'
    BLUE='\033[0;34m'
    NC='\033[0m' # No Color
fi

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to find Python executable
find_python() {
    # Try different Python commands in order of preference
    for cmd in python3 python py; do
        if command -v "$cmd" &> /dev/null; then
            # Verify it's Python 3
            if "$cmd" -c "import sys; exit(0 if sys.version_info[0] >= 3 else 1)" 2>/dev/null; then
                echo "$cmd"
                return 0
            fi
        fi
    done

    # On Windows, try common installation paths
    if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
        for path in "/c/Python3*/python.exe" "/c/Users/*/AppData/Local/Programs/Python/Python3*/python.exe" "/c/Program Files/Python3*/python.exe"; do
            if [[ -f $path ]]; then
                echo "$path"
                return 0
            fi
        done

        # Try conda/miniconda paths
        for path in "/c/Users/*/miniconda3/python.exe" "/c/Users/*/anaconda3/python.exe" "/c/ProgramData/miniconda3/python.exe" "/c/ProgramData/anaconda3/python.exe"; do
            if [[ -f $path ]]; then
                echo "$path"
                return 0
            fi
        done
    fi

    return 1
}

# Check if required tools are installed
check_dependencies() {
    print_status "Checking dependencies..."

    # Find Python and make it global
    PYTHON_CMD=$(find_python)
    if [[ $? -ne 0 ]]; then
        print_error "Python 3 is required but not found"
        print_error "Please install Python 3 or ensure it's in your PATH"
        if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
            print_error "On Windows, try: winget install Python.Python.3"
        fi
        exit 1
    fi

    print_status "Found Python: $PYTHON_CMD"

    # Check and install required packages
    if ! "$PYTHON_CMD" -c "import toml" 2>/dev/null; then
        print_warning "toml library not found, installing..."
        "$PYTHON_CMD" -m pip install toml
    fi

    if ! "$PYTHON_CMD" -c "import packaging" 2>/dev/null; then
        print_warning "packaging library not found, installing..."
        "$PYTHON_CMD" -m pip install packaging
    fi
    
    if ! command -v git &> /dev/null; then
        print_error "Git is required but not installed"
        exit 1
    fi
    
    print_success "All dependencies available"
}

# Extract current version from pyproject.toml
extract_version() {
    print_status "Extracting current version from pyproject.toml..."
    
    if [ ! -f "pyproject.toml" ]; then
        print_error "pyproject.toml not found in current directory"
        exit 1
    fi
    
    CURRENT_VERSION=$("$PYTHON_CMD" -c "import toml; print(toml.load('pyproject.toml')['project']['version'])")
    print_success "Current version: $CURRENT_VERSION"
}

# Check if version changed in current commit
check_version_changed() {
    print_status "Checking if version changed in current commit..."
    
    # Check if we're in a git repository
    if ! git rev-parse --git-dir > /dev/null 2>&1; then
        print_warning "Not in a git repository, assuming version not changed"
        VERSION_CHANGED="false"
        return
    fi
    
    # Check if pyproject.toml was modified in current commit
    if git show HEAD --name-only 2>/dev/null | grep -q "pyproject.toml"; then
        # pyproject.toml was modified, check if version line specifically changed
        if git show HEAD -- pyproject.toml 2>/dev/null | grep -E "^[+-].*version.*="; then
            VERSION_CHANGED="true"
            print_success "Version changed in current commit"
        else
            VERSION_CHANGED="false"
            print_status "pyproject.toml changed but version line unchanged"
        fi
    else
        VERSION_CHANGED="false"
        print_status "pyproject.toml not modified in current commit"
    fi
}

# Calculate dev version if needed
calculate_version() {
    print_status "Calculating final version..."
    
    if [ "$VERSION_CHANGED" = "true" ]; then
        # Use the version from pyproject.toml as-is (real release)
        FINAL_VERSION="$CURRENT_VERSION"
        RELEASE_TYPE="release"
        print_success "Using real release version: $FINAL_VERSION"
    else
        # Generate dev version
        print_status "Generating dev version..."

        # Find the LAST commit where current version was set (most recent)
        if git rev-parse --git-dir > /dev/null 2>&1; then
            VERSION_COMMIT=$(git log --oneline -p -- pyproject.toml 2>/dev/null | grep -B1 "+version = \"$CURRENT_VERSION\"" | head -1 | grep "^[a-f0-9]" | cut -d' ' -f1)

            if [ -z "$VERSION_COMMIT" ]; then
                # Fallback: find any commit that mentions this version
                VERSION_COMMIT=$(git log --oneline --grep="$CURRENT_VERSION" 2>/dev/null | head -1 | cut -d' ' -f1)
            fi

            if [ -z "$VERSION_COMMIT" ]; then
                # Ultimate fallback: count from first commit
                DEV_NUMBER=$(git rev-list --count HEAD 2>/dev/null || echo "1")
                print_status "[FALLBACK] No version history found, counting all commits: $DEV_NUMBER"
            else
                # Count commits AFTER the version was set (this is the key fix)
                DEV_NUMBER=$(git rev-list --count ${VERSION_COMMIT}..HEAD 2>/dev/null || echo "1")
                print_status "Found $DEV_NUMBER commits since version $CURRENT_VERSION was last set ($VERSION_COMMIT)"

                # If no commits since version was set, this means we're at the release commit
                if [ "$DEV_NUMBER" -eq 0 ]; then
                    # We're at the exact release commit, no dev version needed
                    FINAL_VERSION="$CURRENT_VERSION"
                    RELEASE_TYPE="release"
                    print_success "At release commit, using version as-is: $FINAL_VERSION"
                    print_success "Would create a RELEASE: $FINAL_VERSION"
                    return 0
                fi
            fi
        else
            # Not in git repo, default to dev1
            DEV_NUMBER=1
            print_warning "Not in git repository, defaulting to dev1"
        fi
        
        FINAL_VERSION="${CURRENT_VERSION}.dev${DEV_NUMBER}"
        RELEASE_TYPE="prerelease"
        print_success "Generated dev version: $FINAL_VERSION"
    fi
}

# Test version update
test_version_update() {
    print_status "Testing version update (dry run)..."
    
    # Create Python script to test version update
    cat > test_update_version.py << 'EOF'
import toml
import sys
import os

final_version = os.environ.get('FINAL_VERSION', 'unknown')

try:
    # Read current pyproject.toml
    with open('pyproject.toml', 'r') as f:
        data = toml.load(f)
    
    original_version = data['project']['version']
    
    # Update version (in memory only)
    data['project']['version'] = final_version
    
    print(f'[OK] Would update pyproject.toml version: {original_version} -> {final_version}')
    print(f'[OK] Version update test successful')
    
except Exception as e:
    print(f'[ERROR] Version update test failed: {e}')
    sys.exit(1)
EOF
    
    # Run the test script
    FINAL_VERSION="$FINAL_VERSION" "$PYTHON_CMD" test_update_version.py
    
    # Clean up
    rm test_update_version.py
    
    print_success "Version update test completed"
}

# Test package building
test_package_building() {
    print_status "Testing package building capability..."
    
    # Check if build tools are available
    if "$PYTHON_CMD" -c "import build" 2>/dev/null; then
        print_success "build module available"
    else
        print_warning "build module not available (would be installed in CI)"
    fi

    if "$PYTHON_CMD" -c "import twine" 2>/dev/null; then
        print_success "twine module available"
    else
        print_warning "twine module not available (would be installed in CI)"
    fi
    
    print_success "Package building test completed"
}

# Test README.md update
test_readme_update() {
    print_status "Testing README.md update (dry run)..."

    # Create Python script to test README update with persistent markers
    cat > test_readme_update.py << 'EOF'
import re
import os
from packaging import version
from datetime import datetime

def calculate_status(version_str: str) -> str:
    """Calculate status based on version ranges."""
    # Remove dev suffix for status calculation
    clean_version = re.sub(r'\.dev\d+$', '', version_str)
    v = version.parse(clean_version)

    if v < version.parse("0.5.0"):
        return "Alpha"
    elif v < version.parse("1.0.0"):
        return "Beta"
    elif v < version.parse("2.0.0"):
        return "Release Candidate"
    else:
        return "Stable Release"

def get_disclaimer_note(status: str) -> str:
    """Generate appropriate disclaimer note based on status."""
    if status == "Alpha":
        return "> **Note**: This software is currently in alpha development. APIs may change."
    elif status == "Beta":
        return "> **Note**: This software is in beta. Some features may be unstable."
    elif status == "Release Candidate":
        return "> **Note**: This software is a release candidate. Please report any issues."
    else:  # Stable Release
        return "> **Note**: This software is stable and production-ready."

# Get version from environment
final_version = os.environ.get('FINAL_VERSION', 'unknown')
status = calculate_status(final_version)
disclaimer = get_disclaimer_note(status)
test_results = "[OK] 42 passed, [FAIL] 1 failed, [ERROR] 0 error"
timestamp = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')

# Read current README.md
try:
    with open('README.md', 'r', encoding='utf-8') as f:
        content = f.read()

    # Check for persistent markers
    marker_pattern = r'<!-- KUZUALCHEMY-AUTO-UPDATE-START -->.*?<!-- KUZUALCHEMY-AUTO-UPDATE-END -->'

    if re.search(marker_pattern, content, re.DOTALL):
        print(f'[OK] README.md contains persistent markers')
        print(f'[OK] Would update: Version -> {final_version}')
        print(f'[OK] Would update: Status -> {status}')
        print(f'[OK] Would update: Tests -> {test_results}')
        print(f'[OK] Would update: Disclaimer -> {disclaimer}')
        print(f'[OK] README.md update test successful')
    else:
        print(f'[ERROR] README.md missing persistent markers')
        print(f'[INFO] Looking for: <!-- KUZUALCHEMY-AUTO-UPDATE-START -->')
        exit(1)
except Exception as e:
    print(f'[ERROR] README.md update test failed: {e}')
    exit(1)
EOF

    # Run the test script
    FINAL_VERSION="$FINAL_VERSION" "$PYTHON_CMD" test_readme_update.py

    if [ $? -ne 0 ]; then
        print_error "README.md update test failed"
        rm -f test_readme_update.py
        exit 1
    fi

    # Clean up
    rm -f test_readme_update.py
    print_success "README.md update test completed"
}

# Print summary
print_summary() {
    echo ""
    echo "Test Results Summary"
    echo "======================="
    echo "Current Version:    $CURRENT_VERSION"
    echo "Version Changed:    $VERSION_CHANGED"
    echo "Final Version:      $FINAL_VERSION"
    echo "Release Type:       $RELEASE_TYPE"
    echo "Git Repository:     $(git rev-parse --git-dir > /dev/null 2>&1 && echo "Yes" || echo "No")"
    echo "Current Commit:     $(git rev-parse HEAD 2>/dev/null || echo "N/A")"
    echo "Current Branch:     $(git branch --show-current 2>/dev/null || echo "N/A")"
    echo ""
    
    if [ "$RELEASE_TYPE" = "release" ]; then
        print_success "Would create a REAL RELEASE: $FINAL_VERSION"
    else
        print_success "Would create a DEV RELEASE: $FINAL_VERSION"
    fi
    
    echo ""
    echo "Next steps to test with act:"
    echo "  act push -W .github/workflows/test-auto-release.yml"
    echo ""
    echo "To test the full workflow on GitHub:"
    echo "  git checkout -b test-auto-release"
    echo "  git push origin test-auto-release"
}

# Main execution
main() {
    check_dependencies
    extract_version
    check_version_changed
    calculate_version
    test_version_update
    test_package_building
    test_readme_update
    print_summary
}

# Run the script
main "$@"
