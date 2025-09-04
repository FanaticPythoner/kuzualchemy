## HOW DOES THE VERSIONING WORK?

### **SCENARIO 1: PUSHING WITHOUT SPECIFYING A NEW VERSION (Dev Release)**

**What happens**: Creates automatic dev release like `0.1.0.dev1`, `0.1.0.dev2`, etc.

**Steps**:
1. Make your code changes (add features, fix bugs, etc.)
2. **DO NOT** touch the version in `pyproject.toml` - leave it as `version = "0.1.0"`
3. Commit and push to main:
   ```bash
   git add .
   git commit -m "Add new feature X"
   git push origin main
   ```

**What the system does automatically**:
- Detects that `pyproject.toml` version didn't change
- Counts commits since last version change (e.g., 3 commits)
- Generates dev version: `0.1.0.dev3`
- Updates `pyproject.toml` to `version = "0.1.0.dev3"`
- Updates `README.md` with current version, calculated status, and test results using persistent markers
- Commits the updated files back to repository
- Builds package with version `0.1.0.dev3`
- Creates GitHub pre-release tagged `v0.1.0.dev3`
- Publishes `kuzualchemy-0.1.0.dev3` to PyPI

**Result**: You get `0.1.0.dev3` on PyPI and GitHub releases

---

### **SCENARIO 2: PUSHING WITH A NEW VERSION (Real Release)**

**What happens**: Creates real release with your specified version.

**Steps**:
1. Make your code changes
2. **MANUALLY EDIT** `pyproject.toml` and change the version:
   ```toml
   [project]
   name = "kuzualchemy"
   version = "0.1.1"  # <-- CHANGE THIS LINE
   ```
3. Commit and push to main:
   ```bash
   git add .
   git commit -m "Release version 0.1.1"
   git push origin main
   ```

**What the system does automatically**:
- Detects that `pyproject.toml` version changed in this commit
- Uses your specified version: `0.1.1`
- Updates `README.md` with current version, calculated status, and test results using persistent markers
- Commits the updated files back to repository
- Builds package with version `0.1.1`
- Creates GitHub full release tagged `v0.1.1`
- Publishes `kuzualchemy-0.1.1` to PyPI

**Result**: You get `0.1.1` on PyPI and GitHub releases

---

### **EXAMPLE WORKFLOW**

```bash
# Current version in pyproject.toml: "0.1.0"

# 1. Regular development (creates dev releases)
git commit -m "Fix bug A"
git push origin main
# → Creates 0.1.0.dev1

git commit -m "Add feature B"  
git push origin main
# → Creates 0.1.0.dev2

git commit -m "Update docs"
git push origin main  
# → Creates 0.1.0.dev3

# 2. Ready for real release
# Edit pyproject.toml: version = "0.1.1"
git add pyproject.toml
git commit -m "Release version 0.1.1"
git push origin main
# → Creates 0.1.1 (real release)

# 3. Back to development
git commit -m "Post-release fix"
git push origin main
# → Creates 0.1.1.dev1
```

---

### **KEY RULES**

1. **For dev releases**: Don't touch `pyproject.toml` version
2. **For real releases**: Edit `pyproject.toml` version manually
3. **Always push to main branch** - that's what triggers the system
4. **Tests must pass** - release is blocked if tests fail
5. **README.md is updated automatically** - version and status are calculated and updated
6. **No manual intervention needed** - everything happens automatically

### **STATUS CALCULATION**

The system automatically calculates the project status based on version ranges:

- **Alpha**: Versions < 0.5.0 (e.g., 0.1.0, 0.4.9)
- **Beta**: Versions 0.5.0 to < 1.0.0 (e.g., 0.5.0, 0.9.9)
- **Release Candidate**: Versions 1.0.0 to < 2.0.0 (e.g., 1.0.0, 1.5.0)
- **Stable Release**: Versions >= 2.0.0 (e.g., 2.0.0, 3.1.0)

This status is automatically updated in README.md with each release.

---

### **MONITORING RELEASES**

- **GitHub Actions**: Check the "Actions" tab to see workflow progress
- **GitHub Releases**: Check the "Releases" section for created releases  
- **PyPI**: Check https://pypi.org/project/kuzualchemy/ for published packages

---

### **TROUBLESHOOTING**

- **Tests fail**: Fix tests and push again
- **PyPI conflict**: System uses `skip-existing` so it won't fail, just skip
- **Wrong version**: The system counts commits mathematically, so it's always correct
- **Want to test**: Use `./test-versioning-local.bat` (Windows) or `./test-versioning-local.sh` (Linux/macOS) to see what version would be generated

**That's it. The system handles everything else automatically.**



## HOW TO TEST LOCALLY?

When you run `./test-versioning-local.bat` (Windows) or `./test-versioning-local.sh` (Linux/macOS), here's exactly what you'll see:

### **EXPECTED OUTPUT**

```
Testing KuzuAlchemy Automatic Versioning Logic Locally
========================================================
[INFO] Checking dependencies...
[INFO] Found Python: python
[SUCCESS] All dependencies available
[INFO] Extracting current version from pyproject.toml...
[SUCCESS] Current version: 0.1.0
[INFO] Checking if version changed in current commit...
[INFO] pyproject.toml not modified in current commit
[INFO] Calculating final version...
[INFO] Generating dev version...
[INFO] Found 1 commits since last version change (+version)
[SUCCESS] Generated dev version: 0.1.0.dev1
[INFO] Testing version update (dry run)...
[OK] Would update pyproject.toml version: 0.1.0 -> 0.1.0.dev1
[OK] Version update test successful
[SUCCESS] Version update test completed
[INFO] Testing package building capability...
[SUCCESS] build module available
[WARNING] twine module not available (would be installed in CI)
[SUCCESS] Package building test completed
[INFO] Testing README.md update (dry run)...
[OK] README.md contains placeholders
[OK] Would update: Version={VERSION} -> 0.1.0.dev1
[OK] Would update: Status={STATUS} -> Alpha
[OK] README.md update test successful
[SUCCESS] README.md update test completed

Test Results Summary
=======================
Current Version:    0.1.0
Version Changed:    false
Final Version:      0.1.0.dev1
Release Type:       prerelease
Git Repository:     Yes
Current Commit:     48c7a37721fd387535d79d9adbc09eb442ad7194
Current Branch:     main

[SUCCESS] Would create a DEV RELEASE: 0.1.0.dev1
```

### **WHAT EACH SECTION MEANS**

1. **Dependencies Check**: Verifies Python, git, and required libraries are available
2. **Version Extraction**: Reads current version from `pyproject.toml`
3. **Change Detection**: Checks if you changed the version in your last commit
4. **Version Calculation**:
   - If version changed → uses that version (real release)
   - If version unchanged → generates dev version by counting commits
5. **Update Test**: Simulates updating `pyproject.toml` (doesn't actually change it)
6. **Build Test**: Checks if build tools are available
7. **README Test**: Simulates updating README.md with version and status
8. **Summary**: Shows what would happen if you pushed to main

### **DIFFERENT SCENARIOS YOU'LL SEE**

#### **Scenario A: Regular Development (No Version Change)**
```
Version Changed:    false
Final Version:      0.1.0.dev5
Release Type:       prerelease
[SUCCESS] Would create a DEV RELEASE: 0.1.0.dev5
```

#### **Scenario B: You Changed Version in Last Commit**
```
Version Changed:    true
Final Version:      0.1.1
Release Type:       release
[SUCCESS] Would create a REAL RELEASE: 0.1.1
```

### **THE DEV NUMBER CALCULATION**

The dev number (like `dev10`) comes from counting commits since the last time someone changed the version in `pyproject.toml`. So:

- If 5 commits happened since last version change → `0.1.0.dev5`
- If 12 commits happened since last version change → `0.1.0.dev12`
- If this is the first commit ever → `0.1.0.dev1`

### **WHAT THE SCRIPT DOES NOT DO**

- ❌ Does NOT actually change `pyproject.toml`
- ❌ Does NOT actually change `README.md`
- ❌ Does NOT create releases
- ❌ Does NOT publish to PyPI
- ❌ Does NOT build packages
- ❌ Does NOT commit anything

It's a **dry run** that shows you exactly what would happen when you push to main.

### **HOW TO INTERPRET THE RESULTS**

- **"Would create a DEV RELEASE"** = Your next push to main will create a development release
- **"Would create a REAL RELEASE"** = Your next push to main will create a stable release
- **Dev number matches your expectation** = The versioning logic is working correctly
- **All tests pass** = The system is ready to work on GitHub Actions

### **WHEN TO RUN THIS**

- Before pushing to main (to see what version will be created)
- After making changes (to verify the logic works)
- When debugging version issues
- To understand how the dev numbering works

**This gives you confidence that the automatic system will work correctly when you push to main.**

### **CROSS-PLATFORM TESTING**

The repository includes testing scripts for both platforms:

- **Windows**: `./test-versioning-local.bat` - Uses PowerShell and Windows Python
- **Linux/macOS**: `./test-versioning-local.sh` - Uses Bash and detects Python automatically

Both scripts perform identical tests and produce the same output format. The shell script includes robust Python detection that works with:
- Standard Python installations (`python3`, `python`, `py`)
- Conda/Miniconda environments
- Windows Python installations in common paths
- Both Git Bash and native Linux/macOS environments

### **RUNNING THE TESTS**

**On Windows:**
```bash
# PowerShell or Command Prompt
./test-versioning-local.bat

# Git Bash (if you prefer)
./run_bash.ps1 ./test-versioning-local.sh
```

**On Linux/macOS:**
```bash
# Make executable (first time only)
chmod +x test-versioning-local.sh

# Run the test
./test-versioning-local.sh
```
