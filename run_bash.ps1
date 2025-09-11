# PowerShell script to run Bash scripts using Git Bash
# Usage: .\run-bash.ps1 path\to\script.sh [arg1 arg2 ...]

param (
    [Parameter(Mandatory=$true, Position=0)]
    [string]$BashScript,
    
    [Parameter(ValueFromRemainingArguments=$true)]
    [string[]]$ScriptArgs
)

# Potential Git Bash locations
$gitBashPaths = @(
    "C:\Program Files\Git\bin\bash.exe",
    "C:\Program Files (x86)\Git\bin\bash.exe",
    "${env:ProgramFiles}\Git\bin\bash.exe",
    "${env:ProgramFiles(x86)}\Git\bin\bash.exe",
    # Add more potential paths if needed
    "C:\Git\bin\bash.exe"
)

# Find Git Bash executable
$gitBashExe = $null
foreach ($path in $gitBashPaths) {
    if (Test-Path $path) {
        $gitBashExe = $path
        break
    }
}

if ($null -eq $gitBashExe) {
    Write-Error "Git Bash executable not found. Please make sure Git is installed."
    exit 1
}

# Convert Windows path to Unix-style path
function ConvertToUnixPath {
    param (
        [string]$WindowsPath
    )
    
    # Get the absolute path
    $absolutePath = Resolve-Path $WindowsPath -ErrorAction SilentlyContinue
    
    if ($null -eq $absolutePath) {
        # If file doesn't exist, just convert the format
        return $WindowsPath.Replace('\', '/').Replace('C:', '/c')
    }
    
    # Convert to Unix-style path
    $unixPath = $absolutePath.Path.Replace('\', '/').Replace('C:', '/c')
    return $unixPath
}

# Convert the script path to Unix style
$unixScriptPath = ConvertToUnixPath $BashScript

# Create the arguments string
$argString = ""
if ($ScriptArgs.Count -gt 0) {
    $argString = " " + ($ScriptArgs -join " ")
}

# Run the script with Git Bash
Write-Host "Running $BashScript with Git Bash..."
& $gitBashExe -c "cd `"$(ConvertToUnixPath $PWD)`"; bash `"$unixScriptPath`"$argString"

# Check the exit code
if ($LASTEXITCODE -ne 0) {
    Write-Host "Script execution failed with exit code $LASTEXITCODE" -ForegroundColor Red
    exit $LASTEXITCODE
} else {
    Write-Host "Script execution completed successfully" -ForegroundColor Green
}