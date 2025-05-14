# PowerShell script to initialize the Git repository for the RACE Dataset Analysis project

# Initialize Git repo if it doesn't exist
if (-not (Test-Path .git)) {
    Write-Host "Initializing Git repository..." -ForegroundColor Green
    git init
} else {
    Write-Host "Git repository already exists." -ForegroundColor Yellow
}

# Make sure .gitignore is respected
Write-Host "Cleaning any ignored files that might have been previously tracked..." -ForegroundColor Green
git rm -r --cached .
git add .
git status

Write-Host "`nNext steps:" -ForegroundColor Cyan
Write-Host "1. Review the files being tracked above" -ForegroundColor Cyan
Write-Host "2. Make your initial commit:" -ForegroundColor Cyan
Write-Host "   git commit -m 'Initial commit: RACE dataset analysis project'" -ForegroundColor Cyan
Write-Host "3. Add your GitHub repository as remote:" -ForegroundColor Cyan
Write-Host "   git remote add origin https://github.com/yourusername/race-nlp-analysis.git" -ForegroundColor Cyan
Write-Host "4. Push to GitHub:" -ForegroundColor Cyan
Write-Host "   git push -u origin main" -ForegroundColor Cyan
