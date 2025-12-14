# GitHub Setup Guide

## Pre-Push Checklist

### ✅ Completed
- [x] Updated `.gitignore` to exclude sensitive files
- [x] Updated README with setup instructions
- [x] Created backend `.env.example` template

### ⚠️ Manual Steps Required

1. **Create `.env` files** (these are gitignored):
   ```bash
   # Backend
   cd backend
   cp .env.example .env
   # Edit .env with your MongoDB URI and JWT secret
   
   # Frontend (if needed)
   cd ../web
   # Create .env if you need custom settings
   ```

2. **Review `.gitignore`** to ensure:
   - All sensitive files are excluded
   - Model files (*.joblib) are excluded (they're large)
   - Temporary files are excluded
   - Node modules are excluded

3. **Check for sensitive data**:
   - No API keys in code
   - No passwords in code
   - No real MongoDB connection strings
   - No JWT secrets committed

4. **Initialize Git** (if not already done):
   ```bash
   git init
   git add .
   git commit -m "Initial commit: Network Intrusion Detection System"
   ```

5. **Add remote and push**:
   ```bash
   git remote add origin https://github.com/yourusername/your-repo-name.git
   git branch -M main
   git push -u origin main
   ```

## Files Excluded from Git

The following are automatically excluded:
- `node_modules/` (backend and web)
- `venv/` and `venv311/` (Python virtual environments)
- `*.env` files (except `.env.example`)
- `*.joblib` (model files - too large)
- `*.csv`, `*.txt` (data files)
- `tmp_uploads/`, `tmp_*.csv` (temporary files)
- `models/` directory (trained models)
- `reports/` directory (generated reports)

## Project Structure

```
major_project1/
├── backend/          # Node.js/Express API
│   ├── src/
│   ├── .env.example  # Environment template
│   └── package.json
├── web/              # React frontend
│   ├── src/
│   └── package.json
├── models/           # ML models (gitignored)
├── reports/          # Metrics reports (gitignored)
├── *.py             # Python scripts
├── .gitignore
└── README.md
```

## Important Notes

- **Model files are NOT committed** - users must train their own models
- **Environment variables are NOT committed** - use `.env.example` as template
- **Large data files are NOT committed** - users should download NSL-KDD dataset separately

