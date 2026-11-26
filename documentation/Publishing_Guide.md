# Publishing Guide - GitHub and Kaggle

Nothing was published automatically. These are the steps to publish the project yourself.
(لم يتم النشر تلقائياً - هاي الخطوات لتنشر المشروع بنفسك.)

## 1. GitHub (suggested repository name: `Assignment-4-Kernel-Multiclass-SVM-Five-Optimizers`)

### Option A - web upload (no git needed)
1. github.com → **New repository** → name `Assignment-4-Kernel-Multiclass-SVM-Five-Optimizers` → Public → *Create*.
2. **Add file → Upload files** → drag the contents of the project folder **except** `data/UCI HAR Dataset/`
   (the raw data is excluded by `.gitignore`; GitHub's web upload limit is 25 MB per file).
3. *Commit changes*. The README is shown on the repository page.

### Option B - git (VS Code terminal)
```bash
cd Assignment_4_Kernel_Multiclass_SVM
git init
git add .
git commit -m "Assignment 4: kernel multi-class SVM with five optimizers"
git branch -M main
git remote add origin https://github.com/<your-user>/Assignment-4-Kernel-Multiclass-SVM-Five-Optimizers.git
git push -u origin main
```
Never commit tokens, passwords or `kaggle.json`.

## 2. Kaggle (five notebooks)
### Option A - web
1. kaggle.com → **Create → New Notebook → File → Import Notebook** → choose e.g. `notebooks/01_Kernel_SVM_GD.ipynb`.
2. Right panel → **Add Input** → search `uciml/human-activity-recognition-with-smartphones` → *Add*.
3. **Run All** (CPU is enough; about 5-11 minutes per notebook) → **Save Version → Save & Run All**.
4. Repeat for the other four notebooks. Set each one to *Public* when you want the professor to see it.

### Option B - Kaggle CLI
`kaggle/<notebook>/kernel-metadata.json` is prepared for the username `yousifotoom12` (edit the `id` if needed):
```bash
cp notebooks/01_Kernel_SVM_GD.ipynb kaggle/01_Kernel_SVM_GD/
kaggle kernels push -p kaggle/01_Kernel_SVM_GD
```
(requires your own Kaggle API token in `~/.kaggle/`; notebooks are created private - switch to public on the website.)

## 3. One link for the professor
Put the five Kaggle notebook links in the README section "Kaggle notebooks" and share only the GitHub link.
