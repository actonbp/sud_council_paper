# SUD Counselors Research Project

**Complete mixed-methods research study examining factors influencing undergraduate interest in SUD counseling careers, published as an APA-formatted Quarto academic manuscript.**

---

## 🎯 **FOR ERICA: QUICK START GUIDE**

### **What You Need to Know:**
This repository contains a **complete, publication-ready academic study** with:
- ✅ **Study 1:** Machine learning analysis (L1-regularized logistic regression) 
- ✅ **Study 2:** Text analysis of focus group discussions
- ✅ **Complete manuscript:** APA-formatted paper ready for journal submission
- ✅ **All analysis scripts:** Clean, documented, and reproducible

### **Key Files You'll Use:**
- 📄 **`sud_council_paper.qmd`** - Main manuscript (edit this)
- 📄 **`sud_council_paper.docx`** - Compiled Word version
- 📁 **`scripts/r/`** - All analysis scripts (5 total, clearly named)
- 📁 **`results/`** - All figures, tables, and outputs
- 📄 **`references.bib`** - Bibliography (add new references here)

---

## 🚀 **HOW TO COMPILE THE MANUSCRIPT**

### **🎯 ERIKA: START HERE - Complete Setup Guide**

#### **Step 1: Install Required Software**
```bash
# 1. Install Quarto (REQUIRED)
# Download from: https://quarto.org/docs/get-started/
# Choose installer for your operating system

# 2. Install R and RStudio (REQUIRED)  
# Download from: https://posit.co/downloads/
# Install R first, then RStudio

# 3. Verify installations
quarto --version    # Should show version 1.3+ 
R --version        # Should show R 4.0+
```

#### **Step 2: Setup Repository**
```bash
# 1. Clone repository
git clone https://github.com/actonbp/sud_council_paper.git
cd sud_council_paper

# 2. Install R packages (THIS IS CRITICAL)
Rscript scripts/r/r_package_requirements.R

# 3. Verify APA extension exists (should already be included)
ls _extensions/wjschne/apaquarto/  # Should show many files
```

#### **Step 3: Add Your Data Files**
```bash
# See DATA_REQUIREMENTS.md for exact files needed
# Add data files to:
# - data/survey/
# - data/focus_group/  
# - data/processed/

# Verify data setup
Rscript -e "source('scripts/r/r_package_requirements.R'); cat('Data check complete!')"
```

#### **Step 4: Compile Manuscript**
```bash
# 🎯 THE CRITICAL COMMAND (use this EXACT command):
quarto render sud_council_paper.qmd --to apaquarto-docx

# ❌ NEVER use this (breaks APA formatting):
# quarto render sud_council_paper.qmd --to docx
```

#### **Step 5: Verify Success**
```bash
# Check outputs were created
ls sud_council_paper.docx          # Main manuscript
ls sud_council_paper_files/        # Supporting files

# Open in Word to verify APA formatting:
# - Title page with running head
# - Tables moved to end
# - Proper APA citations
# - Double spacing, Times New Roman
```

### **🔧 Quick Compilation (After Setup)**
```bash
# For daily work after initial setup:
quarto render sud_council_paper.qmd --to apaquarto-docx && open sud_council_paper.docx
```

---

## 📈 **HOW TO RUN THE ANALYSES**

### **🎯 ERIKA: Analysis Reproduction Guide**

#### **Study 1: Quantitative Analysis (START HERE)**
```bash
# 1. Verify data files exist (see DATA_REQUIREMENTS.md)
ls data/survey/survey_raw.csv        # Must exist
ls data/survey/ml_ready_survey_data.csv  # Must exist

# 2. Run complete Study 1 analysis (takes ~5-10 minutes)
Rscript scripts/r/study1_main_analysis.R

# 3. Check outputs were created
ls results/r/study1_logistic_fs_modern/  # Should contain:
# - final_fit.rds            (trained model)
# - final_metrics.csv        (performance metrics)  
# - model_coefficients.csv   (effect sizes)
# - features_used.txt        (selected variables)
```

#### **Study 2: Text Analysis (Run in Order)**
```bash
# 1. Verify focus group data exists
ls data/focus_group/*.csv | wc -l    # Should show 7 files

# 2. Run Study 2 pipeline (each script ~2-5 minutes)
Rscript scripts/r/study2_text_preprocessing.R        # Step 1: Text processing
Rscript scripts/r/study2_cooccurrence_analysis.R     # Step 2: Theme analysis  
Rscript scripts/r/study2_methodology_validation.R    # Step 3: Validation tables
Rscript scripts/r/study2_create_visualizations.R     # Step 4: Publication figures

# 3. Check Study 2 outputs
ls results/study2_*.png              # Visualization files
ls results/study2_*.csv              # Analysis tables
ls results/study2_*.html             # Interactive methodology demo
```

#### **🔍 Quick Analysis Verification**
```r
# Run this in R console to verify everything worked:
source("scripts/r/r_package_requirements.R")

# Check Study 1 results
if (file.exists("results/r/study1_logistic_fs_modern/final_metrics.csv")) {
  cat("✅ Study 1 completed successfully!\n")
} else {
  cat("❌ Study 1 needs to be run\n")
}

# Check Study 2 results  
study2_files <- length(list.files("results/", pattern = "study2_.*\\.png"))
if (study2_files >= 3) {
  cat("✅ Study 2 completed successfully!\n")
} else {
  cat("❌ Study 2 needs to be run\n")
}
```

#### **⚡ Full Reproduction (Both Studies)**
```bash
# Complete analysis pipeline from scratch (15-20 minutes total)
echo "Running complete analysis pipeline..."
Rscript scripts/r/study1_main_analysis.R
Rscript scripts/r/study2_text_preprocessing.R  
Rscript scripts/r/study2_cooccurrence_analysis.R
Rscript scripts/r/study2_methodology_validation.R
Rscript scripts/r/study2_create_visualizations.R
echo "✅ All analyses complete! Ready to compile manuscript."
```

---

## 📊 **STUDY OVERVIEW**

### **Study 1: Quantitative Analysis (N=391)**
- **Method:** L1-regularized logistic regression (Lasso) using tidymodels
- **Key Finding:** Students uncertain about mental health careers show 74% higher odds of SUD counseling interest
- **Performance:** Cross-validation ROC AUC = 0.787 [95% CI: 0.766, 0.809]
- **Script:** `scripts/r/study1_main_analysis.R`

### **Study 2: Qualitative Analysis (N=19, 7 focus groups)**
- **Method:** Conservative text analysis with Porter stemming and co-occurrence analysis
- **Key Finding:** Four natural themes emerge (Professional-Field 45.3%, Personal-Emotional 27.5%, People-Centered 18.8%, Service-Helping 8.4%)
- **Scripts:** 4 scripts for preprocessing, analysis, validation, and visualization

### **Mixed-Methods Integration**
- Qualitative themes validate and explain quantitative predictors
- Career uncertainty pathway supported across both studies
- Comprehensive manuscript integrates findings with Social Cognitive Career Theory

---

## 📁 **REPOSITORY STRUCTURE**

```
sud_council_paper/
├── 📄 sud_council_paper.qmd          # ✅ MAIN MANUSCRIPT (edit this)
├── 📄 sud_council_paper.docx          # ✅ Compiled Word document  
├── 📄 references.bib                 # ✅ Bibliography (APA format)
├── 📄 README.md                      # This file
├── 📄 CLAUDE.md                      # AI agent instructions
├── 
├── 📁 scripts/r/                     # ✅ ALL ANALYSIS SCRIPTS
│   ├── r_package_requirements.R           # Install required packages
│   ├── study1_main_analysis.R             # Study 1: Tidymodels analysis  
│   ├── study2_text_preprocessing.R        # Study 2: Text processing
│   ├── study2_cooccurrence_analysis.R     # Study 2: Co-occurrence analysis
│   ├── study2_methodology_validation.R    # Study 2: Validation tables
│   └── study2_create_visualizations.R     # Study 2: Publication figures
│
├── 📁 results/                       # ✅ ALL OUTPUTS
│   ├── r/study1_logistic_fs_modern/       # Study 1 model outputs
│   ├── study2_*.csv                       # Study 2 analysis tables
│   ├── study2_*.png                       # Study 2 publication figures
│   └── study2_interactive_methodology.html # Interactive methodology demo
│
├── 📁 _extensions/wjschne/apaquarto/  # ✅ APA FORMATTING SYSTEM
├── 📁 config/                        # Analysis configuration
├── 📁 data/                          # Data files (not in Git)
├── 📁 archive/                       # Non-essential files moved here
└── 📁 venv/                          # Python environment (legacy)
```

---

## 🔬 **RUNNING THE ANALYSES**

### **Prerequisites:**
```r
# Install all required R packages
Rscript scripts/r/r_package_requirements.R
```

### **Study 1 (Quantitative):**
```r
# Complete tidymodels analysis pipeline
Rscript scripts/r/study1_main_analysis.R
```

### **Study 2 (Qualitative) - Run in Order:**
```r
# 1. Text preprocessing with conservative SUD detection
Rscript scripts/r/study2_text_preprocessing.R

# 2. Co-occurrence analysis and thematic clustering  
Rscript scripts/r/study2_cooccurrence_analysis.R

# 3. Methodology validation and documentation
Rscript scripts/r/study2_methodology_validation.R

# 4. Create publication-quality visualizations
Rscript scripts/r/study2_create_visualizations.R
```

---

## 📝 **WORKING WITH REFERENCES**

### **Adding New Citations:**
1. Add entries to `references.bib` in standard BibTeX format:
```bibtex
@article{author2024,
  title={Article Title},
  author={Author Name},
  journal={Journal Name},
  year={2024}
}
```

2. Cite in manuscript using: `[@author2024]` or `@author2024`

3. Recompile: `quarto render sud_council_paper.qmd --to apaquarto-docx`

### **Citation Style:**
- Uses APA 7th edition automatically
- References appear at end of document
- In-text citations formatted properly

---

## 🎨 **APA FORMATTING FEATURES**

The `apaquarto` extension provides:
- ✅ **Proper title page** with running head
- ✅ **Tables at end** with "INSERT TABLE X ABOUT HERE" placeholders
- ✅ **APA 7th edition citations** and references  
- ✅ **Times New Roman, double spacing, 1-inch margins**
- ✅ **Proper headers** and page numbers
- ✅ **Figure and table numbering**

**Important:** Always use `--to apaquarto-docx` to maintain APA formatting!

---

## 📊 **KEY FINDINGS SUMMARY**

### **Study 1: Quantitative Results**
- **Career Uncertainty Pathway:** Students "unsure" about mental health careers show 74% higher odds of SUD counseling interest
- **Professional Familiarity Effect:** Exposure to SUD counselors increases interest (OR = 1.33)
- **Strong Model Performance:** ROC AUC 0.787 with robust cross-validation
- **Effect Sizes:** Cohen's d = 0.764 (strong for behavioral research)

### **Study 2: Qualitative Themes**
1. **Professional-Field Recognition (45.3%)** - SUD counseling as legitimate career
2. **Personal-Emotional Framework (27.5%)** - Family experience and emotional connection  
3. **People-Centered Orientation (18.8%)** - Helping and relational focus
4. **Service-Helping Identity (8.4%)** - Counselor role conceptualization

### **Mixed-Methods Integration**
- Qualitative themes validate quantitative predictors
- Personal experience pathway emerges in both studies
- Career uncertainty supported by professional field recognition
- Theoretical alignment with Social Cognitive Career Theory

---

## 🛠️ **TROUBLESHOOTING**

### **🚨 Most Common Issues for Erika**

#### **1. Quarto Compilation Fails**
```bash
# Problem: "quarto: command not found" or "unknown format"

# Solution A: Check Quarto installation
quarto --version               # Should show 1.3+
# If missing: Download from https://quarto.org/docs/get-started/

# Solution B: Verify APA extension
ls _extensions/wjschne/apaquarto/    # Should show many .lua files  
# If missing: Extension should be included in repository

# Solution C: Check R packages
Rscript scripts/r/r_package_requirements.R
# Should install all required packages without errors
```

#### **2. Wrong APA Formatting in Word**
```bash
# Problem: Generic Word format instead of APA

# ✅ CORRECT command:
quarto render sud_council_paper.qmd --to apaquarto-docx

# ❌ WRONG commands (don't use these):
# quarto render sud_council_paper.qmd --to docx
# quarto render sud_council_paper.qmd

# Verify output: Tables should be at END, not in text
```

#### **3. R Analysis Scripts Fail**
```bash
# Problem: "Error: object not found" or "cannot open connection"

# Solution: Check data files exist (see DATA_REQUIREMENTS.md)
ls data/survey/survey_raw.csv              # Study 1 data
ls data/focus_group/*.csv | wc -l          # Should show 7 files

# Solution: Install packages properly
Rscript scripts/r/r_package_requirements.R

# Solution: Check R version  
R --version    # Should be R 4.0+
```

#### **4. Missing Analysis Results**
```bash
# Problem: No figures or results files

# Solution: Run analyses in correct order
# Study 1:
Rscript scripts/r/study1_main_analysis.R

# Study 2 (run all 4 in order):
Rscript scripts/r/study2_text_preprocessing.R
Rscript scripts/r/study2_cooccurrence_analysis.R  
Rscript scripts/r/study2_methodology_validation.R
Rscript scripts/r/study2_create_visualizations.R

# Verify outputs:
ls results/r/study1_logistic_fs_modern/    # Study 1 results
ls results/study2_*.png                   # Study 2 figures
```

#### **5. Quarto Renders But R Code Fails**
```bash
# Problem: "Execution halted" or R chunk errors in Quarto

# Solution: Run R scripts individually first
cd /path/to/sud_council_paper
Rscript scripts/r/study1_main_analysis.R   # Must complete successfully

# Then try Quarto again:
quarto render sud_council_paper.qmd --to apaquarto-docx
```

### **📞 Getting Help**

#### **Check These First:**
1. **Data files**: Review `DATA_REQUIREMENTS.md` for exact files needed
2. **Dependencies**: Run `Rscript scripts/r/r_package_requirements.R`  
3. **Versions**: Ensure Quarto 1.3+, R 4.0+, RStudio 2023+
4. **Extensions**: Verify `_extensions/wjschne/apaquarto/` exists

#### **Still Having Issues?**
- **Technical details**: Check `CLAUDE.md` for comprehensive documentation
- **Error messages**: Copy exact error text when seeking help
- **Environment**: Include OS version, R version, Quarto version

---

## 📚 **FOR FURTHER HELP**

- **Quarto Documentation:** https://quarto.org/docs/
- **APA Extension Guide:** Check `_extensions/wjschne/apaquarto/` folder
- **Tidymodels Reference:** https://www.tidymodels.org/
- **Repository Issues:** Check `CLAUDE.md` for detailed technical documentation

---

**Last Updated:** June 2024 - Cleaned and organized for colleague handoff  
**Status:** ✅ Publication-ready manuscript with complete analysis pipeline  
**Contact:** Use GitHub issues for questions about the repository