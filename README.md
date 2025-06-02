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

### **CRITICAL: Use the Correct Command**
```bash
# ✅ CORRECT (APA formatting):
quarto render sud_council_paper.qmd --to apaquarto-docx

# ❌ WRONG (breaks APA styling):
quarto render sud_council_paper.qmd --to docx
```

### **Prerequisites:**
1. **Install Quarto:** https://quarto.org/docs/get-started/
2. **Install R and RStudio:** https://posit.co/downloads/
3. **Install R packages:** `Rscript scripts/r/r_package_requirements.R`

### **To Edit and Recompile:**
1. Edit `sud_council_paper.qmd` in RStudio or VS Code
2. Run: `quarto render sud_council_paper.qmd --to apaquarto-docx`
3. Open `sud_council_paper.docx` to see APA-formatted result

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

### **Common Issues:**

**Quarto won't render:**
- Check Quarto is installed: `quarto --version`
- Install missing R packages: `Rscript scripts/r/r_package_requirements.R`

**Wrong Word formatting:**
- Make sure you use `--to apaquarto-docx` NOT `--to docx`
- Check `_extensions/wjschne/apaquarto/` folder exists

**Script errors:**
- Check data files exist in `data/` folder
- Install packages: `Rscript scripts/r/r_package_requirements.R`
- Scripts expect specific data structure

**Missing figures:**
- Run Study 2 visualization script: `Rscript scripts/r/study2_create_visualizations.R`
- Check `results/` folder for output files

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