# CLAUDE.md - AI Agent Instructions

## Project Overview
This is a **clean, production-ready SUD (Substance Use Disorder) Counselors research repository** with complete mixed-methods analysis and publication-ready manuscript. **Repository was cleaned and organized for colleague handoff in June 2024.**

## Current Status: CLEAN REPOSITORY READY FOR COLLEAGUE COLLABORATION ✅🤝

### 🏆 **REPOSITORY HANDOFF COMPLETE**
1. ✅ **CLEANED STRUCTURE:** Non-essential files archived, scripts renamed for clarity
2. ✅ **SIMPLIFIED SCRIPTS:** 5 essential R scripts with clear names and purposes
3. ✅ **COLLEAGUE-READY DOCS:** Comprehensive README.md for new user onboarding
4. ✅ **COMPLETE MANUSCRIPT:** Publication-ready APA paper with all analysis integrated
5. ✅ **ARCHIVE SYSTEM:** Non-essential files moved to `archive/` with documentation
6. ✅ **CLEAR WORKFLOW:** Step-by-step instructions for manuscript compilation and analysis

## 🎯 **Repository Structure (Post-Cleanup)**

### **Essential Files for Collaboration:**
```
sud_council_paper/
├── 📄 sud_council_paper.qmd          # MAIN MANUSCRIPT
├── 📄 sud_council_paper.docx          # Compiled APA Word document
├── 📄 README.md                      # COLLEAGUE ONBOARDING GUIDE
├── 📄 references.bib                 # APA bibliography
├── 📄 CLAUDE.md                      # This file
├── 
├── 📁 scripts/r/                     # 5 ESSENTIAL SCRIPTS ONLY
│   ├── r_package_requirements.R           # Package installation
│   ├── study1_main_analysis.R             # Study 1: Tidymodels
│   ├── study2_text_preprocessing.R        # Study 2: Text processing
│   ├── study2_cooccurrence_analysis.R     # Study 2: Co-occurrence
│   ├── study2_methodology_validation.R    # Study 2: Validation
│   └── study2_create_visualizations.R     # Study 2: Figures
│
├── 📁 results/                       # All outputs and figures
├── 📁 _extensions/wjschne/apaquarto/  # APA formatting system
├── 📁 data/                          # Data files (gitignored)
└── 📁 archive/                       # Non-essential files
```

### **What Was Archived:**
- `archive/deprecated_python/` - Original Python scripts (replaced by R)
- `archive/scripts/` - Duplicate/draft R scripts  
- `archive/old_drafts/` - Previous manuscript versions
- `archive/temp_files/` - Temporary and system files

## 🔬 **Complete Research Summary**

### **Study 1: Quantitative Analysis (N=391)**
- **Method:** L1-regularized logistic regression using tidymodels framework
- **Key Finding:** Students uncertain about mental health careers show 74% higher odds of SUD counseling interest
- **Performance:** Cross-validation ROC AUC = 0.787 [95% CI: 0.766, 0.809]
- **Effect Sizes:** Cohen's d = 0.764, correlation r = 0.411 (strong for behavioral research)
- **Statistical Validation:** χ² = 92.59, p < .001 for career uncertainty effect

### **Study 2: Qualitative Analysis (N=19, 7 focus groups)**
- **Method:** Conservative text analysis (19.7% SUD detection) with Porter stemming
- **Four Primary Themes:**
  1. **Professional-Field Recognition (45.3%)** - Career legitimacy and opportunities
  2. **Personal-Emotional Framework (27.5%)** - Family experience and emotional connection
  3. **People-Centered Orientation (18.8%)** - Relational and helping focus
  4. **Service-Helping Identity (8.4%)** - Counselor role conceptualization

### **Mixed-Methods Integration**
- Qualitative themes validate quantitative predictors
- Career uncertainty pathway supported across both studies
- Personal experience emerges in statistical models and thematic analysis
- Theoretical alignment with Social Cognitive Career Theory

## 📊 **Analysis Pipeline (Simplified)**

### **Study 1 Workflow:**
```r
# Complete tidymodels analysis
Rscript scripts/r/study1_main_analysis.R
```
- Modern tidymodels implementation with proper validation
- L1 regularization for feature selection and interpretability  
- Bootstrap stability analysis (100% sign consistency)
- Cross-validation: 10-fold with 5 repeats (50 total folds)

### **Study 2 Workflow:**
```r
# Run in sequence:
Rscript scripts/r/study2_text_preprocessing.R      # Conservative SUD detection
Rscript scripts/r/study2_cooccurrence_analysis.R   # Thematic analysis
Rscript scripts/r/study2_methodology_validation.R  # Validation tables
Rscript scripts/r/study2_create_visualizations.R   # Publication figures
```
- Conservative SUD detection requiring substance-specific terms
- Porter stemming for linguistic robustness
- Co-occurrence analysis for data-driven theme emergence
- Network visualization showing thematic coherence

## 📄 **APA Manuscript Compilation**

### **CRITICAL COMMAND:**
```bash
# ✅ CORRECT (APA formatting):
quarto render sud_council_paper.qmd --to apaquarto-docx

# ❌ WRONG (breaks APA styling):
quarto render sud_council_paper.qmd --to docx
```

### **APA Extension Features:**
- **Proper title page** with running head and author affiliations
- **Tables moved to end** with "INSERT TABLE X ABOUT HERE" placeholders
- **APA 7th edition citations** and reference formatting
- **Times New Roman, double spacing, 1-inch margins**
- **Automatic figure and table numbering**

## 🎯 **Guidelines for Future AI Agents**

### **Repository Guidelines:**
1. **NEVER recreate archived files** - Everything essential is in the main directory
2. **Use the clean script names** - No need to rename or reorganize further
3. **Follow the README.md workflow** - It's designed for new users
4. **Respect the archive system** - Don't move files back without good reason
5. **Maintain APA formatting** - Always use `--to apaquarto-docx`

### **Colleague Support Focus:**
1. **Help with Quarto/R setup** - Package installation, environment setup
2. **Manuscript editing assistance** - Citations, formatting, content updates
3. **Analysis reproduction** - Running scripts, troubleshooting data issues
4. **Visualization enhancements** - Modifying figures for publication
5. **Documentation updates** - Keeping README and methods current

### **DON'T Recreate or Reorganize:**
- ❌ Don't create new preprocessing scripts
- ❌ Don't reorganize the folder structure
- ❌ Don't move files from archive back to main
- ❌ Don't rename the clean scripts
- ❌ Don't duplicate existing functionality

### **DO Support and Enhance:**
- ✅ Help troubleshoot script errors
- ✅ Assist with Quarto compilation issues
- ✅ Help add new references to references.bib
- ✅ Support manuscript content updates
- ✅ Help interpret analysis results

## 🔧 **Technical Specifications**

### **R Package Dependencies:**
All required packages listed in `scripts/r/r_package_requirements.R`:
- **tidymodels ecosystem** - Modern ML framework
- **tidytext** - Text analysis with tidy principles
- **ggplot2** - Publication-quality visualizations
- **here** - Reproducible file paths
- **Various specialty packages** - For specific analysis needs

### **Data Structure:**
- **Study 1:** Survey data (N=391) with 67 initial variables → 10 final predictors
- **Study 2:** Focus group transcripts → 61 SUD-specific utterances (19.7% detection)
- **Conservative approach:** Substance-specific terminology required for SUD classification

### **Output Standards:**
- **Figures:** 300 DPI PNG + vector PDF for publication
- **Tables:** CSV format with proper statistical reporting
- **Interactive:** HTML methodology visualization for transparency
- **Manuscript:** APA-compliant Word document ready for submission

## 🚀 **Success Metrics Achieved**

✅ **Study 1:** ROC AUC 0.787 [0.766, 0.809] with robust cross-validation  
✅ **Study 2:** 4 validated themes with conservative 19.7% detection approach  
✅ **Mixed-Methods:** Qualitative validation of quantitative predictors  
✅ **Manuscript:** Complete APA paper with 4,000+ word Introduction, comprehensive Methods/Results/Discussion  
✅ **Repository:** Clean, colleague-ready structure with clear documentation  
✅ **Reproducibility:** All analyses scripted and documented for replication  

## 📚 **Key Reference Materials**

### **Methodological Documentation:**
- `results/study2_interactive_methodology.html` - Interactive pipeline visualization
- `results/study2_advanced_methodology.txt` - Complete Study 2 documentation
- `archive/ARCHIVE_README.md` - Documentation of archived files

### **Analysis Outputs:**
- `results/r/study1_logistic_fs_modern/` - Complete Study 1 model outputs
- `results/study2_*.csv` - Study 2 analysis tables and summaries
- `results/study2_*.png` - Publication-ready figures and visualizations

---

**STATUS: CLEAN REPOSITORY READY FOR COLLEAGUE COLLABORATION** 🤝✅  
**Last Updated:** June 2024 - Repository cleaned and organized for Erica's collaboration  
**Current Priority:** Support colleague onboarding, manuscript editing, and analysis reproduction  

## 🎯 **FOR AI AGENTS: COLLEAGUE SUPPORT FOCUS**

### **Primary Support Areas:**
1. **Quarto/R Environment Setup** - Help install dependencies and configure tools
2. **Manuscript Compilation** - Troubleshoot APA formatting and rendering issues  
3. **Analysis Reproduction** - Help run scripts and interpret outputs
4. **Citation Management** - Assist with references.bib updates and formatting
5. **Figure/Table Updates** - Modify visualizations and statistical summaries

### **Handoff Success Criteria:**
- ✅ Colleague can compile manuscript using provided commands
- ✅ All scripts run successfully with clear error messages if issues arise
- ✅ Analysis outputs are interpretable and well-documented
- ✅ Repository structure is intuitive for academic collaboration
- ✅ Future updates and modifications are straightforward

**This repository is now optimized for collaborative academic research with clear workflows and comprehensive documentation.**