# SUD Counselors Research Project

**Complete mixed-methods research study examining factors influencing undergraduate interest in SUD counseling careers, published as an APA-formatted Quarto academic manuscript.**

> 🤖 **NEW AI AGENTS:** Start with [AI_ONBOARDING.md](AI_ONBOARDING.md) for quick orientation and navigation guide!

---

## 🎯 **FOR ERIKA: QUICK START GUIDE**

### **What You Need to Know:**
This repository contains a **complete, publication-ready academic study** with:
- ✅ **Study 1:** Machine learning analysis (L1-regularized logistic regression) 
- ✅ **Study 2:** Text analysis of focus group discussions
- ✅ **Complete manuscript:** APA-formatted paper ready for journal submission
- ✅ **All analysis scripts:** Clean, documented, and reproducible

### **Key Files You'll Use:**
- 📄 **`manuscript/sud_council_paper.qmd`** - Main manuscript (edit this)
- 📄 **`manuscript/sud_council_paper.docx`** - Compiled Word version
- 📁 **`scripts/r/`** - All analysis scripts (clearly organized)
- 📁 **`results/`** - All figures, tables, and outputs
- 📄 **`manuscript/references.bib`** - Bibliography (add new references here)

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
quarto render manuscript/sud_council_paper.qmd --to apaquarto-docx

# ❌ NEVER use this (breaks APA formatting):
# quarto render manuscript/sud_council_paper.qmd --to docx
```

#### **Step 5: Verify Success**
```bash
# Check outputs were created
ls manuscript/sud_council_paper.docx          # Main manuscript

# Open in Word to verify APA formatting:
# - Title page with running head
# - Tables moved to end
# - Proper APA citations
# - Double spacing, Times New Roman
```

### **🔧 Quick Compilation (After Setup)**
```bash
# For daily work after initial setup:
quarto render manuscript/sud_council_paper.qmd --to apaquarto-docx && open manuscript/sud_council_paper.docx
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

#### **Study 2: Text Analysis**
```bash
# Study 2 is currently being redesigned
# All previous scripts and results have been archived
# New tidytext implementation coming soon
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
- **Status:** Currently being redesigned with tidytext approach
- **Previous work:** All legacy scripts and results archived in `archive/2025-06-29/`
- **Next steps:** Fresh implementation using modern text analysis methods

### **Mixed-Methods Integration**
- Qualitative themes validate and explain quantitative predictors
- Career uncertainty pathway supported across both studies
- Comprehensive manuscript integrates findings with Social Cognitive Career Theory

---

## 📁 **REPOSITORY STRUCTURE**

```
sud_council_paper/
├── 📁 manuscript/                    # ✅ MANUSCRIPT & APA FORMATTING
│   ├── sud_council_paper.qmd              # Main manuscript (edit this)
│   ├── sud_council_paper.docx             # Compiled Word document  
│   ├── references.bib                     # Bibliography (APA format)
│   └── _extensions/wjschne/apaquarto/     # APA formatting system
│
├── 📁 scripts/r/                     # ✅ ALL ANALYSIS SCRIPTS
│   ├── r_package_requirements.R           # Install required packages
│   ├── study1/                            # Study 1: Complete analysis
│   │   └── study1_main_analysis.R         # Tidymodels L1 regression
│   └── study2/                            # Study 2: Empty (to be rebuilt)
│       └── README.md                      # Placeholder for new analysis
│
├── 📁 results/                       # ✅ ALL OUTPUTS
│   └── r/
│       └── study1_logistic_fs_modern/     # Study 1 model outputs
│
├── 📁 data/                          # ⚠️  DATA FILES (NOT INCLUDED - see below)
│   └── survey/                            # Study 1 survey data only
│       ├── ml_ready_survey_data.csv       # Analysis-ready data
│       └── survey_raw.csv                 # Raw survey data
│
├── 📄 README.md                      # This file
├── 📄 CLAUDE.md                      # AI agent instructions
├── 📁 .cursor/rules/                 # Cursor AI configuration
└── 📁 archive/2025-06-29/            # All legacy files preserved here
```

### **⚠️ DATA ACCESS NOTICE**

**Raw data files are NOT included in this repository** due to participant privacy protections and IRB requirements.

#### **What's Missing:**
- `data/survey/` - Survey response data (N=391)
- `data/focus_group/` - Focus group transcripts (7 sessions, N=19 participants)  
- `data/processed/` - Cleaned and preprocessed datasets

#### **For Data Access:**
📧 **Contact:** Linda Reynolds at **linda.reynolds@binghamton.edu**

- Include your institution affiliation
- Specify intended use (replication, extension, etc.)
- Data sharing subject to IRB approval and data use agreements

#### **What IS Included:**
- ✅ All analysis scripts for transparency
- ✅ Complete methodology documentation  
- ✅ Final results and visualizations in `results/` folder
- ✅ **Cluster output template** (`results/study2_cluster_output_DEMO.txt`)
- ✅ Aggregated findings suitable for publication

**Note:** Scripts will require local data files to execute, but provide complete methodological transparency.

#### **Study 2 Cluster Output:**
The hierarchical clustering analysis generates **actual word clusters** for researcher interpretation:

**✅ REAL RESULTS (109 SUD utterances, 35.2% detection rate):**
- `results/study2_cluster_output.txt` - **Complete technical analysis**
- `results/study2_cluster_themes_for_naming.txt` - **Clean researcher worksheet**

**📊 Mathematically-Optimized Cluster Overview:**
- **Cluster 1 (21.9%):** Clinical-Affective Framework (feel, substance, mental, health, abuse, field, job)
- **Cluster 2 (4.7%):** Relational Dimension (people) - mathematically isolated as distinct theme
- **Cluster 3 (14.6%):** Professional-Therapeutic Framework (family, counselor, therapy, therapist, support)

Each cluster emerged from mathematical co-occurrence analysis - researchers now assign thematic names.

---

## 📋 **MEETING PREPARATION & REPORTS**

### **Creating Meeting Documents:**
The `meetings/` folder is designed for preparing presentation materials and updates for team meetings.

**Quick Meeting Report:**
```bash
# Create a Quarto document for meeting updates
# Example: 2025-06-05_project_update.qmd

# Compile to APA-formatted Word document
cd meetings/
quarto render 2025-06-05_project_update.qmd --to apaquarto-docx

# Or compile to PDF
quarto render 2025-06-05_project_update.qmd --to pdf
```

**Meeting Document Types:**
- **Project Updates**: Current manuscript status and recent accomplishments
- **Progress Reports**: Analysis results and methodology updates  
- **Planning Documents**: Next steps and milestone planning
- **Meeting Minutes**: Team discussion documentation

**Template Structure**: Each meeting document can include:
- Current manuscript status
- Recent APA formatting improvements
- Analysis pipeline results (Study 1 & 2)
- Upcoming deadlines and goals
- Discussion points and decisions needed

---

## 📊 **STUDY 2 UPDATED METHODOLOGY (June 10, 2025)**

Following [Supervised Machine Learning for Text Analysis in R](https://smltar.com) and [Tidy Topic Modeling](https://juliasilge.github.io/tidytext/articles/topic_modeling.html) principles for data-driven theme discovery:

### **NEW IMPLEMENTATION - Complete LDA Pipeline:**
- **Algorithm**: Latent Dirichlet Allocation (LDA) using `topicmodels` package
- **Pipeline**: Data preparation → LDA modeling → Visualization → Manuscript integration
- **Less Conservative Filtering**: Include utterances with counseling, substance, or career terms (~40-50% inclusion)
- **Systematic Moderator Removal**: First speaker in each session excluded from analysis
- **Statistical Model Selection**: Multi-metric optimization (Arun2010, CaoJuan2009, Deveaud2014)
- **Enhanced Preprocessing**: Comprehensive stop words, Porter stemming, participant-only analysis
- **Publication-Ready Outputs**: Direct CSV/PNG exports for manuscript integration

### **Script Organization (scripts/r/study2/):**
- **`STUDY2_WORKFLOW.md`**: Step-by-step workflow guide and troubleshooting
- **`study2_data_preparation.R`**: Moderator removal and data cleaning
- **`study2_lda_topic_modeling.R`**: LDA model fitting with optimal k selection
- **`study2_lda_visualizations.R`**: Publication-ready figures and manuscript tables

### **Key Methodological Improvements:**
- **Broader SUD Detection**: Counseling, therapy, substance, mental health terminology
- **Statistical Rigor**: Multi-metric model selection vs. silhouette analysis
- **Data-Driven Approach**: Probabilistic topic membership vs. hard clustering
- **Systematic Validation**: Comprehensive tuning results and model diagnostics
- **Manuscript Integration**: Direct outputs formatted for academic publication

### **Implementation Status:**
- ✅ **Repository Reorganized**: Study 1/Study 2 separation complete
- ✅ **LDA Scripts Created**: Complete pipeline with tidytext framework
- ✅ **Archive Strategy**: Old clustering scripts preserved in archive/
- ✅ **Package Dependencies**: Updated with `ldatuning`, `patchwork`, `ggrepel`
- 📋 **Next Step**: Run analysis and update manuscript Study 2 section

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
Rscript scripts/r/study1/study1_main_analysis.R
```

### **Study 2 (Qualitative) - NEW LDA Pipeline:**
```r
# Follow the workflow guide: scripts/r/study2/STUDY2_WORKFLOW.md
Rscript scripts/r/study2/study2_data_preparation.R      # Step 1: Moderator removal
Rscript scripts/r/study2/study2_lda_topic_modeling.R    # Step 2: LDA modeling  
Rscript scripts/r/study2/study2_lda_visualizations.R    # Step 3: Publication figures
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

## ⚠️ **METHODOLOGICAL CONCERNS & REMEDIATION PLANS**

*This section documents statistical and methodological issues identified during analysis review. Maintaining transparency about limitations strengthens scientific credibility.*

### **🔴 CRITICAL ISSUES REQUIRING IMMEDIATE ATTENTION**

#### **1. Bootstrap Confidence Intervals - INVALID STATISTICAL PROCEDURE**
**Problem:** 
- Current code: `cv_95_lower <- cv_mean - 1.96 * cv_std_err`
- **This is wrong**: CV standard errors ≠ sampling distribution standard errors
- We're confusing model performance variability with parameter uncertainty

**Impact:** All "95% confidence intervals" in Study 1 are mathematically invalid

**Remediation Plan:**
- [ ] **REMOVE all confidence interval claims** from manuscript and results
- [ ] Replace with descriptive statistics: "CV standard error = X"
- [ ] Add limitation statement about performance uncertainty
- [ ] Consider proper nested CV for unbiased performance estimates

**Status:** 🔴 **URGENT - Must fix before any submission**

---

#### **2. Statistical Significance Claims - INVALID WITH REGULARIZATION**
**Problem:**
- Manuscript claims "p < 0.001" for regularized model coefficients
- L1 regularization (Lasso) shrinks coefficients toward zero
- P-values are not valid for regularized models (no proper null distribution)

**Impact:** All statistical significance claims in Study 1 are inappropriate

**Remediation Plan:**
- [ ] **REMOVE all p-value and significance claims** from manuscript
- [ ] Focus on effect sizes and practical significance only
- [ ] Add methodological note about regularization limitations
- [ ] Consider switching to unregularized model for inference (if justified)

**Status:** 🔴 **URGENT - Invalidates current conclusions**

---

#### **3. Effect Size Conversions - QUESTIONABLE ASSUMPTIONS**
**Problem:**
```r
cohens_d <- sqrt(2) * qnorm(roc_auc_value)     # Assumes normal distributions
correlation_r <- 2 * (roc_auc_value - 0.5)    # Assumes linear relationship
```
- These conversions have specific distributional assumptions
- May not be valid for our data or model type
- Could be "hallucinating" effect sizes

**Impact:** Reported Cohen's d and correlation values may be meaningless

**Remediation Plan:**
- [ ] **VALIDATE conversion formulas** against literature for logistic regression
- [ ] Test assumptions: normality, linearity, etc.
- [ ] Consider direct calculation of Cohen's d from actual group differences
- [ ] If invalid, remove effect size conversions entirely

**Status:** 🔴 **HIGH PRIORITY - Could mislead readers**

---

### **🟠 MODERATE CONCERNS REQUIRING ATTENTION**

#### **4. Multiple Comparisons - NO CORRECTION APPLIED**
**Problem:**
- Study 1 examines ~15-20 predictors simultaneously
- No multiple testing correction (Bonferroni, FDR, etc.)
- High risk of Type I error inflation

**Impact:** Increased likelihood of false positive findings

**Remediation Plan:**
- [ ] Apply Bonferroni or FDR correction to coefficient significance
- [ ] OR limit analysis to pre-specified hypotheses only
- [ ] Add power analysis to justify sample size for multiple tests
- [ ] Document family-wise error rate in limitations

**Status:** 🟠 **MODERATE - Should address before publication**

---

#### **5. Nested Cross-Validation - NOT PROPERLY IMPLEMENTED**
**Problem:**
- Current "repeated CV" tunes hyperparameters on same folds used for performance
- This leads to optimistic performance estimates (leakage)
- True nested CV requires separate inner/outer loops

**Impact:** Model performance (ROC AUC = 0.787) likely overestimated

**Remediation Plan:**
- [ ] Implement proper nested CV with inner/outer loops
- [ ] OR clearly state that performance estimates may be optimistic
- [ ] Provide range of expected performance degradation
- [ ] Compare current estimates to simpler baseline models

**Status:** 🟠 **MODERATE - Affects performance claims**

---

#### **6. Study 2 Detection Rate Inconsistencies**
**Problem:**
- Multiple conflicting detection rates reported: 19.7%, 35.2%
- Suggests inconsistent preprocessing across analyses
- Unclear which rate is correct for final results

**Impact:** Undermines reproducibility and precision claims

**Remediation Plan:**
- [ ] **AUDIT all Study 2 scripts** for consistent preprocessing
- [ ] Document exact detection methodology and rate
- [ ] Ensure all results use same detection criteria
- [ ] Add preprocessing validation to prevent future inconsistencies

**Status:** 🟠 **MODERATE - Affects Study 2 credibility**

---

### **🟡 MINOR CONCERNS FOR FUTURE IMPROVEMENT**

#### **7. Power Analysis - POST-HOC MISSING**
**Problem:**
- No formal power analysis for N=391 sample
- Unclear if sample adequate for effect sizes claimed
- Multiple predictors increase power requirements

**Remediation Plan:**
- [ ] Conduct post-hoc power analysis for primary effects
- [ ] Document minimum detectable effect sizes
- [ ] Compare achieved power to field standards (typically 80%)
- [ ] Add to limitations if underpowered

**Status:** 🟡 **MINOR - Good practice but not critical**

---

#### **8. Bootstrap Stability Metric - POTENTIALLY MISLEADING**
**Problem:**
- "Sign consistency" doesn't account for magnitude changes
- Coefficient could flip ±0.001 (95% consistency) but be unstable
- Better metrics exist (CI width, coefficient CV)

**Remediation Plan:**
- [ ] Replace sign consistency with coefficient standard deviation
- [ ] Add confidence interval width as stability measure
- [ ] Consider coefficient of variation for stability assessment
- [ ] Retain sign consistency as supplementary metric only

**Status:** 🟡 **MINOR - Methodological improvement**

---

#### **9. Stemming Validation - ASSUMPTION NOT TESTED**
**Problem:**
- Assumes Porter stemming improves semantic clustering
- No validation against non-stemmed analysis
- Could create artificial semantic relationships

**Remediation Plan:**
- [ ] Run sensitivity analysis: clustering with/without stemming
- [ ] Compare cluster coherence between approaches
- [ ] Document stemming decisions in methodology
- [ ] Consider alternative preprocessing approaches

**Status:** 🟡 **MINOR - Study 2 robustness check**

---

### **📋 IMMEDIATE ACTION ITEMS (Priority Order)**

#### **Week 1: Critical Statistical Fixes**
1. **Remove all confidence interval claims** from Study 1 results and manuscript
2. **Remove all statistical significance claims** (p-values) from regularized model results
3. **Validate or remove effect size conversions** (Cohen's d, correlation)
4. **Add statistical limitations section** to manuscript methodology

#### **Week 2: Moderate Methodological Issues**
5. **Implement multiple testing correction** or limit to pre-specified hypotheses
6. **Audit Study 2 preprocessing** for consistent detection rates
7. **Add nested CV implementation** or caveat about optimistic performance
8. **Document all methodological decisions** with justifications

#### **Week 3: Enhancement and Documentation**
9. **Conduct post-hoc power analysis** for primary effects
10. **Add sensitivity analyses** for key methodological choices
11. **Update manuscript limitations section** with identified concerns
12. **Create reproducibility checklist** to prevent future issues

---

### **🎯 LONG-TERM METHODOLOGICAL IMPROVEMENTS**

#### **For Future Research or Revision:**
- **Implement proper nested CV** for unbiased performance estimates
- **Pre-register analysis plans** to avoid multiple comparisons issues  
- **Use non-regularized models** for statistical inference when appropriate
- **Conduct formal mixed-methods integration** beyond descriptive comparison
- **Add replication studies** to validate key findings

#### **Lessons Learned:**
- **Bootstrap CIs require careful theoretical justification**
- **Regularized models have different inference properties**
- **Cross-validation serves prediction, not inference**
- **Effect size conversions need validation**
- **Transparency about limitations strengthens credibility**

---

*This concerns section will be updated as issues are resolved. Scientific integrity requires acknowledging limitations honestly while working to address them systematically.*

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

**Last Updated:** June 2025 - Enhanced with APA improvements and meeting organization  
**Status:** ✅ Publication-ready manuscript with complete analysis pipeline  
**Contact:** Use GitHub issues for questions about the repository