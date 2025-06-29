# CLAUDE.md - AI Agent Instructions

## Project Overview
This is a **SUD (Substance Use Disorder) Counselors research repository** with mixed-methods analysis and publication-ready manuscript. **Repository underwent major cleanup on June 29, 2025 for maximum parsimony.**

## Current Status: STUDY 2 RESET - READY FOR FRESH APPROACH ✅🔄

### 🧹 **MAJOR CLEANUP JUNE 29, 2025**
1. ✅ **STUDY 2 RESET:** All 26 experimental scripts archived - clean slate for new approach
2. ✅ **MAXIMUM PARSIMONY:** Minimal documentation, simple folder structure (study1, study2)
3. ✅ **STUDY 1 COMPLETE:** Tidymodels analysis finished, needs only statistical claim fixes
4. ✅ **MANUSCRIPT READY:** APA paper exists, needs Study 2 section when analysis complete
5. ✅ **AGGRESSIVE ARCHIVING:** Removed clutter while preserving all work
6. ✅ **SIMPLE STRUCTURE:** Essential files only in root, everything else archived

## 🎯 **Repository Structure (Post-Cleanup)**

### **Current Clean Structure:**
```
sud_council_paper/
├── 📄 sud_council_paper.qmd          # MAIN MANUSCRIPT
├── 📄 sud_council_paper.docx          # Compiled APA Word document  
├── 📄 README.md                      # Main documentation
├── 📄 CLAUDE.md                      # This file (AI agent instructions)
├── 📄 references.bib                 # APA bibliography
├── 
├── 📁 scripts/r/                     # SIMPLE STRUCTURE
│   ├── r_package_requirements.R           # Package installation
│   ├── study1/                           # Study 1 complete 
│   │   └── study1_main_analysis.R             # Tidymodels analysis (DONE)
│   └── study2/                           # Study 2 empty - ready for fresh start
│
├── 📁 results/                       # Analysis outputs
│   ├── figures/                           # Empty - Study 2 figures archived
│   └── r/
│       ├── study1_logistic_fs_modern/     # Study 1 results (KEEP)
│       └── study2/                        # Empty - ready for new results
│
├── 📁 data/                          # Only essential data files remain
├── 📁 _extensions/wjschne/apaquarto/  # APA formatting system
└── 📁 archive/                       # ALL archived work (nothing lost)
    ├── study2_all_approaches/             # 26 Study 2 scripts archived
    ├── study2_results/                    # 12 result folders archived  
    ├── study2_outputs/                    # 21 output files archived
    └── old_documentation/                 # Extra markdown files archived
```

### **Major Archive Additions (June 29, 2025):**
- **study2_all_approaches/**: All 26 experimental Study 2 scripts (LDA, topic modeling, embeddings, etc.)
- **study2_results/**: 12 result folders from different analysis attempts
- **study2_outputs/**: 21 loose files (figures, RDS files, methodology docs)
- **old_documentation/**: Archived extra markdown files for parsimony

## 🔬 **Complete Research Summary**

### **Study 1: Quantitative Analysis (N=391)**
- **Method:** L1-regularized logistic regression using tidymodels framework
- **Key Finding:** Students uncertain about mental health careers show 74% higher odds of SUD counseling interest
- **Performance:** Cross-validation ROC AUC = 0.787 [95% CI: 0.766, 0.809]
- **Effect Sizes:** Cohen's d = 0.764, correlation r = 0.411 (strong for behavioral research)
- **Statistical Validation:** χ² = 92.59, p < .001 for career uncertainty effect

### **Study 2: Qualitative Analysis (N=19, 7 focus groups) - RESET FOR FRESH APPROACH**
- **Status:** All previous experimental approaches archived (26 scripts)
- **Data:** Focus group transcripts ready for analysis (7 CSV files in data/focus_group/)
- **Goal:** Simple, effective tidytext analysis using R
- **Previous Attempts:** LDA, BTM, embeddings, clustering - all archived in archive/study2_all_approaches/
- **Lesson Learned:** Simple frequency analysis worked best (archived in study2_simple_frequency/)
- **Next Step:** Start fresh with clean, simple tidytext approach

### **Mixed-Methods Integration - TO BE UPDATED**
- Study 1 complete with strong quantitative findings
- Study 2 ready for new qualitative analysis to complement Study 1
- Manuscript needs Study 2 section once new analysis complete
- Goal: Align qualitative themes with quantitative career uncertainty findings

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

### **Study 2 Workflow - TO BE CREATED:**
```r
# Study 2 folder is empty - ready for fresh tidytext approach
# scripts/r/study2/ [EMPTY]
# results/r/study2/ [EMPTY]

# Next steps:
# 1. Create simple tidytext preprocessing script
# 2. Create basic frequency/theme analysis 
# 3. Focus on parsimony - simple, interpretable approach
# 4. Use tidytext principles throughout
```
- **26 previous scripts archived** in archive/study2_all_approaches/
- **All results archived** in archive/study2_results/ and archive/study2_outputs/
- **Clean slate approach** - start simple with tidytext
- **Lesson from archives:** Simple frequency analysis worked best
- **Goal:** Complement Study 1 findings with qualitative insights

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

## 🔥 **CURRENT PRIORITIES (June 29, 2025)**

### **Immediate Next Steps:**
1. **Study 2 Fresh Start** - Create simple tidytext analysis (scripts/r/study2/ is empty and ready)
2. **Data Focus** - Essential Study 1 data: `data/survey/ml_ready_survey_data.csv` 
3. **Manuscript Completion** - Study 2 section needs updating once new analysis complete
4. **Statistical Fixes** - Study 1 needs confidence interval and p-value claims removed (documented in README)

### **What's Complete:**
- ✅ Study 1 analysis (just needs statistical claim fixes)
- ✅ Manuscript structure and APA formatting
- ✅ Repository cleanup and parsimony achieved
- ✅ All previous work safely archived

### **Repository Philosophy - MAXIMUM PARSIMONY:**
- **Simple names:** study1, study2 (not "next-gen" or complex names)
- **Minimal docs:** Only CLAUDE.md, README.md, .cursor files
- **R + tidytext:** All analysis in R using tidytext principles
- **Archive aggressively:** Remove clutter but preserve all work

## 🎯 **Guidelines for Future AI Agents**

### **PARSIMONY PRINCIPLES (Updated June 29, 2025):**
1. **Keep it simple** - No complex folder names, no extra documentation files
2. **Focus on essentials** - Only CLAUDE.md, README.md, .cursor rules matter for docs  
3. **R + tidytext only** - All Study 2 work should use R with tidytext
4. **Archive don't delete** - Move clutter to archive/ but preserve all work
5. **Study 2 restart** - Don't use archived approaches, start fresh and simple

### **Current Support Focus:**
1. **Study 2 fresh approach** - Help create simple tidytext analysis from scratch
2. **Data management** - Focus on essential files only (survey data for Study 1)  
3. **Manuscript integration** - Update Study 2 section when analysis complete
4. **Statistical fixes** - Help remove invalid claims from Study 1 results
5. **APA compilation** - Always use `--to apaquarto-docx`

### **CRITICAL DON'Ts:**
- ❌ Don't use archived Study 2 approaches - start fresh
- ❌ Don't create complex documentation - stick to CLAUDE.md/README.md
- ❌ Don't overcomplicate Study 2 - simple tidytext frequency analysis worked best
- ❌ Don't reorganize structure - current simple setup is intentional
- ❌ Don't move things back from archive without explicit request

## 🔧 **Technical Specifications**

### **R Package Dependencies:**
All required packages listed in `scripts/r/r_package_requirements.R`:
- **tidymodels ecosystem** - Modern ML framework
- **tidytext** - Text analysis with tidy principles
- **ggplot2** - Publication-quality visualizations
- **here** - Reproducible file paths
- **Various specialty packages** - For specific analysis needs

### **Current Data Structure:**
- **Study 1:** Survey data ready - `data/survey/ml_ready_survey_data.csv` (N=391)
- **Study 2:** Focus group transcripts ready - `data/focus_group/*.csv` (7 files, N=19)
- **Study 1 Results:** Complete in `results/r/study1_logistic_fs_modern/`
- **Study 2 Results:** Empty `results/r/study2/` - ready for new analysis

### **Output Standards:**
- **Figures:** 300 DPI PNG + vector PDF for publication
- **Tables:** CSV format with proper statistical reporting  
- **Manuscript:** APA-compliant Word document using `--to apaquarto-docx`
- **Keep it simple:** Focus on essential outputs only

## 🚀 **Current Status Summary (June 29, 2025)**

✅ **Study 1:** ROC AUC 0.787 [0.766, 0.809] with robust cross-validation - COMPLETE
⏸️ **Study 2:** RESET - Previous approaches archived, ready for fresh tidytext approach  
⏸️ **Mixed-Methods:** Awaiting Study 2 completion for integration
✅ **Manuscript:** APA paper structure complete, needs Study 2 section update
✅ **Repository:** Maximum parsimony achieved - clean, simple structure  
✅ **Reproducibility:** All work preserved in archive, essential files accessible  

## 📚 **Key Reference Materials**

### **Current Reference Materials:**
- `archive/ARCHIVE_README.md` - Complete documentation of June 29 cleanup
- `results/r/study1_logistic_fs_modern/` - Study 1 results (KEEP - analysis complete)
- `archive/study2_all_approaches/` - All 26 archived Study 2 scripts for reference
- `archive/study2_simple_frequency/` - Final working approach before reset

### **Essential Files Only:**
- Study 1 complete - results in `results/r/study1_logistic_fs_modern/`
- Study 2 reset - empty folders ready for fresh tidytext approach
- Manuscript ready for Study 2 section update once analysis complete

---

## 📅 **FINAL STATUS: JUNE 29, 2025 - STUDY 2 RESET & REPOSITORY CLEANUP**

### **🎯 WHERE WE ARE NOW**

**Repository Philosophy:** Maximum parsimony achieved - simple structure, minimal documentation, essential files only.

**Study 1:** ✅ **COMPLETE** - Tidymodels analysis finished (ROC AUC 0.787), just needs statistical claim fixes documented in README.md

**Study 2:** 🔄 **RESET FOR FRESH START** - All 26 experimental scripts archived, empty folders ready for simple tidytext approach

**Manuscript:** ✅ **READY** - APA structure complete, needs Study 2 section update once new analysis finished

### **🗂️ What Was Archived Today**
- **26 Study 2 scripts** → `archive/study2_all_approaches/`
- **12 result folders** → `archive/study2_results/`  
- **21 output files** → `archive/study2_outputs/`
- **4 extra markdown files** → `archive/old_documentation/`

### **📋 Next Steps for AI Agents**
1. **Start fresh** with Study 2 using simple tidytext approach
2. **Keep it simple** - parsimony over complexity
3. **Use archived simple frequency analysis** as reference (worked best)
4. **Focus on manuscript integration** once Study 2 analysis complete

---

**CURRENT STATUS:** Study 2 reset complete, ready for fresh tidytext approach  
**Last Updated:** June 29, 2025 - Major cleanup and reset completed  
**Current Priority:** Create simple, effective Study 2 analysis to complete manuscript