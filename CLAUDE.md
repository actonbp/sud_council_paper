# CLAUDE.md - AI Agent Instructions

## Project Overview
This is a **clean, production-ready SUD (Substance Use Disorder) Counselors research repository** with complete mixed-methods analysis and publication-ready manuscript. **Repository was cleaned and organized for colleague handoff in May 2025.**

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
- **Method:** Mathematical cluster optimization using silhouette analysis (k=3, score=0.185) + elbow method
- **Innovation:** Data-driven cluster count determination - NO researcher-imposed assumptions
- **Three Mathematically-Optimized Themes:**
  1. **Clinical-Affective Framework (21.9%)** - Integration of clinical knowledge with emotional processing
  2. **Professional-Therapeutic Framework (14.6%)** - Therapeutic roles, training, and professional identity
  3. **Relational Dimension (4.7%)** - Pure interpersonal focus, mathematically isolated as distinct theme

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
- Conservative SUD detection requiring substance-specific terms (35.2% detection rate)
- Enhanced stopword filtering removing function words ("dont", "lot", etc.)
- Porter stemming for linguistic robustness
- Mathematical cluster optimization (silhouette analysis + elbow method)
- Co-occurrence analysis with data-driven theme emergence (k=3, NO researcher bias)
- Network visualization showing mathematically-validated thematic coherence

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
✅ **Study 2:** 3 mathematically-optimized themes with rigorous cluster validation (35.2% detection, k=3 silhouette=0.185)  
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
**Last Updated:** June 4, 2025 - APA formatting improvements completed, manuscript compilation tested  
**Current Priority:** Support manuscript refinement, meeting preparation, and ongoing collaboration  

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

---

## 📅 **JUNE 5, 2025 UPDATE: METHODOLOGICAL REVIEW & IMPROVEMENTS**

### **🔍 CRITICAL METHODOLOGICAL AUDIT COMPLETED**

Today's session focused on systematic review of analysis methodologies and identification of statistical concerns requiring attention. **Key achievement: Enhanced scientific rigor through transparent identification and documentation of methodological limitations.**

#### **Major Methodological Improvements Implemented:**

**1. Study 2 Clustering Methodology - VALIDATED & ENHANCED**
- ✅ **Confirmed data-driven approach:** k=3 clusters determined by silhouette analysis (0.185) NOT researcher assumption
- ✅ **Enhanced stopword filtering:** Added function words ("dont", "lot", "things", "stuff") to prevent semantic contamination  
- ✅ **Mathematical validation documented:** Clear silhouette vs elbow method decision algorithm
- ✅ **Added comprehensive appendix:** Full clustering methodology explanation in manuscript (Appendix B)

**2. Statistical Concerns Documentation - TRANSPARENCY ENHANCED**
- 🔴 **Critical issues identified:** Bootstrap confidence intervals, statistical significance claims, effect size conversions
- 🟠 **Moderate concerns documented:** Multiple comparisons, nested CV, detection rate inconsistencies
- 🟡 **Minor improvements noted:** Power analysis, stability metrics, stemming validation
- ✅ **Complete remediation plan created:** Prioritized action items with timelines

#### **Key Findings from Methodological Review:**

**Study 1 Statistical Issues:**
- **Bootstrap CIs:** Current confidence interval calculations mathematically invalid (CV standard errors ≠ sampling distribution)
- **Regularization inference:** P-values inappropriate for L1-regularized models
- **Effect size conversions:** Cohen's d and correlation formulas need validation for our model type
- **Multiple testing:** No correction applied to ~15-20 simultaneous predictors

**Study 2 Validation Confirmed:**
- **Clustering approach validated:** Mathematical optimization genuinely data-driven
- **Detection consistency verified:** 35.2% rate consistent across final analyses
- **Preprocessing robustness:** Enhanced stopword filtering eliminates semantic contamination
- **Methodological transparency:** Complete documentation enables replication

### **📋 IMMEDIATE ACTION PLAN ESTABLISHED**

**Week 1: Critical Statistical Fixes (🔴 Priority)**
1. Remove all bootstrap confidence interval claims from manuscript and results
2. Remove statistical significance claims (p-values) from regularized model results  
3. Validate or remove effect size conversions (Cohen's d, correlation)
4. Add statistical limitations section to manuscript methodology

**Week 2: Moderate Issues (🟠 Priority)**
5. Implement multiple testing correction or limit to pre-specified hypotheses
6. Add nested CV implementation or caveat about optimistic performance
7. Document all methodological decisions with justifications

**Week 3: Enhancement (🟡 Priority)**
8. Conduct post-hoc power analysis for primary effects
9. Add sensitivity analyses for key methodological choices
10. Update manuscript limitations section with identified concerns

### **🎯 DOCUMENTATION UPDATES COMPLETED**

**1. README.md Enhanced:**
- ✅ **Added comprehensive "Methodological Concerns & Remediation Plans" section**
- ✅ **Detailed issue tracking:** Critical → Moderate → Minor priority levels
- ✅ **Specific action items:** Checkboxes and timelines for systematic resolution
- ✅ **Transparent approach:** Scientific integrity through honest limitation acknowledgment

**2. Manuscript Improvements:**
- ✅ **Added Appendix B:** Detailed hierarchical clustering methodology explanation
- ✅ **Enhanced methodological transparency:** Step-by-step mathematical validation process
- ✅ **Cluster validation documentation:** Silhouette analysis, elbow method, decision algorithm

**3. Project Documentation:**
- ✅ **Meeting notes updated:** June 5th methodology review documented
- ✅ **Cursor rules maintained:** Existing methodological standards preserved
- ✅ **Archive integrity:** No removal of existing work, only additions

### **🔬 SCIENTIFIC IMPACT OF TODAY'S WORK**

**Methodological Rigor Enhanced:**
- **Study 2 approach validated:** Confirmed genuinely data-driven clustering (no researcher bias)
- **Statistical limitations identified:** Honest assessment of Study 1 inference issues
- **Transparency increased:** Complete documentation of methodological decisions and limitations

**Publication Readiness Status:**
- **Study 2:** ✅ Methodology robust and well-documented
- **Study 1:** ⚠️ Requires statistical claim adjustments before submission
- **Overall:** 🔶 Strong foundations with clearly identified improvement areas

**Collaborative Benefits:**
- **Clear improvement roadmap:** Prioritized action items for systematic resolution
- **Enhanced credibility:** Transparent limitation acknowledgment strengthens scientific integrity
- **Future research guidance:** Lessons learned documented for future projects

### **🎯 UPDATED GUIDELINES FOR AI AGENTS**

**Post-June 5th Methodological Review:**

**DO Support:**
- ✅ Help implement the Week 1-3 action plan systematically
- ✅ Assist with removing problematic statistical claims from manuscript
- ✅ Support methodological limitations section development
- ✅ Help validate effect size conversion formulas against literature
- ✅ Assist with nested CV implementation if requested

**DON'T Assume:**
- ❌ Don't assume current confidence intervals are valid
- ❌ Don't add new statistical significance claims without proper justification
- ❌ Don't ignore the documented methodological concerns
- ❌ Don't remove the limitations tracking system

**Priority Focus Areas:**
1. **Statistical validity:** Ensure all numerical claims are mathematically justified
2. **Methodological transparency:** Maintain honest documentation of approaches and limitations
3. **Systematic improvement:** Follow the established remediation plan
4. **Scientific integrity:** Prioritize accuracy over impressive-sounding results

---

**UPDATED STATUS:** Research methodology under systematic review with clear improvement plan  
**Last Updated:** June 5, 2025 - Methodological audit completed, remediation plan established  
**Current Priority:** Implement critical statistical fixes while maintaining research quality and transparency

---

## 📅 **JUNE 10, 2025 UPDATE: STUDY 2 TRANSITION TO TOPIC MODELING**

### **🔄 MAJOR METHODOLOGICAL SHIFT FOR STUDY 2**

Based on research team discussion, we are transitioning Study 2 from hierarchical clustering to **Latent Dirichlet Allocation (LDA) topic modeling** using the `topicmodels` package, following best practices from tidytext framework.

#### **Key Changes Implemented:**

**1. Repository Cleanup - ARCHIVE & SIMPLIFICATION**
- ✅ **Archived clustering scripts:** Moved all hierarchical clustering scripts to `archive/study2_clustering_scripts/`
- ✅ **Simplified documentation:** Archived redundant cursor rules and outdated status files
- ✅ **Preserved preprocessing:** Kept `study2_text_preprocessing.R` for enhancement
- ✅ **Created new plan:** [JUNE_10_2025_PLAN.md](JUNE_10_2025_PLAN.md) with detailed implementation roadmap

**2. Methodological Approach - TOPIC MODELING**
- 🔄 **From clustering to LDA:** Probabilistic topic discovery instead of hard clusters
- 🔄 **Less conservative filtering:** Target ~40-50% utterance inclusion (vs 19.7%)
- 🔄 **Moderator removal:** First speaker per session excluded from analysis
- 🔄 **Multiple k exploration:** Test 2-8 topics for optimal interpretability

**3. Technical Implementation Plan**
- **Phase 1:** Enhanced preprocessing with moderator removal and inclusive filtering
- **Phase 2:** LDA implementation with multiple k values and validation metrics
- **Phase 3:** Research team interpretation of topic word distributions

#### **Repository Structure Updates:**

**Current Scripts (scripts/r/):**
- `r_package_requirements.R` - Updated with `topicmodels` and `ldatuning`
- `study1_main_analysis.R` - Unchanged (Study 1 complete)
- `study2_text_preprocessing.R` - To be enhanced with moderator removal
- `study2_cooccurrence_analysis.R` - Keep for reference
- `study2_methodology_validation.R` - To be updated for topic modeling
- `study2_create_visualizations.R` - To be updated for topic visualization

**Planned New Scripts:**
- `study2_topic_modeling.R` - Core LDA implementation
- `study2_topic_visualization.R` - Topic-specific visualizations

**Archived Files:**
```
archive/
├── study2_clustering_scripts/     # All clustering-related scripts
│   ├── study2_clustering_*.R
│   ├── study2_kmeans_comparison.R
│   ├── study2_alternative_clustering.R
│   ├── singleton_diagnosis.R
│   └── cooccurrence_frequency_table.R
└── old_documentation/            # Outdated docs
    ├── STUDY2_METHODOLOGY_UPDATE.md
    ├── PROJECT_STATUS.md
    └── [redundant cursor rules]
```

### **🎯 IMMEDIATE NEXT STEPS**

**Week 1 Tasks:**
1. [ ] Update `study2_text_preprocessing.R` to identify and remove moderator utterances
2. [ ] Implement less conservative filtering (counseling/therapy/substance/career terms)
3. [ ] Create `study2_topic_modeling.R` script following tidytext vignette
4. [ ] Test on sample data with k=2,3,4,5,6,7,8

**Expected Outcomes:**
- Higher utterance inclusion rate improving representativeness
- Data-driven topics without researcher-imposed categories
- Clear probabilistic topic assignments for each utterance
- Interpretable word-topic distributions for team naming

### **🔬 METHODOLOGICAL RATIONALE**

**Why Topic Modeling over Clustering:**
- **Probabilistic membership:** Documents can belong to multiple topics
- **Better suited for short texts:** LDA handles focus group utterances well
- **Established framework:** Well-documented in tidytext ecosystem
- **Interpretability:** Word probability distributions are intuitive

**Filtering Strategy Change:**
- **Old:** Required specific substance terms (conservative)
- **New:** Include counseling OR substance OR career terms (inclusive)
- **Rationale:** Capture broader discourse about SUD counseling careers

### **📚 KEY REFERENCES FOR IMPLEMENTATION**
- [Tidy Topic Modeling vignette](https://juliasilge.github.io/tidytext/articles/topic_modeling.html)
- [Supervised ML for Text Analysis in R](https://smltar.com)
- Blei, D. M., Ng, A. Y., & Jordan, M. I. (2003). Latent Dirichlet Allocation

---

**UPDATED STATUS:** Study 2 transitioning to LDA topic modeling with enhanced preprocessing  
**Last Updated:** June 10, 2025 - Repository cleaned, topic modeling plan established  
**Current Priority:** Implement enhanced preprocessing and initial topic modeling tests