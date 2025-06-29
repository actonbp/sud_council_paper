# Required Data Files for Analysis Reproduction

**These files are NOT in the GitHub repository for privacy protection but are REQUIRED to run the analyses.**

## 📁 **Exact Data Files Needed**

### Study 1 (Quantitative Analysis)
```
data/survey/
├── survey_raw.csv                    # Original SONA survey export
├── ml_ready_survey_data.csv         # Preprocessed for analysis  
└── ai_generated_dictionary_detailed.csv # Variable definitions
```

### Study 2 (Text Analysis)  
```
data/focus_group/
├── 11_4_2024_11am_processed.csv     # Session 1 transcripts
├── 11_6_2024_130pm_processed.csv    # Session 2 transcripts  
├── 11_6_2024_5_pm_processed.csv     # Session 3 transcripts
├── 11_8_2024_12pm_processed.csv     # Session 4 transcripts
├── 11_11_2024_4pm_processed.csv     # Session 5 transcripts
├── 11_12_2024_11am_processed.csv    # Session 6 transcripts
└── 11_14_2024_4pm_processed.csv     # Session 7 transcripts
```

### Additional Required Files
```
data/
├── focus_group_comprehensive_preprocessed.csv  # Combined sessions
├── focus_group_substantive.csv                # Substantive responses only
└── processed/
    ├── survey_processed.csv                    # Study 1 analysis-ready data
    ├── X_train.csv                            # Training features
    ├── X_test.csv                             # Test features  
    ├── y_train.csv                            # Training outcomes
    └── y_test.csv                             # Test outcomes
```

## 🔍 **Data File Verification**

### Before Running Analyses, Check:
```bash
# Verify Study 1 data exists
ls data/survey/survey_raw.csv
ls data/survey/ml_ready_survey_data.csv

# Verify Study 2 data exists  
ls data/focus_group/*.csv | wc -l    # Should show 7 files

# Verify processed data exists
ls data/processed/*.csv | wc -l      # Should show 5 files
```

## 📊 **Expected Data Structure**

### Survey Data Columns (Study 1)
- `sud_counseling_interest` - Primary outcome (Yes/No/Maybe)
- `mental_health_career_interest` - Key predictor (Yes/No/Unsure)
- `professional_familiarity_*` - SUD counselor familiarity scales
- Demographics: `year_in_school`, `race_ethnicity`, etc.
- **N ≈ 391 participants**

### Focus Group Data Columns (Study 2)  
- `session_id` - Focus group session identifier
- `response_id` - Unique utterance ID
- `Speaker` - Anonymized participant ID
- `cleaned_text` - Preprocessed transcript text
- `word_count` - Utterance length
- **N = 310 substantive utterances across 7 sessions**

## 🚨 **What to Do If Data is Missing**

### Contact Research Team:
1. **Request access** to IRB-approved data files
2. **Verify file formats** match expected structure above
3. **Confirm preprocessing** has been completed
4. **Test small sample** before full analysis

### Data Location Options:
- Secure institutional file sharing
- Encrypted external drives  
- IRB-approved cloud storage
- Direct researcher-to-researcher transfer

## ⚙️ **Quick Setup Test**

```r
# Test data availability before analysis
if (!file.exists("data/survey/survey_raw.csv")) {
  stop("Missing Study 1 data files - see DATA_REQUIREMENTS.md")
}

if (length(list.files("data/focus_group/", pattern = "*.csv")) != 7) {
  stop("Missing Study 2 data files - see DATA_REQUIREMENTS.md") 
}

cat("✅ All required data files present - ready for analysis!")
```

---

**Note**: These data files contain participant information and must be handled according to IRB protocols. Never commit these files to version control.