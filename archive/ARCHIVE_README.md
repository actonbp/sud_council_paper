ARCHIVE CONTENTS - Updated June 29, 2025
=========================

This folder contains files moved during repository cleanup for colleague handoff and ongoing maintenance.

## Major Cleanup June 29, 2025: Study 2 Reset

### Why This Cleanup?
After extensive experimentation with various text analysis approaches (topic modeling, embeddings, etc.), we decided to reset Study 2 with a clean slate. The simple frequency analysis was the final working approach before this reset.

### New Archive Folders Added:
- **study2_all_approaches/**: Contains 26 R scripts and 2 workflow markdown files from Study 2 experiments
  - LDA, BTM, NMF topic modeling attempts
  - Various embedding approaches
  - Tidymodels implementations
  - Simple frequency analysis (the final working approach)
- **study2_results/**: 12 result folders from different Study 2 analysis attempts
- **study2_outputs/**: 21 loose files including figures, RDS files, and methodology documentation

### Archived Documentation:
- **old_documentation/**: Extra markdown files moved from root to reduce clutter
  - AI_ONBOARDING.md
  - DATA_REQUIREMENTS.md
  - JUNE_10_2025_PLAN.md
  - study2_analysis_summary.md

### Repository Philosophy:
- Maximum parsimony - simple folder structure (study1, study2)
- Minimal documentation - rely on CLAUDE.md, README.md, and .cursor files
- Archive everything but keep structure simple for future AI agents

## Previous Archive Contents:

### Folders:
- deprecated_python/: Original Python analysis scripts (replaced by R)
- scripts/: Duplicate/draft R scripts (replaced by final versions)
- old_drafts/: Previous manuscript versions
- temp_files/: Temporary files, logs, and system files

### Files Added June 4, 2025:
- refactoring_plan.md: Completed Python-to-R transition plan (archived as implementation is complete)
- sud_council_paper.tex: LaTeX compilation artifact (can be regenerated)
- sud_council_paper.html: HTML compilation artifact (can be regenerated)
- SCRIPTS_OVERVIEW.md: Script inventory with formatting issues (info now in README.md)

All archived items can be regenerated from current source files or are no longer needed for active development.
