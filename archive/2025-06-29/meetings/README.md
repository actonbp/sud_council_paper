# Meeting Preparation & Quarto Reports

This folder is designed for creating professional meeting materials and project updates for the SUD Counseling Research Project. Documents here compile to APA-formatted Word documents or PDFs that can be brought to meetings.

## Organization

### Meeting Notes Format
- **Filename convention**: `YYYY-MM-DD_meeting_topic.md` or `.qmd`
- **Example**: `2025-06-05_project_update.qmd`

### Document Types
- **Project Updates**: Current manuscript status and recent accomplishments for team meetings
- **Progress Reports**: Analysis results, methodology updates, and key findings summaries
- **Meeting Preparation**: Pre-meeting materials outlining discussion points and decisions needed
- **Planning Documents**: Next steps, milestone planning, and future work direction

## Creating New Meeting Documents

### Option 1: Markdown Files (.md)
- Simple markdown files for quick notes
- Can be converted to Word if needed

### Option 2: Quarto Documents (.qmd) **[RECOMMENDED]**
- Compile to professional APA-formatted Word documents or PDFs
- Perfect for formal meeting presentations and project updates
- Compile to Word: `quarto render filename.qmd --to apaquarto-docx`
- Compile to PDF: `quarto render filename.qmd --to pdf`
- Can include tables, figures, and formatted text automatically

### Option 3: Direct Word Documents
- Create `.docx` files directly for immediate sharing

## Typical Meeting Document Content

### **For Project Update Meetings:**
- **Manuscript Status**: Current version, recent changes, compilation success
- **APA Formatting**: Recent improvements (e.g., page breaks between tables)
- **Analysis Pipeline**: Study 1 & 2 results, any new findings
- **Technical Updates**: R package updates, script improvements, repository organization
- **Next Steps**: Immediate priorities, upcoming deadlines, decisions needed

### **For Progress Reports:**
- **Research Findings**: Key results from Study 1 (quantitative) and Study 2 (qualitative)
- **Methodology Updates**: Analysis approach refinements, validation results
- **Publication Status**: Manuscript readiness, submission timeline
- **Collaboration Items**: Co-author contributions, review feedback

### **Quick Start Example:**
```bash
# Create tomorrow's meeting update
cd meetings/
# Create: 2025-06-05_project_update.qmd
# Add: current status, APA improvements, next steps
quarto render 2025-06-05_project_update.qmd --to apaquarto-docx
# Result: Professional Word document ready for meeting
```

## Templates

Templates for common meeting document types can be added here as needed.

## Current Meeting Documents

**2025-06-05_study2_methodology_review.qmd/docx** - Study 2 methodology review and resolution  
- **Status**: ✅ **COMPLETED with actual results**
- **Contains**: Methodology concerns, systematic solutions, and real data-driven clustering results  
- **Key Results**: 4 mathematically-derived word clusters (35.2% SUD detection rate)
- **Output**: Professional/field framework (18.8%), Substance/personal frame (11.9%), Emotional (4.4%), Social (4.4%) dimensions
- **Action Items**: Research team to assign thematic names based on word cluster groupings

---
**Created**: June 4, 2025  
**Updated**: June 5, 2025 - Added actual clustering results to meeting document  
**Purpose**: Create professional meeting materials using Quarto compilation 