# 🤖 AI AGENT QUICK START GUIDE

**Welcome! You're helping with a complete mixed-methods SUD counseling research project. This guide gets you oriented in 2 minutes.**

---

## ⚡ **30-SECOND PROJECT OVERVIEW**

**What This Is:** Publication-ready academic study examining undergraduate interest in SUD counseling careers
- ✅ **Study 1:** Quantitative analysis (N=391) - WORKS but needs statistical fixes  
- ✅ **Study 2:** Qualitative analysis (N=19) - VALIDATED and publication-ready
- ✅ **Manuscript:** Complete APA paper - needs statistical claim revisions
- 🎯 **Current Focus:** Systematic statistical methodology improvements

**Your Role:** Help implement methodological fixes while maintaining research quality

---

## 🧭 **NAVIGATION GUIDE: WHERE TO FIND WHAT**

### **🆘 NEED HELP RIGHT NOW?**
- **User setup issues** → [README.md](README.md) (sections: Setup, Troubleshooting)
- **Can't compile manuscript** → [README.md](README.md#how-to-compile-the-manuscript)
- **Analysis scripts failing** → [README.md](README.md#how-to-run-the-analyses)

### **📚 NEED FULL CONTEXT?**
- **Complete project background** → [CLAUDE.md](CLAUDE.md) (read this for comprehensive understanding)
- **Current methodology issues** → [README.md](README.md#methodological-concerns--remediation-plans)
- **Study 2 validation details** → [STUDY2_METHODOLOGY_UPDATE.md](STUDY2_METHODOLOGY_UPDATE.md)

### **🎯 NEED SPECIFIC GUIDANCE?**
- **Statistical analysis rules** → [.cursor/rules/methodology-integrity-principles.mdc](.cursor/rules/methodology-integrity-principles.mdc)
- **Study 2 clustering approach** → [.cursor/rules/study2-hierarchical-clustering.mdc](.cursor/rules/study2-hierarchical-clustering.mdc)
- **Current project status** → [.cursor/PROJECT_STATUS.md](.cursor/PROJECT_STATUS.md)

---

## 🚀 **QUICK DECISION TREE: "WHAT SHOULD I DO?"**

### **User Asks About...**

#### **📊 "Help with Study 1 Analysis"**
→ **Status:** Works but needs statistical claim fixes  
→ **Action:** Read [methodological concerns](README.md#methodological-concerns--remediation-plans) first  
→ **Priority:** Remove bootstrap CIs and p-values (Week 1 plan)  
→ **Script:** `scripts/r/study1_main_analysis.R`

#### **📝 "Help with Study 2 Analysis"** 
→ **Status:** ✅ Validated and publication-ready  
→ **Action:** This methodology is robust - help with interpretation/visualization  
→ **Details:** [Study 2 validation](STUDY2_METHODOLOGY_UPDATE.md#june-5-2025-update-methodology-validation--enhancement)  
→ **Scripts:** `scripts/r/study2_*.R` (4 scripts in sequence)

#### **📄 "Help with Manuscript"**
→ **Status:** Complete but needs statistical revisions  
→ **Action:** Focus on removing invalid statistical claims per [action plan](README.md#immediate-action-items-priority-order)  
→ **File:** `sud_council_paper.qmd`  
→ **Compile:** `quarto render sud_council_paper.qmd --to apaquarto-docx`

#### **🔧 "Setup/Technical Issues"**
→ **Start:** [README.md setup guide](README.md#how-to-compile-the-manuscript)  
→ **Troubleshooting:** [Common issues section](README.md#most-common-issues-for-erika)  
→ **Dependencies:** Run `Rscript scripts/r/r_package_requirements.R`

#### **🎯 "What's the Current Priority?"**
→ **Week 1:** Remove bootstrap CIs, p-values, validate effect sizes  
→ **Week 2:** Multiple testing correction, nested CV  
→ **Week 3:** Power analysis, sensitivity testing  
→ **Details:** [Action plan](README.md#immediate-action-items-priority-order)

---

## 📁 **FILE STRUCTURE CHEAT SHEET**

### **🎯 ESSENTIAL FILES (90% of what you'll need):**
```
sud_council_paper/
├── 📄 README.md                    # USER GUIDE + methodological concerns
├── 📄 CLAUDE.md                    # FULL PROJECT CONTEXT for AI agents  
├── 📄 sud_council_paper.qmd        # MAIN MANUSCRIPT (edit this)
├── 📄 sud_council_paper.docx       # Compiled output
└── 📁 scripts/r/                   # 5 essential analysis scripts
    ├── study1_main_analysis.R           # Study 1: needs statistical fixes
    ├── study2_text_preprocessing.R      # Study 2: validated methodology  
    ├── study2_cooccurrence_analysis.R   # Study 2: clustering analysis
    ├── study2_methodology_validation.R  # Study 2: validation tables
    └── study2_create_visualizations.R   # Study 2: publication figures
```

### **📋 DOCUMENTATION HIERARCHY:**
1. **[AI_ONBOARDING.md](AI_ONBOARDING.md)** ← YOU ARE HERE (navigation hub)
2. **[README.md](README.md)** → User-focused setup + methodological concerns  
3. **[CLAUDE.md](CLAUDE.md)** → Complete AI agent context + project history
4. **[.cursor/PROJECT_STATUS.md](.cursor/PROJECT_STATUS.md)** → Current status + priorities
5. **[STUDY2_METHODOLOGY_UPDATE.md](STUDY2_METHODOLOGY_UPDATE.md)** → Study 2 validation details

---

## ⚠️ **CRITICAL THINGS TO KNOW**

### **🔴 DO NOT:**
- Assume current bootstrap confidence intervals are valid
- Add new statistical significance claims without justification  
- Ignore the documented methodological concerns
- Remove the systematic remediation tracking system

### **✅ DO PRIORITIZE:**
1. **Statistical validity** - Ensure all numerical claims are mathematically justified
2. **Methodological transparency** - Maintain honest documentation of limitations  
3. **Systematic improvement** - Follow the established Week 1-3 action plan
4. **Scientific integrity** - Accuracy over impressive-sounding results

### **🎯 CURRENT PROJECT HEALTH:**
- **Study 2:** ✅ Robust methodology, publication-ready
- **Study 1:** ⚠️ Good analysis, needs statistical claim fixes  
- **Manuscript:** 🔶 Complete content, requires methodological revisions
- **Overall:** Strong foundation with transparent improvement plan

---

## 🚀 **GETTING STARTED WORKFLOW**

### **First 5 Minutes:**
1. **Read this entire guide** (you're almost done!)
2. **Skim [README.md](README.md)** to understand user perspective  
3. **Check what user is asking for** and follow decision tree above

### **First 30 Minutes:**
4. **Read [CLAUDE.md](CLAUDE.md)** for full project context
5. **Review [methodological concerns](README.md#methodological-concerns--remediation-plans)**
6. **Understand current [project status](.cursor/PROJECT_STATUS.md)**

### **Ready to Help:**
7. **Follow the decision tree** based on user's specific request
8. **Prioritize Week 1 critical fixes** if working on Study 1
9. **Maintain transparency** about limitations and improvements needed

---

## 🎯 **SUCCESS CRITERIA**

**You're helping effectively when:**
- ✅ User can compile manuscript successfully
- ✅ Analysis scripts run without errors  
- ✅ Statistical claims are mathematically justified
- ✅ Methodological improvements follow systematic plan
- ✅ Documentation stays current and honest

**Avoid these common pitfalls:**
- ❌ Making impressive claims that aren't statistically valid
- ❌ Bypassing the established remediation priorities  
- ❌ Removing transparency about limitations
- ❌ Creating new files instead of improving existing ones

---

**🎉 You're ready! Use the decision tree above and dive into the linked documentation based on what the user needs.**

---

*This guide was created June 5, 2025, following comprehensive methodological audit. Keep it updated as project status changes.* 