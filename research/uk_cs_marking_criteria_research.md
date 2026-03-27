# UK CS Marking Criteria -- what actually matters for my thesis

So i went through a bunch of marking criteria and rubrics from Russell Group and other UK universities to figure out what actually matters for a final year CS project. Looked at Cambridge, Edinburgh, Imperial, Bristol, Sussex, Warwick, Cardiff, Glasgow, and obviously Manchester (my own COMP30040 docs).

The main takeaways: (1) the written report dominates everything, usually 55-80% of the total mark. (2) Implementation/achievements matter but they're secondary at 25-45%. (3) Evaluation quality is what separates the good from the great -- every single university says this. (4) Publishable results are NOT expected from undergrads -- they only come into play for 80+ marks. (5) A well-framed research question matters a lot because it shapes the whole report narrative, but implementation quality carries more raw marks in most schemes.

---

## 1. How different universities split up the marks

### 1.1 Direct comparison

| University | Report / Dissertation | Implementation / Achievements | Presentation / Viva / Screencast | Other |
|---|---|---|---|---|
| **Manchester (COMP30040)** | 55% | 30% | 15% (screencast) | Q&A not directly assessed |
| **Cambridge (Part II)** | 100% (dissertation only, broken into sub-categories below) | -- | -- | -- |
| **Warwick (CS310)** | 75% (final report + viva combined) | Embedded in report mark | 15% (progress report) | 10% (project management) |
| **Exeter (ECM3401)** | ~90% (report + presentation) | Embedded in report mark | Embedded | 10% (supervisor report) |
| **Imperial (MSc, indicative of UG approach)** | 85% (report) | Embedded in report mark | 15% (poster/presentation) | -- |

### 1.2 Cambridge Part II breakdown

Cambridge gives the most transparent breakdown of what markers actually look at within the dissertation. Marked out of 100:

| Category | Marks (out of 100) | What it covers |
|---|---|---|
| **Introduction and Preparation** | 30 | Problem statement, lit review, project planning, background research, understanding of related work |
| **Implementation and Contribution** | 45 | Technical work, design decisions, code quality, system architecture, the actual "doing" |
| **Evaluation and Conclusions** | 25 | Testing, benchmarking, critical analysis of results, reflection, future work |

Source: [Cambridge CST Assessment](https://www.cst.cam.ac.uk/teaching/part-ii/projects/assessment)

### 1.3 Edinburgh -- tiered criteria

Edinburgh does it differently, more qualitative than percentage-based, with three tiers:

**Basic criteria** (must be satisfied to pass):
- Understanding of the problem
- Completion of the work
- Quality of the work
- Quality of the dissertation

**Additional criteria** (needed for good marks, 60+):
- Knowledge of the literature
- Critical evaluation of previous work
- Critical evaluation of own work
- Justification of design decisions
- Solution of conceptual problems
- Amount of work

**Exceptional criteria** (needed for 80+):
- Evidence of outstanding merit, e.g. originality
- Inclusion of material worthy of publication

Source: [Edinburgh DISS Project Assessment](https://opencourse.inf.ed.ac.uk/diss/project-assessment)

### 1.4 Imperial -- three equal pillars

Imperial assesses across three equally weighted dimensions:

1. **Technical Achievement** (design, correctness, completeness, elegance, significance)
2. **Background Research** (awareness of prior work, specification development)
3. **Quality of Dissertation** (material selection, organisation, prose clarity, diagrams)

Important rule from Imperial: a project can't receive a Pass/Merit/Distinction if the report itself isn't at that level. Technical excellence can't compensate for poor writing.

Source: [Imperial Projects Guide](https://www.doc.ic.ac.uk/lab/mac-mcs-projects/ProjectsGuide.html)

### 1.5 Cardiff -- four equal dimensions

Cardiff marks on four equally weighted dimensions (25% each):
1. **Project Approach** -- professional methods and tools
2. **Argument** -- structure, justification of conclusions
3. **Products** -- appropriateness, contribution to knowledge
4. **Reflection** -- insight and understanding developed

Source: [Cardiff Dissertation Guide](https://pats.cs.cf.ac.uk/wiki/doku.php?id=masters_dissertation_guide)

### 1.6 What a "typical" breakdown looks like

Cross-referencing all the schemes, here's roughly where marks fall for a UK CS undergrad project:

| Assessment Dimension | Typical Weight | Range Across Universities |
|---|---|---|
| **Literature Review / Background** | 15-20% | 10-30% |
| **Design & Implementation** | 30-40% | 25-45% |
| **Evaluation & Testing** | 15-20% | 15-25% |
| **Writing Quality & Presentation** | 15-20% | 15-25% |
| **Novelty / Originality** | 0-5% (bonus territory) | 0-10% |
| **Project Management / Process** | 5-10% | 0-15% |

Worth noting: most UK universities don't actually assign fixed percentages to these sub-dimensions within the report itself. The report is marked holistically. These figures are inferred from Cambridge's breakdown, Imperial's three pillars, Edinburgh's tiered criteria, and how Manchester separates "Report" (55%) from "Achievements" (30%).

### 1.7 My Manchester COMP30040 breakdown

| Component | Weight | Notes |
|---|---|---|
| **Report** | 55% | Marked independently by supervisor and second marker, then agreed |
| **Achievements** | 30% | Project output: software, hardware, research, experimentation |
| **Screencast** | 15% | 8-minute video presentation of work |
| **Q&A** | 0% (indirect) | 25-minute interview with second marker; informs achievements mark but not directly scored |

The report must contain:
- Elucidation of the problem and objectives
- In-depth investigation of context and literature, and similar products
- Description of life cycle stages undertaken
- Description of verification and validation
- Description of tools used
- Critical appraisal with rationale for design/implementation decisions
- Lessons learnt and evaluation of outcome
- Description of any research hypothesis

---

## 2. What separates a first (70+) from a 2:1 (60-69)

This is the question that actually matters. The evidence across universities is pretty consistent.

### 2.1 Edinburgh's detailed scheme

**A3 band (70-79, First Class):**
- Covers up-to-date material in a scholarly/professional way
- Good command of the subject and current theory
- Deep thinking, logical rigour
- Some personal creativity
- Fully explored assigned tasks
- Sensible, well-justified software/experimental design
- Code compiles without errors, strong testing

**B band (60-69, Upper Second):**
- Firm grasp of the subject but there may be gaps
- Shows initiative and clear critical evaluation
- Sound conclusions but with minor weaknesses
- Design is adequate but may not be fully justified
- Generally correct behaviour with minor bugs possible

Source: [Edinburgh Common Marking Scheme](https://informatics.ed.ac.uk/taught-students/all-students/your-studies/common-marking-scheme)

### 2.2 Sussex

**70-79%:**
- Complete understanding of all aspects of the project material
- Reasonably demanding project objectives fully achieved
- Thorough evaluation with clear evidence of background research

**60-69%:**
- Competent work with substantially achieved primary objectives
- Clear technical understanding with problem analysis
- Reasonable presentation standards

**80-89%:**
- Essentially faultless outcomes
- Full objective achievement AND originality
- All objectives fully achieved with evidence of going beyond what was required

**90-100%:**
- Truly outstanding project with publishable-quality outcomes
- Original thinking that pushes boundaries

Source: [Sussex Report Marking Criteria](https://www.sussex.ac.uk/ei/internal/forstudents/informatics/undergraduate/finalyearprojects/reportmarkingcriteria)

### 2.3 Bristol -- the independence factor

Bristol has a revealing indicator about student independence:
- **60-69%**: "Some self directed work with significant supervisor input"
- **70+**: Greater autonomy implied, with students driving their own decisions

Source: [Bristol CS Individual Project FAQ](https://cs-uob-individual-project.github.io/qanda/)

### 2.4 What actually separates the grades

Based on all the universities i looked at, here's what distinguishes each band:

| Dimension | 2:1 (60-69) | First (70-79) | High First (80+) |
|---|---|---|---|
| **Literature Review** | Covers relevant work, may miss some key papers, adequate synthesis | Good coverage, solid synthesis, identifies gap clearly | Scholarly treatment, identifies subtle connections between works, positions project expertly |
| **Implementation** | Works, may have minor issues, achieves primary objectives | Works well, clean design, achieves all objectives, some technical sophistication | Elegant, well-engineered, potentially novel techniques, exceeds objectives |
| **Evaluation** | Some testing done, basic analysis of results | Appropriate metrics, honest discussion of limitations | Rigorous methodology, statistical confidence, comparison with baselines, publishable quality |
| **Critical Analysis** | Acknowledges some limitations | Honest, nuanced discussion of what worked and what didn't | Deep self-awareness, identifies exactly why things failed and what would fix them |
| **Writing** | Clear enough, some structural issues, adequate referencing | Well-structured, clear prose, good figures, proper referencing | Professional quality, reads like a technical report or conference paper |
| **Independence** | Significant supervisor guidance needed | Self-directed with supervisor as advisor | Student drives the project, supervisor learns from the student |
| **Originality** | Follows existing approaches | Shows some personal creativity in approach | Genuine novel contribution, new technique or insight |

### 2.5 The biggest differentiator: evaluation quality

This came up again and again across every marking scheme. The evaluation chapter is what separates grades. Cambridge says explicitly that assessors look for "a nuanced discussion of the successes and failures" and that "a graph that does not indicate confidence intervals or has poorly chosen axes will generally leave a professional scientist with a negative impression."

The pattern is pretty clear:
- A 2:1 student builds something that works and describes what they built.
- A first-class student builds something that works, critically evaluates WHY it works (or doesn't), compares it against alternatives, and reflects honestly on the limitations.

---

## 3. Are publishable results expected from undergrads?

No. Not at all.

### 3.1 Evidence

**Edinburgh** lists "Inclusion of material worthy of publication" under its Exceptional Criteria -- meaning it's only expected for 80-90+ marks and is NOT needed for a first.

**Sussex** reserves "publishable-quality outcomes" for the 90-100% band, explicitly labelled "Exceptional" and "Truly outstanding."

**Student Room / general UK academic consensus**: Multiple sources confirm that "originality" at the undergrad level means not plagiarising and not resubmitting prior work. There's "not usually any requirement to deliver anything new, although if you did and it was properly done, you'd be in First territory." The consensus is that "original contribution to knowledge" is a PhD criterion, not an undergrad one.

**Cambridge**: Their Part II assessment notes that "where a project makes a genuine contribution to the field, that is strong evidence that it should score in the 36-45 band [out of 45] for implementation and contribution. However, this is not a requirement of the band." So you can score in the top band without making a genuine field contribution.

**Imperial**: Mentions assessing "the size and quality of the part of your project report that could be published" but frames this as a factor for distinguishing the very top marks, not a baseline expectation.

### 3.2 What IS expected

For a solid first (70-79%):
- Competent, well-evaluated implementation
- Thorough understanding of the problem space
- Clear methodology and honest evaluation
- Good writing quality
- Some degree of independent thinking and personal creativity

For 80+, markers start looking for:
- Work that could potentially appear in a workshop paper or student conference
- Novel approaches or insights
- Particularly elegant or ambitious implementation
- Evaluation methodology that goes beyond the obvious

