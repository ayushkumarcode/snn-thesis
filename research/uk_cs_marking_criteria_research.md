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

### 3.3 What this means for my project

For an SNN / neuromorphic computing project at Manchester, i don't need to produce a publishable paper. What i need is:
- A well-framed problem with clear objectives
- Evidence that i understand the existing literature (SNN architectures, neuromorphic hardware, DVS sensors, etc.)
- A working implementation that achieves the stated objectives
- A rigorous evaluation comparing my approach against relevant baselines
- Honest critical analysis of what works, what doesn't, and why

If i happen to achieve results competitive with published work, that would push into high-first territory, but it's not the threshold for a first.

---

## 4. Research question vs implementation quality

### 4.1 The weight of evidence

Looking at Cambridge's breakdown:
- Introduction and Preparation (which includes the research question): **30 marks**
- Implementation and Contribution: **45 marks**

So implementation carries about 1.5x the raw marks of the research framing.

At Manchester, the "Report" (55%) covers both the research question AND the write-up, while "Achievements" (30%) is purely the practical output. So the implementation component is separately assessed and worth nearly a third of the total mark.

### 4.2 Why the research question still matters a lot

Despite implementation carrying more raw marks, the research question might actually be more important in practice because:

1. **It frames everything else.** A vague or poorly defined research question makes it impossible to write a good evaluation, because you can't evaluate something if you haven't clearly stated what you're trying to achieve.

2. **It determines the lit review quality.** A specific, well-scoped research question naturally leads to a focused, relevant lit review. A vague question leads to a meandering, unfocused background chapter.

3. **It defines the evaluation criteria.** If the research question is "Can SNNs achieve comparable accuracy to ANNs on gesture recognition using DVS128?", then the evaluation has a clear target: compare accuracy, latency, and energy metrics. If the question is vague ("explore SNNs"), markers have nothing to evaluate success against.

4. **It signals ambition level.** A well-formulated research question that's specific, testable, and grounded in the literature signals to markers that you understand the field and have thought carefully about what you're doing.

### 4.3 The interaction

The relationship isn't "either/or" but "both, and they need to be aligned":

- A brilliant research question with poor implementation = 2:1 (you identified something interesting but couldn't execute it)
- A brilliant implementation with no clear research question = 2:1 (you built something impressive but nobody knows why or what it proves)
- A good research question with good implementation AND good evaluation = first (the holy trinity)

### 4.4 Manchester-specific guidance

The COMP30040 report guidance says the report must contain "an elucidation of the problem and the objectives of the project" and "a description of any research hypothesis." So Manchester expects a clear problem statement with objectives. It also expects "a critical appraisal of the project, indicating the rationale for any design/implementation decisions, lessons learnt during the course of the project, and evaluation (with hindsight) of the project outcome."

The word "hypothesis" is notable -- it suggests Manchester values projects that frame themselves around a testable proposition, not just a building exercise.

---

## 5. University-by-university details

### 5.1 Manchester (COMP30040) -- my university

**Assessment:**
- Report: 55%
- Achievements: 30%
- Screencast: 15%
- Q&A: Not directly assessed (informs achievements mark)

**Double marking:** Both supervisor and second marker assess independently, then discuss and agree a final mark.

**Report requirements:**
- Problem elucidation and project objectives
- In-depth context and literature investigation
- Life cycle stages (where appropriate)
- Verification and validation description
- Tools used in development
- Critical appraisal with rationale for decisions
- Research hypothesis description
- LaTeX recommended (Overleaf with university account)
- Formal writing style: third person, passive voice, past tense

The assessment criteria doc (project_assessment.pdf) is on Canvas but i couldn't find it publicly. Should download and study that carefully since it has the detailed grade descriptors.

**Manchester generic grade descriptors:**
- 70-85%: "Designed and developed a substantial, well-rounded product of good quality in all aspects, with all its parts working and overall showing a high degree of creativity"

### 5.2 Cambridge (Part II)

**Assessment:** 100% dissertation (no separate implementation mark)

**Breakdown:**
- Introduction and Preparation: 30/100
- Implementation and Contribution: 45/100
- Evaluation and Conclusions: 25/100

**Key requirements:**
- Max 12,000 words / 40 pages for main body
- Strict chapter structure required (Introduction, Preparation, Implementation, Evaluation, Conclusions)
- Write for reader with general CS knowledge but no specialist knowledge in your area
- Must demonstrate ethical and professional approaches
- Must include repository overview section in implementation chapter

**What makes a first at Cambridge:**
- Professional approach to design and implementation
- Evaluation with confidence metrics
- Clear, literate writing
- Structured design planning and design-for-test considerations
- Honest, nuanced discussion of successes and failures

Source: [Cambridge Part II Projects](https://www.cst.cam.ac.uk/teaching/part-ii/projects)

### 5.3 Edinburgh

**Assessment:** Holistic marking using Common Marking Scheme (A1-H bands)

**Grade bands:**
- A1 (90-100): "Often faultless. Well beyond that expected."
- A2 (80-89): "Truly scholarly and/or professional, often with an absence of errors."
- A3 (70-79): "Good command of subject, deep thinking, some personal creativity."
- B (60-69): "Firm grasp but may be gaps. Shows initiative, sound conclusions with minor weaknesses."
- C (50-59): "Sound but limited knowledge. Adequate understanding, limited analysis."
- D (40-49): "Basic knowledge with potential inaccuracies. Superficial understanding."

**UG4 degree weighting:** Years 3 and 4 are equally weighted (50% each).

Source: [Edinburgh Common Marking Scheme](https://informatics.ed.ac.uk/taught-students/all-students/your-studies/common-marking-scheme)

### 5.4 Imperial

**Assessment dimensions (equally weighted):**
1. Technical Achievement
2. Background Research
3. Quality of Dissertation

**Grade bands:**
- Distinguished (D*): 85-100% -- Exceptional work
- Distinction (D): 73-84% -- "Significant breadth and depth", outstanding implementation or theoretical work
- Merit (M): 63-67% -- "Both breadth and depth", high technical competence, at least moderate risk
- Pass (P): 53-57% -- Competence in "well defined, moderately low-risk problems"

**Critical rule:** The report quality acts as a ceiling -- you can't get a grade higher than what your report deserves, regardless of technical achievement.

Source: [Imperial Projects Guide](https://www.doc.ic.ac.uk/lab/mac-mcs-projects/ProjectsGuide.html)

### 5.5 Bristol

**Assessment:** Uses university-wide Level 6 marking criteria. The individual project (COMS30044) involves a written thesis, viva (~20 minutes), and potentially a demo.

Key insight from their FAQ: the 60-69% band is characterised by "some self directed work with significant supervisor input." For 70+, substantially more autonomy is expected. And this quote really stood out: "It doesn't matter how much work you've done, if you don't write it up well, you will fail."

Source: [Bristol CS Individual Project FAQ](https://cs-uob-individual-project.github.io/qanda/)

### 5.6 Warwick (CS310)

**Assessment:**
- Progress Report: 15%
- Final Report + Viva: 75%
- Project Management Reflection: 10%

**Report:** Expected 8,000-10,000 words. Viva is 45 minutes (presentation + questions).

**Four key assessed capabilities:**
1. Planning and managing a significant individual project
2. Building substantial software systems or conducting research from design to documentation
3. Presenting work orally and in writing
4. Demonstrating critical reflection on outcomes

Source: [Warwick CS310](https://warwick.ac.uk/fac/sci/dcs/teaching/modules/cs310/)

### 5.7 Sussex

Sussex has the most detailed publicly available grade descriptors i could find for CS projects.

For **final reports**:
- 90-100%: "Truly outstanding project" with publishable-quality outcomes and original thinking
- 80-89%: "Essentially faultless outcomes" with full objective achievement and originality
- 70-79%: Complete understanding, demanding objectives fully achieved, thorough evaluation
- 60-69%: Competent work with substantially achieved primary objectives
- 50-59%: Competent in most respects, reasonably achieved objectives
- 40-49%: Basic methodology understanding, fair progress

Source: [Sussex Report Marking Criteria](https://www.sussex.ac.uk/ei/internal/forstudents/informatics/undergraduate/finalyearprojects/reportmarkingcriteria)

---

## 6. How important is writing quality?

Very. It's not a minor factor -- every university flags it as make-or-break.

### 6.1 What universities say directly

**Cambridge:** "Better grades will arise from clarity and ease of reading, good pictures, clear explanations, minimal jargon and appropriate use of equations." They recommend allocating "at least four to six weeks" for writing.

**Imperial:** A project CANNOT receive a Pass/Merit/Distinction if the report isn't independently at that standard. This is the strongest statement i found -- writing quality is a hard ceiling on your grade.

**Bristol:** "It doesn't matter how much work you've done, if you don't write it up well, you will fail."

**Manchester (COMP30040):** Report is 55% of the total mark. Formal writing style recommended: third person, passive voice, past tense. LaTeX is recommended. Spelling and grammar errors "might make it harder for the examiners to understand what you're attempting to communicate."

### 6.2 Other advice

The Laramee guidelines for BSc CS dissertations (used at Swansea, Nottingham, and referenced across UK CS departments) emphasise that "a longer dissertation won't mean a better dissertation" and that markers value accuracy and methodological thoroughness over code quality or engineering practices unrelated to project objectives.

A student who got 90% and the Distinguished Dissertation Prize (Dominik Rys) emphasises: be aware of the "curse of knowledge" -- don't assume readers understand your specialised concepts. Write for clarity, not to impress.

### 6.3 What this means practically

Using LaTeX (as Manchester recommends) immediately signals professionalism. Good figures, clear tables, proper referencing, and logical chapter structure aren't optional extras -- they're fundamental to getting a first.

---

## 7. What to prioritise

Based on everything i've gone through, here's what matters most for targeting a first at Manchester COMP30040:

| Priority | Action | Impact on Mark | Why |
|---|---|---|---|
| 1 | Write an excellent, well-structured report | Very High (55% of total) | The report IS the project in the eyes of markers |
| 2 | Deliver working, well-evaluated implementation | Very High (30% of total) | Achievements are directly assessed; Q&A informs this |
| 3 | Conduct rigorous evaluation with clear metrics | High (embedded in report + achievements) | The single biggest differentiator between 2:1 and first |
| 4 | Frame a clear, testable research question/hypothesis | High (shapes entire report quality) | Without this, you can't write a good evaluation |
| 5 | Thorough, well-synthesised literature review | Moderate-High (part of report mark) | Must identify gap/motivation clearly |
| 6 | Produce a professional, clear screencast | Moderate (15% of total) | Easy marks if done well; hard to recover if done poorly |
| 7 | Show independence and self-direction | Moderate (implicit in all marks) | Distinguishes first from 2:1 |
| 8 | Demonstrate originality/novelty | Low-Moderate (bonus, not required) | Nice to have, not expected for a first |

---

## 8. What i couldn't find and confidence levels
