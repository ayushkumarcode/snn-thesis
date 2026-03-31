# ICONS 2026 EasyChair Submission Guide

## Paper Details

- **Title:** SpiNNaker Deployment of Pruned Spiking Neural Networks for 50-Class Sound Classification
- **Authors:** Ayush Kumar (University of Manchester), Oliver Rhodes (University of Manchester)
- **Conference:** ICONS 2026 --- International Conference on Neuromorphic Systems, Washington DC, July 29 -- August 1, 2026
- **Submission system:** EasyChair (https://easychair.org/cfp/ACM-ICONS-2026)
- **Format:** 6-page full paper, ACM sigconf format
- **PDF file:** `paper/latex/main.pdf`

---

## Abstract (Plain Text)

We present the first spiking neural network (SNN) evaluation on ESC-50, a 50-class environmental sound benchmark, including deployment on SpiNNaker neuromorphic hardware. A Rhythm SNN with learnable oscillatory threshold modulation achieves 61.10%---only 2.75 pp below the matched ANN (63.85%). We deploy pruned models across 10 sparsity levels (50--95%) on SpiNNaker via current-injection, running over 22,000 hardware inferences. The key finding: 50% pruning improves SpiNNaker accuracy from 57.35% to 60.45% (+3.1 pp), acting as implicit regularization for hardware deployment. At 85% pruning, the hardware gap shrinks to just 0.6 pp while energy drops 3.5x. At two pruning levels, SpiNNaker exceeds the software simulation.

---

## Keywords

spiking neural networks, environmental sound classification, neuromorphic computing, SpiNNaker, pruning, spike encoding

---

## Step-by-Step EasyChair Submission Procedure

### Step 1: Create or Log In to EasyChair Account

1. Navigate to https://easychair.org/
2. If you do not have an account, click "Create an account" and fill in:
   - First name: Ayush
   - Last name: Kumar
   - Email: ayush.kumar@student.manchester.ac.uk
   - Affiliation: University of Manchester
3. Confirm the account via the verification email.
4. If you already have an account, log in with your credentials.

### Step 2: Navigate to the ICONS 2026 Submission Portal

1. Go to https://easychair.org/cfp/ACM-ICONS-2026
2. Click "Submit a paper" or "New Submission" (the exact label depends on the conference configuration).
3. You will be taken to the submission form for ICONS 2026.

### Step 3: Enter Author Information

**Author 1:**
- First name: Ayush
- Last name: Kumar
- Email: ayush.kumar@student.manchester.ac.uk
- Affiliation: University of Manchester
- Country: United Kingdom
- Corresponding author: Yes

**Author 2:**
- First name: Oliver
- Last name: Rhodes
- Email: oliver.rhodes@manchester.ac.uk
- Affiliation: University of Manchester
- Country: United Kingdom
- Corresponding author: No

Click "Add author" to add the second author. Ensure the order is Ayush Kumar first, Oliver Rhodes second.

### Step 4: Enter Title

Enter exactly:

```
SpiNNaker Deployment of Pruned Spiking Neural Networks for 50-Class Sound Classification
```

Do not include a line break. EasyChair may or may not display this with your LaTeX line break; the PDF is authoritative.

### Step 5: Enter Abstract

Paste the plain text abstract from above into the abstract field. Do not use LaTeX formatting (no `\textbf{}`, no `$...$`, no `---`). The abstract field typically accepts plain text only.

Ensure:
- All dashes are rendered as plain hyphens or em-dashes depending on what the field supports.
- The "pp" abbreviation (percentage points) is spelled out if the system strips special characters.
- Numbers like "22,000" use commas.
- The multiplication sign "3.5x" uses a plain "x".

### Step 6: Enter Keywords

Enter the following keywords, one per line or comma-separated (depending on interface):

```
spiking neural networks
environmental sound classification
neuromorphic computing
SpiNNaker
pruning
spike encoding
```

### Step 7: Select Paper Type and Track

- **Paper type:** Full paper (6 pages, 20-minute oral presentation)
- **Track:** If ICONS 2026 has multiple tracks, select the most relevant. Likely candidates:
  - "Neuromorphic Applications" (primary -- we deploy on SpiNNaker for audio)
  - "Algorithms and Models" (secondary -- Rhythm SNN, encoding comparison)
  - "Hardware and Systems" (tertiary -- SpiNNaker deployment pipeline)
- If only one general track is offered, select that.

### Step 8: Select Topics / Subject Areas

ICONS typically asks you to select relevant topics. Check all that apply:

- [x] Event/spike-based systems
- [x] Energy-efficient edge AI
- [x] Neuromorphic benchmarks
- [x] Domain-specific implementations
- [x] Supervised learning for neuromorphic systems
- [x] Non-von Neumann computing
- [x] Simulation techniques

### Step 9: Upload the PDF

1. Click "Choose File" or "Upload" in the paper upload section.
2. Navigate to: `paper/latex/main.pdf`
3. Upload the file.
4. Verify:
   - File size is reasonable (typically under 10 MB; ours should be ~500 KB--1 MB with figures).
   - The PDF renders correctly in the EasyChair preview.
   - All 6 pages are visible.
   - Figures (architecture diagram, SpiNNaker pipeline, encoding bar chart, Pareto curve) render correctly.
   - The ACM sigconf formatting is applied (two-column, correct margins, correct font).

### Step 10: Supplementary Materials (Optional)

If ICONS allows supplementary uploads:
- Upload the code repository link: https://github.com/ayushkumarcode/snn-thesis
- Do NOT upload the full codebase as a zip; link to the repository instead.
- If hardware verification logs are requested, they are in `spinnaker/` directory.

### Step 11: Conflict of Interest Declaration

Declare conflicts of interest:
- Oliver Rhodes (co-author, supervisor) -- University of Manchester
- Steve Furber -- University of Manchester (SpiNNaker project lead; our work uses SpiNNaker hardware his group built)
- Any other Manchester APT group members who may serve as reviewers

### Step 12: Review and Submit

1. Review all entered information on the confirmation page.
2. Double-check:
   - Author names and order are correct.
   - Title matches the PDF exactly.
   - Abstract matches the PDF abstract.
   - Keywords are entered.
   - PDF is uploaded and renders correctly.
3. Click "Submit" or "Confirm Submission."
4. Save the confirmation number / submission ID.
5. You will receive a confirmation email at ayush.kumar@student.manchester.ac.uk.

---

## Pre-Submission Checklist

### Formatting Compliance
- [x] ACM sigconf format (`\documentclass[sigconf,nonacm=false]{acmart}`)
- [x] 6 pages (within ICONS full paper limit)
- [x] Two-column layout
- [x] Correct ACM CCS concepts included
- [x] `\acmConference` set to ICONS '26
- [x] Figures are vector PDF (architecture, pipeline, bar chart, Pareto)
- [x] All tables use booktabs formatting
- [x] References in ACM-Reference-Format bibliography style
- [x] No page numbers in author version (handled by acmart)

### Content Compliance
- [x] Abstract under 250 words
- [x] Code/models link provided in abstract: https://github.com/ayushkumarcode/snn-thesis
- [x] All results are 5-fold cross-validated on ESC-50 predefined folds
- [x] Statistical significance reported (paired t-test, p-values)
- [x] Limitations section present in Discussion
- [x] Prior ICONS work cited (Yarga et al. 2022, Seekings et al. 2024)
- [x] Acknowledgements for SpiNNaker access and CSF3 HPC

### PDF Quality
- [x] No broken references (no "??" in text)
- [x] All figures referenced in text
- [x] All tables referenced in text
- [x] No orphaned text or widows
- [x] Fonts are embedded (verify with `pdffonts main.pdf`)
- [x] File is not encrypted or password-protected

---

## Post-Submission Timeline (Estimated)

| Date | Event |
|------|-------|
| April 1, 2026 (or April 8 AoE) | Submission deadline |
| May--June 2026 | Review period |
| June 2026 (est.) | Author notification |
| June--July 2026 | Camera-ready preparation (if accepted) |
| July 29 -- August 1, 2026 | ICONS 2026, Washington DC |

---

## Fallback Plan

If the paper is rejected from ICONS 2026:
1. Address reviewer feedback.
2. Submit to EUSIPCO 2026 (European Signal Processing Conference).
3. Consider arXiv preprint to establish priority (Larroza et al. already on arXiv for ESC-10).
4. Other targets: IJCNN 2027, IEEE BioCAS 2026, ACM NICE 2027.

---

## Contact

- Ayush Kumar: ayush.kumar@student.manchester.ac.uk
- Oliver Rhodes: oliver.rhodes@manchester.ac.uk
