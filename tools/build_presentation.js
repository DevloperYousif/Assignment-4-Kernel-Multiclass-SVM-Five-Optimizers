// Builds presentation/Assignment_4_Kernel_SVM.pptx from results/summary.json (all numbers come from the executed notebooks).
// Run: python tools/collect_results.py && node tools/build_presentation.js && python tools/fix_notes.py presentation/Assignment_4_Kernel_SVM.pptx
const pptxgen = require("pptxgenjs");
const path = require("path");
const fs = require("fs");
const ROOT = path.join(__dirname, "..");
const S = JSON.parse(fs.readFileSync(path.join(ROOT, "results", "summary.json"), "utf8"));
const FIG = path.join(ROOT, "results", "figures") + "/";
const OUT = path.join(ROOT, "presentation", "Assignment_4_Kernel_SVM.pptx");
const makeNotes = require("./presentation_notes.js");

// ---------------------------------------------------------------- data helpers
const C_ = Object.fromEntries(S.comparison.map(r => [r.optimizer, r]));
const ORDER = ["GD", "SGD", "MBGD", "AdaGrad", "Adam"];
const NAME = { GD: "Gradient Descent", SGD: "SGD", MBGD: "Mini-Batch GD", AdaGrad: "AdaGrad", Adam: "Adam" };
const f4 = x => Number(x).toFixed(4), f3 = x => Number(x).toFixed(3), f1 = x => Number(x).toFixed(1);
const pc = x => (100 * Number(x)).toFixed(2) + "%";
const argmax = key => ORDER.reduce((a, b) => (C_[b][key] > C_[a][key] ? b : a));
const argmin = key => ORDER.reduce((a, b) => (C_[b][key] < C_[a][key] ? b : a));
const H = S.hyper;
const LVK = S.linear_vs_kernel;
const LVK_COLS = Object.keys(LVK);
const KC = S.kernel_comparison;
const BS = S.basis_size;
const SEL = S.kernel_selection;
const V = S.verdict;
const D = { C_, ORDER, NAME, f4, f3, f1, pc, argmax, argmin, H, LVK, LVK_COLS, KC, BS, SEL, V, S };

// ---------------------------------------------------------------- design system
const COL = { GD: "1F77B4", SGD: "FF7F0E", MBGD: "2CA02C", AdaGrad: "9467BD", Adam: "D62728",
  navy: "14213D", ink: "1F2937", slate: "475569", gray: "64748B", teal: "0F766E", gold: "B45309", light: "F1F5F9" };
const TINT = { GD: "E8F1FA", SGD: "FFF1E5", MBGD: "EAF6EA", AdaGrad: "F2ECF9", Adam: "FCEAEA", navy: "E9EDF5", teal: "E6F4F2", gold: "FDF3E3", gray: "F1F5F9" };
const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";
pres.title = "Kernel-Based Multi-Class SVM with Five Optimizers";
pres.theme = { headFontFace: "Georgia", bodyFontFace: "Calibri" };

pres.defineSlideMaster({ title: "TITLE", background: { color: COL.navy }, objects: [] });
pres.defineSlideMaster({ title: "CONTENT", background: { color: "FFFFFF" }, objects: [
  { rect: { x: 0, y: 0, w: 13.333, h: 0.12, fill: { color: COL.navy } } },
  { text: { text: "Assignment 4 · Kernel Multi-Class SVM · UCI HAR", options: { x: 0.6, y: 7.05, w: 8, h: 0.3, fontSize: 10, color: COL.gray, margin: 0 } } },
], slideNumber: { x: 12.3, y: 7.05, w: 0.5, h: 0.3, fontSize: 10, color: COL.gray } });

const tb = (s, text, o) => s.addText(text, { isTextBox: true, margin: 0, fontFace: "Calibri", color: COL.ink, ...o });
const title = (s, t, sub) => {
  tb(s, t, { x: 0.6, y: 0.3, w: 12.1, h: 0.7, fontSize: 28, bold: true, color: COL.navy, fontFace: "Georgia", valign: "middle" });
  if (sub) tb(s, sub, { x: 0.6, y: 0.98, w: 12.1, h: 0.4, fontSize: 14, color: COL.slate, italic: true });
};
function img(s, f, x, y, w, maxH) {
  const sz = S.fig_sizes[f]; if (!sz) throw new Error("missing figure " + f);
  let h = w * sz[1] / sz[0];
  if (maxH && h > maxH) { h = maxH; w = h * sz[0] / sz[1]; }
  s.addImage({ path: FIG + f, x, y, w, h });
  return { w, h };
}
function card(s, x, y, w, h, key, head, body, size = 14) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.08, fill: { color: TINT[key] || TINT.gray }, line: { color: TINT[key] || TINT.gray } });
  s.addShape(pres.shapes.RECTANGLE, { x, y: y + 0.12, w: 0.07, h: h - 0.24, fill: { color: COL[key] || COL.navy }, line: { color: COL[key] || COL.navy } });
  tb(s, head, { x: x + 0.25, y: y + 0.12, w: w - 0.4, h: 0.45, fontSize: 18, bold: true, color: COL[key] || COL.navy });
  tb(s, body, { x: x + 0.25, y: y + 0.6, w: w - 0.4, h: h - 0.7, fontSize: size + 3, valign: "top", paraSpaceAfter: 4 });
}
const bullets = (items, size = 15) => items.map((t, i) => ({ text: t, options: { bullet: { indent: 16 }, fontSize: size, paraSpaceAfter: 6, breakLine: i < items.length - 1 } }));
function stat(s, x, y, w, num, label, key) {
  tb(s, num, { x, y, w, h: 0.7, fontSize: 28, bold: true, color: COL[key] || COL.navy, fontFace: "Georgia" });
  tb(s, label, { x, y: y + 0.72, w, h: 0.55, fontSize: 12, color: COL.slate, valign: "top" });
}
function eq(s, text, x, y, w, h, size = 20) {
  s.addShape(pres.shapes.RECTANGLE, { x, y, w, h, fill: { color: "F8FAFC" }, line: { color: "CBD5E1", width: 0.75 } });
  tb(s, text, { x: x + 0.2, y, w: w - 0.4, h, fontSize: size, fontFace: "Cambria Math", align: "center", valign: "middle", color: COL.navy });
}
function table(s, rows, x, y, w, colW, o = {}) {
  const hdr = rows[0].map(t => ({ text: t, options: { bold: true, color: "FFFFFF", fill: { color: COL.navy }, fontSize: o.hs || 12, align: "center" } }));
  const body = rows.slice(1).map((r, i) => r.map((t, j) => ({ text: String(t), options: { fontSize: o.fs || 12, align: j === 0 ? "left" : "center",
    fill: { color: o.hl && o.hl(i, j) ? "FEF3C7" : (i % 2 ? "F8FAFC" : "FFFFFF") }, bold: !!(o.bold && o.bold(i, j)), color: (o.color && o.color(i, j)) || COL.ink } })));
  s.addTable([hdr, ...body], { x, y, w, colW, border: { type: "solid", color: "E2E8F0", pt: 0.75 }, fontFace: "Calibri", rowH: o.rowH || 0.36, valign: "middle" });
}
const NOTES = makeNotes(D);
let slideNo = 0;
function newSlide(master = "CONTENT") { const s = pres.addSlide({ masterName: master }); s.addNotes(NOTES[slideNo]); slideNo += 1; return s; }

// ================================================================= 1 Title
let s = newSlide("TITLE");
s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 6.2, w: 13.333, h: 1.3, fill: { color: "0B1528" }, line: { color: "0B1528" } });
tb(s, "ASSIGNMENT 4  ·  MACHINE LEARNING", { x: 0.8, y: 0.9, w: 11, h: 0.4, fontSize: 14, bold: true, color: "FCA311", charSpacing: 4 });
tb(s, "Human Activity Recognition Using Kernel-Based Multi-Class Support Vector Machines", { x: 0.8, y: 1.5, w: 11.7, h: 1.9, fontSize: 38, bold: true, color: "FFFFFF", fontFace: "Georgia", valign: "top" });
tb(s, "RBF kernel · Crammer-Singer hinge loss · RKHS regularisation · five optimizers written from scratch in NumPy", { x: 0.8, y: 3.55, w: 11.7, h: 0.5, fontSize: 18, color: "CBD5E1" });
ORDER.forEach((o, i) => {
  const x = 0.8 + i * 2.4;
  s.addShape(pres.shapes.OVAL, { x, y: 4.6, w: 0.5, h: 0.5, fill: { color: COL[o] }, line: { color: COL[o] } });
  tb(s, String(i + 1), { x, y: 4.6, w: 0.5, h: 0.5, fontSize: 16, bold: true, color: "FFFFFF", align: "center", valign: "middle" });
  tb(s, NAME[o], { x: x + 0.6, y: 4.6, w: 1.75, h: 0.5, fontSize: 15, color: "FFFFFF", valign: "middle" });
});
tb(s, "Dataset: UCI Human Activity Recognition Using Smartphones (30 volunteers, 6 activities, 561 features)", { x: 0.8, y: 6.5, w: 11.7, h: 0.5, fontSize: 14, color: "94A3B8" });

// ================================================================= 2 Overview
s = newSlide(); title(s, "Project overview", "One model, one objective, five optimizers - a controlled experiment");
[["UCI HAR data", "10,299 windows\n561 features"], ["Subject-aware split", "train / val / test\nby person"], ["Standardise + basis", "M = 2,000 training\nsamples"],
 ["RBF kernel matrices", "H = k(X, Z)\nG = k(Z, Z)"], ["Kernel SVM", "S = HA + b\nCrammer-Singer"], ["5 optimizers", "GD · SGD · MBGD\nAdaGrad · Adam"], ["Evaluation", "Macro F1 on\nunseen subjects"]].forEach(([a, b], i) => {
  const x = 0.6 + i * 1.75;
  s.addShape(pres.shapes.PENTAGON, { x, y: 1.75, w: 1.7, h: 1.25, fill: { color: i === 5 ? COL.Adam : (i === 4 ? COL.teal : COL.navy) }, line: { color: "FFFFFF" } });
  tb(s, [{ text: a, options: { bold: true, breakLine: true, fontSize: 12 } }, { text: b, options: { fontSize: 10 } }], { x: x + 0.08, y: 1.75, w: 1.35, h: 1.25, color: "FFFFFF", align: "center", valign: "middle" });
});
card(s, 0.6, 3.45, 3.9, 3.3, "navy", "What is identical", "Data, split, scaling, RBF kernel, gamma, kernel basis, initial parameters (A = 0, b = 0), lambda, objective, 100-epoch budget and metrics.", 14);
card(s, 4.72, 3.45, 3.9, 3.3, "Adam", "What changes", "Only the optimizer and its learning rate (each tuned on validation with the same budget: 5 candidate values x 100 epochs).", 14);
card(s, 8.84, 3.45, 3.9, 3.3, "teal", "Extra analyses", "Kernel selection (gamma, lambda), basis size up to the exact kernel, Linear / Polynomial / RBF kernels, Linear SVM and scikit-learn SVC baselines.", 14);

// ================================================================= 3 Problem
s = newSlide(); title(s, "Problem statement", "What is the person doing - from the phone's motion sensors?");
card(s, 0.6, 1.6, 6.0, 2.4, "teal", "Real-world motivation", "Activity monitoring for elderly people, rehabilitation and fitness apps. A phone on the waist records 3-axis acceleration and rotation at 50 Hz; every 2.56-second window must be assigned to one activity.", 14);
card(s, 0.6, 4.2, 6.0, 2.5, "navy", "Why it is a hard multi-class problem", "K = 6 classes, not 2. Moving activities are easy to tell apart from static ones, but SITTING and STANDING produce very similar signals. The model must also generalise to NEW people.", 14);
stat(s, 7.1, 1.6, 2.6, "6", "activity classes (multi-class)", "Adam");
stat(s, 9.9, 1.6, 2.8, "561", "engineered time & frequency features", "GD");
stat(s, 7.1, 3.3, 2.6, "30", "volunteers (21 train / 9 test)", "MBGD");
stat(s, 9.9, 3.3, 2.8, "10,299", "labelled windows", "AdaGrad");
tb(s, "Question of the assignment: can a non-linear RBF kernel SVM - trained by five different first-order optimizers - recognise these activities on unseen subjects, and how do the optimizers compare?", { x: 7.1, y: 5.0, w: 5.6, h: 1.6, fontSize: 15, italic: true, color: COL.slate });

// ================================================================= 4 Objectives
s = newSlide(); title(s, "Project objectives");
[["1", "Kernels from scratch", "Linear, polynomial, RBF (primary) and exploratory sigmoid in NumPy", "navy"],
 ["2", "Genuine kernel SVM", "Representer form S = HA + b with RKHS-regularised Crammer-Singer hinge loss", "teal"],
 ["3", "Five optimizers", "GD, SGD, Mini-Batch GD, AdaGrad, Adam implemented by hand", "Adam"],
 ["4", "No data leakage", "Subject-aware validation; all statistics, basis and gamma from training data only", "MBGD"],
 ["5", "Fair comparison", "Same everything except the optimizer; learning rate tuned on validation", "GD"],
 ["6", "Honest evaluation", "Untouched official test subjects, Macro F1, Linear SVM and SVC baselines", "AdaGrad"]].forEach(([n, h, b, k], i) => {
  const x = 0.6 + (i % 3) * 4.1, y = 1.45 + Math.floor(i / 3) * 2.75;
  s.addShape(pres.shapes.OVAL, { x, y, w: 0.65, h: 0.65, fill: { color: COL[k] }, line: { color: COL[k] } });
  tb(s, n, { x, y, w: 0.65, h: 0.65, fontSize: 20, bold: true, color: "FFFFFF", align: "center", valign: "middle" });
  tb(s, h, { x: x + 0.85, y, w: 3.0, h: 0.65, fontSize: 18, bold: true, color: COL[k], valign: "middle" });
  tb(s, b, { x: x + 0.85, y: y + 0.75, w: 3.0, h: 1.5, fontSize: 14, valign: "top" });
});

// ================================================================= 5 Dataset
s = newSlide(); title(s, "Dataset: UCI HAR Using Smartphones", "Anguita et al., ESANN 2013 · Kaggle mirror: uciml/human-activity-recognition-with-smartphones");
table(s, [["Property", "Value"], ["Recording", "Samsung Galaxy S II on the waist, 50 Hz"], ["Window", "2.56 s, 50% overlap (128 readings)"], ["Features", "561 (time + frequency domain), scaled to [-1, 1]"],
  ["Official train", "7,352 windows · 21 subjects"], ["Official test", "2,947 windows · 9 other subjects"], ["Missing / duplicates", "0 / 0"], ["Licence", "CC BY 4.0 (UCI); cite the ESANN 2013 paper"]],
  0.6, 1.55, 7.0, [2.3, 4.7], { fs: 13, rowH: 0.5 });
card(s, 8.0, 1.55, 4.7, 2.2, "teal", "Verified, not assumed", "Row counts, subject lists, labels and the official partition were checked in code (describe_partition).", 13);
card(s, 8.0, 3.95, 4.7, 2.6, "gold", "Our split", `Official train → train (5,551 windows, 16 subjects) + validation (1,801 windows, 5 subjects) by GroupShuffleSplit, seed 42.\nOfficial test (2,947) used once, at the end.`, 13);

// ================================================================= 6 Classes
s = newSlide(); title(s, "Target classes", "Six activities, mildly imbalanced (largest / smallest = 1.4 in the training data)");
img(s, "eda_class_distribution.png", 0.6, 1.5, 7.6, 5.2);
[["Dynamic", "WALKING · WALKING_UPSTAIRS · WALKING_DOWNSTAIRS", "MBGD"], ["Static", "SITTING · STANDING · LAYING", "GD"], ["Hardest pair", "SITTING vs STANDING - almost the same posture signals", "Adam"]].forEach(([h, b, k], i) =>
  card(s, 8.5, 1.5 + i * 1.75, 4.2, 1.6, k, h, b, 13));

// ================================================================= 7 EDA
s = newSlide(); title(s, "Exploratory data analysis");
img(s, "eda_pca_projection.png", 0.6, 1.25, 6.4, 5.6);
s.addText(bullets(["No missing values, no duplicated rows, all 561 features numeric", "Features already in [-1, 1], but with different spreads → standardisation still needed for RBF distances",
  "Many features strongly correlated (derived from the same signals)", "PCA (visualisation only): dynamic vs static activities separate clearly; SITTING / STANDING overlap",
  "Outliers kept: real sensor readings; the RBF kernel is bounded in (0, 1]"], 17), { x: 7.3, y: 1.4, w: 5.4, h: 5.3, valign: "top", isTextBox: true });

// ================================================================= 8 Preprocessing
s = newSlide(); title(s, "Preprocessing without data leakage");
const box = (x, w, t, sub, col) => { s.addShape(pres.shapes.RECTANGLE, { x, y: 1.6, w, h: 1.0, fill: { color: col }, line: { color: "FFFFFF", width: 2 } });
  tb(s, [{ text: t, options: { bold: true, breakLine: true } }, { text: sub, options: { fontSize: 11 } }], { x, y: 1.6, w, h: 1.0, fontSize: 14, color: "FFFFFF", align: "center", valign: "middle" }); };
box(0.6, 6.1, "Train  5,551 windows", "16 subjects", COL.navy); box(6.7, 2.0, "Validation 1,801", "5 subjects", COL.teal); box(8.7, 4.0, "Test  2,947 windows", "9 official test subjects", COL.gold);
tb(s, "official training set (21 subjects)  →  GroupShuffleSplit(test_size = 0.2, random_state = 42)", { x: 0.6, y: 2.7, w: 8.1, h: 0.35, fontSize: 12, italic: true, color: COL.slate });
card(s, 0.6, 3.3, 3.9, 3.35, "navy", "Fitted on train only", "StandardScaler mean / std\nKernel basis: 2,000 class-stratified training samples\nRBF median heuristic gamma0", 14);
card(s, 4.72, 3.3, 3.9, 3.35, "teal", "Validation", "Kernel selection (gamma, lambda)\nLearning-rate selection for each optimizer\nNo test data involved", 14);
card(s, 8.84, 3.3, 3.86, 3.35, "gold", "Never used as features", "Subject id - only for grouping\nThe 9 test subjects are never seen during training or tuning", 14);

// ================================================================= 9 Intro SVM
s = newSlide(); title(s, "Introduction to Support Vector Machines");
card(s, 0.6, 1.5, 6.0, 2.5, "navy", "Idea", "Find the decision boundary that separates the classes with the LARGEST margin. Only the closest points - the support vectors - define it.", 15);
card(s, 0.6, 4.2, 6.0, 2.5, "teal", "Soft margin", "Real data overlap → allow violations, penalised by the hinge loss max(0, 1 − y·f(x)).\nλ (or C ≈ 1/(λN)) trades margin width against training errors.", 15);
eq(s, "min (λ/2)‖w‖² + (1/N) Σ max(0, 1 − yᵢf(xᵢ))", 7.0, 1.5, 5.7, 1.2, 16);
s.addText(bullets(["Hard margin: no violations allowed - only for perfectly separable data", "Soft margin: robust to noise and overlap (used here)", "Hinge loss is zero for points beyond the margin → sparse influence", "Convex problem → a single global optimum"], 17), { x: 7.0, y: 3.0, w: 5.7, h: 3.6, valign: "top", isTextBox: true });

// ================================================================= 10 Hyperplane & margin (diagram)
s = newSlide(); title(s, "Hyperplane and maximum margin");
s.addShape(pres.shapes.RECTANGLE, { x: 0.6, y: 1.4, w: 6.4, h: 5.3, fill: { color: "F8FAFC" }, line: { color: "E2E8F0" } });
const pts1 = [[1.2, 2.0], [1.7, 2.7], [1.1, 3.4], [2.2, 1.9], [2.5, 2.9], [1.6, 4.3], [3.59, 2.86]], pts2 = [[5.0, 4.6], [5.6, 3.6], [4.7, 5.6], [6.2, 5.0], [5.9, 2.7], [6.3, 4.0], [4.0, 4.86]];
pts1.forEach(([x, y]) => s.addShape(pres.shapes.OVAL, { x, y, w: 0.28, h: 0.28, fill: { color: COL.GD }, line: { color: COL.GD } }));
pts2.forEach(([x, y]) => s.addShape(pres.shapes.OVAL, { x, y, w: 0.28, h: 0.28, fill: { color: COL.Adam }, line: { color: COL.Adam } }));
s.addShape(pres.shapes.LINE, { x: 2.2, y: 1.6, w: 3.4, h: 4.9, line: { color: COL.navy, width: 3 }, flipH: true });
s.addShape(pres.shapes.LINE, { x: 1.3, y: 1.6, w: 3.4, h: 4.9, line: { color: COL.gray, width: 1.5, dashType: "dash" }, flipH: true });
s.addShape(pres.shapes.LINE, { x: 3.1, y: 1.6, w: 3.4, h: 4.9, line: { color: COL.gray, width: 1.5, dashType: "dash" }, flipH: true });
[[3.73, 3.0], [4.14, 5.0]].forEach(([cx, cy]) => s.addShape(pres.shapes.OVAL, { x: cx - 0.22, y: cy - 0.22, w: 0.44, h: 0.44, fill: { type: "none" }, line: { color: COL.gold, width: 2.5 } }));
tb(s, "wᵀx + b = 0", { x: 2.4, y: 6.15, w: 1.8, h: 0.4, fontSize: 13, bold: true, color: COL.navy });
tb(s, "margin = 2 / ‖w‖ (between the dashed lines)", { x: 0.8, y: 1.45, w: 3.6, h: 0.35, fontSize: 12, color: COL.gray });
tb(s, "◯ support vectors (on the margin)", { x: 4.4, y: 6.2, w: 2.5, h: 0.4, fontSize: 12, color: COL.gold, bold: true });
card(s, 7.4, 1.4, 5.3, 1.65, "navy", "Hyperplane", "A flat boundary wᵀx + b = 0 (a line in 2-D). w is its normal vector, b shifts it.", 14);
card(s, 7.4, 3.2, 5.3, 1.65, "gold", "Maximum margin", "Among all separating hyperplanes choose the widest street: margin 2/‖w‖ → minimise ‖w‖².", 14);
card(s, 7.4, 5.0, 5.3, 1.7, "Adam", "Support vectors", "The points on the margin. Moving any other point does not change the solution.", 14);

// ================================================================= 11 Linear vs non-linear
s = newSlide(); title(s, "Linear versus non-linear classification", "Synthetic 2-D illustration (not HAR data) - trained with our own code");
img(s, "03_synthetic_decision_boundaries.png", 0.6, 1.5, 8.2, 5.3);
s.addText(bullets(["Three concentric rings: no straight line can separate them", "Linear SVM: straight boundaries → most points misclassified", "RBF kernel SVM: curved boundaries follow the rings",
  "In HAR the boundaries between activities in 561-D space may also be curved - the kernel lets the model learn that"], 17), { x: 9.1, y: 1.6, w: 3.6, h: 5.0, valign: "top", isTextBox: true });

// ================================================================= 12 Kernel trick
s = newSlide(); title(s, "The kernel trick");
eq(s, "k(x, z) = ⟨φ(x), φ(z)⟩", 0.6, 1.5, 5.8, 1.0, 24);
card(s, 0.6, 2.8, 5.8, 3.9, "teal", "Example: degree-2 polynomial in 2-D", "φ(x) = (x₁², √2·x₁x₂, x₂²)\n⟨φ(x), φ(z)⟩ = (x₁z₁ + x₂z₂)² = (xᵀz)²\n\n→ the kernel gives the inner product in the 3-D feature space with ONE dot product in 2-D. For RBF the feature space is infinite-dimensional, so φ can never be built - only k is computed.", 14);
s.addText(bullets(["SVM training and prediction need only inner products → replace them by k(x, z)", "Non-linear boundaries without constructing high-dimensional features",
  "Valid kernel ⇔ Gram matrix positive semidefinite (Mercer)", "Cost moves from the feature dimension to the number of samples (N × M kernel matrix)"], 18), { x: 6.8, y: 1.6, w: 5.9, h: 5.0, valign: "top", isTextBox: true });

// ================================================================= 13 Feature space / Gram
s = newSlide(); title(s, "Kernel feature space, Gram matrix and RKHS");
img(s, "01_gram_matrix_rbf.png", 0.6, 1.3, 5.6, 5.6);
card(s, 6.6, 1.4, 6.1, 1.65, "navy", "Gram matrix G", "Gᵢⱼ = k(zᵢ, zⱼ): all pairwise similarities. Bright diagonal blocks = windows of the same activity are similar.", 13);
card(s, 6.6, 3.2, 6.1, 1.65, "teal", "Positive semidefinite", `aᵀGa ≥ 0 for every a. Our RBF G: symmetric, diagonal = 1, smallest eigenvalue ${f3(KC.find(k => k.kernel === "rbf").min_eig_G)} > 0.`, 13);
card(s, 6.6, 5.0, 6.1, 1.75, "Adam", "RKHS + Representer Theorem", "f(x) = Σⱼ aⱼ k(x, zⱼ) with ‖f‖² = aᵀGa. The optimal f is a weighted sum of kernels on the training points → optimise A instead of an infinite w.", 13);

// ================================================================= 14 Linear kernel
s = newSlide(); title(s, "Linear kernel", "k(x, z) = xᵀz");
eq(s, "k(x, z) = xᵀz        f(x) = Σⱼ aⱼ xᵀzⱼ = wᵀx,  w = Zᵀa", 0.6, 1.5, 12.1, 1.0, 20);
card(s, 0.6, 2.8, 3.9, 3.9, "navy", "Relationship to the Linear SVM", "Feature space = input space. With RKHS norm ‖f‖² = ‖w‖² this is exactly the Linear SVM of the previous assignment - same boundary type: a hyperplane.", 14);
const kLin = KC.find(k => k.kernel === "linear");
card(s, 4.72, 2.8, 3.9, 3.9, "gold", "In coefficient space", `Trained through A (2,000 × 6) the linear kernel is badly conditioned (huge, correlated kernel values): test Macro F1 only ${f4(kLin.test_macro_f1)}, objective ${f3(kLin.train_objective)}.`, 14);
card(s, 8.84, 2.8, 3.86, 3.9, "MBGD", "In the primal", `The same model trained directly with W (561 × 6) - our controlled Linear SVM - reached test Macro F1 ${f4(LVK[LVK_COLS[0]]["Macro F1"])}. Parametrisation matters for first-order optimisers.`, 14);

// ================================================================= 15 Polynomial
s = newSlide(); title(s, "Polynomial kernel", "k(x, z) = (γ·xᵀz + r)ᵈ");
eq(s, "k(x, z) = (γ xᵀz + r)ᵈ      γ = 1/561,  r = 1,  d ∈ {2, 3}", 0.6, 1.5, 12.1, 1.0, 20);
card(s, 0.6, 2.8, 3.9, 3.9, "navy", "Parameters", "γ scales the inner product\nr ≥ 0 weights the lower-order terms\nd = degree: interactions of up to d features (e.g. x₁·x₂·x₇ for d = 3)", 14);
card(s, 4.72, 2.8, 3.9, 3.9, "Adam", "Risk", "Number of implicit features grows combinatorially with d → overfitting and numerical growth of (γxᵀz + r)ᵈ. PSD for integer d ≥ 1, γ > 0, r ≥ 0.", 14);
const p2 = KC[1], p3 = KC[2];
card(s, 8.84, 2.8, 3.86, 3.9, "MBGD", "Our results (supplementary)", `d = 2: val ${f4(p2.val_macro_f1)} · test ${f4(p2.test_macro_f1)}\nd = 3: val ${f4(p3.val_macro_f1)} · test ${f4(p3.test_macro_f1)}\nd = 3 fits validation better but generalises worse - a sign of overfitting.`, 14);

// ================================================================= 16 RBF
s = newSlide(); title(s, "RBF (Gaussian) kernel - the primary kernel", "k(x, z) = exp(−γ‖x − z‖²)");
img(s, "01_rbf_similarity.png", 0.6, 1.5, 6.5, 3.7);
const best = SEL.reduce((a, b) => (b.val_macro_f1 > a.val_macro_f1 ? b : a));
s.addText(bullets([`Similarity = 1 for identical windows, decays with Euclidean distance (local influence)`, `Small γ: wide bumps, smooth near-linear boundaries → underfitting`, `Large γ: narrow bumps, G → I, every point its own island → overfitting`,
  `Median heuristic γ₀ = 1/median‖xᵢ − xⱼ‖² = ${H.gamma0.toExponential(3)}`, `Selected on validation: γ = ${best.gamma_multiplier} × γ₀ = ${H.gamma.toExponential(3)}`, `Distances need standardised features - otherwise large-range features dominate`], 14), { x: 7.4, y: 1.5, w: 5.3, h: 5.2, valign: "top", isTextBox: true });
eq(s, "‖x − z‖² = ‖x‖² + ‖z‖² − 2xᵀz   (clipped at 0)", 0.6, 5.5, 6.5, 0.9, 16);

// ================================================================= 17 Kernelized multi-class SVM
s = newSlide(); title(s, "Kernelized direct multi-class SVM");
eq(s, "fₖ(x) = Σⱼ Aⱼₖ · k(x, zⱼ) + bₖ        S = H·A + b        ŷ = argmaxₖ Sᵢₖ", 0.6, 1.4, 12.1, 1.0, 19);
table(s, [["Strategy", "Models", "Prediction", "Where"], ["One-vs-Rest", "K = 6", "largest score", "many linear libraries"], ["One-vs-One", "K(K−1)/2 = 15", "voting", "scikit-learn SVC (libsvm)"], ["Direct Crammer-Singer", "1 joint problem", "largest score", "this project"]],
  0.6, 2.7, 7.4, [2.3, 1.6, 1.6, 1.9], { fs: 13, rowH: 0.48, hl: (i) => i === 2 });
card(s, 8.3, 2.7, 4.4, 4.0, "teal", "Shapes in our model", "H: 5,551 × 2,000 (train-to-basis kernel)\nG: 2,000 × 2,000 (basis Gram)\nA: 2,000 × 6 coefficients\nb: 6 biases\nS: 5,551 × 6 class scores", 14);
tb(s, "Mode B - restricted basis: the Representer expansion uses 2,000 class-stratified training samples (identical for all optimizers). Not the exact full-basis SVM; the exact M = N case is measured on slide 27.", { x: 0.6, y: 5.0, w: 7.4, h: 1.6, fontSize: 14, italic: true, color: COL.slate });

// ================================================================= 18 Objective
s = newSlide(); title(s, "Mathematical objective and subgradients");
eq(s, "J(A, b) = (λ/2)·Tr(AᵀGA) + (1/N) Σᵢ max(0, 1 + maxⱼ≠yᵢ Sᵢⱼ − Sᵢ,yᵢ)", 0.6, 1.4, 12.1, 1.0, 20);
card(s, 0.6, 2.7, 4.0, 4.0, "navy", "Crammer-Singer hinge", "The true class must beat the BEST wrong class by margin 1. The inner max excludes the true class. Zero loss when the margin is satisfied.", 14);
card(s, 4.75, 2.7, 4.0, 4.0, "teal", "RKHS regularisation", "Tr(AᵀGA) = Σₖ ‖fₖ‖²_H - the kernel norm, not ‖A‖². λ = 1e-4 (selected). Bias not regularised. G ⪰ 0 ⇒ J is convex.", 14);
card(s, 8.9, 2.7, 3.8, 4.0, "Adam", "Subgradients", "D: −1 at yᵢ, +1 at the strongest wrong class (if violated), else 0; ties → smallest index.\n∂J/∂A = HᵀD/N + λGA\n∂J/∂b = Σᵢ Dᵢ / N\nChecked numerically (tests).", 13);

// ================================================================= 19-23 optimizer slides
const OPT_INFO = {
  GD: ["Gradient Descent (full batch)", "θ ← θ − η·∇J(θ)", "All 5,551 windows per update → 1 update per epoch. Exact subgradient, deterministic; but only 100 updates in 100 epochs, and the ill-conditioned kernel problem makes progress slow; too large steps oscillate (lr = 3 blows up).", ["01_objective_parts.png", "01_lr_curves.png"]],
  SGD: ["Stochastic Gradient Descent", "θ ← θ − η·(∇Lᵢ(θ) + λGA)", "One window per update → 5,551 updates per epoch, reproducible shuffling. Noisy but cheap. Exact rotated update C = UᵀA makes each step O(MK) instead of O(M²K) - verified identical to naive SGD.", ["02_sgd_within_epoch.png", "02_lr_curves.png"]],
  MBGD: ["Mini-Batch Gradient Descent", "θ ← θ − η·(1/|B|) Σᵢ∈B ∇Lᵢ(θ) − η·λGA", "Batches of 64 → 87 updates per epoch (last batch has 47 samples). Less noise than SGD, vectorised matrix products, more updates than GD.", ["03_objective_parts.png", "03_lr_curves.png"]],
  AdaGrad: ["AdaGrad", "s ← s + g²     θ ← θ − η·g / (√s + ε)", "Own step size for every coefficient; accumulator s (not the Gram matrix!) only grows → effective learning rate decays. Batch 64, ε = 1e-8.", ["04_adagrad_effective_lr.png", "04_lr_curves.png"]],
  Adam: ["Adam", "m ← β₁m + (1−β₁)g,   v ← β₂v + (1−β₂)g²\nθ ← θ − η·m̂ / (√v̂ + ε)", "Momentum + per-parameter scaling + bias correction m̂ = m/(1−β₁ᵗ), v̂ = v/(1−β₂ᵗ). β₁ = 0.9, β₂ = 0.999, ε = 1e-8, batch 64, global counter t.", ["05_objective_parts.png", "05_lr_curves.png"]],
};
ORDER.forEach(o => {
  const [t, rule, desc, fig] = OPT_INFO[o]; const r = C_[o];
  s = newSlide(); title(s, `${t}`, `Notebook 0${ORDER.indexOf(o) + 1} · selected learning rate η = ${r.learning_rate}`);
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0.12, w: 0.18, h: 7.38, fill: { color: COL[o] }, line: { color: COL[o] } });
  eq(s, rule, 0.6, 1.5, 6.4, 1.0, o === "Adam" ? 15 : 17);
  tb(s, desc, { x: 0.6, y: 2.7, w: 6.4, h: 1.6, fontSize: 16, valign: "top" });
  stat(s, 0.6, 4.35, 2.0, pc(r.accuracy), "test accuracy", o);
  stat(s, 2.75, 4.35, 2.1, f4(r.macro_f1), "test Macro F1", o);
  stat(s, 4.75, 4.35, 2.4, Number(r.updates).toLocaleString("en-US"), "parameter updates", o);
  tb(s, `Validation Macro F1 ${f4(r.val_macro_f1)} · final J ${f4(r.train_objective)} · training ${f1(r.training_time_s)} s`, { x: 0.6, y: 5.85, w: 6.4, h: 0.4, fontSize: 13, color: COL.slate });
  img(s, fig[0], 7.3, 1.4, 5.4, 2.7);
  img(s, fig[1], 7.3, 4.2, 5.4, 2.7);
});

// ================================================================= 24 Methodology
s = newSlide(); title(s, "Experimental methodology");
table(s, [["Kept identical", "Value"], ["Kernel / basis", `RBF, γ = ${H.gamma.toExponential(2)}, M = 2,000 (Mode B)`], ["Regularisation", "λ = 1e-4"], ["Initial parameters", "A = 0, b = 0"], ["Epoch budget", "100 epochs"], ["Data", "same split, scaling, shuffling seed 42"], ["Selection metric", "validation Macro F1"]],
  0.6, 1.45, 6.2, [2.2, 4.0], { fs: 13, rowH: 0.48 });
table(s, [["Optimizer", "LR grid (5 values, 100 epochs each)", "Chosen"], ...ORDER.map(o => [NAME[o], (S.lr_search[o] || []).map(r => r.lr).join(", "), String(C_[o].learning_rate)])],
  7.1, 1.45, 5.6, [1.5, 3.1, 1.0], { fs: 12, rowH: 0.48 });
tb(s, "Kernel configuration chosen BEFORE the optimizer comparison: γ ∈ {0.5,1,2,4,8}×γ₀ and λ ∈ {1e-5 … 1e-2}, one fixed solver (Adam, lr 1e-3, 30 epochs). An epoch is not an update: GD makes 100 updates, SGD 555,100. Test set used only once at the end.", { x: 0.6, y: 5.1, w: 12.1, h: 1.5, fontSize: 14, italic: true, color: COL.slate });

// ================================================================= 25 Metrics + confusion
s = newSlide(); title(s, "Evaluation metrics and confusion matrices", "Independent test set: 2,947 windows from 9 unseen subjects");
img(s, "05_confusion_matrices_all.png", 0.6, 1.45, 12.1, 3.0);
s.addText(bullets(["Accuracy, macro / weighted precision-recall-F1, per-class metrics - computed by hand from the confusion matrix and verified against scikit-learn", "Macro F1 = primary metric (every activity counts equally)",
  "ROC-AUC: One-vs-Rest from raw scores (rank-based); no log-loss - SVM scores are not probabilities", "Main error for all optimizers: SITTING ↔ STANDING; LAYING is almost perfect"], 16), { x: 0.6, y: 4.7, w: 12.1, h: 2.1, valign: "top", isTextBox: true });

// ================================================================= 26 Optimizer comparison
s = newSlide(); title(s, "Optimizer comparison (test set)");
const bestF1 = argmax("macro_f1"), bestAcc = argmax("accuracy"), bestVal = argmax("val_macro_f1"), lowJ = argmin("train_objective"), fast = argmin("training_time_s");
table(s, [["Optimizer", "LR", "Updates", "Accuracy", "Macro P", "Macro R", "Macro F1", "Val F1", "Hinge", "Reg", "J", "Train s", "Pred ms"],
  ...ORDER.map(o => { const r = C_[o]; return [NAME[o], r.learning_rate, Number(r.updates).toLocaleString("en-US"), pc(r.accuracy), f4(r.macro_precision), f4(r.macro_recall), f4(r.macro_f1), f4(r.val_macro_f1), f4(r.train_hinge), f4(r.regularization), f4(r.train_objective), f1(r.training_time_s), (1000 * r.prediction_time_s).toFixed(1)]; })],
  0.6, 1.4, 12.1, [1.55, 0.6, 0.95, 0.95, 0.85, 0.85, 0.9, 0.85, 0.85, 0.75, 0.85, 0.8, 0.8], { fs: 11, hs: 11, rowH: 0.42,
    bold: (i, j) => (j === 6 && ORDER[i] === bestF1) || (j === 3 && ORDER[i] === bestAcc) || (j === 7 && ORDER[i] === bestVal) || (j === 10 && ORDER[i] === lowJ) || (j === 11 && ORDER[i] === fast),
    hl: (i, j) => (j === 6 && ORDER[i] === bestF1) || (j === 7 && ORDER[i] === bestVal) });
[["Best by validation (selection rule)", NAME[bestVal], "Adam"], ["Highest test accuracy / Macro F1", `${NAME[bestAcc]} / ${NAME[bestF1]}`, "AdaGrad"], ["Lowest training objective", NAME[lowJ], "SGD"], ["Fastest training", NAME[fast], "GD"]].forEach(([h, b, k], i) =>
  card(s, 0.6 + i * 3.05, 4.25, 2.9, 1.25, k, b, h, 12));
tb(s, `Top four within ${(100 * (C_[bestF1].macro_f1 - Math.min(...["SGD", "MBGD", "AdaGrad", "Adam"].map(o => C_[o].macro_f1)))).toFixed(1)} Macro-F1 points; GD lags because 100 epochs = only 100 updates. Validation and test winners differ → the top optimizers are statistically close.`, { x: 0.6, y: 5.7, w: 12.1, h: 0.9, fontSize: 14, italic: true, color: COL.slate });

// ================================================================= 27 Hyperparameter analysis
s = newSlide(); title(s, "Kernel hyperparameter analysis (validation)");
img(s, "01_gamma_sensitivity.png", 0.6, 1.35, 4.1, 2.7);
img(s, "01_lambda_sensitivity.png", 4.75, 1.35, 4.1, 2.7);
img(s, "02_basis_size_analysis.png", 0.6, 4.2, 8.25, 2.6);
const ex = BS.find(b => b.exact), b2 = BS.find(b => b.basis_M === 2000);
card(s, 9.1, 1.35, 3.6, 2.6, "Adam", "γ and λ", `Best: γ = ${best.gamma_multiplier}×γ₀, λ = ${best.lam} (val F1 ${f4(best.val_macro_f1)}). γ = 0.5×γ₀ underfits; λ = 1e-2 over-regularises.`, 13);
card(s, 9.1, 4.1, 3.6, 2.7, "teal", "Basis size M", `M = 2,000: val F1 ${f4(b2.val_macro_f1)}, ${f1(b2.train_time_s)} s\nExact M = ${Number(ex.basis_M).toLocaleString("en-US")}: ${f4(ex.val_macro_f1)}, ${f1(ex.train_time_s)} s\n(30 epochs, Adam)`, 13);

// ================================================================= 28 Linear vs Kernel
s = newSlide(); title(s, "Linear SVM versus Kernel SVM", "Same data, split, scaling and epoch budget · Kernel SVM = optimizer with the best validation Macro F1");
const rowsL = ["Accuracy", "Macro Precision", "Macro Recall", "Macro F1", "Training Time (s)", "Prediction Time (s)", "Model Complexity", "Decision Boundary", "Model Memory"];
const fmt = (k, v) => (["Accuracy"].includes(k) ? pc(v) : (["Macro Precision", "Macro Recall", "Macro F1"].includes(k) ? f4(v) : (k.includes("Time") ? Number(v).toFixed(3) : v)));
table(s, [["Metric", ...LVK_COLS], ...rowsL.map(k => [k, ...LVK_COLS.map(c => fmt(k, LVK[c][k]))])], 0.6, 1.5, 8.6, [2.0, 2.0, 2.4, 2.2], { fs: 11, hs: 11, rowH: 0.48 });
card(s, 9.5, 1.5, 3.2, 5.2, "gold", "Kernel ≠ always better", "On HAR the engineered 561 features are already almost linearly separable.\nThe test subjects are new people.\nOur kernel model is limited by M = 2,000 and 100 epochs of first-order training.\nThe kernel model needs the basis at prediction time → slower and larger.", 13);

// ================================================================= 29 Limitations
s = newSlide(); title(s, "Limitations and future improvements");
card(s, 0.6, 1.5, 6.0, 5.2, "Adam", "Limitations", "• Restricted basis (M = 2,000), not the exact full-basis SVM\n• Fixed 100-epoch budget favours methods with many updates; GD is far from convergence\n• Kernel selection used one solver (Adam, 30 epochs)\n• Only 5 validation subjects → noisy model selection (validation vs test winners differ)\n• No learning-rate schedules; single random seed\n• Sigmoid kernel is not PSD → exploratory only", 14);
card(s, 6.84, 1.5, 5.86, 5.2, "MBGD", "Future improvements", "• Larger basis or Nystroem features (better conditioning, same model class)\n• Learning-rate decay / early stopping on validation\n• Leave-subjects-out cross-validation and several seeds\n• Probability calibration (Platt scaling) for ROC / log-loss\n• Raw-signal models (1-D CNN / LSTM) as comparison\n• GPU / float32 kernels for the exact M = N solution", 14);

// ================================================================= 30 Conclusion
s = newSlide(); title(s, "Conclusion and references");
s.addText(bullets([`Genuine kernelized direct multi-class SVM (RBF, Crammer-Singer, RKHS norm) and five hand-written optimizers on the same objective`,
  `Best by validation: ${NAME[bestVal]}; best test accuracy ${NAME[bestAcc]} (${pc(C_[bestAcc].accuracy)}); GD ${pc(C_.GD.accuracy)} after 100 updates`,
  `Linear SVM ${pc(LVK[LVK_COLS[0]]["Accuracy"])} vs our Kernel SVM ${pc(LVK[LVK_COLS[1]]["Accuracy"])} and scikit-learn SVC ${pc(LVK[LVK_COLS[2]]["Accuracy"])}: the kernel is not automatically better`,
  "Lesson: the optimizer changes speed and stability; the parametrisation and conditioning matter as much as the model"], 18), { x: 0.6, y: 1.4, w: 12.1, h: 3.0, valign: "top", isTextBox: true });
tb(s, "References: Anguita et al. (ESANN 2013) · Cortes & Vapnik (1995) · Crammer & Singer (JMLR 2001) · Schölkopf, Herbrich & Smola (COLT 2001) · Schölkopf & Smola, Learning with Kernels (2002) · Williams & Seeger (NIPS 2001) · Rahimi & Recht (NIPS 2007) · Duchi, Hazan & Singer (JMLR 2011) · Kingma & Ba (ICLR 2015) · Chang & Lin, LIBSVM (2011)",
  { x: 0.6, y: 4.6, w: 12.1, h: 1.3, fontSize: 12, color: COL.slate });
tb(s, "Thank you - questions?", { x: 0.6, y: 6.0, w: 12.1, h: 0.7, fontSize: 26, bold: true, color: COL.navy, fontFace: "Georgia" });

if (slideNo !== NOTES.length) throw new Error(`slides ${slideNo} != notes ${NOTES.length}`);
pres.writeFile({ fileName: OUT }).then(() => console.log("wrote", OUT, slideNo, "slides"));
