# Experimentation — Interpretation and Recommendations

This document explains how the model behaves when predicting the activity of real compounds — including **anti‑tuberculosis** drugs (isoniazid, pyrazinamide, ethambutol, ethionamide, rifampicin, bedaquiline) and a comparison set of **non‑TB compounds** (aspirin, ibuprofen, caffeine, lidocaine, metformin, etc.).

The same compounds were evaluated using two different trained models, each trained on a different PubChem datatable. 

---

## 🎯 Aim

The goal of this project is to understand how datatable context shapes model predictions**.

Using the same molecular structures, its observed that:
- The model predicts almost everything Active
  OR
- Predicts everything Inactive

This can emphasize the importance of considering training context when interpreting model predictions.

---

## 📊 The Two Datatables

### FBA Datatable — Mostly Active

In the FBA dataset, most compounds are labeled **Active**. When the model is trained on this data, it learns a strong prior that activity is common.

When TB drugs (like isoniazid or rifampicin) and non‑TB drugs (like aspirin or caffeine) are passed through the model, prediction presents:

- Nearly all compounds being predicted as **Active**
- Confidence scores are often high
- Inhibition values remain low and tightly clustered

This doesn't mean all of the compounds are biologically active in the same way. It means the model has learned what a typical compound in the FBA dataset looks like.

---

### MenB Datatable — All Inactive

The MenB dataset represents the opposite: every compound is confirmed **Inactive**.

When the model is trained on this dataset it learns that inactivity is the default.

When the same TB and non‑TB compounds are evaluated:

- Every compound is predicted **Inactive**
- Confidence is again very high
- Inhibition values shift upward to a different baseline

The chemistry did not change but the **training context** did.

---

## 🔍 To Infer

These results demonstrate that the weighted model mostly learns dataset bias, not general biological truth.

The model tries to understand whether a molecule resembles what it usually sees in the datatable, rather then whether the molecule is an anti-TB drug.

This can explain why known TB drugs and common non‑TB drugs can have similar predictions.

---

## ⚠️ Confidence, Weighted Loss, and Misleading Certainty

Classification confidence in the model reflects label frequency rather than biological certainty.

Using a weighted loss (`pos_weight`) can make this worse:
- It pushes the model toward one class
- But it also misrepresents confidence values

As a result, probabilities near 0.99 may simply indicate dataset imbalance, not strong evidence. In this case, removing weighted loss and fixing class balance directly would likely lead to more interpretable outputs.

---

## 🧪 Inhibition Change Between Models

The inhibition head is trained only in the context of predicted activity. Because of this:

- In mostly‑Active datasets, inhibition values stay low
- In all‑Inactive datasets, inhibition values shift to a higher baseline

Inhibition strength is therefore:
- Table‑specific
- Model‑dependent
- Best used for relative comparisons within the same datatable, not across models

---

## ❓ Recommendations

### Option 1 — Enforce a True 50/50 Active–Inactive Ratio

If the table is mostly active, the model will predict as so because it will choose the safest answer by default. If the table presents a 50/50 split, the model will have no safer option. It will instead focus on ring systems and atomic environments.

By enforcing a **50% Active / 50% Inactive** ratio:
- The model can no longer rely on a default answer
- Decision boundaries become meaningful
- Confidence scores regain interpretability

This approach is would be better than using weighted loss, which can distort probabilities even when accuracy improves.

---

### Option 2 — Multi‑AID Training (Recommended for Biology)

A robust approach can be to train one shared model across multiple datatables.

This forces the model to encounter situations where:
- The same compound is Active in one table
- Inactive in another

From here the model is able to learn chemistry‑driven selectivity rather than dataset priors. 
