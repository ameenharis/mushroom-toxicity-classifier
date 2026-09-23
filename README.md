# Mushroom Toxicity Classifier — an interpretable decision tree

Classifying mushrooms as **edible or poisonous** from observable physical characteristics, using a deliberately shallow decision tree so the output can be read as a set of field rules rather than a black box.

The rules are meant to be usable by a person identifying mushrooms, where the two kinds of error are not remotely equivalent.

---

## The problem

8,124 mushroom samples across 23 species, each described by 22 categorical features: cap shape and colour, odour, gill characteristics, stalk properties, habitat, and so on. The task is to predict toxicity from those observable traits.

Two constraints shape the whole approach:

**Interpretability is a requirement, not a preference.** The output needs to be something a person can apply in a field, without a model running. That rules out anything opaque and argues for a shallow tree whose branches can be written down as rules.

**The errors are asymmetric.** Classifying a poisonous mushroom as edible could kill someone. Classifying an edible mushroom as poisonous means a missed meal. Optimising overall accuracy treats these as equivalent, which is the wrong objective — recall on the poisonous class matters far more.

## Approach

**Data preparation.** Checked duplicates and nulls. The `stalk-root` column encodes missing values as `?` rather than leaving them blank, so those were handled explicitly. Dropped `veil-type`, which takes a single constant value across all 8,124 rows and therefore carries no information.

**Feature selection.** A correlation heatmap across all features against the target pointed to odour, stalk surface above ring, stalk surface below ring and a handful of others as the strongest signals. A count plot of odour against class made the pattern visible directly: foul, fishy and spicy odours are almost exclusively poisonous.

**Confirming it statistically.** Rather than trusting the visual, a Chi-square test of independence between odour and class:

- H₀: class is independent of odour
- χ² = 7659.73, dof = 8, **p < 0.001**
- Null rejected — odour is a genuine predictor, not an artefact of the plot

**Model.** A decision tree classifier with `max_depth=3`, features one-hot encoded, target mapped to 0 (edible) / 1 (poisonous). Stratified 70/30 train-test split to preserve class balance.

The depth limit is the deliberate part. A deeper tree scores better but stops being readable; three levels produce a rule set short enough to put on a page.

## Results

Overall accuracy on the held-out test set: **98%**.

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Edible | 1.00 | 0.97 | 0.99 | 1,263 |
| **Poisonous** | 0.97 | **1.00** | 0.98 | 1,175 |

The number that matters is **recall on the poisonous class**. The confusion matrix shows 1,171 of 1,175 poisonous mushrooms correctly identified and **4 misclassified as edible** — a recall of 99.7%, which the classification report rounds to 1.00. Those 4 are the errors that could hurt someone, so they matter more than the headline accuracy.

The other errors fall on the safe side: 33 edible mushrooms were flagged as poisonous (edible recall 0.97). For a field guide, erring that way round is the right trade-off.

Feature importances confirm odour dominates the model's decisions, consistent with the Chi-square result. The full tree is plotted in the notebook and can be transcribed directly into rules.

## Running it

```bash
pip install pandas matplotlib seaborn scipy scikit-learn
```

The notebook pulls the dataset straight from the UCI repository by default. To use a local copy:

```bash
export MUSHROOM_DATA=data/agaricus-lepiota.data
```

Then run `mushroom_classifier.ipynb` top to bottom.

## What I would do differently

- **Push the false negatives to zero.** A depth-3 tree still lets 4 poisonous mushrooms through. Weighting the poisonous class more heavily, or lowering the decision threshold, would trade a few more false alarms for fewer dangerous misses.
- **Cross-validate rather than relying on one split.** A single stratified split at depth 3 is likely stable, but k-fold would confirm the rules aren't an artefact of this particular partition.
- **Test depth sensitivity explicitly.** Depth 3 was chosen for readability; showing how accuracy and poisonous-recall change across depths 2–6 would make the trade-off between interpretability and performance explicit rather than asserted.
- **Note the dataset's limits.** These are hypothetical samples drawn from a field guide, not observations in the wild, so the clean separability here is unlikely to survive contact with real foraging conditions.

## Files

- `mushroom_classifier.ipynb` — full analysis: preparation, exploratory analysis, Chi-square test, model, evaluation, plotted tree

Data: [UCI Machine Learning Repository — Mushroom](https://archive.ics.uci.edu/dataset/73/mushroom)
