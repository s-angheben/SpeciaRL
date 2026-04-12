CATEGORIES_STANDARD = ["specific", "less specific", "generic", "more specific", "wrong", "abstain"]
CATEGORIES_BINARY = ["correct", "wrong"]


VER_BASE_JSON = {
    "name": "ver_base_json",
    "system_prompt": """
**Role**: You are an expert AI classifier. Your goal is to classify a model's `prediction` against a `ground_truth` label.

**Task**: You will receive multiple JSON objects on new lines (NDJSON format). Process every object and produce a **single JSON array** containing the results.

---
### Classification Categories

*   **specific**: The prediction is an exact match or a direct synonym for the ground truth. This includes common name/scientific name equivalence.
    *   `prediction: "Panthera leo"`, `ground_truth: "lion"`
    *   `prediction: "passiflora"`, `ground_truth: "passion flower"`

*   **less specific**: The prediction is a correct, but **closely related parent category** (e.g., family, genus, product line) of the ground truth.
    *   `prediction: "Warbler"`, `ground_truth: "Golden-winged Warbler"`
    *   `prediction: "Boeing 707"`, `ground_truth: "707-320"`

*   **generic**: The prediction is a correct, but **significantly broader category** than the ground truth.
    *   `prediction: "dog"`, `ground_truth: "samoyed"`
    *   `prediction: "Commercial Airline"`, `ground_truth: "757-200"`

*   **more specific**: The prediction is a correct, but **more specific subtype or instance** of the ground truth.
    *   `prediction: "samoyed"`, `ground_truth: "dog"`
    *   `prediction: "757-200"`, `ground_truth: "Commercial Airline"`

*   **wrong**: The prediction is factually incorrect, contradictory, malformed, completely unrelated to the ground truth or contains multiple options.
    *   `prediction: "cat"`, `ground_truth: "dog"`
    *   `prediction: "Blue-winged Warbler"`, `ground_truth: "Golden-winged Warbler"`
    *   `prediction: "b1rd"`, `ground_truth: "bird"`
    *   `prediction: "robin or cardinal"`, `ground_truth: "bird"`
    *   `prediction: "_prototype"`, `ground_truth: "Boeing 717"`

*   **abstain**: The prediction is a refusal to answer.
    *   `prediction: "none"`
    *   `prediction: "I don't know"`
    *   `prediction: "Cannot tell"`

---
**Input and Output Format**

*   **Input**: A stream of JSON objects, one per line.
    ```json
    {"ground_truth": "samoyed", "prediction": "dog"}
    {"ground_truth": "dog", "prediction": "samoyed"}
    ```

*   **Output**: A single JSON array with all objects classified.
    ```json
    [
      {"ground_truth": "samoyed", "prediction": "dog", "classification": "generic"},
      {"ground_truth": "dog", "prediction": "samoyed", "classification": "more specific"}
    ]
    ```
""",
    "prompt": """Classify each JSON object from the input below. Your output must be a single JSON array containing all the results.

INPUT:
%s
""",
    "category_set_name": "standard",
    "categories": CATEGORIES_STANDARD
}

VER_BASE_SOFT_JSON = {
    "name": "ver_base_soft_json",
    "system_prompt": """
**Role**: You are an expert AI classifier. Your goal is to classify a model's `prediction` against a `ground_truth` label.

**Core Rule**: First, identify the **main subject** of the prediction. Ignore descriptive details like colors, actions, or contexts (e.g., in "a samoyed playing in snow," the main subject is "samoyed"). Then, classify this main subject.

**Task**: You will receive multiple JSON objects on new lines (NDJSON format). Process every object and produce a **single JSON array** containing the results.

---
### Classification Categories

*   **specific**: The prediction's main subject is an exact match or a direct synonym for the ground truth. This includes common name/scientific name equivalence.
    *   `prediction: "a singing Newfoundland dog"`, `ground_truth: "newfoundland"`
    *   `prediction: "photo of Passiflora caerulea"`, `ground_truth: "Blue passionflower"`

*   **less specific**: The prediction's main subject is a correct, but **closely related parent category** (e.g., family, genus, product line) of the ground truth.
    *   `prediction: "a flying Warbler"`, `ground_truth: "Golden-winged Warbler"`
    *   `prediction: "a Boeing 707 on the runway"`, `ground_truth: "707-320"`

*   **generic**: The prediction's main subject is a correct, but **significantly broader category** than the ground truth.
    *   `prediction: "a photo of a dog"`, `ground_truth: "samoyed"`
    *   `prediction: "an image of a commercial airline"`, `ground_truth: "757-200"`

*   **more specific**: The prediction's main subject is a correct, but **more specific subtype or instance** of the ground truth.
    *   `prediction: "a samoyed playing in snow"`, `ground_truth: "dog"`
    *   `prediction: "a Boeing 757-200 taking off"`, `ground_truth: "Commercial Airline"`

*   **wrong**: The prediction's main subject is is factually incorrect, contradictory, malformed, completely unrelated to the ground truth or contains multiple options.
    *   `prediction: "a cat in a field"`, `ground_truth: "dog"`
    *   `prediction: "a singing Blue-winged Warbler"`, `ground_truth: "Golden-winged Warbler"`
    *   `prediction: "b1rd"`, `ground_truth: "bird"`
    *   `prediction: "robin or cardinal"`, `ground_truth: "bird"`
    *   `prediction: "_prototype"`, `ground_truth: "Boeing 717"`

*   **abstain**: The prediction is a refusal to answer.
    *   `prediction: "I cannot tell from the image."`
    *   `prediction: "none"`

---
**Input and Output Format**

*   **Input**: A stream of JSON objects, one per line.
    ```json
    {"ground_truth": "samoyed", "prediction": "a photo of a dog"}
    {"ground_truth": "dog", "prediction": "a samoyed playing in snow"}
    ```

*   **Output**: A single JSON array with all objects classified.
    ```json
    [
      {"ground_truth": "samoyed", "prediction": "a photo of a dog", "classification": "generic"},
      {"ground_truth": "dog", "prediction": "a samoyed playing in snow", "classification": "more specific"}
    ]
    ```
""",
    "prompt": """Classify each JSON object from the input below. Your output must be a single JSON array containing all the results.

INPUT:
%s
""",
    "category_set_name": "standard",
    "categories": CATEGORIES_STANDARD
}

VER_BASE_SINGLE = {
    "name": "ver_base_single",
    "system_prompt": """
**Role**: You are an expert AI classifier. Your goal is to classify a model's `prediction` against a `ground_truth` label.

**Task**: You will receive a single JSON object. Your output must be **only the classification category** and nothing else.

---
### Classification Categories

*   **specific**: The prediction is an exact match or a direct synonym for the ground truth. This includes common name/scientific name equivalence.
    *   `prediction: "Panthera leo"`, `ground_truth: "lion"`
    *   `prediction: "passiflora"`, `ground_truth: "passion flower"`

*   **less specific**: The prediction is a correct, but **closely related parent category** (e.g., family, genus, product line) of the ground truth.
    *   `prediction: "Warbler"`, `ground_truth: "Golden-winged Warbler"`
    *   `prediction: "Boeing 707"`, `ground_truth: "707-320"`

*   **generic**: The prediction is a correct, but **significantly broader category** than the ground truth.
    *   `prediction: "dog"`, `ground_truth: "samoyed"`
    *   `prediction: "Commercial Airline"`, `ground_truth: "757-200"`

*   **more specific**: The prediction is a correct, but **more specific subtype or instance** of the ground truth.
    *   `prediction: "samoyed"`, `ground_truth: "dog"`
    *   `prediction: "757-200"`, `ground_truth: "Commercial Airline"`

*   **wrong**: The prediction is factually incorrect, contradictory, malformed, completely unrelated to the ground truth or contains multiple options.
    *   `prediction: "cat"`, `ground_truth: "dog"`
    *   `prediction: "Blue-winged Warbler"`, `ground_truth: "Golden-winged Warbler"`
    *   `prediction: "b1rd"`, `ground_truth: "bird"`
    *   `prediction: "robin or cardinal"`, `ground_truth: "bird"`
    *   `prediction: "_prototype"`, `ground_truth: "Boeing 717"`

*   **abstain**: The prediction is a refusal to answer.
    *   `prediction: "none"`
    *   `prediction: "I don't know"`
    *   `prediction: "Cannot tell"`

**Input Format**:
You will receive a single JSON object with the following structure:
```json
{"ground_truth": "<the_ground_truth_label>", "prediction": "<the_vlm_prediction>"}
```

**Output Format**:
Your response must be exactly one of the listed category labels.
""",
    "prompt": """Classify the prediction in the following JSON object based on the rules provided. Your output must be exactly one of the listed category labels.

INPUT:
%s
""",
    "category_set_name": "standard",
    "categories": CATEGORIES_STANDARD
}

VER_BASE_SOFT_SINGLE = {
    "name": "ver_base_soft_single",
    "system_prompt": """
**Role**: You are an expert AI classifier. Your goal is to classify a model's `prediction` against a `ground_truth` label.

**Core Rule**: First, identify the **main subject** of the prediction. Ignore descriptive details like colors, actions, or contexts (e.g., in "a samoyed playing in snow," the main subject is "samoyed"). Then, classify this main subject.

**Task**: You will receive a single JSON object. Your output must be **only the classification category** and nothing else.

---
### Classification Categories

*   **specific**: The prediction's main subject is an exact match or a direct synonym for the ground truth. This includes common name/scientific name equivalence.
    *   `prediction: "a singing Newfoundland dog"`, `ground_truth: "newfoundland"`
    *   `prediction: "photo of Passiflora caerulea"`, `ground_truth: "Blue passionflower"`

*   **less specific**: The prediction's main subject is a correct, but **closely related parent category** (e.g., family, genus, product line) of the ground truth.
    *   `prediction: "a flying Warbler"`, `ground_truth: "Golden-winged Warbler"`
    *   `prediction: "a Boeing 707 on the runway"`, `ground_truth: "707-320"`

*   **generic**: The prediction's main subject is a correct, but **significantly broader category** than the ground truth.
    *   `prediction: "a photo of a dog"`, `ground_truth: "samoyed"`
    *   `prediction: "an image of a commercial airline"`, `ground_truth: "757-200"`

*   **more specific**: The prediction's main subject is a correct, but **more specific subtype or instance** of the ground truth.
    *   `prediction: "a samoyed playing in snow"`, `ground_truth: "dog"`
    *   `prediction: "a Boeing 757-200 taking off"`, `ground_truth: "Commercial Airline"`

*   **wrong**: The prediction's main subject is is factually incorrect, contradictory, malformed, completely unrelated to the ground truth or contains multiple options.
    *   `prediction: "a cat in a field"`, `ground_truth: "dog"`
    *   `prediction: "a singing Blue-winged Warbler"`, `ground_truth: "Golden-winged Warbler"`
    *   `prediction: "b1rd"`, `ground_truth: "bird"`
    *   `prediction: "robin or cardinal"`, `ground_truth: "bird"`
    *   `prediction: "_prototype"`, `ground_truth: "Boeing 717"`

*   **abstain**: The prediction is a refusal to answer.
    *   `prediction: "none"`
    *   `prediction: "I cannot tell from the image."`

**Input Format**:
You will receive a single JSON object with the following structure:
```json
{"ground_truth": "<the_ground_truth_label>", "prediction": "<the_vlm_prediction>"}
```

**Output Format**:
Your response must be exactly one of the listed category labels.
""",
    "prompt": """Classify the prediction in the following JSON object based on the rules provided. Your output must be exactly one of the listed category labels.

INPUT:
%s
""",
    "category_set_name": "standard",
    "categories": CATEGORIES_STANDARD
}

VER_BASE_JSON_BINARY = {
    "name": "ver_base_json_binary",
    "system_prompt": """
**Role**: You are an expert AI classifier. Your goal is to classify a model's `prediction` against a `ground_truth` label as either 'correct' or 'wrong'.

**Task**: You will receive multiple JSON objects on new lines (NDJSON format). Process every object and produce a **single JSON array** containing the results.

---
### Classification Categories

*   **correct**: The prediction is factually accurate in relation to the ground truth. This includes synonyms, parent categories (e.g., prediction: "dog", ground_truth: "samoyed"), or more specific, valid subtypes (e.g., prediction: "samoyed", ground_truth: "dog").
    *   `prediction: "Panthera leo"`, `ground_truth: "lion"` -> correct
    *   `prediction: "Warbler"`, `ground_truth: "Golden-winged Warbler"` -> correct
    *   `prediction: "samoyed"`, `ground_truth: "dog"` -> correct

*   **wrong**: The prediction is factually incorrect, contradictory, malformed, unrelated, a refusal to answer, or contains multiple options.
    *   `prediction: "cat"`, `ground_truth: "dog"` -> wrong
    *   `prediction: "Blue-winged Warbler"`, `ground_truth: "Golden-winged Warbler"` -> wrong
    *   `prediction: "I don't know"`, `ground_truth: "bird"` -> wrong
    *   `prediction: "robin or cardinal"`, `ground_truth: "bird"` -> wrong

---
**Input and Output Format**

*   **Input**: A stream of JSON objects, one per line.
    ```json
    {"ground_truth": "samoyed", "prediction": "dog"}
    {"ground_truth": "dog", "prediction": "cat"}
    ```

*   **Output**: A single JSON array with all objects classified.
    ```json
    [
      {"ground_truth": "samoyed", "prediction": "dog", "classification": "correct"},
      {"ground_truth": "dog", "prediction": "cat", "classification": "wrong"}
    ]
    ```
""",
    "prompt": """Classify each JSON object from the input below as 'correct' or 'wrong'. Your output must be a single JSON array containing all the results.

INPUT:
%s
""",
    "category_set_name": "binary",
    "categories": CATEGORIES_BINARY
}

VER_BASE_SINGLE_BINARY = {
    "name": "ver_base_single_binary",
    "system_prompt": """
**Role**: You are an expert AI classifier. Your goal is to classify a model's `prediction` against a `ground_truth` label as either 'correct' or 'wrong'.

**Task**: You will receive a single JSON object. Your output must be **only the classification category** ('correct' or 'wrong') and nothing else.

---
### Classification Categories

*   **correct**: The prediction is factually accurate in relation to the ground truth. This includes synonyms, parent categories (e.g., prediction: "dog", ground_truth: "samoyed"), or more specific, valid subtypes (e.g., prediction: "samoyed", ground_truth: "dog").
    *   `prediction: "Panthera leo"`, `ground_truth: "lion"` -> correct
    *   `prediction: "Warbler"`, `ground_truth: "Golden-winged Warbler"` -> correct
    *   `prediction: "samoyed"`, `ground_truth: "dog"` -> correct

*   **wrong**: The prediction is factually incorrect, contradictory, malformed, unrelated, a refusal to answer, or contains multiple options.
    *   `prediction: "cat"`, `ground_truth: "dog"` -> wrong
    *   `prediction: "Blue-winged Warbler"`, `ground_truth: "Golden-winged Warbler"` -> wrong
    *   `prediction: "I don't know"`, `ground_truth: "bird"` -> wrong
    *   `prediction: "robin or cardinal"`, `ground_truth: "bird"` -> wrong

---
**Input Format**:
You will receive a single JSON object with the following structure:
```json
{"ground_truth": "<the_ground_truth_label>", "prediction": "<the_vlm_prediction>"}
```

**Output Format**:
Your response must be a **single word**: 'correct' or 'wrong'.
""",
    "prompt": """Classify the prediction in the following JSON object as 'correct' or 'wrong'. Your output must be a single word.

INPUT:
%s
""",
    "category_set_name": "binary",
    "categories": CATEGORIES_BINARY
}

VER_DECISION_TREE_SINGLE = {
    "name": "ver_decision_tree_single",
    "system_prompt": r"""
**Role**: You are an expert AI verifier. You must classify a model's `prediction` against a `ground_truth`.

**Task**: You will receive exactly one JSON object. Output **only one category label** and nothing else.

### Allowed Categories (output exactly one)
specific, less specific, generic, more specific, wrong, abstain

### Canonical meanings
- **specific**: exact match or direct synonym (including common name ↔ scientific name equivalence).
- **less specific**: correct but only a *closely related parent* of ground truth (nearby hypernym such as genus/family/model-variant parent).
- **generic**: correct but *significantly broader* than ground truth (coarse hypernym).
- **more specific**: prediction is *more specific* than ground truth (a subtype/instance under the ground truth).
- **wrong**: incorrect, contradictory, malformed, unrelated, or contains multiple options/hedged alternatives.
- **abstain**: refusal/uncertainty/none.

### Deterministic decision procedure (apply in order)
1) If `prediction` is an abstention/refusal/uncertainty (e.g., "none", "cannot tell", "I don't know"): output **abstain**.
2) If `prediction` is malformed, nonsense, unrelated, contradictory, or gives multiple options (e.g., "A or B", lists): output **wrong**.
3) If `prediction` and `ground_truth` denote the same entity via exact match or direct synonym (incl. common/scientific name): output **specific**.
4) If `prediction` is a *parent category* of `ground_truth`:
   - if the parent is close (e.g., genus for species; exact product line for a variant): output **less specific**.
   - if the parent is broad/coarse (e.g., animal for dog; aircraft for 757-200): output **generic**.
5) If `prediction` is a *child/subtype/instance* of `ground_truth`: output **more specific**.
6) Otherwise: output **wrong**.

**Input Format**:
{"ground_truth": "<the_ground_truth_label>", "prediction": "<the_vlm_prediction>"}

**Output Format**:
Exactly one label from the allowed categories.
""",
    "prompt": r"""Apply the decision procedure to classify the following JSON object. Output exactly one category label.

INPUT:
%s
""",
    "category_set_name": "standard",
    "categories": CATEGORIES_STANDARD
}

VER_NORMALIZE_STRICT_SINGLE = {
    "name": "ver_normalize_strict_single",
    "system_prompt": r"""
**Role**: You are an expert AI classifier (verifier). Your goal is to label the relationship between `prediction` and `ground_truth`.

**Task**: You will receive one JSON object. Output must be **only** one category label.

### Categories
- specific
- less specific
- generic
- more specific
- wrong
- abstain

### Pre-processing rules (do before judging)
- Treat case, punctuation, and surrounding whitespace as irrelevant (normalize).
- Treat common name ↔ scientific name equivalence as a valid synonym match.
- If the prediction contains multiple candidates, alternatives, disjunctions ("or", "/", ",") or a list of labels, classify as **wrong** (even if one option is correct).
- If the prediction expresses refusal/uncertainty/no-answer, classify as **abstain**.

### Semantics
- **specific**: normalized exact match or direct synonym of ground truth.
- **less specific**: correct but a *nearby hypernym* (close parent category).
- **generic**: correct but a *coarse hypernym* (much broader).
- **more specific**: correct but a *hyponym* (more specific than ground truth).
- **wrong**: anything else (incorrect, contradictory, malformed, unrelated, multi-answer).
- **abstain**: refusal/no-answer.

**Input**:
{"ground_truth": "<the_ground_truth_label>", "prediction": "<the_vlm_prediction>"}

**Output**:
One label: specific | less specific | generic | more specific | wrong | abstain
""",
    "prompt": r"""Normalize then classify the following JSON. Output exactly one category label.

INPUT:
%s
""",
    "category_set_name": "standard",
    "categories": CATEGORIES_STANDARD
}

VER_CHECKLIST_SINGLE = {
    "name": "ver_checklist_single",
    "system_prompt": r"""
**Role**: You are an expert verifier for label correctness and specificity.

**Task**: Given one JSON object with `ground_truth` and `prediction`, output **only** the correct category label.

### Output Categories
specific, less specific, generic, more specific, wrong, abstain

### Checklist (use internally; do NOT output the checklist)
A) Abstention?
- If prediction is "none" / refusal / uncertainty → **abstain**

B) Invalid / multi-answer?
- If prediction is malformed, gibberish, contradictory, unrelated, or includes multiple options/hedges → **wrong**

C) Same meaning?
- Exact same entity or direct synonym (incl. common/scientific name equivalence) → **specific**

D) Correct but different specificity?
- If prediction is a parent category of ground truth:
  - close parent → **less specific**
  - broad parent → **generic**
- If prediction is a child/subtype/instance under ground truth → **more specific**

E) Otherwise → **wrong**

**Input Format**:
{"ground_truth": "<the_ground_truth_label>", "prediction": "<the_vlm_prediction>"}

**Output Format**:
Return exactly one label from the category set and nothing else.
""",
    "prompt": r"""Classify the following JSON object. Return exactly one category label.

INPUT:
%s
""",
    "category_set_name": "standard",
    "categories": CATEGORIES_STANDARD
}

PROMPT_REGISTRY = {
    "ver_base_json": VER_BASE_JSON,
    "ver_base_soft_json": VER_BASE_SOFT_JSON,
    "ver_base_single": VER_BASE_SINGLE,
    "ver_base_soft_single": VER_BASE_SOFT_SINGLE,
    "ver_base_json_binary": VER_BASE_JSON_BINARY,
    "ver_base_single_binary": VER_BASE_SINGLE_BINARY,
    "ver_decision_tree_single": VER_DECISION_TREE_SINGLE,
    "ver_normalize_strict_single": VER_NORMALIZE_STRICT_SINGLE,
    "ver_checklist_single": VER_CHECKLIST_SINGLE,
}
