# Slide 1: Fine-Tuning Gemma 2B for Math Reasoning (GSM8K)

* Comparing Full SFT, LoRA, and PPO
* **Presenter**: *Anshul Sawant* 

---

# Slide 2: The Problem & How We Measured Success

* **Problem:** Solving Grade School Math Word Problems (GSM8K dataset).
    * Requires multi-step reasoning, calculation accuracy, and understanding language.
    * Example: _"Natalia sold 48 clips in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"_
* **Evaluation Method:**
    * **Exact Match (EM) Accuracy:** Standard for GSM8K.
    * Extract the final number after `####` in both model output and ground truth.
    * Mark as correct *only* if the numbers match exactly.
    * Evaluated on a 256-sample test set (derived from untouched test split).

---
# Slide 3: Bonus

* **Tinier-Zero** Learning resource developed as part of this project in collaboration with Gemini 2.5 Pro.
* From scratch, **single file PPO implementation**.
* Pedagogical part WIP, but still useful.
* https://github.com/anshulsawant/llms-11-667/tree/main/tinier-zero

---
# Slide 4: Key Insight 1: Training Data Matters More Than Training Loss!

* **Experiment:** We fine-tuned using two GSM8K variants:
    * `main`: Standard Q&A with reasoning.
    * `socratic`: More detailed, step-by-step reasoning format.
* **Observation:** Models trained on the `socratic` dataset achieved **lower final training loss**.
* **BUT:** Models trained on the `main` dataset achieved **significantly higher test accuracy** (e.g., Full SFT: 31% vs 23%).
* **Insight:** Minimizing training loss doesn't guarantee better real-world performance. The training data's format and representativeness of the *evaluation task* are critical for generalization. Socratic dataset has longer answers, simpler sentences. Probably easier from LM perspective.
* **Takeaway for Class:** Don't solely rely on loss curves; evaluate on a relevant hold-out set. Ensure training data matches the target task distribution/style.

---

# Slide 5: Key Insight 2: Fine-Tuning Trade-offs & RL Challenges

* **Comparing Methods (Accuracy %):**
    * Base Model (ICL One-Shot + Instruction): 6%
    * LoRA SFT (Main Dataset): 20%
    * Full SFT (Main Dataset): **31%**
    * PPO RL (from Full SFT Main, Exact Match Reward): 28%
* **Insight 1 (SFT/LoRA):**
    * Both SFT methods drastically improved over ICL baseline.
    * Full SFT outperformed LoRA (`r=16`) in accuracy, despite LoRA having higher training throughput. Demonstrates the accuracy vs. efficiency trade-off.
* **Insight 2 (PPO RL):**
    * Applying PPO with a direct exact match reward *did not* improve upon the SFT result and even slightly decreased accuracy.
    * Qualitative analysis showed PPO fixed one SFT error but introduced *new* errors/regressions on other problems.
* **Takeaway for Class:** Fine-tuning is powerful but has trade-offs (Full vs. PEFT). RL alignment is complex; optimizing directly for a metric doesn't guarantee better overall reasoning and can even cause regressions without careful reward design and tuning. Debugging interactions (e.g., LoRA + Gradient Checkpointing) is often required.
