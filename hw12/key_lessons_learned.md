# Key Lessons Learned: Fine-Tuning Gemma 2B for GSM8K

This project involved fine-tuning `google/gemma-2b-it` on the GSM8K dataset using Full SFT and LoRA, comparing different dataset variants (Main vs. Socratic). Several valuable lessons emerged during the development, debugging, and analysis phases:

1.  **Baseline Evaluation Informs Model Choice:**
    * **Observation:** Initial consideration of `gemma-9b-it` revealed very high baseline accuracy (~86%) with simple prompting, potentially masking the effects of fine-tuning.
    * **Lesson:** Always perform baseline evaluations before committing to extensive fine-tuning. Choose a model whose baseline performance allows room for measurable improvement relevant to your project goals. Sometimes a smaller model (like `gemma-2b-it`) is more suitable for demonstrating SFT impact.

2.  **Training Data Quality & Relevance are Crucial for Generalization:**
    * **Observation:** Training on the GSM8K `socratic` dataset yielded lower final training loss compared to the `main` dataset for both Full SFT and LoRA. However, models trained on `main` achieved significantly higher accuracy on the test set.
    * **Lesson:** Minimizing training loss doesn't guarantee better performance on unseen data. The training dataset must be representative of the target task and evaluation distribution. Overfitting to specific data styles (like the Socratic format) can harm generalization. Evaluate rigorously on a hold-out test set.

3.  **Understand Full SFT vs. PEFT (LoRA) Trade-offs:**
    * **Observation:** Full SFT achieved the highest accuracy (31% on Main), while LoRA (`r=16`, extensive targets) reached lower accuracy (20% on Main) but demonstrated higher training throughput (samples/sec). LoRA initially caused OOM errors when Gradient Checkpointing was disabled, highlighting memory differences.
    * **Lesson:** PEFT methods like LoRA offer resource efficiency (memory, potentially faster steps) but may not reach the same peak performance as full fine-tuning. The optimal choice depends on resource constraints, performance requirements, and the specific model/task. For smaller models, the speed benefit of LoRA might be less pronounced as the forward pass dominates.

4.  **Debugging Interactions is Essential (Especially with PEFT & Optimizations):**
    * **Observation:** Significant time was spent debugging a persistent `RuntimeError: element 0 of tensors does not require grad...` when using LoRA with Gradient Checkpointing (GC) enabled. The fix involved setting `gradient_checkpointing_kwargs={'use_reentrant': False}`.
    * **Lesson:** Combining techniques like PEFT, quantization (via BitsAndBytes optimizers), and gradient checkpointing introduces complex interactions. Debugging requires systematic isolation, careful reading of tracebacks and warnings, checking library versions, and understanding configuration options (like `use_reentrant`). Don't assume default settings are always compatible.

5.  **Configuration Management Streamlines Experimentation:**
    * **Observation:** Using a base `config.yaml` and smaller override YAML files for specific runs (e.g., `config_lora_main.yaml`) managed by `OmegaConf` proved effective.
    * **Lesson:** Employing a structured configuration system is vital for managing hyperparameters, paths, and experiment variants, ensuring reproducibility and easier modification.

6.  **Evaluation Strategy Requires Care:**
    * **Observation:** Comparing fine-tuned models required fair base model evaluation (using one-shot prompting).
    * **Lesson:** Design evaluation protocols carefully. Ensure fair comparisons to baselines.

**Overall:** This project highlighted that successful fine-tuning involves not just implementing the core training loop but also careful model/data selection, robust configuration management, systematic debugging of complex library interactions, and thoughtful evaluation design.

