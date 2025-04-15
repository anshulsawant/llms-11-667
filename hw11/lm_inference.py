import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import seaborn as sns
# Make sure vllm is installed: pip install vllm
from vllm import LLM, SamplingParams
import time # Added for potential debugging, although CUDA events are better

BATCH_SIZE = 4
# Reduced NEW_TOKENS for quicker testing, feel free to revert
NEW_TOKENS = [5, 10, 50] # Original: [5, 10, 50]
REPEATS = 5 # Number of times to repeat the generation for averaging

model_id = "Qwen/Qwen2.5-7B"


def timed_generate_huggingface():
    print(f"Loading Hugging Face model: {model_id}")
    # Note: Consider adding quantization (BitsAndBytesConfig) if needed for large models
    # and GPU memory, although it might affect speed comparison fairness if vLLM isn't quantized.
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # Ensure tokenizer has a padding token if it doesn't
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto", # Use "auto" for better multi-GPU handling if applicable
        # use_auth_token=True # Deprecated, use huggingface_hub login or token argument if needed
        # token="YOUR_HUGGINGFACE_TOKEN" # Add if needed for gated models
        torch_dtype=torch.float16 # Use float16 for faster inference if GPU supports it
    )
    model.eval() # Set model to evaluation mode

    total_time_dict = {}
    text = [
            "hello"
        ] * BATCH_SIZE
    # Ensure padding side is consistent, typically 'left' for Causal LMs during generation batching
    tokenizer.padding_side = 'left'
    inputs = tokenizer(text, return_tensors="pt", padding=True).to(model.device) # Ensure inputs are on the same device as the model

    print("Starting Hugging Face generation timing...")
    for num_new_tokens in tqdm(NEW_TOKENS, desc="HF Generate"):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        # Warmup step (optional but recommended)
        with torch.no_grad():
             _ = model.generate(**inputs, max_new_tokens=5, pad_token_id=tokenizer.pad_token_id)

        torch.cuda.synchronize() # Synchronize after warmup

        total_elapsed_time = 0.0
        for _ in range(REPEATS):
             torch.cuda.empty_cache()
             torch.cuda.synchronize()
             start_event.record()
             # Use torch.no_grad() for inference
             with torch.no_grad():
                # TODO: implement model.generate() here
                # We pass pad_token_id to suppress warnings when using padding
                _ = model.generate(
                    **inputs,
                    max_new_tokens=num_new_tokens,
                    pad_token_id=tokenizer.pad_token_id # Important for batched generation with padding
                )
             end_event.record()
             torch.cuda.synchronize()
             total_elapsed_time += start_event.elapsed_time(end_event) # Time in milliseconds

        timing = (total_elapsed_time / REPEATS) * 1.0e-3 # Average time in seconds
        total_time_dict[f"{num_new_tokens}"] = timing
        print(f"HF - {num_new_tokens} new tokens: {timing:.4f} s")

    del model # Clean up GPU memory
    del tokenizer
    torch.cuda.empty_cache()
    return total_time_dict


def timed_generate_vllm():
    print(f"Loading vLLM engine for model: {model_id}")
    # vLLM automatically handles device placement. Add tensor_parallel_size=N for multi-GPU.
    # Add dtype='float16' if needed, though auto usually works well.
    # gpu_memory_utilization=0.9 can help manage memory.
    llm = LLM(model=model_id, trust_remote_code=True) # Add trust_remote_code=True if necessary
    total_time_dict={}
    text = [
            "hello"
        ] * BATCH_SIZE

    print("Starting vLLM generation timing...")
    for num_new_tokens in tqdm(NEW_TOKENS, desc="vLLM Generate"):
        # TODO: implement sampling_params = SamplingParams() with the correct arguments
        # We only need max_tokens for this benchmark.
        # Other params like temperature, top_p, top_k can be added if needed
        # but might slightly affect performance vs simple greedy.
        # Setting temperature=0 and top_p=1 essentially means greedy decoding.
        sampling_params = SamplingParams(max_tokens=num_new_tokens, temperature=0)

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        # Warmup step for vLLM (optional but recommended)
        _ = llm.generate(text, SamplingParams(max_tokens=5, temperature=0))
        torch.cuda.synchronize() # Synchronize after warmup

        total_elapsed_time = 0.0
        for _ in range(REPEATS):
             torch.cuda.empty_cache() # Less critical for vLLM but doesn't hurt
             torch.cuda.synchronize()
             start_event.record()
             # TODO: implement llm.generate() here
             _ = llm.generate(text, sampling_params)
             end_event.record()
             torch.cuda.synchronize()
             total_elapsed_time += start_event.elapsed_time(end_event) # time in milliseconds

        timing = (total_elapsed_time / REPEATS) * 1.0e-3 # Average time in seconds
        total_time_dict[f"{num_new_tokens}"] = timing
        print(f"vLLM - {num_new_tokens} new tokens: {timing:.4f} s")

    del llm # Clean up GPU memory
    torch.cuda.empty_cache()
    return total_time_dict


# --- Main Execution ---
print("Running Hugging Face benchmark...")
total_time_dict_huggingface = timed_generate_huggingface()
print("\nHugging Face Results:", total_time_dict_huggingface)

print("\nRunning vLLM benchmark...")
total_time_dict_vllm = timed_generate_vllm()
print("\nvLLM Results:", total_time_dict_vllm)


# --- Plotting ---
print("\nGenerating plot...")
sns.set(style="darkgrid")

# Prepare data for seaborn lineplot which prefers lists or arrays
hf_tokens = [int(k) for k in total_time_dict_huggingface.keys()]
hf_times = list(total_time_dict_huggingface.values())

vllm_tokens = [int(k) for k in total_time_dict_vllm.keys()]
vllm_times = list(total_time_dict_vllm.values())

# plot both lines
plt.figure(figsize=(10, 6)) # Adjust figure size for better readability
sns.lineplot(x=hf_tokens, y=hf_times, marker='o', color="blue", label="huggingface-generate")
sns.lineplot(x=vllm_tokens, y=vllm_times, marker='o', color="red", label="vllm-generate")

plt.ylabel("Average inference time (s)")
plt.xlabel("Number of New Tokens Generated")
plt.title(f"Inference Speed Comparison (Batch Size: {BATCH_SIZE}, Model: {model_id})", fontsize = 10) # More informative title

# Ensure x-axis ticks match the tested token counts
plt.xticks(NEW_TOKENS)
plt.legend()
plt.grid(True) # Ensure grid is visible

# save plot
plot_filename = "inference_comparison_plot.jpg"
plt.savefig(plot_filename, dpi=300)
print(f"Plot saved as {plot_filename}")

# Optional: Show the plot
