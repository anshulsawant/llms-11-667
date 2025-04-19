# Approximation of Trainable LoRA Parameters

This calculation estimates the number of trainable parameters added by LoRA based on the project's configuration and known dimensions for the `google/gemma-2b-it` model.

## 1. Assumptions and Configuration

* **Base Model:** `google/gemma-2b-it`
* **LoRA Rank (`r`):** 16
* **Target Modules:** `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` (Based on `config.yaml`)
* **Model Dimensions:**
    * Hidden Size (`d`): 2048
    * MLP Intermediate Size (`i`): 16384 (8 * hidden_size)
    * Number of Layers (`L`): 18

## 2. LoRA Parameter Formula

For a pre-trained linear layer `W` with input dimension `d_in` and output dimension `d_out`, LoRA introduces two matrices, `A` (`r` x `d_in`) and `B` (`d_out` x `r`). The number of trainable parameters added for this single layer is:

`Parameters = (r * d_in) + (d_out * r) = r * (d_in + d_out)`

*(Note: Bias terms are typically not trained in LoRA, matching `bias: "none"` in the config).*

## 3. Calculation Steps

We apply this formula to all targeted modules across all `L=18` layers.

**a) Attention Layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`)**

* These layers typically operate within the hidden dimension `d`. Approximating all as `d x d` (2048 x 2048) for simplicity (`d_in=2048`, `d_out=2048`).
* Parameters per adapted attention projection layer = `r * (d + d) = 2 * r * d`
* Parameters per projection = `2 * 16 * 2048 = 65,536`
* Total parameters for 4 projections per layer = `4 * 65,536 = 262,144`
* **Total Attention LoRA Params** = `L * 262,144 = 18 * 262,144 = **4,718,592**`

**b) MLP Layers (`gate_proj`, `up_proj`, `down_proj`)**

* `gate_proj`, `up_proj`: Map `d` to `i` (2048 -> 16384) -> `d_in=2048`, `d_out=16384`
    * Params per layer = `r * (d + i) = 16 * (2048 + 16384) = 16 * 18432 = 294,912`
* `down_proj`: Maps `i` to `d` (16384 -> 2048) -> `d_in=16384`, `d_out=2048`
    * Params per layer = `r * (i + d) = 16 * (16384 + 2048) = 16 * 18432 = 294,912`
* Total MLP Params per layer = `Params(gate) + Params(up) + Params(down)`
* Total MLP Params per layer = `294,912 + 294,912 + 294,912 = 884,736`
* **Total MLP LoRA Params** = `L * 884,736 = 18 * 884,736 = **15,925,248**`

## 4. Total Estimated LoRA Parameters

* Total = Total Attention Params + Total MLP Params
* Total = `4,718,592 + 15,925,248 = **20,643,840**`

**Result:** Approximately **20.6 Million** trainable parameters are added using LoRA with this configuration and 18 layers.

## 5. Percentage of Total Parameters

* Gemma 2B Base Parameters: ~2.5 Billion (2,500 Million)
* Percentage Trainable: `(20.6 Million / 2500 Million) * 100% ≈ **0.82%**`
