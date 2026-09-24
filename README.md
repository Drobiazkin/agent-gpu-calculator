# GPU / RAM Calculator for LLM Agent Sessions

A single-file web calculator that sizes GPUs and host RAM for serving **LLM agent sessions** on
[vLLM](https://github.com/vllm-project/vllm) with KV-cache offloading to host memory.

An agent session is not one request: it is a loop of steps (generate a tool call → run the tool →
append the result to the context). Between steps a session is idle on the GPU while its KV cache
waits in VRAM or in a host-RAM pool. The calculator answers how many GPUs, nodes and how much RAM
you need to keep a target number of live agents (or a target number of steps per day) under a
per-session generation speed (SLA).

## Usage

Open `index.html` in any modern browser. No build step, no server, no network access: all code and
styles are inlined. The page can also be served as-is, e.g. with GitHub Pages.

- **Presets** fill the Model, Load and Platform groups; every field stays editable.
- **The current scenario is always in the URL** (`#s=…`). *Copy* / *Apply* share and restore it.
  Fields missing from an older link fall back to defaults, and the page says so.
- **+ add to estimate** collects several workload classes into one mixed estimate.
- Hover the ⓘ icons: every field documents its meaning, units and how to obtain the value.
- *How this is computed* at the bottom of the page describes the model in full.

## What it models

- **One replica = TP GPUs in one NVLink domain.** Weights, the shared prefix and the KV cache are
  sharded across the replica. Procurement is counted in whole nodes.
- **Two independent procurement conditions**, the larger one wins:
  - *memory* — how many sessions fit (VRAM slots and the host-RAM pool);
  - *throughput* — how many steps per second a replica delivers (prefill compute, PCIe swap,
    decode under the SLA).
- **Decode** is memory-bound: weights plus the KV read by attention, stretched by prefill chunks
  that share the same forward pass (chunked prefill). The SLA cuts the decode batch through the
  same pass equation.
- **Prefill** is compute-bound: matrix work plus attention that grows with context length.
- **Concurrency** follows Little’s law (live agents = step rate × cycle time), with the RAM pool
  as a separate limit.

Model features:

- mixture of experts (MoE);
- compressed attention (MLA): vLLM keeps a full copy of the latent KV cache on every TP rank;
- DSA sparse attention: top-k selection budget, and an indexer computed in full on every TP rank;
- hybrid linear-attention models: fixed per-session state, with separate net traffic and memory
  allocated by vLLM;
- speculative decoding (MTP, EAGLE, draft model, n-gram), entered as a measured speedup;
- mixed-precision checkpoints: the effective peak is weighted by active parameters per precision.

## Presets

**Models**:
- Qwen3-Coder (30B-A3B, Next, 480B-A35B), Qwen3-235B-A22B-2507, Qwen3-32B;
- Qwen3.8 (27B, Flash-Next, 2.4T-A95B);
- GLM-4.6, GLM-5.2, GLM-5.3-Flash (FP8 and NVFP4);
- DeepSeek-V3.2-Exp, DeepSeek-V4-Flash.

**Platforms**: 8×H100, 8×H200, 8×B100, 8×B200, 8×B300 (HGX / DGX).

**Load profiles**: code (existing repository / new module), enterprise documents, reasoning, chat in Russian.

The model × platform pair sets:
- the effective peak (weighted by precision; without a hardware FP4 path 4-bit weights run at the
  BF16 peak);
- the starting TP;
- architecture-dependent KV-cache bytes.

An incompatible pair shows a banner.

Each model preset carries a confidence label. Values come from the models’ `config.json` and
safetensors headers on Hugging Face, from vLLM source code, and from NVIDIA datasheets. They were
checked in September 2026: re-check them against the vLLM version you run.

## Limitations

This is a model, not a measurement. Before buying hardware, replace the assumptions below with
measurements on your stack.

- **Efficiency factors are assumptions**:
  - share of the peak at prefill: 0.25 for MoE, 0.45 for dense;
  - effective HBM bandwidth: ~70% of the peak;
  - PCIe: ~50 GB/s per GPU.

  Measure them on your hardware:
  - the vLLM startup log line `GPU KV cache size: N tokens` checks the memory model;
  - `vllm bench serve` at the target context length gives prefill speed (enter it in *Measured
    prefill*);
  - `nvbandwidth` gives the real HBM and PCIe bandwidth.
- **Not modeled**:
  - pure tensor parallelism is the only layout per replica: data-parallel attention
    (DP-attention), pipeline or expert parallelism across nodes are out of scope;
  - disaggregated prefill/decode;
  - remote KV tiers (e.g. over RDMA);
  - memory and compute of a separate draft model for speculative decoding.
- **Near the SLA feasibility edge the result can jump.** When the weight read takes most of the
  per-pass SLA budget, the throughput solver can jump between regimes; the page shows a warning.
  Check neighbouring values in that case.

## Styling

[Tailwind CSS](https://tailwindcss.com) v3.4.17 (MIT License) is compiled from the classes used on
the page and inlined in the `<style>` tag on line 7, so the page has no external dependencies.

After adding a new Tailwind class to the markup or to a JavaScript string, rebuild the CSS.
Scan the page **without** the inlined style line. Otherwise the selectors of the inlined CSS are
taken for page classes, and the CSS grows on every rebuild.

```bash
sed '/tailwindcss v3.4.17/d' index.html > /tmp/tw-src.html
npx tailwindcss@3.4.17 --content /tmp/tw-src.html \
  -i <(printf '@tailwind base;@tailwind components;@tailwind utilities;') --minify -o /tmp/tw.css
```

Then replace the contents of the `<style>` tag on line 7 (after the license comment) with
`/tmp/tw.css`. Write class names in full: a class assembled from string pieces (`'bg-' + x`) is not
seen by the scanner.

## License

MIT, see [LICENSE](LICENSE). Tailwind CSS is © Tailwind Labs, MIT License.
