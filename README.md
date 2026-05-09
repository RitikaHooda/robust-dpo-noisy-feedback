# Robust DPO — Full Experimental Codebase

A systematic empirical study of vanilla DPO,
cDPO, rDPO, and ROPO under symmetric label noise on Qwen2.5-7B-Instruct + HH-RLHF.

## Project Structure

```
robust_dpo/
├── configs/
│   ├── base.yaml              # shared hyperparameters
│   ├── lora.yaml              # LoRA adapter settings
│   ├── noise.yaml             # noise sweep settings
│   └── judge.yaml             # LLM-as-a-judge settings
├── src/
│   ├── config.py              # dataclass configs + YAML loading
│   ├── data.py                # dataset loading, noise injection, dataloaders
│   ├── losses.py              # vanilla_dpo, cdpo, rdpo, ropo loss functions
│   ├── model.py               # model + LoRA setup, sequence_logprob
│   ├── trainer.py             # training loop
│   ├── eval.py                # preference margin, accuracy evaluation
│   ├── judge.py               # GPT-4o LLM-as-a-judge
│   └── utils.py               # logging, seeding, result IO
├── scripts/
│   ├── train.py               # single (method, noise) training run
│   ├── run_sweep.py           # full noise × method grid
│   ├── run_eval.py            # eval a saved checkpoint
│   └── run_judge.py           # LLM-as-a-judge on saved checkpoints
├── notebooks/
│   └── results_analysis.ipynb # tables + plots from saved results
├── requirements.txt
└── README.md
```

## Quickstart

```bash
# 1. Install
pip install -r requirements.txt

# 2. Set your OpenAI key (for GPT-4o judge)
export OPENAI_API_KEY=sk-...

# 3. Smoke test (tiny model, few steps — ~5 min on 1× A100)
python scripts/train.py \
    --method ropo \
    --noise 0.2 \
    --config configs/base.yaml \
    --smoke_test

# 4. Full sweep (all methods × all noise levels)
python scripts/run_sweep.py --config configs/base.yaml

# 5. Run LLM judge over saved checkpoints
python scripts/run_judge.py --results_dir results/

# 6. Analyse results
jupyter notebook notebooks/results_analysis.ipynb
```

## Reproducing Table 2 (paper results)

All defaults in `configs/base.yaml` match the paper exactly:
- Model: `Qwen/Qwen2.5-7B-Instruct`
- LoRA: r=16, α=32, dropout=0.05
- Training: 3 000 examples, 8 000 gradient steps, β=0.1, lr=5e-5
- Noise levels: {0, 10, 20, 30, 40}%
- Eval: 400 clean held-out examples; 50 examples for GPT-4o judge

## Methods

| Method | Unbiased | Noise-tolerant | Needs ε? |
|--------|----------|----------------|----------|
| vanilla DPO | ✓ | ✗ | No |
| cDPO | ✗ | ✗ | Yes (oracle) |
| rDPO | ✓ | ✓ | Yes (oracle) |
| ROPO | — | ✓ | No |