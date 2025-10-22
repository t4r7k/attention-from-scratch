
# Multi-Head Scaled Dot-Product Attention — Challenge

A minimal, CPU-only PyTorch assignment for implementing **Scaled Dot-Product Attention** and **Multi-Head Attention** from scratch.

## Objectives
- Implement Scaled Dot-Product Attention
- Build Multi-Head Attention (configurable heads)
- Validate shapes & gradients on synthetic inputs
- (Optional) Compare vs. `torch.nn.MultiheadAttention`

## Project Layout
```
mha-attention-home-task/
├── .github/workflows/python-tests.yml
├── LICENSE
├── README.md
├── requirements.txt
├── src/
│   └── mha/
│       ├── __init__.py
│       └── attention.py        # <— implement TODOs
├── tests/
│   └── test_acceptance.py
└── notebooks/
    └── (optional) exploration.ipynb
```

## Quickstart
```bash
# 1) Create & activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2) Install deps (CPU-only torch)
pip install -r requirements.txt

# 3) Run tests
pytest -q
```

## Candidate Instructions
- **Do not change public function signatures** in `attention.py`.
- Implement TODOs in:
  - `ScaledDotProductAttention.forward`
  - `MultiHeadAttention.forward`
- Keep the API and shapes as documented in the file.
- Ensure `pytest` passes locally before submitting a PR.
- Include a short write-up (e.g., as `WRITEUP.md` or in your PR description) addressing:
  - Why we scale by `√d_k`
  - Why multi-heads improve expressivity
  - Any numerical/implementation issues encountered

## Optional Enhancements
- Causal masking (triangular)
- Dropout / LayerNorm wrapper
- Attention weight visualization (matplotlib)
- `nn.MultiheadAttention` comparison
- Throughput micro-benchmarks (L=128/256)

## License
MIT
