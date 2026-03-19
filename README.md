# LogitLens
A web-based interactive tool to visualize how Large Language Models build their predictions layer-by-layer, built with Streamlit, Plotly, and TransformerLens.

This tool implements both **Logit Lens** and **Tuned Lens** techniques for LLM interpretability:
- **Logit Lens**: Interrupts the forward pass at intermediate layers, applies final layer normalization and unembedding to reveal what the model "thinks" the next word will be.
- **Tuned Lens**: Uses pre-trained linear probes to provide better-calibrated predictions by accounting for transformations between layers.

## Setup and Start

This project uses [uv](https://docs.astral.sh/uv/). Install if you haven't already:

```bash
# On macOS and Linux.
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Installation & Running
```bash
git clone https://github.com/nathanhoehndorf/LogitLens.git
cd LogitLens

uv run streamlit run interface.py
```

## How to Use
1. **Choose Lens Type:** Select between "Logit Lens" or "Tuned Lens" using the radio buttons.
2. **Enter a Prompt:** Type an incomplete sentence into the text box (e.g., "The capital of France is").
3. **Wait for Inference:** App downloads GPT-2 model and runs the forward pass.
4. **Explore the Heatmap:**
    - The **X-axis** represents the input tokens.
    - The **Y-axis** represents the layers of the model.
    - Hover over any cell to see the top predicted word at that specific layer and its confidence score.

## Model
The tool uses GPT-2, which has pre-trained tuned lenses available for more accurate interpretability.

## Tools Used
`TransformerLens`, `Streamlit`, `Plotly`, `tuned-lens`, and `uv`.
