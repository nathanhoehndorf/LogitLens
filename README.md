# LogitLens
A web-based interactive tool to visualize how Large Language Models build their predictions layer-by-layer, built with Streamlit, Plotly, and TransformerLens.

This tools implements a **Logit Lens**, a foundational interpretability technique introduced in my Large Language Models course that interrupts the forward pass of an LLM at intermediate layers, applies the final layer normalization and unembedding, and reveals what the model "thinks" the next word will be before the final output.

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
1. **Enter a Prompt:** Type an incomplete sentence into the text box (e.g., "The capital of France is").
2. **Wait for Inference:** App downloads a small model and runs the forward pass.
3. **Explore the Heatmap:**
    - The **X-axis** represents the input tokens.
    - The **Y-axis** represents the layers of the model.
    - Hover over any cell to see the top predicted word at that specific layer and its confidence score.

#### Tools Used
`TransformerLens`, `Streamlit`, `Plotly`, and `uv`.
