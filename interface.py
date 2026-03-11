import streamlit as st
import plotly.graph_objects as go
import torch
from transformer_lens import HookedTransformer

# 1. Cache the model so it doesn't reload every time you type a letter in the GUI
@st.cache_resource
def load_model():
    # Pythia-14m is great for fast testing!
    return HookedTransformer.from_pretrained("pythia-14m")

st.title("🔍 Interactive Logit Lens")
st.markdown("Observe how the model builds its predictions layer-by-layer.")

model = load_model()

# 2. The GUI Input
prompt = st.text_input("Enter a prompt:", "The sky is blue and the grass is")

if prompt:
    # Tokenize and run the model
    tokens = model.to_tokens(prompt)
    str_tokens = model.to_str_tokens(tokens)
    
    with st.spinner("Running model and extracting cache..."):
        logits, cache = model.run_with_cache(tokens)
    
    # 3. Get the final predictions (this is our "target" to calculate probabilities against)
    # We want to see how early the model converges on its final decision
    final_token_ids = logits[0].argmax(dim=-1)
    final_token_strs = [model.to_string(t) for t in final_token_ids]
    
    n_layers = model.cfg.n_layers
    seq_len = tokens.shape[1]
    
    # Prepare matrices for the heatmap
    heatmap_probs = torch.zeros((n_layers, seq_len))
    hover_labels = []
    
    # 4. Extract layer-by-layer data
    for layer in range(n_layers):
        hook_name = f"blocks.{layer}.hook_resid_post"
        residual_stream = cache[hook_name][0] # Shape: [seq_len, d_model]
        
        # Apply layer norm and unembed
        normalized_state = model.ln_final(residual_stream)
        layer_logits = model.unembed(normalized_state) # Shape: [seq_len, d_vocab]
        layer_probs = torch.softmax(layer_logits, dim=-1)
        
        layer_hover_text = []
        for pos in range(seq_len):
            # What is the probability of the token that the model ULTIMATELY predicts?
            target_id = final_token_ids[pos]
            prob_of_target = layer_probs[pos, target_id].item()
            heatmap_probs[layer, pos] = prob_of_target
            
            # What is THIS specific layer's top guess?
            top_guess_id = layer_logits[pos].argmax().item()
            top_guess_str = model.to_string(top_guess_id)
            
            # Format the interactive hover text
            hover_text = (
                f"Input Token: {str_tokens[pos]!r}<br>"
                f"Layer {layer} Top Guess: <b>{top_guess_str!r}</b><br>"
                f"Final Prediction: {final_token_strs[pos]!r}<br>"
                f"Confidence in Final: {prob_of_target:.2%}"
            )
            layer_hover_text.append(hover_text)
            
        hover_labels.append(layer_hover_text)
        
    # 5. Build the Interactive Plotly Heatmap
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_probs.numpy(),
        x=str_tokens,
        y=[f"Layer {i}" for i in range(n_layers)],
        text=hover_labels,
        hoverinfo="text",
        colorscale="Viridis", # "Magma" or "Inferno" also look great
        colorbar_title="Probability"
    ))
    
    fig.update_layout(
        title="Evolution of Final Prediction Probability",
        xaxis_title="Input Sequence",
        yaxis_title="Model Layer",
        xaxis_nticks=len(str_tokens)
    )
    
    # Render in Streamlit
    st.plotly_chart(fig, use_container_width=True)