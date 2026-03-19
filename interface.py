import streamlit as st
import plotly.graph_objects as go
import torch
import torch.nn as nn
from transformer_lens import HookedTransformer
from transformers import AutoModelForCausalLM

class TunedLens(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.n_layers = model.cfg.n_layers
        self.d_model = model.cfg.d_model
        self.d_vocab = model.cfg.d_vocab
        
        # Create linear probes for each layer
        self.probes = nn.ModuleList([
            nn.Linear(self.d_model, self.d_vocab) for _ in range(self.n_layers)
        ])
        
    def train_probes(self, texts, batch_size=4, epochs=5, lr=1e-3):
        """Train the probes on a dataset of texts"""
        optimizer = torch.optim.Adam(self.probes.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        # Tokenize all texts
        all_tokens = []
        for text in texts:
            tokens = self.model.to_tokens(text)
            if tokens.shape[1] > 1:  # Ensure we have at least one token to predict
                all_tokens.append(tokens)
        
        if not all_tokens:
            return
            
        # Training loop
        for epoch in range(epochs):
            total_loss = 0
            n_batches = 0
            
            # Shuffle and batch
            indices = torch.randperm(len(all_tokens))
            for i in range(0, len(all_tokens), batch_size):
                batch_indices = indices[i:i+batch_size]
                batch_tokens = [all_tokens[idx] for idx in batch_indices]
                
                # Pad to same length
                max_len = max(t.shape[1] for t in batch_tokens)
                padded_batch = []
                for tokens in batch_tokens:
                    if tokens.shape[1] < max_len:
                        pad_len = max_len - tokens.shape[1]
                        padding = torch.zeros((tokens.shape[0], pad_len), dtype=tokens.dtype, device=tokens.device)
                        tokens = torch.cat([tokens, padding], dim=1)
                    padded_batch.append(tokens)
                
                batch_tokens = torch.cat(padded_batch, dim=0)
                
                # Get final logits and intermediate representations
                with torch.no_grad():
                    final_logits, cache = self.model.run_with_cache(batch_tokens)
                    target_ids = batch_tokens[:, 1:]
                
                # Train each probe
                for layer in range(self.n_layers):
                    optimizer.zero_grad()
                    
                    hook_name = f"blocks.{layer}.hook_resid_post"
                    resid_stream = cache[hook_name][:, :-1]  # Remove last token (no prediction needed)
                    
                    # Get probe predictions
                    probe_logits = self.probes[layer](resid_stream)
                    
                    # Compute loss (cross entropy with true next-token ids)
                    loss = criterion(probe_logits.view(-1, self.d_vocab), target_ids.view(-1))
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    n_batches += 1
            
            if epoch % 2 == 0:
                print(f"Epoch {epoch}, Avg Loss: {total_loss / n_batches:.4f}")
    
    def forward(self, hidden_states, layer):
        """Apply the tuned lens probe for a specific layer"""
        return self.probes[layer](hidden_states)

# 1. Cache the model so it doesn't reload every time you type a letter in the GUI
@st.cache_resource
def load_model():
    # GPT-2 has pre-trained tuned lenses available
    model = HookedTransformer.from_pretrained("gpt2")
    
    # Create and train custom tuned lens
    tuned_lens = TunedLens(model)
    
    # Train on some sample texts
    training_texts = [
        "The sky is blue and the grass is",
        "The capital of France is",
        "Machine learning is",
        "The weather today is",
        "I like to eat",
        "The best programming language is",
        "In the future, AI will",
        "The color of the ocean is",
        "My favorite food is",
        "The largest planet is"
    ]
    
    print("Training tuned lens probes...")
    tuned_lens.train_probes(training_texts, epochs=10)
    print("Tuned lens training complete!")
    
    return model, tuned_lens

st.title("🔍 Interactive Logit Lens")
st.markdown("Observe how the model builds its predictions layer-by-layer.")

model, tuned_lens = load_model()

# 2. The GUI Input
lens_type = st.radio("Choose Lens Type:", ["Logit Lens", "Tuned Lens"])
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
        
        if lens_type == "Logit Lens":
            # Apply layer norm and unembed
            normalized_state = model.ln_final(residual_stream)
            layer_logits = model.unembed(normalized_state) # Shape: [seq_len, d_vocab]
        else:
            # Tuned lens expects [batch, seq, d_model]
            layer_logits = tuned_lens(residual_stream.unsqueeze(0), layer).squeeze(0)
        
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