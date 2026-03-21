import os
import streamlit as st
import plotly.graph_objects as go
import torch
import torch.nn as nn
from transformer_lens import HookedTransformer
from transformers import AutoModelForCausalLM

class TunedLens(nn.Module):
    def __init__(self, n_layers, d_model, d_vocab):
        super().__init__()
        self.n_layers = n_layers
        self.d_model = d_model
        self.d_vocab = d_vocab
        
        # Create linear probes for each layer
        self.probes = nn.ModuleList([
            nn.Linear(self.d_model, self.d_vocab) for _ in range(self.n_layers)
        ])
        
    def train_probes(self, model, texts, batch_size=4, epochs=5, lr=1e-3):
        """Train the probes on a dataset of texts"""
        optimizer = torch.optim.Adam(self.probes.parameters(), lr=lr)
        
        # Tokenize all texts
        all_tokens = []
        for text in texts:
            try:
                tokens = model.to_tokens(text)
                if tokens.shape[1] > 1:  # Ensure we have at least one token to predict
                    all_tokens.append(tokens)
            except:
                continue  # Skip problematic texts
        
        if not all_tokens:
            return
            
        print(f"Training on {len(all_tokens)} text samples for {epochs} epochs")
        
        # Training loop: 5 epochs, up to 10 batches per epoch (no data repetition)
        batches_per_epoch = 10

        all_batches = (len(all_tokens) + batch_size - 1) // batch_size
        batch_count = min(batches_per_epoch, all_batches)

        for epoch in range(epochs):
            print(f"Starting epoch {epoch+1}/{epochs}")
            total_loss = 0.0
            n_batches = 0

            # Shuffle all token indices each epoch
            indices = torch.randperm(len(all_tokens))

            for batch_id in range(batch_count):
                start = batch_id * batch_size
                end = min(start + batch_size, len(all_tokens))
                batch_indices = indices[start:end]
                print(f"Processing batch {batch_id+1}/{batch_count} (data {start}:{end})")
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
                print(f"Batch shape: {batch_tokens.shape}")

                # Get final logits and intermediate representations once
                print("Running model forward pass...")
                with torch.no_grad():
                    final_logits, cache = model.run_with_cache(batch_tokens)
                    final_target_logits = final_logits[:, :-1, :].detach()  # Shift by 1 to align with predictions
                    final_target_probs = torch.softmax(final_target_logits, dim=-1)

                print("Computing losses...")
                layer_losses = []
                for layer in range(self.n_layers):
                    hook_name = f"blocks.{layer}.hook_resid_post"
                    resid_stream = cache[hook_name][:, :-1]
                    probe_logits = self.probes[layer](resid_stream)
                    probe_log_probs = torch.log_softmax(probe_logits, dim=-1)
                    loss = - (final_target_probs.reshape(-1, self.d_vocab) * probe_log_probs.reshape(-1, self.d_vocab)).sum(dim=-1).mean()
                    layer_losses.append(loss)

                print("Updating parameters...")
                optimizer.zero_grad()
                total_batch_loss = sum(layer_losses)
                total_batch_loss.backward()
                optimizer.step()

                total_loss += total_batch_loss.item()
                n_batches += 1

            avg_loss = total_loss / n_batches if n_batches > 0 else float('nan')
            print(f"Epoch {epoch+1}/{epochs}, Avg Loss: {avg_loss:.4f}")

        print("Training finished; GUI is now active")
    
    def forward(self, hidden_states, layer):
        """Apply the tuned lens probe for a specific layer"""
        return self.probes[layer](hidden_states)

# 1. Cache the model so it doesn't reload every time you type a letter in the GUI
@st.cache_resource
def load_model():
    # GPT-2 has pre-trained tuned lenses available
    model = HookedTransformer.from_pretrained("gpt2")
    return model

@st.cache_resource
def load_tuned_lens_state_dict():
    state_path = "tuned_lens_state.pt"
    if os.path.exists(state_path):
        print(f"Loading saved tuned lens state dict from {state_path}")
        return torch.load(state_path)

    # Create a temporary model just for training (this won't be cached)
    model = HookedTransformer.from_pretrained("gpt2")
    
    # Create and train custom tuned lens using ALL available stories
    tuned_lens = TunedLens(model.cfg.n_layers, model.cfg.d_model, model.cfg.d_vocab)
    
    # Load training texts from tinystories_10k.txt
    try:
        with open("tinystories_10k.txt", "r", encoding="utf-8", errors="ignore") as f:
            full_text = f.read()
        # Split by end-of-text token
        training_texts = [story.strip() for story in full_text.split("<|endoftext|>")]
        # Filter out empty or very short stories
        training_texts = [s for s in training_texts if len(s) > 50]
        print(f"Loaded {len(training_texts)} training stories from tinystories_10k.txt")
    except FileNotFoundError:
        print("tinystories_10k.txt not found, using default training texts")
        training_texts = [
            "The sky is blue and the grass is green",
            "The capital of France is Paris",
            "Machine learning is awesome",
            "The weather today is sunny",
            "I like to eat pizza",
            "The best programming language is Python",
            "In the future, AI will rule the world",
            "The color of the ocean is blue",
            "My favorite food is pizza",
            "The largest planet is Jupiter"
        ]
    
    print("Training tuned lens probes...")
    tuned_lens.train_probes(model, training_texts, epochs=5, batch_size=16)
    print("Tuned lens training complete!")

    state_path = "tuned_lens_state.pt"
    torch.save(tuned_lens.state_dict(), state_path)
    print(f"Saved tuned lens state dict to {state_path}")
    
    # Return only the state dict for caching
    return tuned_lens.state_dict()

def get_tuned_lens(model):
    # Load cached state dict and create TunedLens instance
    state_dict = load_tuned_lens_state_dict()
    tuned_lens = TunedLens(model.cfg.n_layers, model.cfg.d_model, model.cfg.d_vocab)
    tuned_lens.load_state_dict(state_dict)
    return tuned_lens

st.title("🔍 Interactive Logit Lens")
st.markdown("Observe how the model builds its predictions layer-by-layer.")

model = load_model()

if "tuned_lens_state" not in st.session_state:
    st.session_state.tuned_lens_state = None

if "tuned_lens_trained" not in st.session_state:
    st.session_state.tuned_lens_trained = False

if "tuned_lens_obj" not in st.session_state:
    st.session_state.tuned_lens_obj = None

if not st.session_state.tuned_lens_trained:
    with st.expander("Tuned Lens setup"):
        st.write("TunedLens is not trained yet.")
        if st.button("Start TunedLens Training"):
            with st.spinner("Training TunedLens — this may take several minutes..."):
                state_dict = load_tuned_lens_state_dict()
                st.session_state.tuned_lens_state = state_dict
                st.session_state.tuned_lens_obj = TunedLens(model.cfg.n_layers, model.cfg.d_model, model.cfg.d_vocab)
                st.session_state.tuned_lens_obj.load_state_dict(state_dict)
                st.session_state.tuned_lens_trained = True
                st.success("TunedLens trained and ready. ✅")

if st.session_state.tuned_lens_trained:
    tuned_lens = st.session_state.tuned_lens_obj
else:
    tuned_lens = None

# 2. The GUI Input
lens_type = st.radio("Choose Lens Type:", ["Logit Lens", "Tuned Lens"])
prompt = st.text_input("Enter a prompt:", "The sky is blue and the grass is")

if prompt:
    if lens_type == "Tuned Lens" and st.session_state.tuned_lens_trained is not True:
        st.warning("TunedLens is not trained yet. Please click the training button.")
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
            # What is THIS specific layer's top guess and its confidence?
            top_guess_id = layer_logits[pos].argmax().item()
            top_guess_prob = layer_probs[pos, top_guess_id].item()
            top_guess_str = model.to_string(top_guess_id)
            
            # Also show confidence in the final prediction
            target_id = final_token_ids[pos]
            prob_of_target = layer_probs[pos, target_id].item()
            
            heatmap_probs[layer, pos] = top_guess_prob
            
            # Format the interactive hover text
            hover_text = (
                f"Input Token: {str_tokens[pos]!r}<br>"
                f"Layer {layer} Top Guess: <b>{top_guess_str!r}</b><br>"
                f"Layer Confidence: {top_guess_prob:.2%}<br>"
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
        colorbar_title="Layer Confidence"
    ))
    
    fig.update_layout(
        title="Evolution of Layer Prediction Confidence",
        xaxis_title="Input Sequence",
        yaxis_title="Model Layer",
        xaxis_nticks=len(str_tokens)
    )
    
    # Render in Streamlit
    st.plotly_chart(fig, width='stretch')