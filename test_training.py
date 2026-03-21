from transformer_lens import HookedTransformer
import torch
import torch.nn as nn
import random

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
        
    def train_probes(self, texts, batch_size=1, epochs=1, lr=1e-3):
        """Train the probes on a dataset of texts"""
        optimizer = torch.optim.Adam(self.probes.parameters(), lr=lr)
        
        # Tokenize all texts
        all_tokens = []
        for text in texts:
            try:
                tokens = self.model.to_tokens(text)
                if tokens.shape[1] > 1:  # Ensure we have at least one token to predict
                    all_tokens.append(tokens)
            except:
                continue  # Skip problematic texts
        
        if not all_tokens:
            return
            
        print(f"Training on {len(all_tokens)} text samples for {epochs} epochs")
        
        # Training loop
        for epoch in range(epochs):
            print(f"Starting epoch {epoch+1}/{epochs}")
            total_loss = 0
            n_batches = 0
            
            # Shuffle and batch
            indices = torch.randperm(len(all_tokens))
            for i in range(0, len(all_tokens), batch_size):
                print(f"Processing batch {i//batch_size + 1}")
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
                print(f"Batch shape: {batch_tokens.shape}")
                
                # Get final logits and intermediate representations once
                print("Running model forward pass...")
                with torch.no_grad():
                    final_logits, cache = self.model.run_with_cache(batch_tokens)
                    final_target_logits = final_logits[:, :-1, :].detach()  # Shift by 1 to align with predictions
                    final_target_probs = torch.softmax(final_target_logits, dim=-1)
                
                print("Computing losses...")
                # Collect all layer losses first, then update
                layer_losses = []
                for layer in range(min(2, self.n_layers)):  # Test with just first 2 layers
                    hook_name = f"blocks.{layer}.hook_resid_post"
                    resid_stream = cache[hook_name][:, :-1]  # Remove last token (no prediction needed)
                    
                    # Get probe predictions
                    probe_logits = self.probes[layer](resid_stream)
                    probe_log_probs = torch.log_softmax(probe_logits, dim=-1)
                    
                    # Compute soft cross-entropy loss to final model distribution
                    loss = - (final_target_probs.reshape(-1, self.d_vocab) * probe_log_probs.reshape(-1, self.d_vocab)).sum(dim=-1).mean()
                    layer_losses.append(loss)
                
                print("Updating parameters...")
                # Update all probes
                optimizer.zero_grad()
                total_batch_loss = sum(layer_losses)
                total_batch_loss.backward()
                optimizer.step()
                
                total_loss += total_batch_loss.item()
                n_batches += 1
            
            print(f"Epoch {epoch+1}/{epochs}, Avg Loss: {total_loss / n_batches:.4f}")

# Test basic functionality
print("Loading model...")
model = HookedTransformer.from_pretrained("gpt2")
print("Model loaded")

# Test text loading
print("Loading texts...")
try:
    with open("tinystories_10k.txt", "r", encoding="utf-8", errors="ignore") as f:
        full_text = f.read()
    training_texts = [story.strip() for story in full_text.split("<|endoftext|>")]
    training_texts = [s for s in training_texts if len(s) > 50]
    training_texts = random.sample(training_texts, min(2, len(training_texts)))  # Just 2 samples
    print(f"Loaded {len(training_texts)} texts")
except Exception as e:
    print(f"Error loading texts: {e}")
    training_texts = ["The sky is blue and the grass is"]

# Test TunedLens
print("Creating TunedLens...")
tuned_lens = TunedLens(model)
print("TunedLens created")

print("Testing training...")
tuned_lens.train_probes(training_texts, epochs=1, batch_size=1)

print("Test completed successfully!")