from transformer_lens import HookedTransformer
import torch

# 1. Load the model
model: HookedTransformer = HookedTransformer.from_pretrained("gpt2")

# 2. Define the prompt (we leave off "green" to see how the model predicts it)
prompt = "The sky is blue and the grass is"
logits, cache = model.run_with_cache(prompt)

# Print the input tokens just to see how the model tokenizes our prompt
tokens = model.to_tokens(prompt)
str_tokens = model.to_str_tokens(tokens)
print(f"Input tokens: {str_tokens}\n")

print("--- Logit Lens: Predicting the next word layer by layer ---")

# 3. Implement the Logit Lens
# We iterate through every layer in the model
for layer in range(model.cfg.n_layers):
    
    # A. Extract the residual stream after the attention and MLP blocks for this layer
    hook_name = f"blocks.{layer}.hook_resid_post"
    residual_stream = cache[hook_name] 
    
    # B. Get the hidden state for the last token
    # We slice as [:, -1:, :] to keep the shape [batch=1, pos=1, d_model] for the layernorm
    last_token_hidden_state = residual_stream[:, -1:, :]
    
    # C. Apply the final Layer Normalization
    # (The model usually applies this right before the final unembedding)
    normalized_state = model.ln_final(last_token_hidden_state)
    
    # D. Project to the vocabulary space using the unembedding layer
    layer_logits = model.unembed(normalized_state)
    
    # E. Find the most likely next token at this intermediate layer
    top_token_id = layer_logits[0, 0].argmax().item()
    top_token_str = model.to_string(top_token_id)
    
    print(f"Layer {layer:2d} prediction: {top_token_str!r}")