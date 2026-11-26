# Security policy

tinylm downloads public datasets over HTTPS from the Hugging Face Hub and loads
PyTorch checkpoints with `torch.load(..., weights_only=False)`. **Only load
checkpoints you produced yourself**: a pickled checkpoint from an untrusted
source can execute code. The GGUF files it writes contain only tensors and
metadata.

## Reporting

Please use GitHub's private vulnerability reporting
(**Security -> Report a vulnerability** on the repository page) rather than a
public issue. Include the command line and the output. Fixes land on `main`;
there are no maintained release branches.
