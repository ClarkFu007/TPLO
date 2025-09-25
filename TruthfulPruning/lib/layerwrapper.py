import torch
import torch.nn as nn

# Define WrappedGPT class
class WrappedGPT:
    """
       This class wraps a GPT layer for specific operations.
    The WrappedGPT class is designed to wrap a GPT layer
    (like nn.Linear) and accumulate statistics about the
    inputs fed into the layer over time. The add_batch method
    processes each batch of inputs, reshaping them if necessary,
    computing L2 norms, and maintaining a running average in
    scaler_row. This kind of functionality is typically used
    for monitoring, debugging, or applying certain transformations
    during model training or inference.
    """
    def __init__(self, layer, layer_id=0, layer_name="none"):
        self.layer = layer  # nn.Linear layer
        self.dev = self.layer.weight.device
        self.rows = layer.weight.data.shape[0]
        self.columns = layer.weight.data.shape[1]

        """
           Accumulate some running statistics (related to norms of 
        inputs) across batches.
        """
        self.scaler_row = torch.zeros(self.columns, device=self.dev)

        self.nsamples = 0
        self.layer_id = layer_id 
        self.layer_name = layer_name

    def add_batch(self, inp, out):
        """
           This method processes a batch of data,
        specifically updating the running statistics
        for the self.scaler_row.
        """
        if len(inp.shape) == 2:
            """
               If the input is 2-dimensional, it adds 
            an additional dimension using unsqueeze(0) 
            to make the input 3-dimensional.
            """
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear):
            if len(inp.shape) == 3:
                """
                   If the input is 3-dimensional (e.g., 
                [batch_size, seq_len, num_features]), it 
                flattens it to [batch_size * seq_len, 
                num_features]. 
                """
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()  # matrix transpose

        """
           Scaling it by the ratio of the number of previously 
        seen samples (self.nsamples) to the total number of samples 
        after adding the current batch (self.nsamples + tmp). 
        """
        self.scaler_row *= self.nsamples / (self.nsamples + tmp)

        """
           Updates the total count of samples processed by 
        adding tmp, the number of samples in the current batch.
        """
        self.nsamples += tmp

        inp = inp.type(torch.float32)
        """
           torch.norm(inp, p=2, dim=1): Computes the L2 norm 
        (Euclidean norm) of each row (along dim=1) of the input. 
        This gives the magnitude of each input vector. This line 
        updates self.scaler_row by adding the square of the L2 
        norm of the input (normalized by the total number of 
        samples processed). It accumulates this information 
        across batches.
        """
        self.scaler_row += torch.norm(inp, p=2, dim=1) ** 2 / self.nsamples
