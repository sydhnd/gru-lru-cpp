import struct
import torch
import numpy as np

def write_tensor(f, tensor):
    tensor = tensor.detach().cpu().contiguous().float()
    data = tensor.numpy().ravel()
    f.write(struct.pack("<I", len(data)))
    f.write(data.astype("float32").tobytes())

def export_gru_binary(model, output_file):
    """
    Exports GRU weights to a custom binary format.
    Format: [Magic(4b)][Version(4b)][InputSize(4b)][HiddenSize(4b)][Tensors...]
    """
    model_cpu = model.detach().cpu() if hasattr(model, "detach") else model.cpu()
    model_cpu.eval()

    gru = model_cpu.gru
    input_size = gru.input_size
    hidden_size = gru.hidden_size
    state = model_cpu.state_dict()

    with open(output_file, "wb") as f:
        f.write(b"GRU1")                # Magic
        f.write(struct.pack("<I", 1))    # Version

        # Metadata (8 bytes)
        f.write(struct.pack("<I", input_size))
        f.write(struct.pack("<I", hidden_size))

        # GRU Layer 0 Parameters
        # PyTorch gate order: [reset, update, new]
        write_tensor(f, state["gru.weight_ih_l0"])
        write_tensor(f, state["gru.weight_hh_l0"])
        write_tensor(f, state["gru.bias_ih_l0"])
        write_tensor(f, state["gru.bias_hh_l0"])

        # Linear Output Layer
        write_tensor(f, state["out.weight"])
        write_tensor(f, state["out.bias"])

    print(f"Exported model to: {output_file}")