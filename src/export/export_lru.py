
import numpy as np
import torch


#Groups all Reals then all Imags; requires a manual C++ loop
def export_lru_compat(model, output_file):
 
    lamda, B, C = model.get_complex_params()
    weights = np.concatenate([
        np.array([model.in_features], dtype=np.float32), 
        np.array([model.hidden_size], dtype=np.float32), 
        np.array([model.window_size], dtype=np.float32), 
        lamda.real.detach().cpu().numpy().flatten(),
        lamda.imag.detach().cpu().numpy().flatten(),
        B.real.detach().cpu().numpy().flatten(),
        B.imag.detach().cpu().numpy().flatten(),
        C.real.detach().cpu().numpy().flatten(),
        C.imag.detach().cpu().numpy().flatten()
    ]).astype(np.float32)
    weights.tofile(output_file)
    print(f"Success! {output_file} saved. ({len(weights) * 4} bytes)")        

#C++ can "zero-copy" read the file
def export_lru_ultra(model, output_file):
    lamda, B, C = model.get_complex_params()

    header = np.array([model.in_features, model.hidden_size, model.window_size], dtype=np.float32)
    lamda_flat = torch.view_as_real(lamda).detach().cpu().numpy().flatten()
    B_flat = torch.view_as_real(B).detach().cpu().numpy().flatten()
    C_flat = torch.view_as_real(C).detach().cpu().numpy().flatten()

    weights = np.concatenate([
        header,
        lamda_flat,
        B_flat,
        C_flat
    ]).astype(np.float32)
    
    weights.tofile(output_file)
    print(f"Success! {output_file} saved. ({len(weights) * 4} bytes)")    