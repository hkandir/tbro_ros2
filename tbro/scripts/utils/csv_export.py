import torch
import numpy as np
import os

def export_tensor_to_csv(tensor,trans_or_rot,base_name):
    tensor_np = tensor.detach().cpu().numpy()
    tensor_np_0 = tensor_np[:,:,0]
    tensor_np_1 = tensor_np[:,:,1]
    tensor_np_2 = tensor_np[:,:,2]
    if trans_or_rot == 'trans':
        np.savetxt(os.path.join(base_name + "x.csv"), tensor_np_0, delimiter = ",")
        np.savetxt(os.path.join(base_name + "y.csv"), tensor_np_1, delimiter = ",")
        np.savetxt(os.path.join(base_name + "z.csv"), tensor_np_2, delimiter = ",")
    elif trans_or_rot == 'rot':
        np.savetxt(os.path.join(base_name + "r.csv"), tensor_np_0, delimiter = ",")
        np.savetxt(os.path.join(base_name + "p.csv"), tensor_np_1, delimiter = ",")
        np.savetxt(os.path.join(base_name + "q.csv"), tensor_np_2, delimiter = ",")
