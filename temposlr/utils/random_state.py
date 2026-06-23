import torch
import random
import numpy as np


class RandomState(object):
    def __init__(self, seed):
        torch.set_num_threads(1)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

    def save_rng_state(self):
        rng_dict = {}
        rng_dict["torch"] = torch.get_rng_state()
        rng_dict["cuda"] = torch.cuda.get_rng_state_all()
        rng_dict["numpy"] = np.random.get_state()
        rng_dict["random"] = random.getstate()
        return rng_dict

    def set_rng_state(self, rng_dict):
        try:
            # Ensure torch RNG state is a CPU ByteTensor
            torch_state = rng_dict.get("torch", None)
            if torch_state is not None:
                if isinstance(torch_state, torch.Tensor):
                    torch_state = torch_state.cpu()
                    if torch_state.dtype != torch.uint8:
                        torch_state = torch_state.to(dtype=torch.uint8)
                else:
                    try:
                        torch_state = torch.tensor(torch_state, dtype=torch.uint8)
                    except Exception:
                        torch_state = torch.tensor(
                            np.asarray(torch_state), dtype=torch.uint8
                        )
                torch.set_rng_state(torch_state)

            # Ensure cuda rng states are list of CPU ByteTensors
            cuda_state = rng_dict.get("cuda", None)
            if cuda_state is not None:
                cuda_states_converted = []
                for s in cuda_state:
                    if isinstance(s, torch.Tensor):
                        s_cpu = s.cpu()
                        if s_cpu.dtype != torch.uint8:
                            s_cpu = s_cpu.to(dtype=torch.uint8)
                    else:
                        try:
                            s_cpu = torch.tensor(s, dtype=torch.uint8)
                        except Exception:
                            s_cpu = torch.tensor(np.asarray(s), dtype=torch.uint8)
                    cuda_states_converted.append(s_cpu)
                torch.cuda.set_rng_state_all(cuda_states_converted)
            np.random.set_state(rng_dict["numpy"])
            random.setstate(rng_dict["random"])
        except Exception as e:
            print(
                f"Warning: Failed to restore RNG state: {e}. Skipping RNG state restoration."
            )
