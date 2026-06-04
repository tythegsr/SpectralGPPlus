import shutil
import tempfile

import torch

from .base import BenchmarkProblem


class MOPTA08Car(BenchmarkProblem):

    r'''
    https://leonard.papenmeier.io/2023/02/09/mopta08-executables.html
    '''

    available_dimensions = 124
    num_objectives = 1

    # 124D objective, 68 constraints, X = n-by-124

    tags = {"single_objective", "constrained", "continuous", "124D", "extra_imports"}

    def __init__(self):
        super().__init__(dim = 124, 
                         num_obj = 1, 
                         num_cons = 68, 
                         bounds = [(0, 1)]*124,
                         optimum = [222.74])

    def evaluate(self, X):

        import os
        import platform
        import stat
        import subprocess
        import sys
        import tempfile
        from pathlib import Path

        import numpy as np

        def MOPTA08_Car_single(x, device_str):

            import os
            import platform
            import shutil
            import stat
            import subprocess
            import sys
            import tempfile
            from pathlib import Path

            import numpy as np

            # Determine the correct executable name based on platform
            machine = platform.machine().lower()
            sysarch = 64 if sys.maxsize > 2 ** 32 else 32

            if machine == "armv7l":
                assert sysarch == 32, "Not supported"
                mopta_exectuable = "mopta08_armhf.bin"
            elif machine == "x86_64":
                assert sysarch == 64, "Not supported"
                mopta_exectuable = "mopta08_elf64.bin"
            elif machine == "i386":
                assert sysarch == 32, "Not supported"
                mopta_exectuable = "mopta08_elf32.bin"
            else:
                raise RuntimeError("Machine with this architecture is not supported")

            # Get the original directory containing Mopta_Data and the executable
            script_dir = Path(__file__).parent
            original_data_dir = script_dir / "Mopta_Data"
            
            # Build the full path to the executable in the original directory
            original_executable_path = original_data_dir / mopta_exectuable
            original_executable_path = os.path.join(original_executable_path)

            # Ensure the executable has execute permissions
            if not os.access(original_executable_path, os.X_OK):
                print(f"Adding execution permissions to: {original_executable_path}")
                os.chmod(original_executable_path, os.stat(original_executable_path).st_mode |
                        stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

            # Create a unique temporary working directory
            with tempfile.TemporaryDirectory() as work_dir:
                # Copy the entire Mopta_Data directory to the temporary directory
                temp_data_dir = os.path.join(work_dir, "Mopta_Data")
                shutil.copytree(original_data_dir, temp_data_dir)

                # Build a new executable path pointing to the temporary directory version
                temp_executable_path = os.path.join(temp_data_dir, mopta_exectuable)
                
                # Write the input file in the temporary directory instead of the original
                input_path = os.path.join(temp_data_dir, "input.txt")
                with open(input_path, "w") as tmp_file:
                    for _x in x:
                        tmp_file.write(f"{_x}\n")
                
                # Run the executable in the temporary directory context
                result = subprocess.run(
                    temp_executable_path,
                    stdout=subprocess.PIPE,
                    cwd=temp_data_dir,  # Use the temporary data directory as CWD
                    shell=True,
                )
                
                # Read the output file from the temporary directory
                output_path = os.path.join(temp_data_dir, "output.txt")
                with open(output_path, "r") as tmp_file:
                    tmp_file.seek(0)
                    output = tmp_file.read().split("\n")
                
                # Clean up the output list: remove whitespace and empty strings, then convert to floats
                output = [m.strip() for m in output if len(m.strip()) > 0]
                output = np.array([float(m) for m in output])
                value = output[0]
                constraints = output[1:]
                
                return constraints, value


        DEVICE = X.device
        X = X.detach().cpu()

        fx = np.zeros((X.shape[0], 1))
        gx = np.zeros((X.shape[0], 68))

        for i in range(X.shape[0]):
            # Get objectives and constraints for each row
            gx[i], fx[i] = MOPTA08_Car_single(X[i,:].numpy(), str(DEVICE))

      #   return torch.from_numpy(gx), torch.from_numpy(fx)
        penalty = torch.clamp(torch.from_numpy(gx), min=0).sum(dim=1, keepdim=True)  # shape [N, 1]
        OUTPUT = -(torch.from_numpy(fx) + 10.0 * penalty)

        return None, OUTPUT.to(DEVICE)