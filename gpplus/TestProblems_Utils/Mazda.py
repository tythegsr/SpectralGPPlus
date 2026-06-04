import time

import numpy as np
import torch

from .base import BenchmarkProblem


class Mazda_SCA(BenchmarkProblem):

    r'''
    https://ladse.eng.isas.jaxa.jp/benchmark/
    '''

    available_dimensions = 148
    num_objectives = 4

    # 222D objective, 54 constraints, X = n-by-222
    # 2 Cars Optimization Case

    tags = {"single_objective", "multi_objective", "constrained", "continuous", "222D", "extra_imports"}

    def __init__(self):
        super().__init__(dim = 148, 
                         num_obj = 4, 
                         num_cons = 36, 
                         bounds = [(0, 1)]*148 # Scaled upon evaluation
                         )

    def evaluate(self, X):
      import os
      import shutil
      import stat
      import subprocess
      import tempfile
      from pathlib import Path

      import numpy as np
      import pandas as pd
      import torch

      DEVICE = X.device
      X = X.detach().cpu()

      # Get the path to the original Mazda_Data directory
      script_dir = Path(__file__).parent
      original_data_dir = script_dir / "Mazda_Data"

      # Create a temporary working directory and copy Mazda_Data into it
      with tempfile.TemporaryDirectory() as work_dir:
          temp_data_dir = os.path.join(work_dir, "Mazda_Data")
          shutil.copytree(original_data_dir, temp_data_dir)

          ##########################################
          # Scaling
          ##########################################
          # Read the Excel file (using the temporary copy)
          file_path = Path(temp_data_dir) / "Info_Mazda_CdMOBP.xlsx"
          dataframe = pd.read_excel(file_path, 
                                    sheet_name='Explain_DV_and_Const.',
                                    engine="openpyxl"
                                    )

          # Get bounds from the Excel file, then rearrange bounds by stacking slices
          bounds = dataframe.values[2:, 3:5].astype(float)
          bounds = np.vstack((bounds[:74], bounds[-74:]))
          bounds_tensor = torch.tensor(bounds, dtype=torch.float32)
          range_bounds = bounds_tensor[:, 1] - bounds_tensor[:, 0]

          # Scale the samples accordingly
          scaled_samples = X * range_bounds + bounds_tensor[:, 0]
          data_numpy_back = scaled_samples.numpy()
          dataframe_back = pd.DataFrame(data_numpy_back)

          # Write the scaled samples to a text file in the temporary directory
          output_file_path = Path(temp_data_dir) / "pop_vars_eval.txt"
          dataframe_back.to_csv(output_file_path, sep='\t', header=False, index=False)

          #####################
          # Run Bash file
          #####################
          # Define the path to the binary (using the temporary copy)
          bin_path = Path(temp_data_dir) / "bin" / "mazda_mop_sca"
          if not os.access(bin_path, os.X_OK):
              print(f"Adding execution permissions to: {bin_path}")
              os.chmod(bin_path, os.stat(bin_path).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

          # Run the bash binary with the temporary copy as its input directory.
          subprocess.run(
              [str(bin_path), str(temp_data_dir)],
              stdout=subprocess.PIPE,
              stderr=subprocess.PIPE,
              start_new_session=True
          )

          #####################
          # Read in objective and constraints
          #####################
          # Read in the objectives file (from the temporary copy)
          objs_file_path = Path(temp_data_dir) / "pop_objs_eval.txt"
          objs_dataframe = pd.read_csv(objs_file_path, delim_whitespace=True, header=None)
          objs_data_numpy = objs_dataframe.values
          objs_data_tensor = torch.tensor(objs_data_numpy, dtype=torch.float32)

          # Read in the constraints file (from the temporary copy)
          cons_file_path = Path(temp_data_dir) / "pop_cons_eval.txt"
          cons_dataframe = pd.read_csv(cons_file_path, delim_whitespace=True, header=None)
          cons_data_numpy = cons_dataframe.values
          cons_data_tensor = torch.tensor(cons_data_numpy, dtype=torch.float32)

          # Compute the objective mean and penalty, then form the final output
          obj_mean = torch.mean(objs_data_tensor, dim=1).reshape(-1, 1)
          penalty = torch.clamp(cons_data_tensor, min=0).sum(dim=1, keepdim=True)
          OUTPUT = -(obj_mean + 10.0 * penalty)

          # Return the results after moving the output back to the original device.
          return None, OUTPUT.to(DEVICE)


    # def evaluate(self, X):

    #     import os
    #     import subprocess
    #     import stat
    #     import pandas as pd
    #     from pathlib import Path

    #     DEVICE = X.device
    #     X = X.detach().cpu()

    #     ##########################################
    #     # Scaling
    #     ##########################################

    #     # Define the path to your Excel file
    #     file_path = Path(__file__).parent / "Mazda_Data" / "Info_Mazda_CdMOBP.xlsx"

    #     # Read the Excel file into a DataFrame
    #     dataframe = pd.read_excel(file_path, sheet_name='Explain_DV_and_Const.')

    #     bounds = dataframe.values[2:, 3:5].astype(float)

    #     bounds = np.vstack((bounds[:74], bounds[-74:]))
        
    #     bounds_tensor = torch.tensor(bounds, dtype=torch.float32)

    #     range_bounds = bounds_tensor[:,1] - bounds_tensor[:,0]

    #     scaled_samples = X * range_bounds + bounds_tensor[:,0]

    #     # Convert the torch tensor to a numpy array
    #     data_numpy_back = scaled_samples.numpy()

    #     # Create a pandas DataFrame from the numpy array
    #     dataframe_back = pd.DataFrame(data_numpy_back)

    #     # Write the DataFrame to a text file with space-separated values
    #     output_file_path = Path(__file__).parent / "Mazda_Data" / f"pop_vars_eval.txt"

    #     dataframe_back.to_csv(output_file_path, sep='\t', header=False, index=False)

    #     #####################
    #     # Run Bash file
    #     #####################

    #     script_dir = Path(__file__).parent
    #     bin_path = script_dir / "Mazda_Data" / "bin" / "mazda_mop_sca"
    #     input_dir = script_dir / "Mazda_Data"

    #     if not os.access(bin_path, os.X_OK):
    #         print(f"Adding execution permissions to: {bin_path}")
    #         os.chmod(bin_path, os.stat(bin_path).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    #     # MUST BE ON A LINUX/UNIX MACHINE
    #     subprocess.run([str(bin_path), str(input_dir)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)

    #     #####################
    #     # Read in objective and constraints
    #     #####################

    #     # Read the data from the file into a pandas DataFrame
    #     file_path = script_dir / "Mazda_Data" / f"pop_objs_eval.txt"
        
    #     objs_dataframe = pd.read_csv(file_path, delim_whitespace=True, header=None)

    #     # Convert the DataFrame to a numpy array
    #     objs_data_numpy = objs_dataframe.values

    #     # Convert the numpy array to a torch tensor
    #     objs_data_tensor = torch.tensor(objs_data_numpy, dtype=torch.float32)
    #     # objs_data_tensor = objs_data_tensor[:,0].reshape(objs_data_tensor.shape[0],1)
    #     objs_data_tensor = objs_data_tensor

    #     # Read the data from the file into a pandas DataFrame
    #     file_path = script_dir / "Mazda_Data" / f"pop_cons_eval.txt"
    #     cons_dataframe = pd.read_csv(file_path, delim_whitespace=True, header=None)

    #     # Convert the DataFrame to a numpy array
    #     cons_data_numpy = cons_dataframe.values

    #     # Convert the numpy array to a torch tensor
    #     cons_data_tensor = torch.tensor(cons_data_numpy, dtype=torch.float32)

    #   #   return cons_data_tensor, -objs_data_tensor # original
    #     obj_mean = torch.mean(objs_data_tensor, dim=1).reshape(-1, 1)
    #   #   print("obj_mean", obj_mean.shape)
    #     penalty = torch.clamp(cons_data_tensor, min=0).sum(dim=1, keepdim=True)
    #   #   print("cons_data_tensor", cons_data_tensor.shape)
    #   #   print("penalty", penalty.shape)
    #     OUTPUT = -(obj_mean + 10.0 * penalty)

    #     return None, OUTPUT.to(DEVICE)


class Mazda(BenchmarkProblem):

    r'''
    https://ladse.eng.isas.jaxa.jp/benchmark/
    '''

    '''
    Meanings of each objective:
    - The first column is total weight of three vehicles.
    - The second column is number of common gauge parts.
    - The third column is weight of SUV.
    - The fourth column is weight of LV.
    - The fifth column is weight of SV.
    '''

    available_dimensions = 222
    num_objectives = 5

    # 222D objective, 54 constraints, X = n-by-222
    # 3 car optimization case

    tags = {"single_objective", "multi_objective", "constrained", "continuous", "222D", "extra_imports"}

    def __init__(self):
        super().__init__(dim = 222, 
                         num_obj = 5, 
                         num_cons = 54, 
                         bounds = [(0, 1)]*222 # Scaled upon evaluation
                         )

    def evaluate(self, X):
      import os
      import shutil
      import stat
      import subprocess
      import tempfile
      from pathlib import Path

      import pandas as pd
      import torch  # Ensure torch is imported

      DEVICE = X.device
      X = X.detach().cpu()

      # Get the original Mazda_Data directory
      script_dir = Path(__file__).parent
      original_data_dir = script_dir / "Mazda_Data"

      # Create a temporary working directory and copy Mazda_Data there
      with tempfile.TemporaryDirectory() as work_dir:
          # Create a temporary copy of the entire Mazda_Data directory.
          temp_data_dir = os.path.join(work_dir, "Mazda_Data")
          shutil.copytree(original_data_dir, temp_data_dir)

          ##########################################
          # Scaling
          ##########################################
          # Read the Excel file from the temporary directory.
          file_path = Path(temp_data_dir) / "Info_Mazda_CdMOBP.xlsx"
          dataframe = pd.read_excel(file_path, 
                                    sheet_name='Explain_DV_and_Const.',
                                    engine="openpyxl"
                                    )
          bounds = dataframe.values[2:, 3:5].astype(float)
          bounds_tensor = torch.tensor(bounds, dtype=torch.float32)
          range_bounds = bounds_tensor[:, 1] - bounds_tensor[:, 0]
          scaled_samples = X * range_bounds + bounds_tensor[:, 0]
          data_numpy_back = scaled_samples.numpy()

          # Create a pandas DataFrame from the scaled samples
          dataframe_back = pd.DataFrame(data_numpy_back)
          # Write the DataFrame to a text file in the temporary directory.
          output_file_path = Path(temp_data_dir) / "pop_vars_eval.txt"
          dataframe_back.to_csv(output_file_path, sep='\t', header=False, index=False)

          #####################
          # Run Bash file
          #####################
          # Prepare the binary path from the temporary copy.
          bin_path = Path(temp_data_dir) / "bin" / "mazda_mop"
          if not os.access(bin_path, os.X_OK):
              print(f"Adding execution permissions to: {bin_path}")
              os.chmod(bin_path, os.stat(bin_path).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

          # The input directory for the bash file is the temporary copy of Mazda_Data.
          subprocess.run([str(bin_path), str(temp_data_dir)],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        start_new_session=True)

          #####################
          # Read in objective and constraints
          #####################
          # Read the objectives file from the temporary directory.
          objs_file_path = Path(temp_data_dir) / "pop_objs_eval.txt"
          objs_dataframe = pd.read_csv(objs_file_path, delim_whitespace=True, header=None)
          objs_data_numpy = objs_dataframe.values
          objs_data_tensor = torch.tensor(objs_data_numpy, dtype=torch.float32)

          # Read the constraints file from the temporary directory.
          cons_file_path = Path(temp_data_dir) / "pop_cons_eval.txt"
          cons_dataframe = pd.read_csv(cons_file_path, delim_whitespace=True, header=None)
          cons_data_numpy = cons_dataframe.values
          cons_data_tensor = torch.tensor(cons_data_numpy, dtype=torch.float32)

          # Compute objective mean and penalty.
          obj_mean = torch.mean(objs_data_tensor, dim=1).reshape(-1, 1)
          penalty = torch.clamp(cons_data_tensor, min=0).sum(dim=1, keepdim=True)

          OUTPUT = -(obj_mean + 10.0 * penalty)
          # All processing that depends on the temporary files is done before exiting the 'with' block.
          return None, OUTPUT.to(DEVICE)

    # def evaluate(self, X):

    #     import os
    #     import subprocess
    #     import stat
    #     import pandas as pd
    #     from pathlib import Path

    #     DEVICE = X.device
    #     X = X.detach().cpu()

    #     ##########################################
    #     # Scaling
    #     ##########################################

    #     # Define the path to your Excel file
    #     file_path = Path(__file__).parent / "Mazda_Data" / "Info_Mazda_CdMOBP.xlsx"

    #     # Read the Excel file into a DataFrame
    #     dataframe = pd.read_excel(file_path, sheet_name='Explain_DV_and_Const.')

    #     bounds = dataframe.values[2:, 3:5].astype(float)
        
    #     bounds_tensor = torch.tensor(bounds, dtype=torch.float32)

    #     range_bounds = bounds_tensor[:,1] - bounds_tensor[:,0]

    #     scaled_samples = X * range_bounds + bounds_tensor[:,0]

    #     # Convert the torch tensor to a numpy array
    #     data_numpy_back = scaled_samples.numpy()

    #     # Create a pandas DataFrame from the numpy array
    #     dataframe_back = pd.DataFrame(data_numpy_back)

    #     # Write the DataFrame to a text file with space-separated values
    #     output_file_path = Path(__file__).parent / "Mazda_Data" / f"pop_vars_eval.txt"

    #     dataframe_back.to_csv(output_file_path, sep='\t', header=False, index=False)

    #     #####################
    #     # Run Bash file
    #     #####################

    #     script_dir = Path(__file__).parent
    #     bin_path = script_dir / "Mazda_Data" / "bin" / "mazda_mop"
    #     input_dir = script_dir / "Mazda_Data"

    #     if not os.access(bin_path, os.X_OK):
    #         print(f"Adding execution permissions to: {bin_path}")
    #         os.chmod(bin_path, os.stat(bin_path).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    #     # MUST BE ON A LINUX/UNIX MACHINE
    #     subprocess.run([str(bin_path), str(input_dir)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)

    #     #####################
    #     # Read in objective and constraints
    #     #####################

    #     # Read the data from the file into a pandas DataFrame
    #     file_path = script_dir / "Mazda_Data" / f"pop_objs_eval.txt"
    #     objs_dataframe = pd.read_csv(file_path, delim_whitespace=True, header=None)

    #     # Convert the DataFrame to a numpy array
    #     objs_data_numpy = objs_dataframe.values

    #     # Convert the numpy array to a torch tensor
    #     objs_data_tensor = torch.tensor(objs_data_numpy, dtype=torch.float32)
    #     # objs_data_tensor = objs_data_tensor[:,0].reshape(objs_data_tensor.shape[0],1)
    #     objs_data_tensor = objs_data_tensor

    #     # Read the data from the file into a pandas DataFrame
    #     file_path = script_dir / "Mazda_Data" / f"pop_cons_eval.txt"
    #     cons_dataframe = pd.read_csv(file_path, delim_whitespace=True, header=None)

    #     # Convert the DataFrame to a numpy array
    #     cons_data_numpy = cons_dataframe.values

    #     # Convert the numpy array to a torch tensor
    #     cons_data_tensor = torch.tensor(cons_data_numpy, dtype=torch.float32)

    #   #   return cons_data_tensor, -objs_data_tensor
    #     obj_mean = torch.mean(objs_data_tensor, dim=1).reshape(-1, 1)
    #     penalty = torch.clamp(cons_data_tensor, min=0).sum(dim=1, keepdim=True)

    #     OUTPUT = -(obj_mean + 10.0 * penalty)
    #     return None, OUTPUT.to(DEVICE)