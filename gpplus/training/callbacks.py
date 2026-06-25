from abc import ABC
from typing import Any, Callable, List, Optional, Tuple, TypedDict
import torch


def _seek_kernel_types() -> tuple[type, ...]:
    try:
        from gpplus.kernels.seek_kernel import SEEKKernel
        from gpplus.kernels.seek_kernel_trunk_head import SEEKKernelTrunkHead

        return (SEEKKernel, SEEKKernelTrunkHead)
    except ImportError:
        return ()


def _is_seek_kernel(kernel: Any) -> bool:
    seek_types = _seek_kernel_types()
    return bool(seek_types) and isinstance(kernel, seek_types)


class CallbackOnEpochStartContext(TypedDict):
    epoch: int
    model: Any
    trainer: Any
    device: str


class CallbackOnEpochEndContext(TypedDict):
    epoch: int
    model: Any
    trainer: Any
    loss: float
    device: str
    jitter: float  # Current jitter value used for this epoch


class CallbackOnTrainStartContext(TypedDict):
    model: Any
    trainer: Any
    device: str


class CallbackOnTrainEndContext(TypedDict):
    epoch: int
    model: Any
    trainer: Any
    best_loss: float
    best_state_dict: Any
    device: str


class Callback(ABC):
    def on_epoch_start(self, context: CallbackOnEpochStartContext):
        """
        Called at the start of each epoch during training.

        Args:
            context (dict): A dictionary containing training state info.
        """
        pass

    def on_epoch_end(self, context: CallbackOnEpochEndContext):
        """
        Called at the end of each epoch during training.

        Args:
            context (dict): A dictionary containing training state info.
        """
        pass

    def on_train_start(self, context: CallbackOnTrainStartContext):
        """
        Called at the start of each training.

        Args:
            context (dict): A dictionary containing training state info.
        """
        pass

    def on_train_end(self, context: CallbackOnTrainEndContext):
        """
        Called at the end of each training.

        Args:
            context (dict): A dictionary containing training state info.
        """
        pass


class PrintLossCallback(Callback):
    def on_epoch_end(self, context: dict):
        print(f"Epoch {context['epoch']} - Loss: {context['loss']:.4f}")


class PrintTrainingMetricsCallback(Callback):
    """Prints NLL, NIS, LOO_NLL, KF, MSE, and R2 each epoch when provided in context['metrics']."""

    def on_epoch_end(self, context: dict):
        metrics = context.get("metrics")
        if not metrics:
            return
        parts = [f"Epoch {context['epoch']} - Loss: {context['loss']:.4f}"]
        if "NLL" in metrics:
            parts.append(f"NLL: {metrics['NLL']:.4f}")
        if "NIS" in metrics:
            parts.append(f"NIS: {metrics['NIS']:.4f}")
        if "LOO_NLL" in metrics:
            parts.append(f"LOO_NLL: {metrics['LOO_NLL']:.4f}")
        if "KF" in metrics:
            parts.append(f"KF: {metrics['KF']:.4f}")
        if "MSE" in metrics:
            parts.append(f"MSE: {metrics['MSE']:.4f}")
        if "R2" in metrics:
            parts.append(f"R2: {metrics['R2']:.4f}")
        print(" | ".join(parts))


class PrintInitialParametersCallback(Callback):
    def on_train_start(self, context: dict):
        print("Initial parameters: ")
        for name, param in context["model"].named_parameters():
            print(name, param.data)


class KernelParameterExtractor:
    """
    Helper class to extract parameters from kernel structures in a nested path format.
    Recursively traverses kernel hierarchies and extracts all parameters without overwriting.
    """
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
    
    def extract_all_parameters(self, model, trainer=None) -> dict:
        """
        Extract all parameters from the model's kernel structure.
        
        Returns:
            dict: Nested dictionary with kernel parameters organized by path
        """
        result = {}
        
        # Extract from covar_module
        if hasattr(model, "covar_module") and model.covar_module is not None:
            result["covar_module"] = self._extract_from_kernel(model.covar_module, path="covar_module")
            # Optional: SEEK internal combination outputs (weight/bias terms used before exponential wrapper)
            # We only record lightweight summary stats, and only if training data is available.
            try:
                if trainer is not None and hasattr(trainer, "train_x") and trainer.train_x is not None:
                    seek_stats = self._extract_seek_internal_outputs(model.covar_module, trainer.train_x)
                    if seek_stats:
                        result["covar_module"]["seek_internal_outputs"] = seek_stats
            except Exception:
                # Never break training/recording if stats extraction fails
                pass
        
        # Extract from likelihood (noise)
        if hasattr(model, "likelihood") and model.likelihood is not None:
            noise_params = self._extract_noise(model.likelihood)
            if noise_params:
                result["likelihood"] = noise_params
        
        # Extract from mean_module
        if hasattr(model, "mean_module") and model.mean_module is not None:
            mean_params = self._extract_mean_module(model.mean_module)
            if mean_params:
                result["mean_module"] = mean_params
        
        return result

    def _extract_seek_internal_outputs(self, kernel, train_x):
        """
        For SEEK kernels, compute lightweight stats of the multiplicative "weight" term(s)
        and additive "bias" term used in the SEEK forward before any exponential wrapper.

        We compute diagonal-only outputs to avoid huge NxN dumps.
        """
        if not _is_seek_kernel(kernel):
            return None

        import torch

        def _stats_1d(vec: torch.Tensor):
            try:
                v = vec.detach()
                if v.is_cuda:
                    v = v.cpu()
                v = v.to(dtype=torch.float64)
                return {
                    "min": float(v.min().item()),
                    "max": float(v.max().item()),
                    "mean": float(v.mean().item()),
                    "std": float(v.std(unbiased=False).item()),
                    "numel": int(v.numel()),
                }
            except Exception:
                return None

        stats = {"weight_kernels_diag": [], "bias_kernel_diag": None}

        with torch.no_grad():
            # SEEK uses encoded features for weight/bias computation (mirrors kernel.forward)
            x = train_x
            if hasattr(kernel, "_encode_for_weights") and hasattr(kernel, "base_kernels") and len(kernel.base_kernels) > 0:
                base_kernel = kernel.base_kernels[0]
                x_encoded, _ = kernel._encode_for_weights(x, base_kernel, epsilon=None)
            else:
                x_encoded = x

            # Weight terms: one per base kernel
            if hasattr(kernel, "weight_kernels") and kernel.weight_kernels is not None:
                for wk in kernel.weight_kernels:
                    try:
                        diag_vec = wk(x_encoded, x_encoded, diag=True)
                        stats["weight_kernels_diag"].append(
                            {"stats": _stats_1d(diag_vec)}
                        )
                    except Exception:
                        stats["weight_kernels_diag"].append({"stats": None})

            # Bias term: optional
            if getattr(kernel, "use_bias", False) and hasattr(kernel, "bias_kernel") and kernel.bias_kernel is not None:
                try:
                    diag_vec = kernel.bias_kernel(x_encoded, x_encoded, diag=True)
                    stats["bias_kernel_diag"] = {"stats": _stats_1d(diag_vec)}
                except Exception:
                    stats["bias_kernel_diag"] = {"stats": None}
            else:
                stats["bias_kernel_diag"] = None

        return stats
    
    def _extract_from_kernel(self, kernel, path: str = "") -> dict:
        """Recursively extract parameters from a kernel object."""
        if kernel is None:
            return {}
        
        result = {}
        kernel_type = type(kernel).__name__
        result["type"] = kernel_type
        
        # Handle LogScaleKernel - wraps another kernel and has outputscale
        # Check this first so we can unwrap and continue extraction
        from gpplus.kernels.log_scale_kernel import LogScaleKernel
        if isinstance(kernel, LogScaleKernel):
            # Extract outputscale
            if hasattr(kernel, "raw_outputscale"):
                result["raw_outputscale"] = self._tensor_to_value(kernel.raw_outputscale)
            if hasattr(kernel, "outputscale"):
                result["outputscale"] = self._tensor_to_value(kernel.outputscale)
            
            # Recursively extract from base_kernel (unwrap LogScaleKernel)
            if hasattr(kernel, "base_kernel") and kernel.base_kernel is not None:
                base_result = self._extract_from_kernel(kernel.base_kernel, path=f"{path}.base_kernel")
                result["base_kernel"] = base_result
                # Return early since we've handled the wrapped kernel
                return result
        
        # Handle SEEKKernel and SEEKKernelTrunkHead - has base_kernels ModuleList, weight_kernels, and bias_kernel
        if _is_seek_kernel(kernel):
            if hasattr(kernel, "base_kernels") and len(kernel.base_kernels) > 0:
                base_kernels_list = []
                for idx, base_kernel in enumerate(kernel.base_kernels):
                    base_result = self._extract_from_kernel(base_kernel, path=f"{path}.base_kernels[{idx}]")
                    base_kernels_list.append(base_result)
                result["base_kernels"] = base_kernels_list
            
            # Extract weight kernels (neural network weights and biases)
            if hasattr(kernel, "weight_kernels") and len(kernel.weight_kernels) > 0:
                weight_kernels_list = []
                for idx, weight_kernel in enumerate(kernel.weight_kernels):
                    weight_result = self._extract_neural_network_params(
                        weight_kernel, path=f"{path}.weight_kernels[{idx}]"
                    )
                    weight_kernels_list.append(weight_result)
                result["weight_kernels"] = weight_kernels_list
            
            # Extract bias kernel (neural network weights and biases)
            if hasattr(kernel, "bias_kernel") and kernel.bias_kernel is not None:
                bias_result = self._extract_neural_network_params(
                    kernel.bias_kernel, path=f"{path}.bias_kernel"
                )
                result["bias_kernel"] = bias_result
        
        # Handle AdditiveKernel / ProductKernel (gpytorch) - have list of sub-kernels
        try:
            from gpytorch.kernels import AdditiveKernel, ProductKernel
        except Exception:  # pragma: no cover - defensive import
            AdditiveKernel = ProductKernel = tuple()  # type: ignore
        if isinstance(kernel, (AdditiveKernel, ProductKernel)):
            if hasattr(kernel, "kernels") and kernel.kernels is not None:
                subkernels = []
                for idx, sub_k in enumerate(kernel.kernels):
                    sub_result = self._extract_from_kernel(sub_k, path=f"{path}.kernels[{idx}]")
                    subkernels.append(sub_result)
                result["kernels"] = subkernels
        
        # Handle MVMFKernel - has cont_kernel, cat_kernel, source_kernel
        from gpplus.kernels.mvmf_kernel import MVMFKernel
        if isinstance(kernel, MVMFKernel):
            if hasattr(kernel, "cont_kernel") and kernel.cont_kernel is not None:
                cont_result = self._extract_from_kernel(kernel.cont_kernel, path=f"{path}.cont_kernel")
                result["cont_kernel"] = cont_result
            
            if hasattr(kernel, "cat_kernel") and kernel.cat_kernel is not None:
                cat_result = self._extract_from_kernel(kernel.cat_kernel, path=f"{path}.cat_kernel")
                result["cat_kernel"] = cat_result
            
            if hasattr(kernel, "source_kernel") and kernel.source_kernel is not None:
                source_result = self._extract_from_kernel(kernel.source_kernel, path=f"{path}.source_kernel")
                result["source_kernel"] = source_result
        
        # Handle base kernels (GaussianKernel, PowerExponentialKernel, MaternKernel, PeriodicKernel, etc.)
        # Extract lengthscale parameters
        if hasattr(kernel, "raw_lengthscale"):
            result["raw_lengthscale"] = self._tensor_to_value(kernel.raw_lengthscale)
        
        if hasattr(kernel, "lengthscale"):
            result["lengthscale"] = self._tensor_to_value(kernel.lengthscale)
        
        # Extract period parameters for periodic kernels
        if hasattr(kernel, "raw_period"):
            result["raw_period"] = self._tensor_to_value(kernel.raw_period)
        
        if hasattr(kernel, "period"):
            result["period"] = self._tensor_to_value(kernel.period)
        
        # Extract power parameter (PowerExponentialKernel)
        if hasattr(kernel, "raw_power"):
            result["raw_power"] = self._tensor_to_value(kernel.raw_power)
        
        if hasattr(kernel, "power"):
            result["power"] = self._tensor_to_value(kernel.power)
        
        # Extract outputscale if not already extracted (for kernels that have it directly)
        if "raw_outputscale" not in result and hasattr(kernel, "raw_outputscale"):
            result["raw_outputscale"] = self._tensor_to_value(kernel.raw_outputscale)
        
        if "outputscale" not in result and hasattr(kernel, "outputscale"):
            result["outputscale"] = self._tensor_to_value(kernel.outputscale)
        
        return result
    
    def _extract_noise(self, likelihood) -> dict:
        """Extract noise parameters from likelihood."""
        result = {}
        if hasattr(likelihood, "raw_noise"):
            result["raw_noise"] = self._tensor_to_value(likelihood.raw_noise)
        if hasattr(likelihood, "noise"):
            result["noise"] = self._tensor_to_value(likelihood.noise)
        return result
    
    def _extract_mean_module(self, mean_module) -> dict:
        """Extract parameters from mean module (e.g., ConstantMean, LinearMean, MultiMean)."""
        result = {}
        result["type"] = type(mean_module).__name__
        
        # Handle MultiMean - has a list of mean modules
        from gpplus.means.multi_mean import MultiMean
        if isinstance(mean_module, MultiMean):
            if hasattr(mean_module, "means") and mean_module.means is not None:
                means_list = []
                for idx, mean in enumerate(mean_module.means):
                    mean_result = self._extract_mean_module(mean)
                    means_list.append(mean_result)
                result["means"] = means_list
            return result
        
        # ConstantMean parameters
        if hasattr(mean_module, "raw_constant"):
            result["raw_constant"] = self._tensor_to_value(mean_module.raw_constant)
        if hasattr(mean_module, "constant"):
            result["constant"] = self._tensor_to_value(mean_module.constant)
        
        # LinearMean parameters - check both attributes and named_parameters
        # LinearMean may use 'weights' attribute or register it as a parameter
        if hasattr(mean_module, "weights"):
            result["weights"] = self._tensor_to_value(mean_module.weights)
        if hasattr(mean_module, "bias"):
            result["bias"] = self._tensor_to_value(mean_module.bias)
        
        # Check for weight and bias as named parameters (LinearMean stores them as parameters)
        # This handles cases where attributes don't exist but parameters do
        for name, param in mean_module.named_parameters():
            if "weight" in name.lower():
                if "weights" not in result:
                    result["weights"] = self._tensor_to_value(param)
                else:
                    # Store with full name if multiple weight parameters exist
                    result[name] = self._tensor_to_value(param)
            elif "bias" in name.lower():
                if "bias" not in result:
                    result["bias"] = self._tensor_to_value(param)
                else:
                    # Store with full name if multiple bias parameters exist
                    result[name] = self._tensor_to_value(param)
        
        return result
    
    def _extract_neural_network_params(self, composite_scale_kernel, path: str = "") -> dict:
        """Extract neural network weights and biases from CompositeScaleKernel."""
        result = {}
        result["type"] = type(composite_scale_kernel).__name__
        
        if hasattr(composite_scale_kernel, "input_transform"):
            network = composite_scale_kernel.input_transform
            
            # Handle InputTransformNet (has .network which is a Sequential)
            if hasattr(network, "network"):
                layers = []
                for layer_idx, layer in enumerate(network.network):
                    layer_dict = {}
                    layer_dict["type"] = type(layer).__name__
                    
                    # Extract weights and biases from Linear layers
                    if hasattr(layer, "weight"):
                        layer_dict["weight"] = self._tensor_to_value(layer.weight)
                        if hasattr(layer.weight, "shape"):
                            layer_dict["weight_shape"] = list(layer.weight.shape)
                    
                    if hasattr(layer, "bias") and layer.bias is not None:
                        layer_dict["bias"] = self._tensor_to_value(layer.bias)
                        if hasattr(layer.bias, "shape"):
                            layer_dict["bias_shape"] = list(layer.bias.shape)
                    
                    layers.append(layer_dict)
                
                result["layers"] = layers
            
            # Handle TrunkHeadNet (has .trunk and .head)
            elif hasattr(network, "trunk") or hasattr(network, "head"):
                # Extract trunk network
                if hasattr(network, "trunk") and hasattr(network.trunk, "network"):
                    trunk_layers = []
                    for layer_idx, layer in enumerate(network.trunk.network):
                        layer_dict = {}
                        layer_dict["type"] = type(layer).__name__
                        if hasattr(layer, "weight"):
                            layer_dict["weight"] = self._tensor_to_value(layer.weight)
                            if hasattr(layer.weight, "shape"):
                                layer_dict["weight_shape"] = list(layer.weight.shape)
                        if hasattr(layer, "bias") and layer.bias is not None:
                            layer_dict["bias"] = self._tensor_to_value(layer.bias)
                            if hasattr(layer.bias, "shape"):
                                layer_dict["bias_shape"] = list(layer.bias.shape)
                        trunk_layers.append(layer_dict)
                    result["trunk_layers"] = trunk_layers
                
                # Extract head network
                # In our codebase (`gpplus/utils/trunk_head_net.py`), `TrunkHeadNet.head` is an `nn.Sequential`,
                # not an object with `.network`. Handle both patterns.
                head_seq = None
                if hasattr(network, "head"):
                    head_seq = network.head
                if head_seq is not None:
                    head_layers = []
                    # If it is a Sequential, iterate over it. Otherwise, try `.network` as fallback.
                    iterable = None
                    if hasattr(head_seq, "__iter__"):
                        iterable = head_seq
                    elif hasattr(head_seq, "network"):
                        iterable = head_seq.network
                    else:
                        iterable = None

                    if iterable is not None:
                        for layer_idx, layer in enumerate(iterable):
                            layer_dict = {}
                            layer_dict["type"] = type(layer).__name__
                            if hasattr(layer, "weight"):
                                layer_dict["weight"] = self._tensor_to_value(layer.weight)
                                if hasattr(layer.weight, "shape"):
                                    layer_dict["weight_shape"] = list(layer.weight.shape)
                            if hasattr(layer, "bias") and layer.bias is not None:
                                layer_dict["bias"] = self._tensor_to_value(layer.bias)
                                if hasattr(layer.bias, "shape"):
                                    layer_dict["bias_shape"] = list(layer.bias.shape)
                            head_layers.append(layer_dict)
                        result["head_layers"] = head_layers
            
            # Fallback: try to extract from named_parameters if structure is unknown
            else:
                params_dict = {}
                for name, param in network.named_parameters():
                    params_dict[name] = self._tensor_to_value(param)
                if params_dict:
                    result["parameters"] = params_dict
        
        return result
    
    def _tensor_to_value(self, tensor) -> Any:
        """Convert tensor to Python value(s)."""
        if tensor is None:
            return None
        try:
            if hasattr(tensor, "data"):
                tensor = tensor.data
            if hasattr(tensor, "detach"):
                tensor = tensor.detach()
            if hasattr(tensor, "cpu"):
                tensor = tensor.cpu()
            
            if hasattr(tensor, "numel"):
                if tensor.numel() == 1:
                    return float(tensor.item())
                else:
                    return tensor.flatten().tolist()
            else:
                return float(tensor) if isinstance(tensor, (int, float)) else tensor
        except Exception as e:
            if self.verbose:
                print(f"Warning: Could not convert tensor to value: {e}")
            return None


class FinalParameterStorageCallback(Callback):
    """
    Callback to store final raw parameters (raw_lengthscales, raw_outputscales, raw_noise)
    from each model after training for later reference in results files.

    Args:
        save_file (str, optional): Path to save the parameters JSON file. If None, no file is written.
        verbose (bool): If True, print parameter values to console when storing. Does not affect saving.
    """

    def __init__(self, save_file: str = None, verbose: bool = False):
        self.save_file = save_file
        self.verbose = verbose
        self.stored_parameters = []
        self._kernel_extractor = KernelParameterExtractor(verbose=verbose)
        self._initial_params = None
        self._initial_params_flat = None
        self._initial_jitter = None
        self._run_count = 0
        self._best_epoch = 0  # Track best epoch when loss improves
        self._current_best_loss = float("inf")
        self._metrics_at_best = None  # NLL, NIS, LOO_NLL, KF, MSE, R2 at best epoch (for trainer analysis JSON)
        self._current_run_index = None
        self._current_fold_index = None
        self._current_num_folds = None
        self._epoch_metrics_list = None  # Per-epoch metrics for this run (list of dicts)

    def _run_fold_label(self, context: dict) -> str:
        """Build label for display: 'Fold X/Y — Init Z' or 'Init Z' when no fold info."""
        run_index = context.get("run_index")
        fold_index = context.get("fold_index")
        num_folds = context.get("num_folds")
        init_label = f"Init {run_index}" if run_index is not None else f"Run {self._run_count}"
        if fold_index is not None and num_folds is not None:
            return f"Fold {fold_index}/{num_folds} — {init_label}"
        return init_label

    def on_train_start(self, context: dict):
        """Capture initial parameters at the start of training."""
        model = context["model"]
        self._run_count += 1
        self._current_run_index = context.get("run_index")
        self._current_fold_index = context.get("fold_index")
        self._current_num_folds = context.get("num_folds")
        self._best_epoch = 0
        self._current_best_loss = float("inf")
        self._metrics_at_best = None
        self._epoch_metrics_list = []
        trainer = context.get("trainer")
        # Capture initial jitter (for initial_jitter / final jitter in each init record)
        if trainer is not None and hasattr(trainer, "cholesky_jitter"):
            j = trainer.cholesky_jitter
            self._initial_jitter = float(j) if j is not None and (isinstance(j, (int, float)) or hasattr(j, "item")) else None
        else:
            self._initial_jitter = None
        try:
            self._initial_params = self._kernel_extractor.extract_all_parameters(model, trainer=trainer)
        except Exception:
            self._initial_params = None
        try:
            self._initial_params_flat = self._extract_final_parameters(model, epoch=0, best_loss=None)
        except Exception:
            self._initial_params_flat = None
        if self.verbose and self._initial_params_flat is not None:
            label = self._run_fold_label(context)
            flat = self._initial_params_flat
            print(f"\n=== Initial Parameters ({label}, Epoch 0) ===")
            print(f"Raw noise: {flat.get('raw_noise')}")
            print(f"Raw outputscale: {flat.get('raw_outputscale')}")
            raw_lengthscales = flat.get("raw_lengthscales", [])
            if raw_lengthscales:
                print(f"Raw lengthscales: {raw_lengthscales} (count: {len(raw_lengthscales)})")
            else:
                print(f"Raw lengthscales: N/A (not found)")
            raw_cat_lengthscales = flat.get("raw_cat_lengthscales", [])
            if raw_cat_lengthscales:
                print(f"Raw cat lengthscales: {raw_cat_lengthscales} (count: {len(raw_cat_lengthscales)})")
            raw_source_lengthscales = flat.get("raw_source_lengthscales", [])
            if raw_source_lengthscales:
                print(f"Raw source lengthscales: {raw_source_lengthscales} (count: {len(raw_source_lengthscales)})")
            periods = flat.get("periods", [])
            if periods:
                print(f"Period (transformed): {periods} (count: {len(periods)})")
            raw_periods = flat.get("raw_periods", [])
            if raw_periods:
                print(f"Raw period: {raw_periods} (count: {len(raw_periods)})")

    @staticmethod
    def _to_float(x: Any) -> float:
        """Convert metric value to JSON-serializable float."""
        if x is None:
            return None
        if hasattr(x, "item"):
            return float(x.item())
        return float(x)

    def on_epoch_end(self, context: dict):
        """Track best epoch when loss improves; save metrics at best epoch; record metrics every epoch."""
        loss = context.get("loss")
        epoch = context.get("epoch", 0)
        if loss is not None and loss < self._current_best_loss:
            self._current_best_loss = loss
            self._best_epoch = epoch
            self._metrics_at_best = context.get("metrics")  # NLL, NIS, LOO_NLL, KF, MSE, R2 when logged
        # Record metrics at every epoch for reporting
        metrics = context.get("metrics") or {}
        epoch_record = {"epoch": int(epoch), "loss": self._to_float(loss)}
        for key in ("NLL", "NIS", "LOO_NLL", "KF", "MSE", "R2"):
            if key in metrics:
                epoch_record[key] = self._to_float(metrics[key])
        self._epoch_metrics_list.append(epoch_record)

    def on_train_end(self, context: dict):
        """Store final parameters at the end of training."""
        model = context["model"]
        epoch = context["epoch"]
        best_loss = context.get("best_loss", None)
        trainer = context.get("trainer", None)
        self._current_run_index = context.get("run_index")
        self._current_fold_index = context.get("fold_index")
        self._current_num_folds = context.get("num_folds")
        # Get cholesky_jitter from trainer if available
        cholesky_jitter = None
        if trainer is not None and hasattr(trainer, "cholesky_jitter"):
            cholesky_jitter = trainer.cholesky_jitter
        # Max jitter actually used during this run (tracked in GPTrainerSingleProcess)
        jitter_max = context.get("jitter_max", None)
        best_state_dict = context.get("best_state_dict", None)

        try:
            # If best_state_dict is available, load it temporarily to extract parameters from best model
            # Otherwise, use the current model state
            if best_state_dict is not None:
                import copy

                temp_model = copy.deepcopy(model)
                temp_model.load_state_dict(best_state_dict)
                model_to_extract = temp_model
            else:
                model_to_extract = model

            # Nested structure (covar_module, mean_module, likelihood) for JSON
            final_params_nested = self._kernel_extractor.extract_all_parameters(model_to_extract, trainer=trainer)
            # Flat structure for metadata and deltas
            final_params = self._extract_final_parameters(
                model_to_extract, epoch, best_loss, cholesky_jitter, self._best_epoch, jitter_max=jitter_max
            )
            best_lbfgs_iter = context.get("best_lbfgs_iter")
            record = self._combine_initial_final(
                self._initial_params, final_params_nested, self._initial_params_flat, final_params,
                best_lbfgs_iter=best_lbfgs_iter,
            )
            lbfgs_stop_reason = context.get("lbfgs_stop_reason")
            if lbfgs_stop_reason is not None:
                record["lbfgs_stop_reason"] = lbfgs_stop_reason
            # Add training metrics at best epoch (NLL, NIS, LOO_NLL, KF, MSE, R2) for trainer analysis JSON
            if self._metrics_at_best:
                for key in ("NLL", "NIS", "LOO_NLL", "KF", "MSE", "R2"):
                    if key in self._metrics_at_best:
                        record[key] = self._metrics_at_best[key]
            # Per-epoch metrics: only when NOT LBFGS (we have lbfgs_inner_metrics instead)
            if best_lbfgs_iter is None and self._epoch_metrics_list:
                record["epoch_metrics"] = self._epoch_metrics_list
            self.stored_parameters.append(record)

            if self.verbose:
                label = self._run_fold_label(context)
                print(f"\n=== Final Parameters ({label}, Epoch {epoch}) ===")
                print(f"Number of epochs: {final_params.get('num_epochs', 'N/A')}")
                if best_lbfgs_iter is not None:
                    print(f"Best iter: {best_lbfgs_iter}")
                else:
                    print(f"Best epoch: {final_params.get('best_epoch', 'N/A')}")
                print(f"Jitter: {final_params.get('jitter', 'N/A')}")
                print(f"Noise (transformed): {final_params.get('noise', 'N/A')}")
                print(f"Outputscale (transformed): {final_params.get('outputscale', 'N/A')}")
                lengthscales = final_params.get("lengthscales", [])
                if lengthscales:
                    print(f"Lengthscales (transformed): {lengthscales} (count: {len(lengthscales)})")
                else:
                    print(f"Lengthscales (transformed): N/A (not found)")

                # Show cat_kernel lengthscales if they exist
                cat_lengthscales = final_params.get("cat_lengthscales", [])
                if cat_lengthscales:
                    print(f"Cat lengthscales (transformed): {cat_lengthscales} (count: {len(cat_lengthscales)})")

                # Show source_kernel lengthscales if they exist
                source_lengthscales = final_params.get("source_lengthscales", [])
                if source_lengthscales:
                    print(
                        f"Source lengthscales (transformed): {source_lengthscales} (count: {len(source_lengthscales)})"
                    )

                print(f"Raw noise: {final_params['raw_noise']}")
                print(f"Raw outputscale: {final_params['raw_outputscale']}")
                raw_lengthscales = final_params.get("raw_lengthscales", [])
                if raw_lengthscales:
                    print(f"Raw lengthscales: {raw_lengthscales} (count: {len(raw_lengthscales)})")
                else:
                    print(f"Raw lengthscales: N/A (not found)")

                # Show raw cat_kernel and source_kernel lengthscales if they exist
                raw_cat_lengthscales = final_params.get("raw_cat_lengthscales", [])
                if raw_cat_lengthscales:
                    print(f"Raw cat lengthscales: {raw_cat_lengthscales} (count: {len(raw_cat_lengthscales)})")

                raw_source_lengthscales = final_params.get("raw_source_lengthscales", [])
                if raw_source_lengthscales:
                    print(f"Raw source lengthscales: {raw_source_lengthscales} (count: {len(raw_source_lengthscales)})")

                periods = final_params.get("periods", [])
                if periods:
                    print(f"Period (transformed): {periods} (count: {len(periods)})")
                else:
                    print("Period (transformed): N/A (not found)")

                raw_periods = final_params.get("raw_periods", [])
                if raw_periods:
                    print(f"Raw period: {raw_periods} (count: {len(raw_periods)})")
                else:
                    print("Raw period: N/A (not found)")

                print(f"Best loss: {best_loss}")
                # If we have initial, also print deltas
                if self._initial_params is not None:
                    deltas = record.get("deltas", {})
                    print("--- Deltas (final - initial) ---")
                    print(f"Delta raw_noise: {deltas.get('raw_noise')}")
                    print(f"Delta raw_outputscale: {deltas.get('raw_outputscale')}")
                    print(f"Delta raw_lengthscales: {deltas.get('raw_lengthscales')}")
                    print(f"Delta raw_period: {deltas.get('raw_periods')}")

            # Save to file
            if self.save_file is not None:
                self._save_parameters()

        except Exception as e:
            print(f"Error storing final parameters: {e}")
            import traceback

            traceback.print_exc()

    def _combine_initial_final(
        self,
        initial_nested: Any,
        final_nested: Any,
        initial_flat: Any,
        final_flat: dict,
        best_lbfgs_iter: Optional[int] = None,
    ) -> dict:
        """Combine initial and final snapshots. Uses nested (covar_module, mean_module, likelihood) when provided."""
        run_value = self._current_run_index if self._current_run_index is not None else self._run_count
        record = {
            "run": run_value,
            "best_iter": best_lbfgs_iter,
            "best_loss": final_flat.get("best_loss"),
            "jitter_max": final_flat.get("jitter_max"),
            "timestamp": final_flat.get("timestamp"),
            "initial": None,
            "final": None,
            "deltas": None,
        }
        if best_lbfgs_iter is None:
            record["epoch"] = final_flat.get("epoch")
            record["num_epochs"] = final_flat.get("num_epochs")
            record["best_epoch"] = final_flat.get("best_epoch")
        if self._current_fold_index is not None:
            record["fold"] = self._current_fold_index
        if self._current_num_folds is not None:
            record["num_folds"] = self._current_num_folds

        # Prefer nested structure (covar_module, mean_module, likelihood); fallback to flat
        if final_nested is not None and isinstance(final_nested, dict):
            record["final"] = dict(final_nested)
        else:
            record["final"] = self._flat_final_to_dict(final_flat)

        if initial_nested is not None and isinstance(initial_nested, dict):
            record["initial"] = dict(initial_nested)
        elif initial_flat is not None:
            record["initial"] = self._flat_initial_to_dict(initial_flat)

        # Jitter lives inside initial_parameters and final_parameters (not top-level)
        if record.get("initial") is not None:
            record["initial"]["jitter"] = getattr(self, "_initial_jitter", None)
        if record.get("final") is not None:
            record["final"]["jitter"] = final_flat.get("jitter")

        # Compute deltas from flat params when available
        if initial_flat is not None and final_flat is not None:
            try:
                noise_init = initial_flat.get("raw_noise")
                noise_final = final_flat.get("raw_noise")
                outscale_init = initial_flat.get("raw_outputscale")
                outscale_final = final_flat.get("raw_outputscale")
                ls_init = initial_flat.get("raw_lengthscales") or []
                ls_final = final_flat.get("raw_lengthscales") or []

                noise_delta = None if (noise_init is None or noise_final is None) else float(noise_final - noise_init)
                outscale_delta = (
                    None if (outscale_init is None or outscale_final is None) else float(outscale_final - outscale_init)
                )
                max_len = max(len(ls_init), len(ls_final))
                ls_init_aligned = list(ls_init) + [0.0] * (max_len - len(ls_init))
                ls_final_aligned = list(ls_final) + [0.0] * (max_len - len(ls_final))
                ls_delta = [float(f - i) for f, i in zip(ls_final_aligned, ls_init_aligned)] if max_len > 0 else []
                period_init = initial_flat.get("raw_periods") or []
                period_final = final_flat.get("raw_periods") or []
                max_period_len = max(len(period_init), len(period_final))
                period_init_aligned = list(period_init) + [0.0] * (max_period_len - len(period_init))
                period_final_aligned = list(period_final) + [0.0] * (max_period_len - len(period_final))
                period_delta = (
                    [float(f - i) for f, i in zip(period_final_aligned, period_init_aligned)]
                    if max_period_len > 0
                    else None
                )
                record["deltas"] = {
                    "raw_noise": noise_delta,
                    "raw_outputscale": outscale_delta,
                    "raw_lengthscales": ls_delta,
                    "raw_periods": period_delta,
                }
            except Exception:
                record["deltas"] = None

        return record

    def _flat_final_to_dict(self, final: dict) -> dict:
        """Build flat final dict for backward compatibility when nested is not used."""
        out = {
            "raw_noise": final.get("raw_noise"),
            "raw_outputscale": final.get("raw_outputscale"),
            "raw_lengthscales": final.get("raw_lengthscales"),
            "raw_constant": final.get("raw_constant"),
            "constant": final.get("constant"),
            "noise": final.get("noise"),
            "outputscale": final.get("outputscale"),
            "lengthscales": final.get("lengthscales"),
            "periods": final.get("periods"),
            "raw_periods": final.get("raw_periods"),
            "kernel_type": final.get("kernel_type"),
            "input_dim": final.get("input_dim"),
        }
        if final.get("raw_cat_lengthscales") is not None:
            out["raw_cat_lengthscales"] = final.get("raw_cat_lengthscales")
        if final.get("raw_source_lengthscales") is not None:
            out["raw_source_lengthscales"] = final.get("raw_source_lengthscales")
        if final.get("cat_lengthscales") is not None:
            out["cat_lengthscales"] = final.get("cat_lengthscales")
        if final.get("source_lengthscales") is not None:
            out["source_lengthscales"] = final.get("source_lengthscales")
        if isinstance(final.get("kernel_type"), str) and "PowerExponential" in final.get("kernel_type", ""):
            out["raw_power"] = final.get("raw_power")
            out["power"] = final.get("power")
        for k, v in final.items():
            if k.startswith("encoder_embedding_") and v is not None:
                out[k] = v
        return out

    def _flat_initial_to_dict(self, initial: dict) -> dict:
        """Build flat initial dict for backward compatibility."""
        out = {
            "raw_noise": initial.get("raw_noise"),
            "raw_outputscale": initial.get("raw_outputscale"),
            "raw_lengthscales": initial.get("raw_lengthscales"),
            "raw_constant": initial.get("raw_constant"),
            "constant": initial.get("constant"),
            "kernel_type": initial.get("kernel_type"),
            "input_dim": initial.get("input_dim"),
        }
        if initial.get("raw_cat_lengthscales") is not None:
            out["raw_cat_lengthscales"] = initial.get("raw_cat_lengthscales")
        if initial.get("raw_source_lengthscales") is not None:
            out["raw_source_lengthscales"] = initial.get("raw_source_lengthscales")
        if isinstance(initial.get("kernel_type"), str) and "PowerExponential" in initial.get("kernel_type", ""):
            out["raw_power"] = initial.get("raw_power")
            out["power"] = initial.get("power")
        if initial.get("raw_periods") is not None:
            out["raw_periods"] = initial.get("raw_periods")
        if initial.get("periods") is not None:
            out["periods"] = initial.get("periods")
        for k, v in initial.items():
            if k.startswith("encoder_embedding_") and v is not None:
                out[k] = v
        return out

    def _extract_final_parameters(
        self,
        model,
        epoch: int,
        best_loss: float = None,
        cholesky_jitter: float = None,
        best_epoch: int = None,
        jitter_max: float = None,
    ):
        """Extract raw and transformed parameters from the model using recursive search."""
        params = {
            "epoch": epoch,
            "num_epochs": epoch + 1,  # +1 because epochs are 0-indexed
            "best_epoch": best_epoch if best_epoch is not None else epoch,
            "best_loss": best_loss,
            "jitter": cholesky_jitter,
            "jitter_max": jitter_max,
            "timestamp": None,
            "raw_noise": None,
            "raw_outputscale": None,
            "raw_lengthscales": None,
            "raw_constant": None,  # mean_module (e.g. ConstantMean)
            "raw_power": None,  # PowerExponentialKernel exponent (raw)
            "raw_periods": None,
            "noise": None,  # Transformed
            "outputscale": None,  # Transformed
            "lengthscales": [],  # Transformed
            "periods": None,
            "constant": None,  # mean_module transformed
            "power": None,  # PowerExponentialKernel exponent (transformed)
            "kernel_type": None,
            "input_dim": None,
        }

        # Add timestamp
        from datetime import datetime

        params["timestamp"] = datetime.now().isoformat()

        # Recursively search for raw parameters
        noise_params = []
        outputscale_params = []
        lengthscale_params = []
        power_params = []
        period_params = []

        self._recursive_parameter_search(
            model, noise_params, outputscale_params, lengthscale_params, power_params, period_params
        )

        # Extract the first found parameter of each type (raw)
        if noise_params:
            params["raw_noise"] = noise_params[0]

        if outputscale_params:
            params["raw_outputscale"] = outputscale_params[0]

        # Always check base_kernel for raw lengthscales - it's the authoritative source
        # This ensures we get all lengthscales for ARD kernels
        
        # Special handling for SEEK kernels (SEEKKernel, SEEKKernelTrunkHead) - they use base_kernels (plural)
        covar_module = getattr(model, "covar_module", None)
        if covar_module is not None:
            from gpplus.kernels.log_scale_kernel import LogScaleKernel

            if _is_seek_kernel(covar_module):
                # SEEK kernels use base_kernels (plural) as a ModuleList
                if hasattr(covar_module, "base_kernels") and len(covar_module.base_kernels) > 0:
                    # Extract from the first base kernel (or aggregate across all if needed)
                    # Unwrap LogScaleKernel if present
                    first_base = covar_module.base_kernels[0]
                    actual_kernel = first_base.base_kernel if isinstance(first_base, LogScaleKernel) else first_base
                    
                    # Unwrap MVMFKernel if present to get to cont_kernel
                    if hasattr(actual_kernel, "cont_kernel") and actual_kernel.cont_kernel is not None:
                        cont_kernel = actual_kernel.cont_kernel
                        if hasattr(cont_kernel, "raw_lengthscale"):
                            try:
                                ls = cont_kernel.raw_lengthscale.data.flatten().tolist()
                                if ls:
                                    params["raw_lengthscales"] = ls
                            except Exception:
                                pass
                    elif hasattr(actual_kernel, "raw_lengthscale"):
                        # Direct kernel (not wrapped in MVMFKernel)
                        try:
                            ls = actual_kernel.raw_lengthscale.data.flatten().tolist()
                            if ls:
                                params["raw_lengthscales"] = ls
                        except Exception:
                            pass
                    
                    # Extract outputscale from LogScaleKernel if present
                    if isinstance(first_base, LogScaleKernel) and hasattr(first_base, "raw_outputscale"):
                        try:
                            outputscale = first_base.raw_outputscale.data
                            if outputscale.numel() == 1:
                                params["raw_outputscale"] = float(outputscale.item())
                            else:
                                params["raw_outputscale"] = outputscale.detach().cpu().flatten().tolist()
                        except Exception:
                            pass
        
        if hasattr(model, "covar_module") and hasattr(model.covar_module, "base_kernel"):
            base_kernel = model.covar_module.base_kernel

            # Special handling for CombinedKernel - extract from cont_kernel, cat_kernel, and source_kernel
            if hasattr(base_kernel, "cont_kernel") and base_kernel.cont_kernel is not None:
                # Extract from cont_kernel (trainable lengthscales)
                cont_kernel = base_kernel.cont_kernel
                if hasattr(cont_kernel, "raw_lengthscale"):
                    try:
                        ls = cont_kernel.raw_lengthscale.data.flatten().tolist()
                        if ls:
                            params["raw_lengthscales"] = ls
                    except Exception:
                        pass

                # Extract from cat_kernel (fixed at 0)
                if hasattr(base_kernel, "cat_kernel") and base_kernel.cat_kernel is not None:
                    cat_kernel = base_kernel.cat_kernel
                    # Check if it's a single kernel or a list (MultCatKs)
                    if isinstance(cat_kernel, list):
                        # MultCatKs - extract from all cat_kernels
                        all_cat_ls = []
                        for idx, ck in enumerate(cat_kernel):
                            if hasattr(ck, "raw_lengthscale"):
                                try:
                                    cat_ls = ck.raw_lengthscale.data.flatten().tolist()
                                    if cat_ls:
                                        all_cat_ls.extend(cat_ls)
                                except Exception:
                                    pass
                        if all_cat_ls:
                            params["raw_cat_lengthscales"] = all_cat_ls
                    else:
                        # Single cat_kernel
                        if hasattr(cat_kernel, "raw_lengthscale"):
                            try:
                                cat_ls = cat_kernel.raw_lengthscale.data.flatten().tolist()
                                if cat_ls:
                                    params["raw_cat_lengthscales"] = cat_ls
                            except Exception:
                                pass

                # Extract from source_kernel (fixed at 0)
                if hasattr(base_kernel, "source_kernel") and base_kernel.source_kernel is not None:
                    source_kernel = base_kernel.source_kernel
                    if hasattr(source_kernel, "raw_lengthscale"):
                        try:
                            source_ls = source_kernel.raw_lengthscale.data.flatten().tolist()
                            if source_ls:
                                params["raw_source_lengthscales"] = source_ls
                        except Exception:
                            pass
            # Special handling for CombinedKernel_MultCatKs (no cont_kernel, but has multiple cat_kernels)
            elif hasattr(base_kernel, "cat_kernel"):
                cat_kernel_attr = base_kernel.cat_kernel
                # Check if cat_kernel is a list (MultCatKs) or if we have cat_kernel_0, cat_kernel_1, etc.
                if isinstance(cat_kernel_attr, list) or hasattr(base_kernel, "cat_kernel_0"):
                    # This is CombinedKernel_MultCatKs
                    all_cat_ls = []

                    # Get list of cat_kernels
                    if isinstance(cat_kernel_attr, list):
                        cat_kernels = cat_kernel_attr
                    else:
                        # Extract all cat_kernel_* modules
                        cat_kernels = []
                        i = 0
                        while hasattr(base_kernel, f"cat_kernel_{i}"):
                            cat_kernels.append(getattr(base_kernel, f"cat_kernel_{i}"))
                            i += 1

                    # Extract raw lengthscales from each cat_kernel
                    for ck in cat_kernels:
                        if hasattr(ck, "raw_lengthscale"):
                            try:
                                cat_ls = ck.raw_lengthscale.data.flatten().tolist()
                                if cat_ls:
                                    all_cat_ls.extend(cat_ls)
                            except Exception:
                                pass

                    if all_cat_ls:
                        params["raw_lengthscales"] = all_cat_ls
            elif hasattr(base_kernel, "raw_lengthscale"):
                try:
                    ls = base_kernel.raw_lengthscale.data.flatten().tolist()
                    if ls:
                        params["raw_lengthscales"] = ls
                except Exception:
                    pass

        # Extract raw power (for PowerExponentialKernel) if found
        if power_params:
            params["raw_power"] = power_params[0]

        if period_params:
            params["raw_periods"] = period_params

        # Fallback to recursive search results if base_kernel didn't have raw_lengthscale
        if not params.get("raw_lengthscales") and lengthscale_params:
            params["raw_lengthscales"] = lengthscale_params

        # Extract encoder embedding positions from MVMFKernel (cat_encoder, source_encoder)
        if hasattr(model, "covar_module") and model.covar_module is not None:
            base = getattr(model.covar_module, "base_kernel", model.covar_module)
            if base is not None:
                # Categorical encoders: single cat_encoder or cat_encoder_0, cat_encoder_1, ...
                cat_enc = getattr(base, "cat_encoder", None)
                if cat_enc is not None:
                    encoders_list = (
                        [cat_enc]
                        if not isinstance(cat_enc, (list, tuple))
                        else list(cat_enc)
                    )
                    for idx, enc in enumerate(encoders_list):
                        try:
                            if hasattr(enc, "projection_matrix") and enc.projection_matrix is not None:
                                # MatrixEncoder: rows = embedding position per category
                                mat = enc.projection_matrix.data.detach().cpu()
                                key = "encoder_embedding_cat" if len(encoders_list) == 1 else f"encoder_embedding_cat_{idx}"
                                params[key] = mat.tolist()
                        except Exception:
                            pass
                # Source encoder
                src_enc = getattr(base, "source_encoder", None)
                if src_enc is not None:
                    try:
                        if hasattr(src_enc, "projection_matrix") and src_enc.projection_matrix is not None:
                            mat = src_enc.projection_matrix.data.detach().cpu()
                            params["encoder_embedding_source"] = mat.tolist()
                    except Exception:
                        pass

        # Extract mean_module parameters (e.g. ConstantMean: raw_constant, constant)
        if hasattr(model, "mean_module") and model.mean_module is not None:
            mean_mod = model.mean_module
            try:
                if hasattr(mean_mod, "raw_constant") and mean_mod.raw_constant is not None:
                    raw_c = mean_mod.raw_constant.data
                    if raw_c.numel() == 1:
                        params["raw_constant"] = float(raw_c.item())
                    else:
                        params["raw_constant"] = raw_c.detach().cpu().flatten().tolist()
            except Exception:
                pass
            try:
                if hasattr(mean_mod, "constant"):
                    c = mean_mod.constant
                    if c is not None and hasattr(c, "numel"):
                        if c.numel() == 1:
                            params["constant"] = float(c.item())
                        else:
                            params["constant"] = c.detach().cpu().flatten().tolist()
            except Exception:
                pass

        # Extract transformed parameters
        self._extract_transformed_parameters(model, params)

        # Try to determine kernel type and input dimension
        params["kernel_type"] = self._determine_kernel_type(model)
        if hasattr(model, "train_inputs") and model.train_inputs:
            params["input_dim"] = model.train_inputs[0].shape[-1]

        return params

    def _extract_transformed_parameters(self, model, params):
        """Extract transformed parameters (not raw) from the model."""
        # Track whether we've already captured the continuous lengthscales
        params.setdefault("_lengthscales_locked", False)

        # Extract transformed noise
        try:
            if hasattr(model, "likelihood") and hasattr(model.likelihood, "noise"):
                noise = model.likelihood.noise
                if noise.numel() == 1:
                    params["noise"] = float(noise.item())
                else:
                    params["noise"] = noise.detach().cpu().numpy().flatten().tolist()
        except Exception:
            pass

        # Recursively extract transformed outputscale, lengthscales, and power
        self._recursive_extract_transformed_kernel_params(model, params)

        # ALWAYS check base_kernel directly for ARD lengthscales and override any previous results
        # This is the authoritative source for ARD kernels wrapped in LogScaleKernel
        # The recursive search might find lengthscales from wrapper kernels, but we want the base_kernel ones
        # Clear any lengthscales found by recursive search before direct extraction to avoid duplication
        if "lengthscales" in params:
            params["lengthscales"] = None
        params["_lengthscales_locked"] = False
        
        if hasattr(model, "covar_module"):
            covar = model.covar_module

            def _has_component(kernel_obj):
                return any(
                    hasattr(kernel_obj, attr) and getattr(kernel_obj, attr) is not None
                    for attr in ("cont_kernel", "cat_kernel", "source_kernel")
                )

            from gpplus.kernels.log_scale_kernel import LogScaleKernel

            if _is_seek_kernel(covar):
                # SEEK kernels use base_kernels (plural) as a ModuleList
                if hasattr(covar, "base_kernels") and len(covar.base_kernels) > 0:
                    # Extract from the first base kernel (or aggregate across all if needed)
                    # Unwrap LogScaleKernel if present
                    first_base = covar.base_kernels[0]
                    actual_kernel = first_base.base_kernel if isinstance(first_base, LogScaleKernel) else first_base
                    
                    try:
                        ls_list = None
                        # Unwrap MVMFKernel if present to get to cont_kernel
                        if hasattr(actual_kernel, "cont_kernel") and actual_kernel.cont_kernel is not None:
                            cont_kernel = actual_kernel.cont_kernel
                            if hasattr(cont_kernel, "raw_lengthscale"):
                                raw_ls = cont_kernel.raw_lengthscale
                                if raw_ls is not None and raw_ls.numel() > 0:
                                    if hasattr(cont_kernel, "raw_lengthscale_constraint"):
                                        constraint = cont_kernel.raw_lengthscale_constraint
                                        transformed = constraint.transform(raw_ls)
                                        ls_list = transformed.detach().cpu().numpy().flatten().tolist()
                                    else:
                                        ls_list = raw_ls.detach().cpu().numpy().flatten().tolist()
                        elif hasattr(actual_kernel, "raw_lengthscale"):
                            # Direct kernel (not wrapped in MVMFKernel)
                            raw_ls = actual_kernel.raw_lengthscale
                            if raw_ls is not None and raw_ls.numel() > 0:
                                if hasattr(actual_kernel, "raw_lengthscale_constraint"):
                                    constraint = actual_kernel.raw_lengthscale_constraint
                                    transformed = constraint.transform(raw_ls)
                                    ls_list = transformed.detach().cpu().numpy().flatten().tolist()
                                else:
                                    ls_list = raw_ls.detach().cpu().numpy().flatten().tolist()
                        
                        # Fallback to lengthscale property
                        if ls_list is None:
                            if hasattr(actual_kernel, "cont_kernel") and actual_kernel.cont_kernel is not None:
                                cont_kernel = actual_kernel.cont_kernel
                                if hasattr(cont_kernel, "lengthscale"):
                                    ls = cont_kernel.lengthscale
                                    if ls is not None and ls.numel() > 0:
                                        ls_list = ls.detach().cpu().numpy().flatten().tolist()
                            elif hasattr(actual_kernel, "lengthscale"):
                                ls = actual_kernel.lengthscale
                                if ls is not None and ls.numel() > 0:
                                    ls_list = ls.detach().cpu().numpy().flatten().tolist()
                        
                        if ls_list is not None:
                            params["lengthscales"] = ls_list
                            params["_lengthscales_locked"] = True
                            if self.verbose:
                                print(f"[DEBUG] Extracted {len(ls_list)} lengthscales from SEEK kernel base_kernels[0]: {ls_list[:5]}..."
                                      if len(ls_list) > 5
                                      else f"[DEBUG] Extracted {len(ls_list)} lengthscales from SEEK kernel base_kernels[0]: {ls_list}")
                        
                        # Extract outputscale from LogScaleKernel if present
                        if isinstance(first_base, LogScaleKernel):
                            if hasattr(first_base, "raw_outputscale"):
                                raw_os = first_base.raw_outputscale
                                if raw_os is not None and raw_os.numel() > 0:
                                    if hasattr(first_base, "raw_outputscale_constraint"):
                                        constraint = first_base.raw_outputscale_constraint
                                        transformed = constraint.transform(raw_os)
                                        if transformed.numel() == 1:
                                            params["outputscale"] = float(transformed.item())
                                        else:
                                            params["outputscale"] = transformed.detach().cpu().numpy().flatten().tolist()
                                    else:
                                        if raw_os.numel() == 1:
                                            params["outputscale"] = float(raw_os.item())
                                        else:
                                            params["outputscale"] = raw_os.detach().cpu().numpy().flatten().tolist()
                            elif hasattr(first_base, "outputscale"):
                                os = first_base.outputscale
                                if os is not None and os.numel() > 0:
                                    if os.numel() == 1:
                                        params["outputscale"] = float(os.item())
                                    else:
                                        params["outputscale"] = os.detach().cpu().numpy().flatten().tolist()
                    except Exception as e:
                        import logging
                        logging.debug(f"Error extracting parameters from SEEK kernel: {e}")

            base_kernel_attr = covar.base_kernel if hasattr(covar, "base_kernel") else None
            combined_kernel = None
            if base_kernel_attr is not None and _has_component(base_kernel_attr):
                combined_kernel = base_kernel_attr
            if combined_kernel is None and _has_component(covar):
                combined_kernel = covar

            if combined_kernel is not None:
                # Special handling for combined kernels - extract from cont_kernel, cat_kernel, and source_kernel
                if hasattr(combined_kernel, "cont_kernel") and combined_kernel.cont_kernel is not None:
                    cont_kernel = combined_kernel.cont_kernel
                    try:
                        ls_list = None
                        # Always use raw_lengthscale to ensure we get all ARD dimensions
                        if hasattr(cont_kernel, "raw_lengthscale"):
                            raw_ls = cont_kernel.raw_lengthscale
                            if raw_ls is not None and raw_ls.numel() > 0:
                                if hasattr(cont_kernel, "raw_lengthscale_constraint"):
                                    constraint = cont_kernel.raw_lengthscale_constraint
                                    transformed = constraint.transform(raw_ls)
                                    ls_list = transformed.detach().cpu().numpy().flatten().tolist()
                                else:
                                    ls_list = raw_ls.detach().cpu().numpy().flatten().tolist()

                        # Fallback to lengthscale property if raw_lengthscale didn't work
                        if ls_list is None and hasattr(cont_kernel, "lengthscale"):
                            try:
                                ls = cont_kernel.lengthscale
                                if ls is not None and ls.numel() > 0:
                                    ls_list = ls.detach().cpu().numpy().flatten().tolist()
                            except (AttributeError, RuntimeError):
                                pass

                        if ls_list is not None:
                            params["lengthscales"] = ls_list
                            params["_lengthscales_locked"] = True
                            if self.verbose:
                                print(
                                    f"[DEBUG] Extracted {len(ls_list)} lengthscales from cont_kernel (shape: {raw_ls.shape if 'raw_ls' in locals() else 'unknown'}): {ls_list[:5]}..."
                                    if len(ls_list) > 5
                                    else f"[DEBUG] Extracted {len(ls_list)} lengthscales from cont_kernel: {ls_list}"
                                )
                    except Exception as e:
                        import logging

                        logging.debug(f"Error extracting lengthscales from cont_kernel: {e}")

                    # Extract from cat_kernel (should be fixed at 0)
                    if hasattr(combined_kernel, "cat_kernel") and combined_kernel.cat_kernel is not None:
                        cat_kernel = combined_kernel.cat_kernel
                        try:
                            cat_ls_list = None
                            # Always use raw_lengthscale to ensure we get all ARD dimensions
                            if hasattr(cat_kernel, "raw_lengthscale"):
                                cat_raw_ls = cat_kernel.raw_lengthscale
                                if cat_raw_ls is not None and cat_raw_ls.numel() > 0:
                                    if hasattr(cat_kernel, "raw_lengthscale_constraint"):
                                        constraint = cat_kernel.raw_lengthscale_constraint
                                        transformed = constraint.transform(cat_raw_ls)
                                        cat_ls_list = transformed.detach().cpu().numpy().flatten().tolist()
                                    else:
                                        cat_ls_list = cat_raw_ls.detach().cpu().numpy().flatten().tolist()

                            # Fallback to lengthscale property if raw_lengthscale didn't work
                            if cat_ls_list is None and hasattr(cat_kernel, "lengthscale"):
                                try:
                                    cat_ls = cat_kernel.lengthscale
                                    if cat_ls is not None and cat_ls.numel() > 0:
                                        cat_ls_list = cat_ls.detach().cpu().numpy().flatten().tolist()
                                except (AttributeError, RuntimeError):
                                    pass

                            if cat_ls_list is not None:
                                params["cat_lengthscales"] = cat_ls_list
                                if self.verbose:
                                    print(
                                        f"[DEBUG] Extracted {len(cat_ls_list)} lengthscales from cat_kernel (shape: {cat_raw_ls.shape if 'cat_raw_ls' in locals() else 'unknown'}): {cat_ls_list[:5]}..."
                                        if len(cat_ls_list) > 5
                                        else f"[DEBUG] Extracted {len(cat_ls_list)} lengthscales from cat_kernel: {cat_ls_list}"
                                    )
                        except Exception as e:
                            import logging

                            logging.debug(f"Error extracting lengthscales from cat_kernel: {e}")

                    # Extract from source_kernel (should be fixed at 0)
                    if hasattr(combined_kernel, "source_kernel") and combined_kernel.source_kernel is not None:
                        source_kernel = combined_kernel.source_kernel
                        try:
                            source_ls_list = None
                            # Always use raw_lengthscale to ensure we get all ARD dimensions
                            if hasattr(source_kernel, "raw_lengthscale"):
                                source_raw_ls = source_kernel.raw_lengthscale
                                if source_raw_ls is not None and source_raw_ls.numel() > 0:
                                    if hasattr(source_kernel, "raw_lengthscale_constraint"):
                                        constraint = source_kernel.raw_lengthscale_constraint
                                        transformed = constraint.transform(source_raw_ls)
                                        source_ls_list = transformed.detach().cpu().numpy().flatten().tolist()
                                    else:
                                        source_ls_list = source_raw_ls.detach().cpu().numpy().flatten().tolist()

                            # Fallback to lengthscale property if raw_lengthscale didn't work
                            if source_ls_list is None and hasattr(source_kernel, "lengthscale"):
                                try:
                                    source_ls = source_kernel.lengthscale
                                    if source_ls is not None and source_ls.numel() > 0:
                                        source_ls_list = source_ls.detach().cpu().numpy().flatten().tolist()
                                except (AttributeError, RuntimeError):
                                    pass

                            if source_ls_list is not None:
                                params["source_lengthscales"] = source_ls_list
                                if self.verbose:
                                    print(
                                        f"[DEBUG] Extracted {len(source_ls_list)} lengthscales from source_kernel (shape: {source_raw_ls.shape if 'source_raw_ls' in locals() else 'unknown'}): {source_ls_list[:5]}..."
                                        if len(source_ls_list) > 5
                                        else f"[DEBUG] Extracted {len(source_ls_list)} lengthscales from source_kernel: {source_ls_list}"
                                    )
                        except Exception as e:
                            import logging

                            logging.debug(f"Error extracting lengthscales from source_kernel: {e}")

                # Also try to extract transformed power from any PowerExponentialKernel inside cont_kernel
                try:
                    possible_kernels = []
                    if hasattr(combined_kernel, "cont_kernel") and combined_kernel.cont_kernel is not None:
                        possible_kernels.append(combined_kernel.cont_kernel)
                    if hasattr(combined_kernel, "base_kernel") and combined_kernel.base_kernel is not None:
                        possible_kernels.append(combined_kernel.base_kernel)

                    for k_obj in possible_kernels:
                        if hasattr(k_obj, "power"):
                            p = k_obj.power
                            if hasattr(p, "item") and p.numel() == 1:
                                params["power"] = float(p.item())
                            else:
                                params["power"] = (
                                    p.detach().cpu().numpy().flatten().tolist()
                                    if hasattr(p, "detach")
                                    else float(p)
                                )
                            break
                except Exception:
                    pass

                # Special handling for CombinedKernel_MultCatKs - extract from all cat_kernel_* modules
                # Check if this is CombinedKernel_MultCatKs by looking for multiple cat_kernel modules
                if hasattr(combined_kernel, "cat_kernel"):
                    cat_kernel_attr = combined_kernel.cat_kernel
                    # Check if cat_kernel is a list (MultCatKs) or if we have cat_kernel_0, cat_kernel_1, etc.
                    if isinstance(cat_kernel_attr, list) or hasattr(combined_kernel, "cat_kernel_0"):
                        # This is CombinedKernel_MultCatKs
                        all_cat_lengthscales = []

                        # Get list of cat_kernels
                        if isinstance(cat_kernel_attr, list):
                            cat_kernels = cat_kernel_attr
                        else:
                            # Extract all cat_kernel_* modules
                            cat_kernels = []
                            i = 0
                            while hasattr(combined_kernel, f"cat_kernel_{i}"):
                                cat_kernels.append(getattr(combined_kernel, f"cat_kernel_{i}"))
                                i += 1

                        # Extract lengthscales from each cat_kernel
                        for idx, cat_kernel in enumerate(cat_kernels):
                            try:
                                cat_ls_list = None
                                # Always use raw_lengthscale to ensure we get all ARD dimensions
                                if hasattr(cat_kernel, "raw_lengthscale"):
                                    cat_raw_ls = cat_kernel.raw_lengthscale
                                    if cat_raw_ls is not None and cat_raw_ls.numel() > 0:
                                        if hasattr(cat_kernel, "raw_lengthscale_constraint"):
                                            constraint = cat_kernel.raw_lengthscale_constraint
                                            transformed = constraint.transform(cat_raw_ls)
                                            cat_ls_list = transformed.detach().cpu().numpy().flatten().tolist()
                                        else:
                                            cat_ls_list = cat_raw_ls.detach().cpu().numpy().flatten().tolist()

                                # Fallback to lengthscale property if raw_lengthscale didn't work
                                if cat_ls_list is None and hasattr(cat_kernel, "lengthscale"):
                                    try:
                                        cat_ls = cat_kernel.lengthscale
                                        if cat_ls is not None and cat_ls.numel() > 0:
                                            cat_ls_list = cat_ls.detach().cpu().numpy().flatten().tolist()
                                    except (AttributeError, RuntimeError):
                                        pass

                                if cat_ls_list is not None:
                                    all_cat_lengthscales.extend(cat_ls_list)
                                    if self.verbose:
                                        print(
                                            f"[DEBUG] Extracted {len(cat_ls_list)} lengthscales from cat_kernel_{idx} (shape: {cat_raw_ls.shape if 'cat_raw_ls' in locals() else 'unknown'}): {cat_ls_list[:5]}..."
                                            if len(cat_ls_list) > 5
                                            else f"[DEBUG] Extracted {len(cat_ls_list)} lengthscales from cat_kernel_{idx}: {cat_ls_list}"
                                        )
                            except Exception as e:
                                import logging

                                logging.debug(f"Error extracting lengthscales from cat_kernel_{idx}: {e}")

                        # Store all cat lengthscales
                        if all_cat_lengthscales:
                            params["cat_lengthscales"] = all_cat_lengthscales
                            if self.verbose:
                                print(
                                    f"[DEBUG] Extracted {len(all_cat_lengthscales)} total cat lengthscales from {len(cat_kernels)} cat_kernels: {all_cat_lengthscales[:10]}..."
                                    if len(all_cat_lengthscales) > 10
                                    else f"[DEBUG] Extracted {len(all_cat_lengthscales)} total cat lengthscales from {len(cat_kernels)} cat_kernels: {all_cat_lengthscales}"
                                )

            # If we haven't extracted lengthscales yet, try regular kernel extraction
            fallback_kernel = base_kernel_attr if base_kernel_attr is not None else covar
            if "lengthscales" not in params or params["lengthscales"] is None:
                try:
                    ls_list = None
                    if hasattr(fallback_kernel, "lengthscale"):
                        try:
                            ls = fallback_kernel.lengthscale
                            if ls is not None and ls.numel() > 0:
                                ls_list = ls.detach().cpu().numpy().flatten().tolist()
                        except (AttributeError, RuntimeError):
                            pass

                    if ls_list is None and hasattr(fallback_kernel, "raw_lengthscale"):
                        raw_ls = fallback_kernel.raw_lengthscale
                        if raw_ls is not None and raw_ls.numel() > 0:
                            if hasattr(fallback_kernel, "raw_lengthscale_constraint"):
                                constraint = fallback_kernel.raw_lengthscale_constraint
                                transformed = constraint.transform(raw_ls)
                                ls_list = transformed.detach().cpu().numpy().flatten().tolist()
                            else:
                                ls_list = raw_ls.detach().cpu().numpy().flatten().tolist()

                    if ls_list is not None:
                        params["lengthscales"] = ls_list
                        if self.verbose:
                            print(
                                f"[DEBUG] Extracted {len(ls_list)} lengthscales from base_kernel: {ls_list[:5]}..."
                                if len(ls_list) > 5
                                else f"[DEBUG] Extracted {len(ls_list)} lengthscales from base_kernel: {ls_list}"
                            )
                except Exception as e:
                    import logging

                    logging.debug(f"Error extracting lengthscales from base_kernel: {e}")
                    pass

        params.pop("_lengthscales_locked", None)

    def _recursive_extract_transformed_kernel_params(self, obj, params, visited=None, depth=0):
        """Recursively search through model components for transformed kernel parameters."""
        if visited is None:
            visited = set()

        # Avoid infinite recursion and limit depth
        obj_id = id(obj)
        if obj_id in visited or depth > 10:
            return
        visited.add(obj_id)

        try:
            # Extract transformed outputscale
            if hasattr(obj, "outputscale") and params["outputscale"] is None:
                try:
                    outputscale = obj.outputscale
                    if outputscale.numel() == 1:
                        params["outputscale"] = float(outputscale.item())
                    else:
                        params["outputscale"] = outputscale.detach().cpu().numpy().flatten().tolist()
                except:
                    pass

            # Extract transformed power (e.g., for PowerExponentialKernel)
            # This handles simple kernel layouts where the kernel is wrapped (e.g., LogScaleKernel(PowerExponentialKernel))
            if hasattr(obj, "power") and params.get("power") is None:
                try:
                    p = obj.power
                    # Tensors: prefer scalar, otherwise flatten
                    if hasattr(p, "numel"):
                        if p.numel() == 1:
                            params["power"] = float(p.item())
                        else:
                            params["power"] = p.detach().cpu().numpy().flatten().tolist()
                    else:
                        # Fallback for non-tensor values
                        params["power"] = float(p)
                except Exception:
                    # If extraction fails for this object, just skip
                    pass

            # Extract transformed period (PeriodicKernel, CosineKernel)
            if hasattr(obj, "period"):
                try:
                    p = obj.period
                    if hasattr(p, "numel") and p.numel() > 0:
                        period_list = p.detach().cpu().numpy().flatten().tolist()
                        if period_list:
                            existing = params.get("periods") or []
                            params["periods"] = list(existing) + period_list
                except Exception:
                    pass

            # Extract transformed lengthscales
            # Skip lengthscale extraction here - we'll get it directly from base_kernel in the fallback
            # This avoids picking up lengthscales from the wrong kernel components
            if hasattr(obj, "lengthscale") and not hasattr(obj, "base_kernel"):
                # Only extract if this is a base kernel (doesn't have base_kernel attribute)
                # This ensures we get the ARD kernel's lengthscales, not from wrapper kernels
                try:
                    lengthscale = obj.lengthscale
                    if lengthscale.numel() > 0:
                        lengthscale_list = lengthscale.detach().cpu().numpy().flatten().tolist()
                        # Only use if this has multiple lengthscales (ARD) or we haven't found any yet
                        if len(lengthscale_list) > 1:
                            # This is an ARD kernel, use it
                            if not params.get("_lengthscales_locked"):
                                params["lengthscales"] = lengthscale_list
                                params["_lengthscales_locked"] = True
                        elif not params.get("lengthscales"):
                            # No lengthscales found yet, use these (might be isotropic)
                            if not params.get("_lengthscales_locked"):
                                params["lengthscales"] = lengthscale_list
                except Exception as e:
                    # Silently continue if extraction fails for this object
                    pass

            # Recursively search through specific attributes that are likely to contain kernels/parameters
            # Search base_kernel first to prioritize ARD kernel lengthscales
            # NOTE: We skip cont_kernel, cat_kernel, and source_kernel here because we extract those directly
            # in _extract_transformed_parameters to avoid finding lengthscales from both the wrapper and the kernel itself
            search_attrs = [
                "base_kernel",  # Search base_kernel first for ARD kernels
                "covar_module",
                "likelihood",
                "mean_module",
                "kernels",
                "noise_covar",
                "cat_encoder",
                "source_encoder",
            ]

            for attr_name in search_attrs:
                if hasattr(obj, attr_name):
                    try:
                        attr = getattr(obj, attr_name)
                        if attr is not None:
                            self._recursive_extract_transformed_kernel_params(attr, params, visited, depth + 1)
                    except:
                        continue

            # Also search through registered modules
            if hasattr(obj, "_modules"):
                for module_name, module in obj._modules.items():
                    if module is not None:
                        self._recursive_extract_transformed_kernel_params(module, params, visited, depth + 1)

        except Exception:
            pass

    def _recursive_parameter_search(
        self, obj, noise_params, outputscale_params, lengthscale_params, power_params, period_params, visited=None, depth=0
    ):
        """Recursively search through model components for raw parameters."""
        if visited is None:
            visited = set()

        # Avoid infinite recursion and limit depth
        obj_id = id(obj)
        if obj_id in visited or depth > 10:
            return
        visited.add(obj_id)

        try:
            # Check if this object has the parameters we're looking for
            if hasattr(obj, "raw_noise"):
                try:
                    if hasattr(obj.raw_noise, "data") and obj.raw_noise.data is not None:
                        noise_val = obj.raw_noise.data.item()
                        noise_params.append(noise_val)
                except:
                    pass

            if hasattr(obj, "raw_outputscale"):
                try:
                    if hasattr(obj.raw_outputscale, "data") and obj.raw_outputscale.data is not None:
                        outputscale_val = obj.raw_outputscale.data.item()
                        outputscale_params.append(outputscale_val)
                except:
                    pass

            if hasattr(obj, "raw_lengthscale"):
                try:
                    if hasattr(obj.raw_lengthscale, "data") and obj.raw_lengthscale.data is not None:
                        lengthscale_val = obj.raw_lengthscale.data.flatten().tolist()
                        lengthscale_params.extend(lengthscale_val)
                except:
                    pass

            # Raw power parameter (PowerExponentialKernel)
            if hasattr(obj, "raw_power"):
                try:
                    if hasattr(obj.raw_power, "data") and obj.raw_power.data is not None:
                        # raw_power is constrained to [1,2], but we store the raw tensor value here
                        power_val = obj.raw_power.data.item()
                        power_params.append(power_val)
                except:
                    pass

            if hasattr(obj, "raw_period"):
                try:
                    if hasattr(obj.raw_period, "data") and obj.raw_period.data is not None:
                        period_val = obj.raw_period.data.flatten().tolist()
                        period_params.extend(period_val)
                except:
                    pass

            # Recursively search through specific attributes that are likely to contain kernels/parameters
            search_attrs = [
                "covar_module",
                "likelihood",
                "mean_module",
                "base_kernel",
                "cat_kernel",
                "cont_kernel",
                "source_kernel",
                "kernels",
                "noise_covar",
                "cat_encoder",
                "source_encoder",
            ]

            for attr_name in search_attrs:
                if hasattr(obj, attr_name):
                    try:
                        attr = getattr(obj, attr_name)
                        if attr is not None:
                            self._recursive_parameter_search(
                                attr,
                                noise_params,
                                outputscale_params,
                                lengthscale_params,
                                power_params,
                                period_params,
                                visited,
                                depth + 1,
                            )
                    except:
                        continue

            # Also search through registered modules
            if hasattr(obj, "_modules"):
                for module_name, module in obj._modules.items():
                    if module is not None:
                        self._recursive_parameter_search(
                            module,
                            noise_params,
                            outputscale_params,
                            lengthscale_params,
                            power_params,
                            period_params,
                            visited,
                            depth + 1,
                        )

        except Exception:
            pass

    def _determine_kernel_type(self, model):
        """Determine the kernel type by examining the covar_module structure.

        Preference order:
        - If there is a base_kernel (e.g. PowerExponentialKernel, GaussianKernel, MaternKernel),
          report its type (optionally wrapped as Combined(...)).
        - Otherwise fall back to the covar_module wrapper type.
        """
        try:
            if hasattr(model, "covar_module"):
                covar_module = model.covar_module
                module_type = type(covar_module).__name__

                # If there is a base_kernel, use its concrete type name
                if hasattr(covar_module, "base_kernel") and covar_module.base_kernel is not None:
                    base = covar_module.base_kernel
                    base_type = type(base).__name__

                    # If this base kernel itself is a combined kernel, label clearly
                    if any(
                        hasattr(base, attr) and getattr(base, attr) is not None
                        for attr in ("cont_kernel", "cat_kernel", "source_kernel")
                    ):
                        return f"Combined({base_type})"
                    return base_type

                # Fallbacks for other structures on the covar_module itself
                if (
                    hasattr(covar_module, "cat_kernel")
                    or hasattr(covar_module, "cont_kernel")
                    or hasattr(covar_module, "source_kernel")
                ):
                    return "CombinedKernel"
                if hasattr(covar_module, "kernels"):
                    return "MultiKernel"
                return module_type

            return "Unknown"
        except Exception:
            return "Unknown"

    def _save_parameters(self):
        """Save stored parameters to JSON file."""
        try:
            import json
            import os

            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.save_file) if os.path.dirname(self.save_file) else ".", exist_ok=True)

            # Save parameters
            with open(self.save_file, "w") as f:
                json.dump(self.stored_parameters, f, indent=2)

            if self.verbose:
                print(f"Final parameters saved to: {self.save_file}")

        except Exception as e:
            print(f"Error saving parameters to file: {e}")

    def get_stored_parameters(self):
        """Get the stored parameters for external use."""
        return self.stored_parameters.copy()

    def get_best_model_metrics(self):
        """
        Get the best model metrics for inclusion in results files.
        Returns metrics from the run with the lowest best_loss.

        If stored_parameters is empty (e.g., in multi-run training where callbacks are deep-copied),
        this will try to load from the saved JSON file.

        Returns:
            dict: Dictionary containing:
                - num_epochs: Number of epochs trained
                - best_epoch: Epoch when best loss was achieved
                - best_loss: Best loss value
                - jitter: Cholesky jitter value used
                - noise: Transformed noise parameter
                - outputscale: Transformed outputscale parameter
                - lengthscales: List of transformed lengthscale parameters
        """
        # If stored_parameters is empty, try to load from file
        if not self.stored_parameters:
            try:
                import json
                import os

                if os.path.exists(self.save_file):
                    with open(self.save_file, "r") as f:
                        self.stored_parameters = json.load(f)
            except Exception:
                pass

        if not self.stored_parameters:
            return None

        # Find the run with the lowest best_loss
        best_run = min(
            [r for r in self.stored_parameters if r.get("best_loss") is not None],
            key=lambda x: x.get("best_loss", float("inf")),
            default=None,
        )

        if best_run is None:
            return None

        # Extract relevant metrics
        final_dict = best_run.get("final", {})
        lengthscales = final_dict.get("lengthscales")

        # Debug output
        if self.verbose:
            print(f"[DEBUG get_best_model_metrics] best_run keys: {list(best_run.keys())}")
            print(f"[DEBUG get_best_model_metrics] final_dict keys: {list(final_dict.keys())}")
            if lengthscales is not None:
                if isinstance(lengthscales, (list, tuple)):
                    print(f"[DEBUG get_best_model_metrics] Found {len(lengthscales)} lengthscales in final dict")
                else:
                    print(
                        f"[DEBUG get_best_model_metrics] lengthscales is not a list: {type(lengthscales)}, value: {lengthscales}"
                    )
            else:
                print(f"[DEBUG get_best_model_metrics] lengthscales is None in final_dict")

        metrics = {
            "num_epochs": best_run.get("num_epochs"),
            "best_epoch": best_run.get("best_epoch"),
            "best_loss": best_run.get("best_loss"),
            "jitter": best_run.get("jitter"),
            "jitter_max": best_run.get("jitter_max"),
            "noise": final_dict.get("noise"),
            "outputscale": final_dict.get("outputscale"),
            "lengthscales": lengthscales,
        }

        return metrics

    def clear_parameters(self):
        """Clear stored parameters."""
        self.stored_parameters = []


class IterationParameterCallback(Callback):
    """
    Callback to record parameters for LBFGSScipy runs. Always saves initial and final
    parameters (same nested format as FinalParameterStorageCallback: covar_module,
    mean_module, likelihood, jitter). Optionally saves parameters every X iterations.

    Only activates when LBFGSScipy optimizer is used.

    Args:
        save_file (str, optional): Path to save the parameters JSON file. If None, no file is written.
        verbose (bool): If True, print to console when storing. Does not affect saving.
        save_every_n_iterations (int, optional): If set, append a parameter snapshot every N
            L-BFGS iterations to "records". If None, only initial and final are stored (no per-iter records).
            Defaults to None (initial + final only).
    """
    
    def __init__(self, save_file: str = None, verbose: bool = False, save_every_n_iterations: Optional[int] = None):
        self.save_file = save_file
        self.verbose = verbose
        self.save_every_n_iterations = save_every_n_iterations
        self.recorded_parameters = []
        self._final_parameters = None
        self.extractor = KernelParameterExtractor(verbose=verbose)
        self._optimizer = None
        self._model = None
        self._trainer = None
        self._current_epoch = 0
        self._is_active = False
        self._last_actual_jitter = None  # Track the last jitter value from warnings
        self._max_actual_jitter = None  # Track the maximum jitter value seen from warnings
        self._warning_handler = None  # Warning handler to capture jitter from linear_operator
        self._initial_parameters = None  # Store initial parameters from ParameterInitializer
        self._run_index = None  # Will be set via set_run_index() by GPTrainer
        self._fold_index = None  # Will be set via set_fold_index() by experiment script
    
    def set_fold_index(self, fold_index: int):
        """Set the fold index for this callback instance. Called by experiment script."""
        self._fold_index = fold_index
    
    def on_train_start(self, context: dict):
        """Store model and trainer references for later use."""
        self._model = context["model"]
        self._trainer = context.get("trainer", None)
        # Optimizer registration will happen via register_with_optimizer() called from trainer
        
        # Capture initial parameters (same nested format as FinalParameterStorageCallback)
        try:
            params = self.extractor.extract_all_parameters(self._model, trainer=self._trainer)
            if params is not None:
                self._initial_parameters = dict(params)
                if self._trainer is not None and hasattr(self._trainer, "cholesky_jitter"):
                    j = self._trainer.cholesky_jitter
                    self._initial_parameters["jitter"] = float(j) if j is not None and (isinstance(j, (int, float)) or hasattr(j, "item")) else None
                else:
                    self._initial_parameters["jitter"] = None
            else:
                self._initial_parameters = None
            if self.verbose:
                print(f"IterationParameterCallback: Captured initial parameters")
        except Exception as e:
            if self.verbose:
                print(f"IterationParameterCallback: Warning - Could not capture initial parameters: {e}")
            self._initial_parameters = None
        
        # Set up warning handler to capture jitter values (same as JitterTrackingCallback)
        try:
            import warnings
            if not hasattr(warnings, '_showwarning_orig'):
                warnings._showwarning_orig = warnings.showwarning
            self._warning_handler = self._create_jitter_warning_handler()
            warnings.showwarning = self._warning_handler
        except Exception as e:
            if self.verbose:
                print(f"IterationParameterCallback: Warning - Could not set up warning handler: {e}")
            self._warning_handler = None
    
    def on_epoch_start(self, context: dict):
        """Update current epoch."""
        self._current_epoch = context.get("epoch", 0)
    
    def register_with_optimizer(self, optimizer, model=None, trainer=None):
        """Register this callback with the LBFGSScipy optimizer."""
        from .optimizers import LBFGSScipy
        
        if model is not None:
            self._model = model
        if trainer is not None:
            self._trainer = trainer
        
        if isinstance(optimizer, LBFGSScipy):
            self._optimizer = optimizer
            self._is_active = True
            # Set the iteration callback - it will be called with iteration and loss
            optimizer.iteration_callback = self._on_iteration
            
            # Store trunk/head parameter indices for tracking changes in flat_params
            if model is not None:
                self._trunk_head_param_indices = []
                self._trunk_head_param_names = []
                self._trunk_head_param_objects = []  # Store param objects for gradient checking
                trunk_head_params = []
                for name, param in model.named_parameters():
                    if 'trunk' in name.lower() or 'head' in name.lower():
                        trunk_head_params.append((name, param))
                
                if trunk_head_params:
                    optimizer_param_ids = {id(p) for p in optimizer._params}
                    # Find which optimizer parameter indices correspond to trunk/head params
                    for idx, opt_param in enumerate(optimizer._params):
                        if id(opt_param) in {id(p) for _, p in trunk_head_params}:
                            self._trunk_head_param_indices.append(idx)
                            # Find the name and store the param object
                            for name, param in trunk_head_params:
                                if id(param) == id(opt_param):
                                    self._trunk_head_param_names.append(name)
                                    self._trunk_head_param_objects.append(param)
                                    break
                    
                    # Calculate offsets for each parameter in flat_params
                    self._trunk_head_offsets = []
                    offset = 0
                    for idx in range(len(optimizer._params)):
                        if idx in self._trunk_head_param_indices:
                            self._trunk_head_offsets.append((offset, offset + optimizer._params[idx].numel()))
                        offset += optimizer._params[idx].numel()
            
            # Diagnostic: Check if TrunkHeadNet parameters are in optimizer and find their indices
            if self.verbose and model is not None:
                trunk_head_params = []
                trunk_head_param_objects = []
                for name, param in model.named_parameters():
                    if 'trunk' in name.lower() or 'head' in name.lower():
                        trunk_head_params.append((name, param))
                        trunk_head_param_objects.append(param)
                
                if trunk_head_params:
                    optimizer_param_ids = {id(p) for p in optimizer._params}
                    in_optimizer = []
                    not_in_optimizer = []
                    param_indices = []
                    
                    # Find which optimizer parameter indices correspond to trunk/head params
                    for idx, opt_param in enumerate(optimizer._params):
                        if id(opt_param) in {id(p) for _, p in trunk_head_params}:
                            param_indices.append(idx)
                    
                    for name, param in trunk_head_params:
                        if id(param) in optimizer_param_ids:
                            in_optimizer.append(name)
                        else:
                            not_in_optimizer.append(name)
                    
                    # Count total elements (not just tensors)
                    total_elements = sum(p.numel() for _, p in trunk_head_params if id(p) in optimizer_param_ids)
                    print(f"IterationParameterCallback: Found {len(trunk_head_params)} trunk/head parameter tensors ({total_elements} total elements)")
                    if in_optimizer:
                        print(f"  ✓ {len(in_optimizer)} parameter tensors in optimizer (indices: {param_indices[:5]}...)" if len(param_indices) > 5 else f"  ✓ {len(in_optimizer)} parameter tensors in optimizer (indices: {param_indices})")
                        # Print parameter names and sizes for debugging
                        if len(in_optimizer) <= 10:
                            print(f"    Parameter details:")
                            for name in in_optimizer:
                                # Find the param object
                                param_obj = next(p for n, p in trunk_head_params if n == name)
                                print(f"      {name}: shape={list(param_obj.shape)}, numel={param_obj.numel()}")
                    if not_in_optimizer:
                        print(f"  ✗ {len(not_in_optimizer)} parameters NOT in optimizer: {not_in_optimizer[:3]}...")
                else:
                    print(f"IterationParameterCallback: No trunk/head parameters found in model")
            
            if self.verbose:
                print(f"IterationParameterCallback: Registered with LBFGSScipy optimizer")
        else:
            self._is_active = False
            if self.verbose:
                print(f"IterationParameterCallback: Not LBFGSScipy optimizer, callback inactive")
    
    def _create_jitter_warning_handler(self):
        """Create a warning handler that captures jitter values from linear_operator warnings."""
        import warnings
        import re
        callback_instance = self  # Capture self for use in closure
        
        def jitter_warning_handler(message, category, filename, lineno, file=None, line=None):
            """Intercept warnings to capture jitter values from linear_operator."""
            try:
                # Call original handler first
                if hasattr(warnings, '_showwarning_orig'):
                    warnings._showwarning_orig(message, category, filename, lineno, file, line)
                else:
                    import sys
                    sys.stderr.write(warnings.formatwarning(message, category, filename, lineno))
                
                # Check if this is a jitter-related warning from linear_operator
                if isinstance(message, (str, Warning)):
                    msg_str = str(message)
                    filename_str = str(filename).lower() if filename else ""
                    if 'cholesky' in filename_str or 'linear_operator' in filename_str or 'jitter' in msg_str.lower():
                        patterns = [
                            r'added jitter of ([\d.]+e[+-]?\d+)',
                            r'added jitter of ([\d.]+)',
                            r'jitter of ([\d.]+e[+-]?\d+)',
                        ]
                        for pattern in patterns:
                            match = re.search(pattern, msg_str, re.IGNORECASE)
                            if match:
                                try:
                                    jitter_value = float(match.group(1))
                                    callback_instance._last_actual_jitter = jitter_value
                                    if callback_instance._max_actual_jitter is None or jitter_value > callback_instance._max_actual_jitter:
                                        callback_instance._max_actual_jitter = jitter_value
                                except (ValueError, AttributeError):
                                    pass
                                break
            except Exception:
                pass
        
        return jitter_warning_handler
    
    def _get_current_jitter(self) -> float:
        """Get current jitter value from settings, trainer, or captured warnings.
        
        Priority (same as JitterTrackingCallback):
        1. Last actual jitter captured from linear_operator warnings (most accurate)
        2. Trainer's current_jitter (updated on NotPSDError)
        3. gpytorch settings cholesky_jitter
        4. Trainer's initial cholesky_jitter
        """
        # First, check if we captured an actual jitter value from warnings
        # Use max jitter if available (most accurate), otherwise use last
        if self._max_actual_jitter is not None:
            return float(self._max_actual_jitter)
        if self._last_actual_jitter is not None:
            return float(self._last_actual_jitter)
        
        # Fallback: try to get from trainer's current_jitter (preferred) / _current_run_jitter (v3)
        if self._trainer is not None:
            for attr in ("current_jitter", "_current_run_jitter"):
                if hasattr(self._trainer, attr):
                    j = getattr(self._trainer, attr)
                    if hasattr(j, "item"):
                        try:
                            j = j.item()
                        except Exception:
                            pass
                    if j is not None:
                        try:
                            return float(j)
                        except (TypeError, ValueError):
                            pass

        # Next: try to get jitter stored on the model (persisted at end of training, may be set elsewhere too)
        if self._model is not None and hasattr(self._model, "cholesky_jitter"):
            j = getattr(self._model, "cholesky_jitter")
            if hasattr(j, "item"):
                try:
                    j = j.item()
                except Exception:
                    pass
            if j is not None:
                try:
                    return float(j)
                except (TypeError, ValueError):
                    pass
        
        try:
            # Try to get from gpytorch settings
            import gpytorch
            if hasattr(gpytorch.settings, "cholesky_jitter"):
                jitter_obj = gpytorch.settings.cholesky_jitter
                # Prefer a simple numeric/tensor "value" attribute if present and NOT callable.
                jitter_value = getattr(jitter_obj, "value", None)
                if callable(jitter_value):
                    # In some gpytorch versions this is a context method that
                    # requires arguments (e.g. dtype). We avoid calling it and
                    # instead fall back to trainer-level settings.
                    jitter_value = None
                # Handle tensors or numpy types
                if hasattr(jitter_value, "item"):
                    try:
                        jitter_value = jitter_value.item()
                    except Exception:
                        pass
                if jitter_value is not None:
                    try:
                        return float(jitter_value)
                    except (TypeError, ValueError):
                        # If it's still not directly convertible, ignore and fall back
                        pass
            if self.verbose:
                print(f"IterationParameterCallback: Could not get jitter value from gpytorch settings")
        except Exception:
            pass
        
        # Final fallback: get from trainer's cholesky_jitter (initial value)
        if self._trainer is not None and hasattr(self._trainer, "cholesky_jitter"):
            return float(self._trainer.cholesky_jitter)
        
        return None
    
    def _on_iteration(self, iteration: int, loss: float = None, flat_params=None):
        """Called by LBFGSScipy optimizer at each iteration. Only appends when save_every_n_iterations is set."""
        if not self._is_active or self._model is None:
            return
        if self.save_every_n_iterations is None:
            return
        try:
            if iteration != 1 and (iteration % self.save_every_n_iterations) != 0:
                return
            params = self.extractor.extract_all_parameters(self._model, trainer=self._trainer)
            jitter = self._get_current_jitter()
            if isinstance(params, dict):
                params = dict(params)
                params["jitter"] = jitter
            record = {
                "iteration": iteration,
                "loss": loss,
                "jitter": jitter,
                "parameters": params,
            }
            record["run_index"] = self._run_index
            record["fold_index"] = self._fold_index
            self.recorded_parameters.append(record)
            if self.verbose and iteration % 250 == 0:
                run_str = f"Run {self._run_index}, " if self._run_index is not None else ""
                print(f"{run_str}Iteration {iteration} - Loss: {loss:.6f}" if loss else f"{run_str}Iteration {iteration}")
            if self.save_file is not None:
                self._save_parameters()
        except Exception as e:
            if self.verbose:
                print(f"Error recording parameters at iteration {iteration}: {e}")
            import traceback
            traceback.print_exc()
    
    def on_epoch_end(self, context: dict):
        """Save parameters at end of each epoch."""
        if self.save_file is not None and self.recorded_parameters:
            self._save_parameters()
    
    def on_train_end(self, context: dict):
        """Capture final parameters (same nested format as initial) and optionally save."""
        import copy
        if self._warning_handler is not None:
            import warnings
            if hasattr(warnings, '_showwarning_orig'):
                warnings.showwarning = warnings._showwarning_orig
        model = context.get("model")
        trainer = context.get("trainer")
        best_state_dict = context.get("best_state_dict")
        if model is not None:
            try:
                model_to_extract = model
                if best_state_dict is not None:
                    model_to_extract = copy.deepcopy(model)
                    model_to_extract.load_state_dict(best_state_dict)
                self._final_parameters = self.extractor.extract_all_parameters(model_to_extract, trainer=trainer)
                if self._final_parameters is not None:
                    self._final_parameters = dict(self._final_parameters)
                    jitter = None
                    if trainer is not None and hasattr(trainer, "cholesky_jitter"):
                        j = trainer.cholesky_jitter
                        jitter = float(j) if j is not None and (isinstance(j, (int, float)) or hasattr(j, "item")) else None
                    self._final_parameters["jitter"] = jitter
            except Exception as e:
                if self.verbose:
                    print(f"IterationParameterCallback: Warning - Could not capture final parameters: {e}")
                self._final_parameters = None
        if self.save_file is not None:
            self._save_parameters()
            if self.verbose:
                run_str = f" (run_index={self._run_index})" if self._run_index is not None else ""
                n_rec = len(self.recorded_parameters)
                print(f"IterationParameterCallback: Saved initial, final" + (f" and {n_rec} iteration records" if n_rec else "") + run_str)
    
    def _save_parameters(self):
        """Save recorded parameters to JSON file.
        
        For single runs, saves directly to save_file.
        For multiple runs (when run_index is set), saves to a temporary file
        that will be aggregated later.
        """
        try:
            import json
            import os
            
            if self.save_file is None:
                return
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.save_file) if os.path.dirname(self.save_file) else ".", exist_ok=True)
            
            # Prepare output data (initial + final same format as JSON; records when save_every_n_iterations set)
            output_data = {
                "initial_parameters": self._initial_parameters,
                "final_parameters": getattr(self, "_final_parameters", None),
                "records": self.recorded_parameters,
            }
            
            # If run_index is set, we're in a multi-run scenario - save to temp file
            if self._run_index is not None:
                # Save to temporary file: {base_path}_temp_run_{run_index}.json
                base_path, ext = os.path.splitext(self.save_file)
                temp_file = f"{base_path}_temp_run_{self._run_index}{ext}"
                with open(temp_file, "w") as f:
                    json.dump(output_data, f, indent=2)
                if self.verbose:
                    print(f"IterationParameterCallback: Saved {len(self.recorded_parameters)} records to temp file: {temp_file}")
            else:
                # Single run - save directly to main file
                with open(self.save_file, "w") as f:
                    json.dump(output_data, f, indent=2)
        except Exception as e:
            if self.verbose:
                print(f"Error saving parameters to file: {e}")
    
    def get_stored_parameters(self):
        """Get the stored parameters for external use.
        
        Returns initial_parameters, final_parameters (same nested format as JSON),
        and records (per-iteration only when save_every_n_iterations is set).
        """
        return {
            "initial_parameters": self._initial_parameters,
            "final_parameters": getattr(self, "_final_parameters", None),
            "records": self.recorded_parameters.copy(),
        }

    @staticmethod
    def aggregate_parameters_from_results(results, save_file: str, verbose: bool = True):
        """
        Aggregate parameter data from multiple training runs into a single file.

        This method should be called after all runs complete to collect parameter data
        from all runs and save it to a single file organized by run_index.

        Args:
            results: List of result dictionaries from GPTrainer.train(), each containing
                    'run_index' and 'callback_data' with 'IterationParameterCallback' data
            save_file: Path to save the aggregated parameter data
            verbose: Whether to print aggregation status
        """
        import json
        import os
        import glob

        if save_file is None:
            return

        # Collect parameter data from all folds and runs
        aggregated_data = {}
        base_path, ext = os.path.splitext(save_file)
        temp_pattern_with_fold = f"{base_path}_temp_fold_*_run_*{ext}"
        temp_pattern_no_fold = f"{base_path}_temp_run_*{ext}"
        temp_files = glob.glob(temp_pattern_with_fold) + glob.glob(temp_pattern_no_fold)

        if verbose:
            print(f"IterationParameterCallback: Aggregating data from {len(results)} results...")
            print(f"  Looking for temp files with patterns: {temp_pattern_with_fold}, {temp_pattern_no_fold}")
            print(f"  Found {len(temp_files)} temp files: {[os.path.basename(f) for f in temp_files]}")

        for temp_file in temp_files:
            try:
                filename = os.path.basename(temp_file)
                base_name = os.path.basename(base_path)
                if f"_temp_fold_" in filename:
                    fold_match = filename.replace(f"{base_name}_temp_fold_", "").replace(ext, "")
                    parts = fold_match.split("_run_")
                    if len(parts) == 2:
                        fold_idx_str, run_idx_str = parts[0], parts[1]
                    else:
                        fold_idx_str = None
                        run_idx_str = fold_match.replace("_run_", "")
                else:
                    fold_idx_str = None
                    run_idx_str = filename.replace(f"{base_name}_temp_run_", "").replace(ext, "")
                with open(temp_file, "r") as f:
                    temp_data = json.load(f)
                if fold_idx_str is None:
                    fold_idx_str = "0"
                if fold_idx_str not in aggregated_data:
                    aggregated_data[fold_idx_str] = {}
                aggregated_data[fold_idx_str][run_idx_str] = temp_data
                if verbose:
                    num_records = len(temp_data.get("records", [])) if isinstance(temp_data, dict) else len(temp_data) if isinstance(temp_data, list) else 0
                    has_initial = temp_data.get("initial_parameters") is not None if isinstance(temp_data, dict) else False
                    print(f"  Fold {fold_idx_str}, Run {run_idx_str}: Loaded {num_records} records from temp file (initial_parameters: {'present' if has_initial else 'null'})")
            except Exception as e:
                if verbose:
                    print(f"  Warning: Could not load temp file {temp_file}: {e}")

        for result in results:
            run_index = result.get("run_index")
            fold_index = result.get("fold_index")
            if run_index is None:
                if verbose:
                    print(f"  Warning: Result missing run_index: {list(result.keys())}")
                continue
            run_idx_str = str(run_index)
            fold_idx_str = str(fold_index) if fold_index is not None else "0"
            if fold_idx_str not in aggregated_data or run_idx_str not in aggregated_data.get(fold_idx_str, {}):
                if fold_idx_str not in aggregated_data:
                    aggregated_data[fold_idx_str] = {}
                callback_data = result.get("callback_data", {})
                param_data = callback_data.get("IterationParameterCallback")
                if param_data is not None:
                    if isinstance(param_data, dict) and "initial_parameters" in param_data and "records" in param_data:
                        aggregated_data[fold_idx_str][run_idx_str] = param_data
                        if verbose:
                            has_initial = param_data.get("initial_parameters") is not None
                            print(f"  Fold {fold_idx_str}, Run {run_index}: Found {len(param_data.get('records', []))} records in callback_data (initial_parameters: {'present' if has_initial else 'null'})")
                    elif isinstance(param_data, list):
                        aggregated_data[fold_idx_str][run_idx_str] = {
                            "initial_parameters": None,
                            "records": param_data
                        }
                        if verbose:
                            print(f"  Fold {fold_idx_str}, Run {run_index}: Found {len(param_data)} records in callback_data (old format)")

        if aggregated_data:
            existing_data = {"folds": {}}
            if os.path.exists(save_file):
                try:
                    with open(save_file, "r") as f:
                        existing_data = json.load(f)
                        if "folds" not in existing_data:
                            existing_data = {"folds": {}}
                except Exception as e:
                    if verbose:
                        print(f"  Warning: Could not read existing aggregated file: {e}")
                    existing_data = {"folds": {}}
            for fold_idx, runs_dict in aggregated_data.items():
                fold_key = f"fold_{fold_idx}"
                if fold_key not in existing_data["folds"]:
                    existing_data["folds"][fold_key] = {"runs": {}}
                existing_data["folds"][fold_key]["runs"].update(runs_dict)
            os.makedirs(os.path.dirname(save_file) if os.path.dirname(save_file) else ".", exist_ok=True)
            with open(save_file, "w") as f:
                json.dump(existing_data, f, indent=2)
            if verbose:
                total_records = 0
                total_folds = len(existing_data["folds"])
                total_runs = 0
                for fold_data in existing_data["folds"].values():
                    runs = fold_data.get("runs", {})
                    total_runs += len(runs)
                    for run_data in runs.values():
                        total_records += len(run_data.get("records", [])) if isinstance(run_data, dict) else len(run_data) if isinstance(run_data, list) else 0
                print(f"IterationParameterCallback: Aggregated {total_folds} folds, {total_runs} runs with {total_records} total records")
                print(f"  Saved to: {save_file}")
            for temp_file in temp_files:
                try:
                    os.remove(temp_file)
                    if verbose:
                        print(f"  Removed temp file: {os.path.basename(temp_file)}")
                except Exception as e:
                    if verbose:
                        print(f"  Warning: Could not remove temp file {temp_file}: {e}")
        else:
            if verbose:
                print(f"IterationParameterCallback: WARNING - No parameter data found in results or temp files!")
                print(f"  Checked {len(results)} results")
                print(f"  Checked {len(temp_files)} temp files")
                print(f"  Save file path: {save_file}")
            os.makedirs(os.path.dirname(save_file) if os.path.dirname(save_file) else ".", exist_ok=True)
            with open(save_file, "w") as f:
                json.dump({"runs": {}}, f, indent=2)
            if verbose:
                print(f"IterationParameterCallback: WARNING - Saved empty aggregation file to {save_file}")


class LBFGSInnerMetricsCallbackV3(Callback):
    """
    Callback that computes per-iteration training metrics (NLL, NIS, LOO_NLL,
    KF, MSE, R2) and optionally parameter snapshots. Loss is logged every
    `log_record_every_n_iters` iters (default 1). Extra metrics are computed every
    `log_metrics_every_n_iters` iters (default 5). When `log_parameters_every_n_iters`
    is set, initial/final and every-N-iter parameter snapshots are stored and
    exposed as lbfgs_parameters (separate from lbfgs_inner_metrics).

    To add more metrics: pass extra_metrics=[("MyMetric", fn), ...] where fn(context) -> float.
    """

    def __init__(
        self,
        log_record_every_n_iters: int = 1,
        log_metrics_every_n_iters: int = 5,
        log_nll: bool = True,
        log_nis: bool = True,
        log_loo: bool = True,
        log_kf: bool = True,
        log_residual_mse: bool = True,
        kf_Nf: Optional[int] = None,
        verbose: bool = False,
        extra_metrics: Optional[List[Tuple[str, Callable[[dict], float]]]] = None,
        log_parameters_every_n_iters: Optional[int] = None,
        save_file: Optional[str] = None,
    ):
        self.log_record_every_n_iters = max(1, int(log_record_every_n_iters))
        self.log_metrics_every_n_iters = max(1, int(log_metrics_every_n_iters))
        self.log_nll = log_nll
        self.log_nis = log_nis
        self.log_loo = log_loo
        self.log_kf = log_kf
        self.log_residual_mse = log_residual_mse
        self.kf_Nf = kf_Nf
        self.verbose = verbose
        self.extra_metrics = list(extra_metrics) if extra_metrics else []
        self.log_parameters_every_n_iters = max(1, int(log_parameters_every_n_iters)) if log_parameters_every_n_iters is not None else None
        self.save_file = save_file
        self._lbfgs_parameters: List[dict] = []
        self._initial_parameters = None
        self._final_parameters = None
        self._model = None
        self._trainer = None
        self._param_extractor = None

    def on_train_start(self, context: dict):
        """Store model/trainer refs and capture initial parameters when log_parameters_every_n_iters is set."""
        self._model = context.get("model")
        self._trainer = context.get("trainer")
        if self.log_parameters_every_n_iters is not None and self._model is not None:
            if self._param_extractor is None:
                self._param_extractor = KernelParameterExtractor(verbose=self.verbose)
            try:
                params = self._param_extractor.extract_all_parameters(self._model, trainer=self._trainer)
                if params is not None:
                    self._initial_parameters = dict(params)
                    jitter = None
                    if self._trainer is not None and hasattr(self._trainer, "cholesky_jitter"):
                        j = self._trainer.cholesky_jitter
                        jitter = float(j) if j is not None and (isinstance(j, (int, float)) or hasattr(j, "item")) else None
                    self._initial_parameters["jitter"] = jitter
            except Exception as e:
                if self.verbose:
                    print(f"LBFGSInnerMetricsCallbackV3: Could not capture initial parameters: {e}")
                self._initial_parameters = None

    def on_lbfgs_iteration(self, context: dict):
        """
        Called from v3 trainer each LBFGS inner iteration.

        Expected context keys:
          - epoch, lbfgs_iter, loss
          - model, trainer, device
          - run_jitter
          - mll (MarginalLogLikelihood instance)
          - record (dict to be updated)
        """
        lbfgs_iter: int = context.get("lbfgs_iter", 0)
        if lbfgs_iter <= 0:
            return

        record = context["record"]

        # When not on a "log record" iteration, ask trainer to skip appending this record
        if lbfgs_iter != 1 and (lbfgs_iter % self.log_record_every_n_iters) != 0:
            record["_skip_append"] = True
            return

        # Only compute/add NLL and extra metrics (NIS, LOO, KF, MSE, R2) every log_metrics_every_n_iters
        if lbfgs_iter != 1 and (lbfgs_iter % self.log_metrics_every_n_iters) != 0:
            return

        model = context["model"]
        trainer = context["trainer"]
        run_jitter = float(context.get("run_jitter", getattr(trainer, "cholesky_jitter", 1e-6)))

        from .training_metrics import compute_training_metrics_batch

        train_x = trainer.train_x.to(dtype=trainer.dtype)
        train_y = trainer.train_y.to(dtype=trainer.dtype)

        # NLL: reuse loss from optimizer (no extra forward)
        if self.log_nll:
            loss_val = context.get("loss")
            if loss_val is not None:
                record["NLL"] = float(loss_val)

        # All other metrics in one pass: single model.eval(), one forward, one likelihood(prior), then model.train()
        need_batch = self.log_nis or self.log_loo or self.log_kf or self.log_residual_mse
        if need_batch:
            batch = compute_training_metrics_batch(
                model,
                train_x,
                train_y,
                cholesky_jitter=run_jitter,
                log_nis=self.log_nis,
                log_loo=self.log_loo,
                log_kf=self.log_kf,
                log_mse=self.log_residual_mse,
                log_r2=self.log_residual_mse,
                kf_Nf=self.kf_Nf,
                kf_seed=lbfgs_iter,
            )
            for k, v in batch.items():
                record[k] = float(v)

        # User-defined extra metrics: callable(context) -> float
        for name, fn in self.extra_metrics:
            try:
                val = fn(context)
                if val is not None and (isinstance(val, (int, float)) or hasattr(val, "item")):
                    record[name] = float(val.item() if hasattr(val, "item") else val)
            except Exception:
                if self.verbose:
                    record[name] = float("nan")

        # Optional: parameter snapshots every N iters (reported separately as lbfgs_parameters)
        if self.log_parameters_every_n_iters is not None and (lbfgs_iter == 1 or lbfgs_iter % self.log_parameters_every_n_iters == 0):
            if self._param_extractor is None and model is not None:
                self._param_extractor = KernelParameterExtractor(verbose=self.verbose)
                self._model = model
                self._trainer = trainer
            if self._param_extractor is not None and model is not None:
                try:
                    params = self._param_extractor.extract_all_parameters(model, trainer=trainer)
                    if params is not None:
                        params = dict(params)
                        params["jitter"] = run_jitter
                        self._lbfgs_parameters.append({
                            "iteration": lbfgs_iter,
                            "loss": context.get("loss"),
                            "jitter": run_jitter,
                            "parameters": params,
                        })
                except Exception as e:
                    if self.verbose:
                        print(f"LBFGSInnerMetricsCallbackV3: Could not capture parameters at iter {lbfgs_iter}: {e}")

    def on_train_end(self, context: dict):
        """Capture final parameters when log_parameters_every_n_iters is set; optionally save to save_file."""
        if self.log_parameters_every_n_iters is None:
            return
        model = context.get("model")
        trainer = context.get("trainer")
        best_state_dict = context.get("best_state_dict")
        if model is not None and self._param_extractor is not None:
            try:
                import copy
                model_to_extract = model
                if best_state_dict is not None:
                    model_to_extract = copy.deepcopy(model)
                    model_to_extract.load_state_dict(best_state_dict)
                self._final_parameters = self._param_extractor.extract_all_parameters(model_to_extract, trainer=trainer)
                if self._final_parameters is not None:
                    self._final_parameters = dict(self._final_parameters)
                    jitter = None
                    if trainer is not None and hasattr(trainer, "cholesky_jitter"):
                        j = trainer.cholesky_jitter
                        jitter = float(j) if j is not None and (isinstance(j, (int, float)) or hasattr(j, "item")) else None
                    self._final_parameters["jitter"] = jitter
            except Exception as e:
                if self.verbose:
                    print(f"LBFGSInnerMetricsCallbackV3: Could not capture final parameters: {e}")
                self._final_parameters = None
        if self.save_file is not None:
            try:
                import json
                import os
                data = self.get_stored_parameters()
                if data:
                    os.makedirs(os.path.dirname(self.save_file) or ".", exist_ok=True)
                    with open(self.save_file, "w") as f:
                        json.dump(data, f, indent=2)
            except Exception as e:
                if self.verbose:
                    print(f"LBFGSInnerMetricsCallbackV3: Could not save to {self.save_file}: {e}")

    def get_stored_parameters(self):
        """Return lbfgs_parameters (and initial/final when log_parameters_every_n_iters is set) for aggregation."""
        if self.log_parameters_every_n_iters is None:
            return None
        return {
            "initial_parameters": self._initial_parameters,
            "final_parameters": self._final_parameters,
            "lbfgs_parameters": getattr(self, "_lbfgs_parameters", []).copy(),
        }

    @staticmethod
    def aggregate_parameters_from_results(results, save_file: str, verbose: bool = True):
        """
        Aggregate parameter data from multiple training runs into a single file.
        
        This method should be called after all runs complete to collect parameter data
        from all runs and save it to a single file organized by run_index.
        
        Args:
            results: List of result dictionaries from GPTrainer.train(), each containing
                    'run_index' and 'callback_data' with 'IterationParameterCallback' data
            save_file: Path to save the aggregated parameter data
            verbose: Whether to print aggregation status
        """
        import json
        import os
        import glob
        
        if save_file is None:
            return
        
        # Collect parameter data from all folds and runs
        # Structure: {fold_index: {run_index: {"initial_parameters": {...}, "records": [...]}}}
        aggregated_data = {}  # Will be organized by fold, then by run
        
        # First, check for temporary files (they have the most complete data)
        base_path, ext = os.path.splitext(save_file)
        # Look for both patterns: with fold_index and without
        temp_pattern_with_fold = f"{base_path}_temp_fold_*_run_*{ext}"
        temp_pattern_no_fold = f"{base_path}_temp_run_*{ext}"
        temp_files = glob.glob(temp_pattern_with_fold) + glob.glob(temp_pattern_no_fold)
        
        if verbose:
            print(f"IterationParameterCallback: Aggregating data from {len(results)} results...")
            print(f"  Looking for temp files with patterns: {temp_pattern_with_fold}, {temp_pattern_no_fold}")
            print(f"  Found {len(temp_files)} temp files: {[os.path.basename(f) for f in temp_files]}")
        
        for temp_file in temp_files:
            try:
                # Extract fold_index and run_index from filename
                filename = os.path.basename(temp_file)
                base_name = os.path.basename(base_path)
                
                # Try pattern with fold: {base_path}_temp_fold_{fold_index}_run_{run_index}{ext}
                if f"_temp_fold_" in filename:
                    # Extract fold_index and run_index
                    fold_match = filename.replace(f"{base_name}_temp_fold_", "").replace(ext, "")
                    parts = fold_match.split("_run_")
                    if len(parts) == 2:
                        fold_idx_str = parts[0]
                        run_idx_str = parts[1]
                    else:
                        # Fallback: treat as run_index only
                        fold_idx_str = None
                        run_idx_str = fold_match.replace("_run_", "")
                else:
                    # Pattern without fold: {base_path}_temp_run_{run_index}{ext}
                    fold_idx_str = None
                    match = filename.replace(f"{base_name}_temp_run_", "").replace(ext, "")
                    run_idx_str = match
                
                with open(temp_file, "r") as f:
                    temp_data = json.load(f)
                
                # Organize by fold, then by run
                if fold_idx_str is None:
                    fold_idx_str = "0"  # Default fold if not specified
                
                if fold_idx_str not in aggregated_data:
                    aggregated_data[fold_idx_str] = {}
                
                aggregated_data[fold_idx_str][run_idx_str] = temp_data
                if verbose:
                    num_records = len(temp_data.get("records", [])) if isinstance(temp_data, dict) else len(temp_data) if isinstance(temp_data, list) else 0
                    has_initial = temp_data.get("initial_parameters") is not None if isinstance(temp_data, dict) else False
                    print(f"  Fold {fold_idx_str}, Run {run_idx_str}: Loaded {num_records} records from temp file (initial_parameters: {'present' if has_initial else 'null'})")
            except Exception as e:
                if verbose:
                    print(f"  Warning: Could not load temp file {temp_file}: {e}")
        
        # Then, try to get data from results (callback_data) for runs not found in temp files
        for result in results:
            run_index = result.get("run_index")
            fold_index = result.get("fold_index")  # May not be present in old format
            if run_index is None:
                if verbose:
                    print(f"  Warning: Result missing run_index: {list(result.keys())}")
                continue
            
            run_idx_str = str(run_index)
            fold_idx_str = str(fold_index) if fold_index is not None else "0"
            
            # Only use callback_data if we don't already have data from temp files
            if fold_idx_str not in aggregated_data or run_idx_str not in aggregated_data.get(fold_idx_str, {}):
                if fold_idx_str not in aggregated_data:
                    aggregated_data[fold_idx_str] = {}
                
                callback_data = result.get("callback_data", {})
                param_data = callback_data.get("IterationParameterCallback")
                
                if param_data is not None:
                    # Data is already in the correct format from callback_data
                    if isinstance(param_data, dict) and "initial_parameters" in param_data and "records" in param_data:
                        aggregated_data[fold_idx_str][run_idx_str] = param_data
                        if verbose:
                            has_initial = param_data.get("initial_parameters") is not None
                            print(f"  Fold {fold_idx_str}, Run {run_index}: Found {len(param_data.get('records', []))} records in callback_data (initial_parameters: {'present' if has_initial else 'null'})")
                    elif isinstance(param_data, list):
                        # Old format: just a list of records
                        aggregated_data[fold_idx_str][run_idx_str] = {
                            "initial_parameters": None,
                            "records": param_data
                        }
                        if verbose:
                            print(f"  Fold {fold_idx_str}, Run {run_index}: Found {len(param_data)} records in callback_data (old format)")
        
        # Save aggregated data organized by folds
        if aggregated_data:
            # Read existing aggregated data if file exists (to merge with previous folds)
            existing_data = {"folds": {}}
            if os.path.exists(save_file):
                try:
                    with open(save_file, "r") as f:
                        existing_data = json.load(f)
                        if "folds" not in existing_data:
                            existing_data = {"folds": {}}
                except Exception as e:
                    if verbose:
                        print(f"  Warning: Could not read existing aggregated file: {e}")
                    existing_data = {"folds": {}}
            
            # Merge new fold data with existing data
            for fold_idx, runs_dict in aggregated_data.items():
                fold_key = f"fold_{fold_idx}"
                if fold_key not in existing_data["folds"]:
                    existing_data["folds"][fold_key] = {"runs": {}}
                # Merge runs for this fold (new runs overwrite old ones if same run_index)
                existing_data["folds"][fold_key]["runs"].update(runs_dict)
            
            # Write merged data back
            os.makedirs(os.path.dirname(save_file) if os.path.dirname(save_file) else ".", exist_ok=True)
            with open(save_file, "w") as f:
                json.dump(existing_data, f, indent=2)
            
            output_data = existing_data
            
            if verbose:
                total_records = 0
                total_folds = len(output_data["folds"])
                total_runs = 0
                for fold_data in output_data["folds"].values():
                    runs = fold_data.get("runs", {})
                    total_runs += len(runs)
                    for run_data in runs.values():
                        total_records += len(run_data.get("records", [])) if isinstance(run_data, dict) else len(run_data) if isinstance(run_data, list) else 0
                print(f"IterationParameterCallback: Aggregated {total_folds} folds, {total_runs} runs with {total_records} total records")
                print(f"  Saved to: {save_file}")
            
            # Clean up temporary files
            for temp_file in temp_files:
                try:
                    os.remove(temp_file)
                    if verbose:
                        print(f"  Removed temp file: {os.path.basename(temp_file)}")
                except Exception as e:
                    if verbose:
                        print(f"  Warning: Could not remove temp file {temp_file}: {e}")
        else:
            if verbose:
                print(f"IterationParameterCallback: WARNING - No parameter data found in results or temp files!")
                print(f"  Checked {len(results)} results")
                print(f"  Checked {len(temp_files)} temp files")
                print(f"  Save file path: {save_file}")
            # Save empty structure
            output_data = {"runs": {}}
            os.makedirs(os.path.dirname(save_file) if os.path.dirname(save_file) else ".", exist_ok=True)
            with open(save_file, "w") as f:
                json.dump(output_data, f, indent=2)
            if verbose:
                print(f"IterationParameterCallback: WARNING - Saved empty aggregation file to {save_file}")


class EpochParameterCallback(Callback):
    """
    Callback to record parameters at each epoch for Adam and other non-LBFGS optimizers.
    Only activates when Adam (or specified optimizers) is used.
    
    This callback records parameters from every run (not just the best one),
    making it useful for analyzing parameter evolution across multiple initializations.
    
    Args:
        save_file (str, optional): Path to save the parameters JSON file. If None, no file is written.
        verbose (bool): If True, print to console when storing. Does not affect saving.
        optimizer_classes (list): List of optimizer classes that should trigger this callback.
        save_every_n_epochs (int): Save to file every N epochs (only if save_file is set). Defaults to 10.
    """
    
    def __init__(self, save_file: str = None, verbose: bool = False, optimizer_classes=None, save_every_n_epochs: int = 10):
        self.save_file = save_file
        self.verbose = verbose
        self.save_every_n_epochs = save_every_n_epochs
        self.recorded_parameters = []
        self.extractor = KernelParameterExtractor(verbose=verbose)
        self._is_active = False
        self._optimizer_classes = optimizer_classes or [torch.optim.Adam]
        # Jitter tracking (warning-capture) fields are always initialized so tests / inactive callbacks don't crash
        self._trainer = None
        self._last_actual_jitter = None
        self._max_actual_jitter = None
        self._warning_handler = None
        self._initial_parameters = None  # Store initial parameters from ParameterInitializer
        self._run_index = None  # Will be set via set_run_index() by GPTrainer
        self._fold_index = None  # Will be set via set_fold_index() by experiment script
    
    def set_run_index(self, run_index: int):
        """Set the run index for this callback instance. Called by GPTrainer."""
        self._run_index = run_index
    
    def set_fold_index(self, fold_index: int):
        """Set the fold index for this callback instance. Called by experiment script."""
        self._fold_index = fold_index
    
    def on_train_start(self, context: dict):
        """Check if we should activate based on optimizer type."""
        trainer = context.get("trainer", None)
        self._trainer = trainer
        # Check if run_index is in context (set by GPTrainer via set_run_index)
        self._run_index = context.get("run_index", self._run_index)
        
        if trainer is not None and hasattr(trainer, "optimizer_class"):
            optimizer_class = trainer.optimizer_class
            # Check if optimizer class matches any in our list
            self._is_active = any(
                optimizer_class == opt_class or 
                (isinstance(optimizer_class, type) and issubclass(optimizer_class, opt_class))
                for opt_class in self._optimizer_classes
            )
            
            if self.verbose:
                if self._is_active:
                    print(f"EpochParameterCallback: Activated for optimizer {optimizer_class.__name__}")
                else:
                    print(f"EpochParameterCallback: Not activated (optimizer: {optimizer_class.__name__})")
        else:
            self._is_active = False
        
        # Capture initial parameters (set by ParameterInitializer before training starts)
        try:
            model = context.get("model", None)
            if model is not None:
                self._initial_parameters = self.extractor.extract_all_parameters(model, trainer=self._trainer)
                if self.verbose:
                    print(f"EpochParameterCallback: Captured initial parameters")
        except Exception as e:
            if self.verbose:
                print(f"EpochParameterCallback: Warning - Could not capture initial parameters: {e}")
            self._initial_parameters = None
        
        # Set up warning handler to capture jitter values (same as JitterTrackingCallback)
        if self._is_active:
            try:
                import warnings
                if not hasattr(warnings, '_showwarning_orig'):
                    warnings._showwarning_orig = warnings.showwarning
                self._warning_handler = self._create_jitter_warning_handler()
                warnings.showwarning = self._warning_handler
            except Exception as e:
                if self.verbose:
                    print(f"EpochParameterCallback: Warning - Could not set up warning handler: {e}")
                self._warning_handler = None
    
    def _create_jitter_warning_handler(self):
        """Create a warning handler that captures jitter values from linear_operator warnings."""
        import warnings
        import re
        callback_instance = self  # Capture self for use in closure
        
        def jitter_warning_handler(message, category, filename, lineno, file=None, line=None):
            """Intercept warnings to capture jitter values from linear_operator."""
            try:
                # Call original handler first
                if hasattr(warnings, '_showwarning_orig'):
                    warnings._showwarning_orig(message, category, filename, lineno, file, line)
                else:
                    import sys
                    sys.stderr.write(warnings.formatwarning(message, category, filename, lineno))
                
                # Check if this is a jitter-related warning from linear_operator
                if isinstance(message, (str, Warning)):
                    msg_str = str(message)
                    filename_str = str(filename).lower() if filename else ""
                    if 'cholesky' in filename_str or 'linear_operator' in filename_str or 'jitter' in msg_str.lower():
                        patterns = [
                            r'added jitter of ([\d.]+e[+-]?\d+)',
                            r'added jitter of ([\d.]+)',
                            r'jitter of ([\d.]+e[+-]?\d+)',
                        ]
                        for pattern in patterns:
                            match = re.search(pattern, msg_str, re.IGNORECASE)
                            if match:
                                try:
                                    jitter_value = float(match.group(1))
                                    callback_instance._last_actual_jitter = jitter_value
                                    if callback_instance._max_actual_jitter is None or jitter_value > callback_instance._max_actual_jitter:
                                        callback_instance._max_actual_jitter = jitter_value
                                except (ValueError, AttributeError):
                                    pass
                                break
            except Exception:
                pass
        
        return jitter_warning_handler
    
    def _get_current_jitter(self) -> float:
        """Get current jitter value from settings, trainer, or captured warnings.
        
        Priority (same as JitterTrackingCallback):
        1. Last actual jitter captured from linear_operator warnings (most accurate)
        2. Trainer's current_jitter (updated on NotPSDError)
        3. gpytorch settings cholesky_jitter
        4. Trainer's initial cholesky_jitter
        """
        # First, check if we captured an actual jitter value from warnings
        # Use max jitter if available (most accurate), otherwise use last
        if self._max_actual_jitter is not None:
            return float(self._max_actual_jitter)
        if self._last_actual_jitter is not None:
            return float(self._last_actual_jitter)
        
        # Fallback: try to get from trainer's current_jitter (preferred) / _current_run_jitter (v3)
        if self._trainer is not None:
            for attr in ("current_jitter", "_current_run_jitter"):
                if hasattr(self._trainer, attr):
                    j = getattr(self._trainer, attr)
                    if hasattr(j, "item"):
                        try:
                            j = j.item()
                        except Exception:
                            pass
                    if j is not None:
                        try:
                            return float(j)
                        except (TypeError, ValueError):
                            pass

        # Next: try to get jitter stored on the model (persisted at end of training, may be set elsewhere too)
        if self._model is not None and hasattr(self._model, "cholesky_jitter"):
            j = getattr(self._model, "cholesky_jitter")
            if hasattr(j, "item"):
                try:
                    j = j.item()
                except Exception:
                    pass
            if j is not None:
                try:
                    return float(j)
                except (TypeError, ValueError):
                    pass
        
        try:
            # Try to get from gpytorch settings
            import gpytorch
            if hasattr(gpytorch.settings, "cholesky_jitter"):
                jitter_value = gpytorch.settings.cholesky_jitter.value
                if jitter_value is not None:
                    return float(jitter_value)
        except Exception:
            pass
        
        # Final fallback: get from trainer's cholesky_jitter (initial value)
        if self._trainer is not None and hasattr(self._trainer, "cholesky_jitter"):
            return float(self._trainer.cholesky_jitter)
        
        return None
    
    def on_epoch_end(self, context: dict):
        """Record parameters at end of each epoch."""
        if not self._is_active:
            return
        
        model = context["model"]
        epoch = context.get("epoch", 0)
        loss = context.get("loss", None)
        jitter = context.get("jitter", None)  # Get jitter from context
        
        # Fallback to getting jitter from warning handler if not in context
        if jitter is None:
            jitter = self._get_current_jitter()
        
        try:
            # Extract parameters using the extractor
            params = self.extractor.extract_all_parameters(model, trainer=self._trainer)
            
            # Create record
            record = {
                "epoch": epoch,
                "loss": loss,
                "jitter": jitter,
                "parameters": params,
            }
            
            # Add run_index to record
            record["run_index"] = self._run_index
            record["fold_index"] = self._fold_index
            self.recorded_parameters.append(record)

            if self.verbose and epoch % 10 == 0:  # Print every 10 epochs
                run_str = f"Run {self._run_index}, " if self._run_index is not None else ""
                print(f"{run_str}Epoch {epoch} - Loss: {loss:.6f}" if loss else f"{run_str}Epoch {epoch}")
            
            # Save to file periodically based on save_every_n_epochs
            if self.save_file is not None:
                should_save = False
                if self.save_every_n_epochs is None:
                    # Only save at end (handled in on_train_end)
                    should_save = False
                elif epoch == 0:
                    # Always save first epoch
                    should_save = True
                elif epoch % self.save_every_n_epochs == 0:
                    should_save = True
                
                if should_save:
                    self._save_parameters()
        
        except Exception as e:
            if self.verbose:
                print(f"Error recording parameters at epoch {epoch}: {e}")
            import traceback
            traceback.print_exc()
    
    def on_train_end(self, context: dict):
        """Final save at end of training."""
        # Restore original warning handler
        if hasattr(self, "_warning_handler") and self._warning_handler is not None:
            import warnings
            if hasattr(warnings, '_showwarning_orig'):
                warnings.showwarning = warnings._showwarning_orig
        
        if self.save_file is not None and self.recorded_parameters:
            self._save_parameters()
            if self.verbose:
                run_str = f" (run_index={self._run_index})" if self._run_index is not None else ""
                print(f"EpochParameterCallback: Saved {len(self.recorded_parameters)} epoch records{run_str}")
    
    def _save_parameters(self):
        """Save recorded parameters to JSON file.
        
        For single runs, saves directly to save_file.
        For multiple runs (when run_index is set), saves to a temporary file
        that will be aggregated later.
        """
        try:
            import json
            import os
            
            if self.save_file is None:
                return
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.save_file) if os.path.dirname(self.save_file) else ".", exist_ok=True)
            
            # Prepare output data
            output_data = {
                "initial_parameters": self._initial_parameters,
                "records": self.recorded_parameters,
            }
            
            # If run_index is set, we're in a multi-run scenario - save to temp file
            # Include fold_index in filename if available
            if self._run_index is not None:
                # Save to temporary file: {base_path}_temp_fold_{fold_index}_run_{run_index}.json
                base_path, ext = os.path.splitext(self.save_file)
                if self._fold_index is not None:
                    temp_file = f"{base_path}_temp_fold_{self._fold_index}_run_{self._run_index}{ext}"
                else:
                    temp_file = f"{base_path}_temp_run_{self._run_index}{ext}"
                with open(temp_file, "w") as f:
                    json.dump(output_data, f, indent=2)
                if self.verbose:
                    print(f"EpochParameterCallback: Saved {len(self.recorded_parameters)} records to temp file: {temp_file}")
            else:
                # Single run - save directly to main file
                with open(self.save_file, "w") as f:
                    json.dump(output_data, f, indent=2)
        except Exception as e:
            if self.verbose:
                print(f"Error saving parameters to file: {e}")
    
    def get_stored_parameters(self):
        """Get the stored parameters for external use.
        
        Returns the full structure including initial_parameters and records,
        matching the format saved to files.
        """
        return {
            "initial_parameters": self._initial_parameters,
            "records": self.recorded_parameters.copy(),
        }
    
    @staticmethod
    def aggregate_parameters_from_results(results, save_file: str, verbose: bool = True):
        """
        Aggregate parameter data from multiple training runs into a single file.
        
        This method should be called after all runs complete to collect parameter data
        from all runs and save it to a single file organized by run_index.
        
        Args:
            results: List of result dictionaries from GPTrainer.train(), each containing
                    'run_index' and 'callback_data' with 'EpochParameterCallback' data
            save_file: Path to save the aggregated parameter data
            verbose: Whether to print aggregation status
        """
        import json
        import os
        import glob
        
        if save_file is None:
            return
        
        # Collect parameter data from all folds and runs
        # Structure: {fold_index: {run_index: {"initial_parameters": {...}, "records": [...]}}}
        aggregated_data = {}  # Will be organized by fold, then by run
        
        # First, check for temporary files (they have the most complete data, saved after on_train_start)
        base_path, ext = os.path.splitext(save_file)
        # Look for both patterns: with fold_index and without
        temp_pattern_with_fold = f"{base_path}_temp_fold_*_run_*{ext}"
        temp_pattern_no_fold = f"{base_path}_temp_run_*{ext}"
        temp_files = glob.glob(temp_pattern_with_fold) + glob.glob(temp_pattern_no_fold)
        
        if verbose:
            print(f"EpochParameterCallback: Aggregating data from {len(results)} results...")
            print(f"  Looking for temp files with patterns: {temp_pattern_with_fold}, {temp_pattern_no_fold}")
            print(f"  Found {len(temp_files)} temp files: {[os.path.basename(f) for f in temp_files]}")
        
        for temp_file in temp_files:
            try:
                # Extract fold_index and run_index from filename
                filename = os.path.basename(temp_file)
                base_name = os.path.basename(base_path)
                
                # Try pattern with fold: {base_path}_temp_fold_{fold_index}_run_{run_index}{ext}
                if f"_temp_fold_" in filename:
                    # Extract fold_index and run_index
                    fold_match = filename.replace(f"{base_name}_temp_fold_", "").replace(ext, "")
                    parts = fold_match.split("_run_")
                    if len(parts) == 2:
                        fold_idx_str = parts[0]
                        run_idx_str = parts[1]
                    else:
                        # Fallback: treat as run_index only
                        fold_idx_str = None
                        run_idx_str = fold_match.replace("_run_", "")
                else:
                    # Pattern without fold: {base_path}_temp_run_{run_index}{ext}
                    fold_idx_str = None
                    match = filename.replace(f"{base_name}_temp_run_", "").replace(ext, "")
                    run_idx_str = match
                
                with open(temp_file, "r") as f:
                    temp_data = json.load(f)
                
                # Organize by fold, then by run
                if fold_idx_str is None:
                    fold_idx_str = "0"  # Default fold if not specified
                
                if fold_idx_str not in aggregated_data:
                    aggregated_data[fold_idx_str] = {}
                
                aggregated_data[fold_idx_str][run_idx_str] = temp_data
                if verbose:
                    num_records = len(temp_data.get("records", [])) if isinstance(temp_data, dict) else len(temp_data) if isinstance(temp_data, list) else 0
                    has_initial = temp_data.get("initial_parameters") is not None if isinstance(temp_data, dict) else False
                    print(f"  Fold {fold_idx_str}, Run {run_idx_str}: Loaded {num_records} records from temp file (initial_parameters: {'present' if has_initial else 'null'})")
            except Exception as e:
                if verbose:
                    print(f"  Warning: Could not load temp file {temp_file}: {e}")
        
        # Then, try to get data from results (callback_data) for runs not found in temp files
        for result in results:
            run_index = result.get("run_index")
            fold_index = result.get("fold_index")  # May not be present in old format
            if run_index is None:
                if verbose:
                    print(f"  Warning: Result missing run_index: {list(result.keys())}")
                continue
            
            run_idx_str = str(run_index)
            fold_idx_str = str(fold_index) if fold_index is not None else "0"
            
            # Only use callback_data if we don't already have data from temp files
            if fold_idx_str not in aggregated_data or run_idx_str not in aggregated_data.get(fold_idx_str, {}):
                if fold_idx_str not in aggregated_data:
                    aggregated_data[fold_idx_str] = {}
                
                callback_data = result.get("callback_data", {})
                param_data = callback_data.get("EpochParameterCallback")
                
                if param_data is not None:
                    # Data is already in the correct format from callback_data
                    if isinstance(param_data, dict) and "initial_parameters" in param_data and "records" in param_data:
                        aggregated_data[fold_idx_str][run_idx_str] = param_data
                        if verbose:
                            has_initial = param_data.get("initial_parameters") is not None
                            print(f"  Fold {fold_idx_str}, Run {run_index}: Found {len(param_data.get('records', []))} records in callback_data (initial_parameters: {'present' if has_initial else 'null'})")
                    elif isinstance(param_data, list):
                        # Old format: just a list of records
                        aggregated_data[fold_idx_str][run_idx_str] = {
                            "initial_parameters": None,
                            "records": param_data
                        }
                        if verbose:
                            print(f"  Fold {fold_idx_str}, Run {run_index}: Found {len(param_data)} records in callback_data (old format)")
        
        # Save aggregated data organized by folds
        if aggregated_data:
            # Read existing aggregated data if file exists (to merge with previous folds)
            existing_data = {"folds": {}}
            if os.path.exists(save_file):
                try:
                    with open(save_file, "r") as f:
                        existing_data = json.load(f)
                        if "folds" not in existing_data:
                            existing_data = {"folds": {}}
                except Exception as e:
                    if verbose:
                        print(f"  Warning: Could not read existing aggregated file: {e}")
                    existing_data = {"folds": {}}
            
            # Merge new fold data with existing data
            for fold_idx, runs_dict in aggregated_data.items():
                fold_key = f"fold_{fold_idx}"
                if fold_key not in existing_data["folds"]:
                    existing_data["folds"][fold_key] = {"runs": {}}
                # Merge runs for this fold (new runs overwrite old ones if same run_index)
                existing_data["folds"][fold_key]["runs"].update(runs_dict)
            
            # Write merged data back
            os.makedirs(os.path.dirname(save_file) if os.path.dirname(save_file) else ".", exist_ok=True)
            with open(save_file, "w") as f:
                json.dump(existing_data, f, indent=2)
            
            output_data = existing_data
            
            if verbose:
                total_records = 0
                total_folds = len(output_data["folds"])
                total_runs = 0
                for fold_data in output_data["folds"].values():
                    runs = fold_data.get("runs", {})
                    total_runs += len(runs)
                    for run_data in runs.values():
                        total_records += len(run_data.get("records", [])) if isinstance(run_data, dict) else len(run_data) if isinstance(run_data, list) else 0
                print(f"EpochParameterCallback: Aggregated {total_folds} folds, {total_runs} runs with {total_records} total records")
                print(f"  Saved to: {save_file}")
            
            # Clean up temporary files
            for temp_file in temp_files:
                try:
                    os.remove(temp_file)
                    if verbose:
                        print(f"  Removed temp file: {os.path.basename(temp_file)}")
                except Exception as e:
                    if verbose:
                        print(f"  Warning: Could not remove temp file {temp_file}: {e}")
        else:
            if verbose:
                print(f"EpochParameterCallback: WARNING - No parameter data found in results or temp files!")
                print(f"  Checked {len(results)} results")
                print(f"  Checked {len(temp_files)} temp files")
                print(f"  Save file path: {save_file}")
            # Save empty structure
            output_data = {"runs": {}}
            os.makedirs(os.path.dirname(save_file) if os.path.dirname(save_file) else ".", exist_ok=True)
            with open(save_file, "w") as f:
                json.dump(output_data, f, indent=2)
            if verbose:
                print(f"EpochParameterCallback: WARNING - Saved empty aggregation file to {save_file}")


class JitterTrackingCallback(Callback):
    """
    Callback to track jitter values at each iteration/epoch for all runs.
    
    This callback records jitter values from every run (not just the best one),
    making it useful for analyzing jitter behavior across multiple initializations.
    The callback works with both LBFGS (iteration-based) and Adam (epoch-based) optimizers.
    
    Args:
        save_file (str, optional): Path to save the jitter tracking JSON file. If None, no file is written.
        verbose (bool): If True, print jitter values to console. Does not affect saving.
    """
    
    def __init__(self, save_file: str = None, verbose: bool = False):
        self.save_file = save_file
        self.verbose = verbose
        self.recorded_jitter = []  # List of jitter records
        self._optimizer = None
        self._model = None
        self._trainer = None
        self._current_epoch = 0
        self._is_active = False
        self._run_index = None  # Will be set via set_run_index() by GPTrainer
        self._current_iteration = 0
        self._last_actual_jitter = None  # Track the last (highest) jitter value from warnings
        self._max_actual_jitter = None  # Track the maximum jitter value seen from warnings
        self._warning_handler = None  # Warning handler to capture jitter from linear_operator
    
    def set_run_index(self, run_index: int):
        """Set the run index for this callback instance. Called by GPTrainer."""
        self._run_index = run_index
    
    def on_train_start(self, context: dict):
        """Store model and trainer references, and check for run_index in context."""
        self._model = context["model"]
        self._trainer = context.get("trainer", None)
        # Check if run_index is in context (set by GPTrainer via set_run_index)
        self._run_index = context.get("run_index", self._run_index)
        self._is_active = True
        
        # Store original warning handler and set up our custom one
        # Wrap in try-except to ensure it doesn't break training if something goes wrong
        try:
            import warnings
            # Ensure warnings are not filtered (we want to capture all warnings)
            # Don't filter - just intercept all warnings via showwarning
            if not hasattr(warnings, '_showwarning_orig'):
                warnings._showwarning_orig = warnings.showwarning
            self._warning_handler = self._create_jitter_warning_handler()
            warnings.showwarning = self._warning_handler
            if self.verbose:
                print(f"JitterTrackingCallback: Warning handler installed")
        except Exception as e:
            # If warning handler setup fails, just log and continue without it
            if self.verbose:
                print(f"JitterTrackingCallback: Warning - Could not set up warning handler: {e}")
                import traceback
                traceback.print_exc()
            self._warning_handler = None
        
        if self.verbose:
            run_str = f" (run_index={self._run_index})" if self._run_index is not None else ""
            print(f"JitterTrackingCallback: Activated{run_str}")
    
    def on_epoch_start(self, context: dict):
        """Update current epoch."""
        self._current_epoch = context.get("epoch", 0)
    
    def _create_jitter_warning_handler(self):
        """Create a warning handler that captures jitter values from linear_operator warnings."""
        import warnings
        import re
        callback_instance = self  # Capture self for use in closure
        
        def jitter_warning_handler(message, category, filename, lineno, file=None, line=None):
            """Intercept warnings to capture jitter values from linear_operator."""
            try:
                # Call original handler first
                if hasattr(warnings, '_showwarning_orig'):
                    warnings._showwarning_orig(message, category, filename, lineno, file, line)
                else:
                    # Fallback to default behavior
                    import sys
                    sys.stderr.write(warnings.formatwarning(message, category, filename, lineno))
                
                # Check if this is a jitter-related warning from linear_operator
                if isinstance(message, (str, Warning)):
                    msg_str = str(message)
                    # Check if this is from linear_operator cholesky (to reduce noise)
                    filename_str = str(filename).lower() if filename else ""
                    if 'cholesky' in filename_str or 'linear_operator' in filename_str or 'jitter' in msg_str.lower():
                        # Pattern: "A not p.d., added jitter of 1.0e-06 to the diagonal"
                        # Try multiple patterns to catch different formats
                        patterns = [
                            r'added jitter of ([\d.]+e[+-]?\d+)',  # Scientific notation: 1.0e-06 (e is required)
                            r'added jitter of ([\d.]+)',  # Decimal fallback
                            r'jitter of ([\d.]+e[+-]?\d+)',  # Alternative format
                        ]
                        matched = False
                        for pattern in patterns:
                            match = re.search(pattern, msg_str, re.IGNORECASE)
                            if match:
                                try:
                                    jitter_value = float(match.group(1))
                                    # Update both last and max jitter
                                    old_max = callback_instance._max_actual_jitter
                                    callback_instance._last_actual_jitter = jitter_value
                                    if callback_instance._max_actual_jitter is None or jitter_value > callback_instance._max_actual_jitter:
                                        callback_instance._max_actual_jitter = jitter_value
                                    
                                    # If jitter increased significantly, immediately record it (in case training crashes)
                                    # This ensures we capture the jitter even if _on_iteration isn't called
                                    if callback_instance._is_active:
                                        # Get current iteration (may be stale if _on_iteration hasn't been called yet)
                                        current_iter = callback_instance._current_iteration if hasattr(callback_instance, '_current_iteration') else None
                                        
                                        # Check if jitter increased significantly (by at least 10x, or reached a threshold)
                                        jitter_increased = old_max is None or jitter_value >= old_max * 10.0
                                        reached_threshold = jitter_value >= 1e-3  # Reached at least 1e-3
                                        
                                        if jitter_increased or reached_threshold:
                                            # Use the last known iteration, or estimate based on last record
                                            if current_iter is None and callback_instance.recorded_jitter:
                                                # Estimate: use last iteration + 1 (or last iteration if we can't determine)
                                                last_iter = callback_instance.recorded_jitter[-1].get("iteration")
                                                if last_iter is not None:
                                                    current_iter = last_iter + 1
                                                else:
                                                    current_iter = 0  # Fallback
                                            elif current_iter is None:
                                                current_iter = 0  # Fallback
                                            
                                            # Immediately record this jitter value
                                            # (This is a safety measure in case training crashes before _on_iteration is called)
                                            try:
                                                record = {
                                                    "run_index": callback_instance._run_index,
                                                    "epoch": callback_instance._current_epoch,
                                                    "iteration": current_iter,
                                                    "loss": None,  # Loss not available yet
                                                    "jitter": jitter_value,
                                                }
                                                # Always append when jitter increases significantly (don't check for duplicates here)
                                                # The _on_iteration method will handle updating existing records
                                                callback_instance.recorded_jitter.append(record)
                                                
                                                # Save immediately if jitter increased significantly
                                                # This ensures we capture data even if training crashes before _on_iteration is called
                                                should_save = False
                                                if jitter_value >= 1e-2:  # Very high - always save
                                                    should_save = True
                                                elif jitter_value >= 1e-3 and (old_max is None or old_max < 1e-3):  # First time reaching 1e-3
                                                    should_save = True
                                                elif jitter_increased:  # Increased by 10x or more
                                                    should_save = True
                                                
                                                if should_save and callback_instance.save_file is not None:
                                                    try:
                                                        callback_instance._save_jitter()
                                                        if callback_instance.verbose:
                                                            print(f"JitterTracking: Saved immediately due to jitter increase to {jitter_value:.2e} (iter {current_iter})")
                                                    except Exception as save_err:
                                                        if callback_instance.verbose:
                                                            print(f"JitterTracking: Warning - Could not save immediately: {save_err}")
                                            except Exception as record_error:
                                                # Don't break training if recording fails
                                                if callback_instance.verbose:
                                                    print(f"JitterTracking: Warning - Could not record jitter immediately: {record_error}")
                                    
                                    # Always print when captured to verify it's working (helps debug)
                                    if callback_instance.verbose:
                                        run_str = f"Run {callback_instance._run_index}, " if callback_instance._run_index is not None else ""
                                        iter_str = f" (iter {callback_instance._current_iteration})" if hasattr(callback_instance, '_current_iteration') else ""
                                        print(f"JitterTracking: {run_str}Captured actual jitter from warning: {jitter_value:.2e}{iter_str}")
                                    matched = True
                                except (ValueError, AttributeError) as e:
                                    if callback_instance.verbose:
                                        print(f"JitterTracking: Warning - Could not parse jitter from warning: {e}, message: {msg_str[:100]}")
                                break  # Found a match, no need to try other patterns
                        
                        # Debug: if we didn't match but it looks like a jitter warning, print it
                        if not matched and callback_instance.verbose and 'jitter' in msg_str.lower():
                            print(f"JitterTracking: Warning handler called but no match - message: {msg_str[:150]}")
            except Exception as e:
                # If anything goes wrong in the warning handler, just pass to avoid breaking training
                # But log it if verbose
                if callback_instance.verbose:
                    print(f"JitterTracking: Error in warning handler: {e}")
                pass
        
        return jitter_warning_handler
    
    def register_with_optimizer(self, optimizer, model=None, trainer=None):
        """Register this callback with the LBFGSScipy optimizer for iteration tracking.
        
        Note: The actual callback registration is handled by the trainer which chains
        multiple callbacks together. This method just stores references.
        """
        from .optimizers import LBFGSScipy
        
        if model is not None:
            self._model = model
        if trainer is not None:
            self._trainer = trainer
        
        if isinstance(optimizer, LBFGSScipy):
            self._optimizer = optimizer
            # Don't set iteration_callback directly here - the trainer will chain callbacks
            # Just mark that we're using LBFGS so we know to use _on_iteration
            if self.verbose:
                run_str = f" (run_index={self._run_index})" if self._run_index is not None else ""
                print(f"JitterTrackingCallback: Registered with LBFGSScipy optimizer{run_str}")
        else:
            # For non-LBFGS optimizers, we'll track jitter at epoch end instead
            if self.verbose:
                run_str = f" (run_index={self._run_index})" if self._run_index is not None else ""
                print(f"JitterTrackingCallback: Not LBFGS optimizer, will track at epoch end{run_str}")
    
    def _get_current_jitter(self) -> float:
        """Get current jitter value from settings, trainer, or captured warnings.
        
        Priority:
        1. Last actual jitter captured from linear_operator warnings (most accurate)
        2. Trainer's current_jitter (updated on NotPSDError)
        3. gpytorch settings cholesky_jitter
        4. Trainer's initial cholesky_jitter
        """
        # First, check if we captured an actual jitter value from warnings
        # Use max jitter if available (most accurate), otherwise use last
        if self._max_actual_jitter is not None:
            return float(self._max_actual_jitter)
        if self._last_actual_jitter is not None:
            return float(self._last_actual_jitter)
        
        # Fallback: try to get from trainer's current_jitter (preferred) / _current_run_jitter (v3)
        if self._trainer is not None:
            for attr in ("current_jitter", "_current_run_jitter"):
                if hasattr(self._trainer, attr):
                    j = getattr(self._trainer, attr)
                    if hasattr(j, "item"):
                        try:
                            j = j.item()
                        except Exception:
                            pass
                    if j is not None:
                        try:
                            return float(j)
                        except (TypeError, ValueError):
                            pass

        # Next: try to get jitter stored on the model (persisted at end of training, may be set elsewhere too)
        if self._model is not None and hasattr(self._model, "cholesky_jitter"):
            j = getattr(self._model, "cholesky_jitter")
            if hasattr(j, "item"):
                try:
                    j = j.item()
                except Exception:
                    pass
            if j is not None:
                try:
                    return float(j)
                except (TypeError, ValueError):
                    pass
        
        try:
            # Try to get from gpytorch settings
            import gpytorch
            if hasattr(gpytorch.settings, "cholesky_jitter"):
                jitter_value = gpytorch.settings.cholesky_jitter.value
                if jitter_value is not None:
                    return float(jitter_value)
        except Exception:
            pass
        
        # Final fallback: get from trainer's cholesky_jitter (initial value)
        if self._trainer is not None and hasattr(self._trainer, "cholesky_jitter"):
            return float(self._trainer.cholesky_jitter)
        
        return None
    
    def _on_iteration(self, iteration: int, loss: float = None):
        """Called by LBFGSScipy optimizer at each iteration."""
        if not self._is_active:
            return
        
        try:
            # Update current iteration first (needed for warning handler to know which iteration we're on)
            self._current_iteration = iteration
            
            # Get current jitter (this will use _max_actual_jitter or _last_actual_jitter if available from warnings)
            # The warnings from the CURRENT iteration's closure should have updated these values
            # (since _on_iteration is called AFTER the closure completes)
            jitter = self._get_current_jitter()
            
            # Check if we already have a record for this iteration (created by warning handler)
            # If so, update it with the loss value; otherwise create a new record
            existing_record = None
            if self.recorded_jitter:
                # Check the last few records to see if this iteration was already recorded
                for i in range(len(self.recorded_jitter) - 1, max(-1, len(self.recorded_jitter) - 5), -1):
                    if i >= 0 and self.recorded_jitter[i].get("iteration") == iteration:
                        existing_record = self.recorded_jitter[i]
                        break
            
            if existing_record is not None:
                # Update existing record with loss and potentially updated jitter
                existing_record["loss"] = loss
                existing_record["jitter"] = jitter  # Update with latest jitter value
            else:
                # Create new record
                record = {
                    "run_index": self._run_index,
                    "epoch": self._current_epoch,
                    "iteration": iteration,
                    "loss": loss,
                    "jitter": jitter,
                }
                self.recorded_jitter.append(record)
            
            if self.verbose and iteration % 50 == 0:  # Print every 50 iterations
                run_str = f"Run {self._run_index}, " if self._run_index is not None else ""
                # Show debug info about captured jitter
                debug_info = ""
                if self._last_actual_jitter is not None:
                    debug_info = f" (last captured: {self._last_actual_jitter:.2e})"
                print(f"JitterTracking: {run_str}Iteration {iteration} (Epoch {self._current_epoch}) - Jitter: {jitter:.2e}{debug_info}")
            
            # Save to file periodically
            # Save more frequently when jitter is high (to catch data before potential crashes)
            save_frequency = 100  # Default: every 100 iterations
            should_save_immediately = False
            
            if jitter is not None:
                if jitter >= 1e-2:  # Very high jitter - save every iteration
                    save_frequency = 1
                    should_save_immediately = True
                elif jitter >= 1e-3:  # High jitter - save every 10 iterations
                    save_frequency = 10
                elif jitter >= 1e-4:  # Medium-high jitter - save every 25 iterations
                    save_frequency = 25
                elif jitter >= 1e-5:  # Medium jitter - save every 50 iterations
                    save_frequency = 50
            
            # Save immediately if jitter reached a critical threshold, or periodically
            if self.save_file is not None:
                if should_save_immediately or iteration % save_frequency == 0 or iteration == 1:
                    self._save_jitter()
        
        except Exception as e:
            if self.verbose:
                print(f"Error recording jitter at iteration {iteration}: {e}")
            import traceback
            traceback.print_exc()
            # Try to save data even if there was an error (to preserve what we have)
            if self.save_file is not None and self.recorded_jitter:
                try:
                    self._save_jitter()
                except Exception as save_error:
                    if self.verbose:
                        print(f"Error saving jitter data after exception: {save_error}")
    
    def on_epoch_end(self, context: dict):
        """Record jitter at end of each epoch (for non-LBFGS optimizers)."""
        if not self._is_active:
            return
        
        # Only record at epoch end if we're not using LBFGS (which uses _on_iteration)
        # Check if we have an optimizer registered (LBFGS uses iteration callback)
        if self._optimizer is not None:
            # LBFGS optimizer - jitter is tracked via _on_iteration
            return
        
        epoch = context.get("epoch", 0)
        loss = context.get("loss", None)
        jitter = context.get("jitter", None)  # Get jitter from context
        
        # Fallback to getting jitter from trainer if not in context
        if jitter is None:
            jitter = self._get_current_jitter()
        
        try:
            # Create record
            record = {
                "run_index": self._run_index,
                "epoch": epoch,
                "iteration": None,  # No iteration for epoch-based optimizers
                "loss": loss,
                "jitter": jitter,
            }
            
            self.recorded_jitter.append(record)
            
            if self.verbose and epoch % 100 == 0:  # Print every 10 epochs
                run_str = f"Run {self._run_index}, " if self._run_index is not None else ""
                print(f"JitterTracking: {run_str}Epoch {epoch} - Jitter: {jitter:.2e}")
            
            # Save to file periodically (every 10 epochs) or at start
            if self.save_file is not None and (epoch % 10 == 0 or epoch == 0):
                self._save_jitter()
        
        except Exception as e:
            if self.verbose:
                print(f"Error recording jitter at epoch {epoch}: {e}")
            import traceback
            traceback.print_exc()
    
    def on_train_end(self, context: dict):
        """Final save at end of training."""
        # Restore original warning handler
        if self._warning_handler is not None:
            import warnings
            if hasattr(warnings, '_showwarning_orig'):
                warnings.showwarning = warnings._showwarning_orig
        
        # Always save if we have a save_file, even if recorded_jitter is empty
        # (this ensures temp files are created for aggregation, and we can see if callback ran but didn't record)
        if self.save_file is not None:
            self._save_jitter()  # Save even if empty - aggregation will handle it
            if self.verbose:
                run_str = f"Run {self._run_index}, " if self._run_index is not None else ""
                if self.recorded_jitter:
                    print(f"JitterTrackingCallback: Saved {len(self.recorded_jitter)} jitter records for {run_str}")
                else:
                    print(f"JitterTrackingCallback: No jitter records recorded for {run_str} (saved empty list)")
    
    def _save_jitter(self):
        """Save recorded jitter values to JSON file.
        
        For single runs, saves directly. For multiple runs (when run_index is set),
        saves to a temporary per-run file that will be aggregated later.
        """
        if self.save_file is None:
            return
        
        try:
            import json
            import os
            
            # Use absolute path to ensure consistency across processes
            save_file_abs = os.path.abspath(self.save_file)
            
            # If run_index is set, we're in a multi-run scenario
            # Save to a temporary file that will be aggregated later
            if self._run_index is not None:
                # Save to temp file for this run
                base_path = os.path.splitext(save_file_abs)[0]
                ext = os.path.splitext(save_file_abs)[1] or ".json"
                temp_file = f"{base_path}_temp_run_{self._run_index}{ext}"
                
                os.makedirs(os.path.dirname(temp_file) if os.path.dirname(temp_file) else ".", exist_ok=True)
                with open(temp_file, "w") as f:
                    json.dump(self.recorded_jitter, f, indent=2)
                
                if self.verbose:
                    print(f"JitterTrackingCallback: Saved {len(self.recorded_jitter)} records to temp file: {temp_file}")
            else:
                # Single run - save directly
                os.makedirs(os.path.dirname(save_file_abs) if os.path.dirname(save_file_abs) else ".", exist_ok=True)
                with open(save_file_abs, "w") as f:
                    json.dump(self.recorded_jitter, f, indent=2)
        except Exception as e:
            if self.verbose:
                print(f"Error saving jitter to file: {e}")
            import traceback
            traceback.print_exc()
    
    @staticmethod
    def aggregate_jitter_from_results(results, save_file: str, verbose: bool = True):
        """
        Aggregate jitter data from multiple training runs into a single file.
        
        This method should be called after all runs complete to collect jitter data
        from all runs and save it to a single file organized by run_index.
        
        Args:
            results: List of result dictionaries from GPTrainer.train(), each containing
                    'run_index' and 'callback_data' with 'JitterTrackingCallback' data
            save_file: Path to save the aggregated jitter data
            verbose: Whether to print aggregation status
        """
        import json
        import os
        import glob
        
        if save_file is None:
            return
        
        # Collect jitter data from all runs
        aggregated_data = {}  # {run_index: [records]}
        
        # First, try to get data from results (callback_data)
        if verbose:
            print(f"JitterTrackingCallback: Aggregating data from {len(results)} results...")
        
        for result in results:
            run_index = result.get("run_index")
            if run_index is None:
                if verbose:
                    print(f"  Warning: Result missing run_index: {list(result.keys())}")
                continue
            
            callback_data = result.get("callback_data", {})
            if verbose:
                print(f"  Run {run_index}: callback_data keys = {list(callback_data.keys())}")
            
            jitter_data = callback_data.get("JitterTrackingCallback", None)
            
            # Debug: print what we actually got
            if verbose:
                if jitter_data is not None:
                    print(f"  Run {run_index}: JitterTrackingCallback data type = {type(jitter_data)}, length = {len(jitter_data) if isinstance(jitter_data, (list, tuple)) else 'N/A'}")
                else:
                    print(f"  Run {run_index}: JitterTrackingCallback key not found or value is None")
            
            # Check if jitter_data exists and is not None (could be empty list, which is valid)
            if jitter_data is not None:
                # jitter_data could be a list (even if empty) or other data structure
                if isinstance(jitter_data, list):
                    if len(jitter_data) > 0:
                        if verbose:
                            print(f"  Run {run_index}: Found {len(jitter_data)} jitter records in callback_data")
                        aggregated_data[run_index] = jitter_data
                    else:
                        if verbose:
                            print(f"  Run {run_index}: JitterTrackingCallback data is empty list (no records) - will check temp files")
                else:
                    # Not a list, but data exists - use it
                    if verbose:
                        print(f"  Run {run_index}: Found jitter data (type: {type(jitter_data)}) in callback_data")
                    aggregated_data[run_index] = jitter_data
            else:
                if verbose:
                    print(f"  Run {run_index}: No JitterTrackingCallback key in callback_data")
        
        # Also check for temporary files (in case callback_data wasn't collected)
        # Use absolute path to ensure we find files regardless of working directory
        save_file_abs = os.path.abspath(save_file)
        base_path = os.path.splitext(save_file_abs)[0]
        ext = os.path.splitext(save_file_abs)[1] or ".json"
        temp_pattern = f"{base_path}_temp_run_*{ext}"
        
        if verbose:
            print(f"  Looking for temp files with pattern: {temp_pattern}")
        
        temp_files = glob.glob(temp_pattern)
        
        if verbose:
            print(f"  Found {len(temp_files)} temp files: {temp_files}")
        
        for temp_file in temp_files:
            try:
                # Extract run_index from filename
                import re
                match = re.search(r'_temp_run_(\d+)', temp_file)
                if match:
                    run_index = int(match.group(1))
                    
                    # Only load if we don't already have data for this run
                    if run_index not in aggregated_data:
                        with open(temp_file, "r") as f:
                            jitter_data = json.load(f)
                            aggregated_data[run_index] = jitter_data
                        if verbose:
                            print(f"  Run {run_index}: Loaded {len(jitter_data)} records from temp file {temp_file}")
                    
                    # Clean up temp file
                    os.remove(temp_file)
                    if verbose:
                        print(f"  Removed temp file: {temp_file}")
            except Exception as e:
                if verbose:
                    print(f"Warning: Could not process temp file {temp_file}: {e}")
                import traceback
                traceback.print_exc()
        
        # If no data was collected, warn the user
        if not aggregated_data:
            if verbose:
                print(f"JitterTrackingCallback: WARNING - No jitter data found in results or temp files!")
                print(f"  Checked {len(results)} results")
                print(f"  Checked {len(temp_files)} temp files")
                print(f"  Save file path: {save_file_abs}")
        
        # Organize data by run_index
        # Format: {"runs": {"run_index": [records]}}
        # Note: JSON keys must be strings, so we use string keys
        output_data = {
            "runs": {}
        }
        
        for run_index in sorted(aggregated_data.keys()):
            output_data["runs"][str(run_index)] = aggregated_data[run_index]
        
        # Save aggregated data (even if empty, so user knows aggregation ran)
        try:
            save_file_abs = os.path.abspath(save_file)
            os.makedirs(os.path.dirname(save_file_abs) if os.path.dirname(save_file_abs) else ".", exist_ok=True)
            with open(save_file_abs, "w") as f:
                json.dump(output_data, f, indent=2)
            
            if verbose:
                total_runs = len(output_data["runs"])
                total_records = sum(len(records) for records in output_data["runs"].values())
                if total_records > 0:
                    print(f"JitterTrackingCallback: Aggregated {total_records} jitter records from {total_runs} runs to {save_file_abs}")
                else:
                    print(f"JitterTrackingCallback: WARNING - Saved empty aggregation file to {save_file_abs}")
        except Exception as e:
            if verbose:
                print(f"Error saving aggregated jitter data: {e}")
            import traceback
            traceback.print_exc()
    
    def get_stored_jitter(self):
        """Get the stored jitter records for external use."""
        return self.recorded_jitter.copy()
    
    def get_stored_parameters(self):
        """Alias for get_stored_jitter for compatibility with GPTrainer."""
        return self.get_stored_jitter()


class ValidationMetricsCallback(Callback):
    """
    Monitor validation NLL and RRMSE during training on held-out validation data.

    Works with Adam (on_epoch_end) and LBFGSScipy (register_with_optimizer iteration hook).
    """

    def __init__(
        self,
        val_x: torch.Tensor,
        val_y: torch.Tensor,
        *,
        verbose: bool = True,
        log_every_n_epochs: int = 1,
        log_every_n_iters: int = 10,
        num_inits: Optional[int] = None,
        cholesky_jitter: Optional[float] = None,
        chunk_size: int = 512,
    ):
        self.val_x = val_x
        self.val_y = val_y
        self.verbose = verbose
        self.log_every_n_epochs = max(1, int(log_every_n_epochs))
        self.log_every_n_iters = max(1, int(log_every_n_iters))
        self.num_inits = num_inits
        self.cholesky_jitter = cholesky_jitter
        self.chunk_size = chunk_size
        self._run_index: Optional[int] = None
        self._fold_index: Optional[int] = None
        self._records: list[dict] = []
        self._trainer = None
        self._lbfgs_registered = False

    def set_run_index(self, run_index: int) -> None:
        self._run_index = run_index

    def set_fold_index(self, fold_index: int) -> None:
        self._fold_index = fold_index

    def _init_label(self, context: dict) -> str:
        run_index = context.get("run_index", self._run_index)
        num_inits = context.get("num_inits", self.num_inits)
        if run_index is not None and num_inits is not None:
            return f"Init {run_index + 1}/{num_inits}"
        if run_index is not None:
            return f"Init {run_index}"
        return "Init ?"

    @staticmethod
    def _to_float(x) -> float:
        if x is None:
            return float("nan")
        if hasattr(x, "item"):
            return float(x.item())
        return float(x)

    def _compute_and_record(
        self,
        context: dict,
        *,
        epoch: Optional[int] = None,
        lbfgs_iter: Optional[int] = None,
    ) -> dict:
        from .training_metrics import compute_validation_metrics

        model = context["model"]
        trainer = context.get("trainer", self._trainer)
        train_loss = context.get("loss")
        jitter = self.cholesky_jitter
        if jitter is None and trainer is not None and hasattr(trainer, "cholesky_jitter"):
            jitter = trainer.cholesky_jitter

        metrics = compute_validation_metrics(
            model,
            self.val_x,
            self.val_y,
            cholesky_jitter=jitter,
            chunk_size=self.chunk_size,
        )
        record = {
            "run_index": context.get("run_index", self._run_index),
            "fold_index": context.get("fold_index", self._fold_index),
            "train_loss": self._to_float(train_loss),
            "val_NLL": metrics["val_NLL"],
            "val_RRMSE": metrics["val_RRMSE"],
        }
        if epoch is not None:
            record["epoch"] = int(epoch)
        if lbfgs_iter is not None:
            record["lbfgs_iter"] = int(lbfgs_iter)
        self._records.append(record)
        context["val_metrics"] = metrics
        return metrics

    def _print_metrics(
        self,
        context: dict,
        metrics: dict,
        *,
        epoch: Optional[int] = None,
        lbfgs_iter: Optional[int] = None,
    ) -> None:
        if not self.verbose:
            return
        label = self._init_label(context)
        train_loss = context.get("loss")
        parts = [f"[{label}]"]
        if lbfgs_iter is not None:
            parts.append(f"LBFGS iter {lbfgs_iter}")
        elif epoch is not None:
            parts.append(f"Epoch {epoch}")
        if train_loss is not None:
            parts.append(f"train_loss={train_loss:.4f}")
        parts.append(f"val_NLL={metrics['val_NLL']:.4f}")
        parts.append(f"val_RRMSE={metrics['val_RRMSE']:.4f}")
        print(" | ".join(parts))

    def on_train_start(self, context: dict) -> None:
        self._trainer = context.get("trainer")
        self._run_index = context.get("run_index", self._run_index)
        self._fold_index = context.get("fold_index", self._fold_index)
        self._records = []

    def on_epoch_end(self, context: dict) -> None:
        if self._lbfgs_registered:
            return
        epoch = context.get("epoch", 0)
        if epoch != 0 and (epoch % self.log_every_n_epochs) != 0:
            return
        metrics = self._compute_and_record(context, epoch=epoch)
        self._print_metrics(context, metrics, epoch=epoch)

    def register_with_optimizer(self, optimizer, model=None, trainer=None) -> None:
        from .optimizers import LBFGSScipy

        if trainer is not None:
            self._trainer = trainer

        if not isinstance(optimizer, LBFGSScipy):
            return

        self._lbfgs_registered = True
        callback_self = self

        def on_lbfgs_iteration(iter_idx: int, loss: float) -> None:
            if iter_idx != 1 and (iter_idx % callback_self.log_every_n_iters) != 0:
                return
            ctx = {
                "model": model,
                "trainer": trainer,
                "loss": loss,
                "run_index": callback_self._run_index,
                "num_inits": callback_self.num_inits,
            }
            metrics = callback_self._compute_and_record(ctx, lbfgs_iter=iter_idx)
            callback_self._print_metrics(ctx, metrics, lbfgs_iter=iter_idx)

        if hasattr(optimizer, "iteration_callback"):
            previous_callback = optimizer.iteration_callback

            def chained_iteration_callback(iter_idx, loss):
                if previous_callback is not None:
                    previous_callback(iter_idx, loss)
                on_lbfgs_iteration(iter_idx, loss)

            optimizer.iteration_callback = chained_iteration_callback

    def on_train_end(self, context: dict) -> None:
        self._run_index = context.get("run_index", self._run_index)

    def get_stored_parameters(self):
        return {"records": self._records.copy()}
