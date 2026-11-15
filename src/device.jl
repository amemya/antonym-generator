using Flux

"""
    DeviceConfig

Execution device configuration holding `:cpu` or `:cuda` mode and whether GPU transfer is active.
"""
struct DeviceConfig
    mode::Symbol
    use_gpu::Bool
end

"""
    resolve_device(mode::Symbol = :auto) -> DeviceConfig

Resolve the compute device (`:auto`, `:cpu`, or `:cuda`).
On macOS or hosts without a functional CUDA GPU, `:auto` falls back to `:cpu` with an informational message.
"""
function resolve_device(mode::Symbol = :auto)
    mode in (:auto, :cpu, :cuda) || error("Unsupported device mode: $mode (expected :auto, :cpu, or :cuda)")

    if mode === :cpu
        return DeviceConfig(:cpu, false)
    end

    cuda_available = false
    try
        @eval import CUDA
        cuda_mod = getfield(@__MODULE__, :CUDA)
        cuda_available = Base.invokelatest(cuda_mod.functional)
    catch
        cuda_available = false
    end

    if cuda_available
        @info "CUDA GPU detected; using GPU execution."
        return DeviceConfig(:cuda, true)
    elseif mode === :cuda
        error(
            "CUDA device was explicitly requested (`--device cuda`), but no functional CUDA GPU is available in this environment. " *
            "Please run GPU training on the Windows machine with CUDA installed (`julia --project=. -e 'using Pkg; Pkg.add(\"CUDA\")'`), or use `--device cpu`."
        )
    else
        @info "CUDA GPU not available on this host (e.g. macOS); falling back to CPU execution."
        return DeviceConfig(:cpu, false)
    end
end

"""
    to_device(cfg::DeviceConfig, x)

Move model or array `x` to the configured device.
"""
to_device(cfg::DeviceConfig, x) = cfg.use_gpu ? Flux.gpu(x) : x

"""
    cpu_device(x)

Move model or array `x` back to CPU.
"""
cpu_device(x) = Flux.cpu(x)
