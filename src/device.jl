using Flux

const _CUDA_MODULE = Ref{Union{Nothing, Module}}(nothing)

function _try_require(pkg_name::String)
    pkg = Base.identify_package(pkg_name)
    pkg === nothing && return nothing
    try
        return Base.invokelatest(Base.require, pkg)
    catch
        return nothing
    end
end

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
Dynamically loads `CUDA` (and `cuDNN` if installed) without module-level `@eval` world-age issues.
On macOS or hosts without a functional CUDA GPU, `:auto` falls back to `:cpu` with an informational message.
"""
function resolve_device(mode::Symbol = :auto)
    mode in (:auto, :cpu, :cuda) || error("Unsupported device mode: $mode (expected :auto, :cpu, or :cuda)")

    if mode === :cpu
        return DeviceConfig(:cpu, false)
    end

    cuda_mod = _try_require("CUDA")
    cuda_available = false
    if cuda_mod !== nothing && isdefined(cuda_mod, :functional)
        try
            cuda_available = Bool(Base.invokelatest(getfield(cuda_mod, :functional)))
        catch
            cuda_available = false
        end
    end

    if cuda_available
        _CUDA_MODULE[] = cuda_mod
        cudnn_mod = _try_require("cuDNN")
        cudnn_available = false
        if cudnn_mod !== nothing && isdefined(cudnn_mod, :functional)
            try
                cudnn_available = Bool(Base.invokelatest(getfield(cudnn_mod, :functional)))
            catch
                cudnn_available = false
            end
        end

        if cudnn_available
            @info "CUDA GPU + cuDNN detected; using GPU execution."
        else
            @info "CUDA GPU detected; using Flux.CUDADevice directly (install cuDNN with `Pkg.add(\"cuDNN\")` for additional cuDNN kernel acceleration)."
        end
        return DeviceConfig(:cuda, true)
    elseif mode === :cuda
        error(
            "CUDA device was explicitly requested (`--device cuda`), but no functional CUDA GPU is available in this environment. " *
            "Please run GPU training on the Windows machine with CUDA installed (`julia --project=. -e 'using Pkg; Pkg.add([\"CUDA\", \"cuDNN\"])'`), or use `--device cpu`."
        )
    else
        @info "CUDA GPU not available on this host (e.g. macOS); falling back to CPU execution."
        return DeviceConfig(:cpu, false)
    end
end

"""
    to_device(cfg::DeviceConfig, x)

Move model or array `x` to the configured device (`Flux.CUDADevice()` on GPU or no-op on CPU).
"""
to_device(cfg::DeviceConfig, x) = cfg.use_gpu ? Base.invokelatest(Flux.CUDADevice(), x) : x

"""
    cpu_device(x)

Move model or array `x` back to CPU.
"""
cpu_device(x) = Base.invokelatest(Flux.CPUDevice(), x)

"""
    reclaim_device_memory!(cfg::DeviceConfig)

Trigger garbage collection and, when using CUDA GPU, reclaim cached GPU memory pool allocations.
"""
function reclaim_device_memory!(cfg::DeviceConfig)
    GC.gc(true)
    if cfg.use_gpu
        cuda_mod = _CUDA_MODULE[]
        if cuda_mod !== nothing && isdefined(cuda_mod, :reclaim)
            Base.invokelatest(getfield(cuda_mod, :reclaim))
        end
    end
    return nothing
end
