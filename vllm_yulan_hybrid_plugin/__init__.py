GREEN = "\033[32m"
RESET = "\033[0m"

def register():
    # Register custom attention op so compilation knows about it
    print(f"{GREEN}[vLLM Plugin] Registering custom attention op...{RESET}")
    from vllm.config import CompilationConfig
    print(f"{GREEN}[vLLM Plugin] {RESET}")
    if "vllm::yulan_hybrid_gdn_attention" not in CompilationConfig._attention_ops:
        CompilationConfig._attention_ops.append("vllm::yulan_hybrid_gdn_attention")
        print(f"{GREEN}[vLLM Plugin] Added 'vllm::yulan_hybrid_gdn_attention' to CompilationConfig._attention_ops{RESET}")

    print(f"{GREEN}[vLLM Plugin] Enable this plugin by setting `os.environ['VLLM_PLUGINS'] = 'register_yulan_hybrid_model'` if this not working{RESET}")

    from vllm import ModelRegistry
    from vllm.transformers_utils.config import _CONFIG_REGISTRY
    import vllm.transformers_utils.configs as configs
    import vllm
    
    # Register configuration class
    from .configuration_yulan_hybrid import YuLanHybridConfig
    
    # Step 1: Dynamically add config class to configs module
    # This allows LazyConfigDict to find it via getattr(configs, "YuLanHybridConfig")
    setattr(configs, "YuLanHybridConfig", YuLanHybridConfig)
    
    # Step 2: Register config class name (as string) to _CONFIG_REGISTRY
    if "yulan_hybrid" not in _CONFIG_REGISTRY:
        _CONFIG_REGISTRY["yulan_hybrid"] = "YuLanHybridConfig" # NOTE use str rather than class
        print(f"{GREEN}[vLLM Plugin] Registered YuLanHybridConfig{RESET}")
    
    # Get vLLM version and select appropriate implementation
    vllm_version = vllm.__version__
    
    if vllm_version.startswith("0.10.2"):
        from .versions.v0_10_2 import YuLanHybridForCausalLM
        print(f"{GREEN}[vLLM Plugin] Loaded implementation for vLLM {vllm_version}{RESET}")
        
        # Register YuLanHybridForCausalLM
        print(f"{GREEN}[vLLM Plugin] Registered YuLanHybridForCausalLM with custom support{RESET}")
        ModelRegistry.register_model("YuLanHybridForCausalLM", YuLanHybridForCausalLM)
        
        # Register YuLanHybridMTP
        from .versions.v0_10_2_mtp import YuLanHybridMTP
        print(f"{GREEN}[vLLM Plugin] Registered YuLanHybridMTP (v0.10.2) with custom support{RESET}")
        ModelRegistry.register_model("YuLanHybridMTP", YuLanHybridMTP)
    elif vllm_version.startswith("0.11.0"):
        from .versions.v0_11_0 import YuLanHybridForCausalLM
        print(f"{GREEN}[vLLM Plugin] Loaded implementation for vLLM {vllm_version}{RESET}")
        
        # Register YuLanHybridForCausalLM
        print(f"{GREEN}[vLLM Plugin] Registered YuLanHybridForCausalLM with custom support{RESET}")
        ModelRegistry.register_model("YuLanHybridForCausalLM", YuLanHybridForCausalLM)
        
        # Register YuLanHybridMTP
        from .versions.v0_11_0_mtp import YuLanHybridMTP
        print(f"{GREEN}[vLLM Plugin] Registered YuLanHybridMTP (v0.11.0) with custom support{RESET}")
        ModelRegistry.register_model("YuLanHybridMTP", YuLanHybridMTP)
    else:
        raise ImportError(f"Unsupported vLLM version: {vllm_version}. Supported versions: 0.10.2, 0.11.0")

