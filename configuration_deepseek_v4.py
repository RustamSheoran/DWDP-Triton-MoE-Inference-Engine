"""DeepSeek-V4 configuration."""
from transformers.configuration_utils import PretrainedConfig


class DeepseekV4Config(PretrainedConfig):
    model_type = "deepseek_v4"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=129280,
        hidden_size=512,
        num_hidden_layers=12,
        num_attention_heads=8,
        num_key_value_heads=1,
        head_dim=64,
        qk_rope_head_dim=32,
        q_lora_rank=256,
        o_lora_rank=256,
        o_groups=2,
        # MoE
        moe_intermediate_size=512,
        n_routed_experts=16,
        n_shared_experts=1,
        num_experts_per_tok=2,
        num_hash_layers=2,
        norm_topk_prob=True,
        scoring_func="sqrtsoftplus",
        topk_method="noaux_tc",
        routed_scaling_factor=1.5,
        # CSA / indexer
        index_n_heads=4,
        index_head_dim=32,
        index_topk=64,
        # Per-layer CSA/HCA dispatch:
        #   0   = pure sliding-window (no compression)
        #   m>0 small (e.g. 4)  = CSA, compression rate m
        #   m>0 large (e.g. 32+)= HCA, compression rate m'
        compress_ratios=None,
        sliding_window=32,
        # mHC
        hc_mult=4,                # n_hc expansion factor
        hc_eps=1e-6,
        hc_sinkhorn_iters=20,
        # SwiGLU
        swiglu_limit=10.0,
        hidden_act="silu",
        # MTP
        num_nextn_predict_layers=1,
        # Norm / init
        rms_norm_eps=1e-6,
        initializer_range=0.02,
        # RoPE
        rope_theta=10000.0,
        compress_rope_theta=160000.0,
        rope_scaling=None,
        max_position_embeddings=1048576,
        # Misc
        attention_bias=False,
        attention_dropout=0.0,
        tie_word_embeddings=False,
        use_cache=True,
        bos_token_id=0,
        eos_token_id=1,
        pad_token_id=None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.q_lora_rank = q_lora_rank
        self.o_lora_rank = o_lora_rank
        self.o_groups = o_groups

        self.moe_intermediate_size = moe_intermediate_size
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.num_hash_layers = num_hash_layers
        self.norm_topk_prob = norm_topk_prob
        self.scoring_func = scoring_func
        self.topk_method = topk_method
        self.routed_scaling_factor = routed_scaling_factor

        self.index_n_heads = index_n_heads
        self.index_head_dim = index_head_dim
        self.index_topk = index_topk

        if compress_ratios is None:
            # Default pattern (V4-style): first 2 layers pure sliding-window,
            # alternating CSA(4) / HCA(big) for the middle, last layer
            # sliding-window. The "big" HCA ratio scales with num_hidden_layers
            # so smaller models still get sensible block sizes.
            big = max(8, min(128, num_hidden_layers * 4))
            compress_ratios = [0, 0]
            for i in range(num_hidden_layers - 3):
                compress_ratios.append(4 if i % 2 == 0 else big)
            compress_ratios.append(0)
            compress_ratios = compress_ratios[:num_hidden_layers]
            # Pad with 0 (sliding-window) if the recipe came out too short
            while len(compress_ratios) < num_hidden_layers:
                compress_ratios.append(0)
        assert len(compress_ratios) == num_hidden_layers, (
            f"compress_ratios length ({len(compress_ratios)}) must equal "
            f"num_hidden_layers ({num_hidden_layers})"
        )
        self.compress_ratios = compress_ratios
        self.sliding_window = sliding_window

        self.hc_mult = hc_mult
        self.hc_eps = hc_eps
        self.hc_sinkhorn_iters = hc_sinkhorn_iters

        self.swiglu_limit = swiglu_limit
        self.hidden_act = hidden_act

        self.num_nextn_predict_layers = num_nextn_predict_layers

        self.rms_norm_eps = rms_norm_eps
        self.initializer_range = initializer_range

        self.rope_theta = rope_theta
        self.compress_rope_theta = compress_rope_theta
        if rope_scaling is None:
            # YaRN scaling identical to real V4 Pro/Flash
            rope_scaling = {
                "beta_fast": 32.0,
                "beta_slow": 1.0,
                "factor": 16.0,
                "original_max_position_embeddings": 65536,
                "type": "yarn",
            }
        self.rope_scaling = rope_scaling
        self.max_position_embeddings = max_position_embeddings

        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.use_cache = use_cache

        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
