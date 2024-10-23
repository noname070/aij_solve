from typing import Optional
from transformers.configuration_utils import PretrainedConfig


class HGRNBitMultimodalConfig(PretrainedConfig):
    model_type = "hgrn_bit_multimodal"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 32000,
        hidden_size: int = 2048,
        num_hidden_layers: int = 24,
        attn_mode: str = "fused_recurrent",
        num_heads: Optional[int] = 1,
        expand_ratio: Optional[int] = 1,
        use_short_conv: bool = False,
        conv_size: int = 4,
        share_conv_kernel: bool = True,
        use_lower_bound: bool = True,
        hidden_ratio: Optional[int] = 4,
        intermediate_size: Optional[int] = None,
        hidden_act: str = "swish",
        rms_norm_eps: float = 1e-6,
        use_cache: bool = True,
        pad_token_id: int = None,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        initializer_range: float = 0.02,
        fuse_cross_entropy: bool = True,
        fusion_method: str = "concat",
        video_frame_size: int = 224,
        video_channels: int = 3,
        audio_feat_size: int = 128,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.attn_mode = attn_mode
        self.num_heads = num_heads
        self.expand_ratio = expand_ratio
        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.share_conv_kernel = share_conv_kernel
        self.use_lower_bound = use_lower_bound
        self.hidden_ratio = hidden_ratio
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.initializer_range = initializer_range
        self.fuse_cross_entropy = fuse_cross_entropy
        self.fusion_method = fusion_method
        self.video_frame_size = video_frame_size
        self.video_channels = video_channels
        self.audio_feat_size = audio_feat_size
