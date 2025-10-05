import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

from transformers.modeling_attn_mask_utils import AttentionMaskConverter


def _prepare_4d_attention_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Creates a non-causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
    `(batch_size, key_value_length)`

    Args:
        mask (`torch.Tensor`):
            A 2D attention mask of shape `(batch_size, key_value_length)`
        dtype (`torch.dtype`):
            The torch dtype the created mask shall have.
        tgt_len (`int`):
            The target length or query length the created mask shall have.
    """
    return AttentionMaskConverter._expand_mask(mask=mask, dtype=dtype, tgt_len=tgt_len)


class SiglipvisionConfig:

    def __init__(
            self,
            hidden_size=768,  # dimention of the mdoel
            intermediate_size=3072,  # linear layer
            num_hidden_layer=12,  # no_hidden_layers
            num_attention_heads=12,
            num_channels=3,
            num_patches=256,
            image_size=224,
            patch_size=16,
            layer_norm_eps=1e-6,
            attention_dropout=0.0,
            num_image_tokens: int = None,
            version="siglip2",
            **kwargs
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layer
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.num_patches = num_patches
        self.patch_size = patch_size
        self.image_size = image_size
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.num_image_tokens = num_image_tokens
        self.version = version


class SiglipVisionEmbedding(nn.Module):
    def __init__(self, config: SiglipvisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid",  # This indicates no padding is added
        )

        if isinstance(self.image_size, (Tuple, list)):
            self.num_patches = (self.image_size[0] // self.patch_size) * (self.image_size[1] // self.patch_size)
        else:
            self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer(
            "position_ids",
            torch.arange(self.num_positions).expand((1, -1)),
            persistent=False,
        )

    def interpolate_pos_encoding(self, embeddings: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """
        This method allows to interpolate the pre-trained position encodings, to be able to use the model on higher resolution
        images. This method is also adapted to support torch.jit tracing and no class embeddings.

        Adapted from:
        - https://github.com/facebookresearch/dino/blob/de9ee3df6cf39fac952ab558447af1fa1365362a/vision_transformer.py#L174-L194, and
        - https://github.com/facebookresearch/dinov2/blob/e1277af2ba9496fbadf7aec6eba56e8d882d1e35/dinov2/models/vision_transformer.py#L179-L211
        """

        num_patches = embeddings.shape[1]
        num_positions = self.position_embedding.weight.shape[0]

        # always interpolate when tracing to ensure the exported model works for dynamic input shapes
        if not torch.jit.is_tracing() and num_patches == num_positions and height == width:
            return self.position_embedding(self.position_ids)

        patch_pos_embed = self.position_embedding.weight.unsqueeze(0)

        dim = embeddings.shape[-1]

        new_height = height // self.patch_size
        new_width = width // self.patch_size

        sqrt_num_positions = int(num_positions ** 0.5)
        patch_pos_embed = patch_pos_embed.reshape(1, sqrt_num_positions, sqrt_num_positions, dim)
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)

        patch_pos_embed = F.interpolate(
            patch_pos_embed,
            size=(new_height, new_width),
            mode="bicubic",
            align_corners=False,
        )

        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return patch_pos_embed

    def forward(self, pixel_values: torch.FloatTensor, interpolate_pos_encoding=False) -> torch.Tensor:
        _, _, height, width = pixel_values.shape  # [Batch_Size, Channels, Height, Width]
        # Convolve the 'patch_size' kernal over the image, with no overlapping patches since the stide is equal
        # The output of the convolution will have shape [Batch_Size, Embed_Dim, Num_Patches_H, Num_Patches_W]
        # where Num_Patches_H = height // patch_size and Num_Patches_W = width //  patch_size
        patch_embeds = self.patch_embedding(pixel_values)
        # [Batch_Size, Embed_Dim, Num_Patches_H, Num_Patches_W] -> [Batch_Size, Embed_dim, Num_Patches]
        # where Num_Patches = Num_Patches_H * Num_Patches_W
        embeddings = patch_embeds.flatten(2)
        # [Batch_Size, Embed_Dim, Num_Patches] -> [Batch_Size, Num_Patches, Embed_Dim]
        embeddings = embeddings.transpose(1, 2)
        # Add positional encodings to each patch. Each Positional encoding is a vector of size
        if interpolate_pos_encoding:
            embeddings = embeddings + self.interpolate_pos_encoding(embeddings, height, width)
        else:
            embeddings = embeddings + self.position_embedding(self.position_ids)
        # [Batch_Size, Num_Patches, Embed_Dim]
        return embeddings


class Siglip2VisionEmbeddings(nn.Module):
    def __init__(self, config: SiglipvisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.patch_size = config.patch_size

        self.patch_embedding = nn.Linear(
            in_features=config.num_channels * self.patch_size * self.patch_size,
            out_features=self.embed_dim,
        )

        self.num_patches = config.num_patches
        self.position_embedding_size = int(self.num_patches**0.5)
        self.position_embedding = nn.Embedding(self.num_patches, self.embed_dim)

    @staticmethod
    def resize_positional_embeddings(
        positional_embeddings: torch.Tensor,
        spatial_shapes: torch.LongTensor,
        max_length: int,
    ) -> torch.Tensor:
        """
        Resize positional embeddings to image-specific size and pad to a fixed size.

        Args:
            positional_embeddings (`torch.Tensor`):
                Position embeddings of shape (height, width, embed_dim)
            spatial_shapes (`torch.LongTensor`):
                Spatial shapes of shape (batch_size, 2) to resize the positional embeddings to
            max_length (`int`):
                Maximum length of the positional embeddings to pad resized positional embeddings to

        Returns:
            `torch.Tensor`: Embeddings of shape (batch_size, max_length, embed_dim)
        """
        batch_size = spatial_shapes.shape[0]
        embed_dim = positional_embeddings.shape[-1]
        source_dtype = positional_embeddings.dtype

        resulted_positional_embeddings = torch.empty(
            (batch_size, max_length, embed_dim),
            device=positional_embeddings.device,
            dtype=source_dtype,
        )

        # (height, width, embed_dim) -> (1, embed_dim, height, width) for interpolation
        positional_embeddings = positional_embeddings.permute(2, 0, 1).unsqueeze(0)

        # Upcast to float32 on CPU because antialias is not supported for bfloat16/float16 on CPU
        if positional_embeddings.device.type == "cpu":
            positional_embeddings = positional_embeddings.to(torch.float32)

        for i in range(batch_size):
            # (1, dim, height, width) -> (1, dim, target_height, target_width)
            height, width = spatial_shapes[i]
            resized_embeddings = F.interpolate(
                positional_embeddings,
                size=(height, width),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )

            # (1, dim, target_height, target_width) -> (target_height * target_width, dim)
            resized_embeddings = resized_embeddings.reshape(embed_dim, height * width).transpose(0, 1)

            # Cast to original dtype
            resized_embeddings = resized_embeddings.to(source_dtype)

            resulted_positional_embeddings[i, : height * width] = resized_embeddings
            resulted_positional_embeddings[i, height * width :] = resized_embeddings[0]

        return resulted_positional_embeddings

    def forward(self, pixel_values: torch.FloatTensor, spatial_shapes: torch.LongTensor) -> torch.Tensor:
        """
        Args:
            pixel_values (`torch.FloatTensor`):
                Pixel values of shape (batch_size, max_num_patches, num_channels * patch_size * patch_size)
            spatial_shapes (`List[Tuple[int, int]]`):
                Spatial shapes of shape (batch_size, 2) to resize the positional embeddings to
        """

        # Apply patch embeddings to already patchified pixel values
        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype))

        # Get positional resized and padded positional embeddings
        positional_embeddings = self.position_embedding.weight.reshape(
            self.position_embedding_size, self.position_embedding_size, -1
        )
        resized_positional_embeddings = self.resize_positional_embeddings(
            positional_embeddings, spatial_shapes, max_length=pixel_values.shape[1]
        )

        # Add positional embeddings to patch embeddings
        embeddings = patch_embeds + resized_positional_embeddings
        return embeddings


class SiglipAttention(nn.Module):
    """Multi-head attention from 'Attention is All You Need' paper """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim ** -0.5  # Equivalent to 1 / sqrt(self.head_dim)
        self.dropout = config.attention_dropout

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # hidden_states: [Batch_Size, Num_Patches, Embed_Dim]
        batch_size, seq_len, _ = hidden_states.size()
        # query-states: [Batch_Size, Num_Patches, Embed_Dim]
        query_states = self.q_proj(hidden_states)
        # key_states: [Batch_Size, Num_Patches, Embed_Dim]
        key_states = self.k_proj(hidden_states)
        # value_states: [Batch_Size, Num_Patches, Embed_Dim]
        value_states = self.v_proj(hidden_states)
        # query_states: [Batch_Size, Num_Patches, Num_Heads, Head_Dim]
        # in transpose [Batch_Size, Num_Patches, Num_Heads, Head_Dim] -> [Batch_Size, Num_Heads, Num_Patches, Head_Dim]
        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # Calculate the attention using the formula Q * K^T / sqrt(d_k) . attn_weights : [Batch_Size, Num_Heads, Num_Patches, Num_Patches]
        attention_weights = (query_states @ key_states.transpose(2, 3) * self.scale)

        if attention_weights.size() != (batch_size, self.num_heads, seq_len, seq_len):
            raise ValueError(
                f"Attention weights should be of size {(batch_size, self.num_heads, seq_len, seq_len)}, but is"
                f"{attention_weights.size()}"
            )

        if attention_mask is not None:
            attention_weights = attention_weights + attention_mask

        # Apply the softmax row-wise atten_weights: [Batch_Size, Num_Heads, Num_Patches, Num_Patches]
        attention_weights = F.softmax(attention_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        # Apply dropout only during training
        attention_weights = F.dropout(attention_weights, p=self.dropout, training=self.training)
        # Multiply the attention weights by the value_states. atten_output: [Batch_Size, Num_Heads, Num_Patches, Head_Dim]
        attention_output = (attention_weights @ value_states)

        if attention_weights.size() != (batch_size, self.num_heads, seq_len, seq_len):
            raise ValueError(
                f"Attention weights should be of size {(batch_size, self.num_heads, seq_len, seq_len)}, but is"
                f" {attention_weights.size()}"
            )

        # [Batch_Size, Num_Heads, Num_Patches, Head_Dim] -> [Batch_Size, Num_Patches, Num_Heads, Head_Dim]
        attention_output = attention_output.transpose(1, 2).contiguous()
        # [Batch_Size, Num_Patches, Num_Heads, Head_Dim] -> [Batch_Size, Num_Patches, Embed_Dim]
        attention_output = attention_output.reshape(batch_size, seq_len, self.embed_dim)
        # [Batch_Size, Num_Patches, Embed_Dim]
        attention_output = self.out_proj(attention_output)
        return attention_output, attention_weights


class Siglip2MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Intermediate_Size]
        hidden_states = self.fc1(hidden_states)
        # hidden_states : [Batch_Size, Num_Patches, Intermediate_Size]
        hidden_states = F.gelu(hidden_states, approximate="tanh")
        # [Batch_Size, Num_Patches, Intermediate_Size] -> [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class SiglipEncoderLayer(nn.Module):
    def __init__(self, config: SiglipvisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = SiglipAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = Siglip2MLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        # residual : [Batch_Size, Num_Patches, Embed_Dim]
        residual = hidden_states
        # [Batch_Size, Num_patches, Embed_Dim] -> [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = self.layer_norm1(hidden_states)
        # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states, _ = self.self_attn(hidden_states=hidden_states, attention_mask=attention_mask)
        # [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = residual + hidden_states
        # residual : [Batch_Size, Num_Patches, Embed_Dim]
        residual = hidden_states
        # [Batch_Size, Number_Patchesm Embed_Dim] -> [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = self.layer_norm2(hidden_states)
        # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = self.mlp(hidden_states)
        # [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = residual + hidden_states

        return hidden_states


class Siglip2Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [SiglipEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )

    def forward(
            self,
            inputs_embeds: torch.Tensor,
            attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        # inputs_embeds: [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = inputs_embeds

        for encoder_layer in self.layers:
            # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Embed_Dim]
            hidden_states = encoder_layer(hidden_states, attention_mask)

        return hidden_states


class SiglipMultiheadAttentionPoolingHead(nn.Module):
    """Multihead Attention Pooling."""

    def __init__(self, config: SiglipvisionConfig):
        super().__init__()

        self.probe = nn.Parameter(torch.randn(1, 1, config.hidden_size))
        self.attention = torch.nn.MultiheadAttention(config.hidden_size, config.num_attention_heads, batch_first=True)
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = Siglip2MLP(config)

    def forward(self, hidden_state):
        batch_size = hidden_state.shape[0]
        probe = self.probe.repeat(batch_size, 1, 1)

        hidden_state = self.attention(probe, hidden_state, hidden_state)[0]

        residual = hidden_state
        hidden_state = self.layernorm(hidden_state)
        hidden_state = residual + self.mlp(hidden_state)

        return hidden_state[:, 0]


class SiglipVisionTransformer(nn.Module):
    def __init__(self, config: SiglipvisionConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = SiglipVisionEmbedding(config) if config.version == "siglip" else Siglip2VisionEmbeddings(config)

        self.encoder = Siglip2Encoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        self.head = SiglipMultiheadAttentionPoolingHead(config)

    @staticmethod
    def convert_image_to_patches(image: torch.FloatTensor, patch_size: int) -> torch.FloatTensor:
        """
        Convert 4D array image of shape (bs, num_channel, image_height, image_width) into 2D array of patches of shape
        (bs, num_patches_height * num_patches_width, patch_size * patch_size * num_channels).
        """
        batch_size, num_channels, image_height, image_width = image.size()
        num_patches_height = image_height // patch_size
        num_patches_width = image_width // patch_size
        patched_image = image.reshape(batch_size, num_channels, num_patches_height, patch_size, num_patches_width, patch_size)
        patched_image = patched_image.permute(0, 2, 4, 3, 5, 1)
        patched_image = patched_image.reshape(batch_size, num_patches_height * num_patches_width, -1)
        return patched_image

    @staticmethod
    def pad_along_first_dim(image: torch.FloatTensor, target_length: int, pad_value: int = 0) -> Tuple[torch.FloatTensor, torch.IntTensor]:
        """
        Pad the array along the first dimension.
        """
        bs, n_patch, _ = image.size()
        current_length = n_patch
        padding_length = target_length - current_length
        mask = torch.ones((bs, target_length), dtype=torch.int32)
        if padding_length > 0:
            paddings = (0, 0, 0, padding_length)
            image = F.pad(image, paddings, mode="constant", value=pad_value)
            mask[:, -padding_length:] = 0
        return image, mask

    def forward(self, pixel_values: torch.Tensor, interpolate_pos_encoding: bool = False, spatial_shapes: torch.LongTensor = None) -> torch.Tensor:
        # pixel_values: [Batch_Size, Channels, Height, Width] -> [Batch_Size, Num_Patches, Embedding_Dimension]
        attention_mask = None
        if self.config.version == "siglip":
            hidden_states = self.embeddings(pixel_values, interpolate_pos_encoding)
        else:
            pixel_values, attention_mask = self.pad_along_first_dim(self.convert_image_to_patches(pixel_values, self.config.patch_size), self.config.num_patches)
            hidden_states = self.embeddings(pixel_values, spatial_shapes)
            attention_mask = _prepare_4d_attention_mask(attention_mask, hidden_states.dtype)

        last_hidden_state = self.encoder(inputs_embeds=hidden_states, attention_mask=attention_mask)

        last_hidden_state = self.post_layernorm(last_hidden_state)

        pooler_output = self.head(last_hidden_state)

        return pooler_output


class SiglipVisionModel(nn.Module):

    def __init__(self, config: SiglipvisionConfig):
        super().__init__()
        self.config = config
        self.vision_model = SiglipVisionTransformer(config)

    def forward(self, pixel_values: torch.Tensor, interpolate_pos_encoding: bool) -> Tuple:
        # [Batch_Size, Channels, Height, Width] -> [Batch_size, Num_Patches, Embed_Dim]
        bs, _, h, w = pixel_values.shape
        spatial_shapes = torch.tensor([h // self.config.patch_size, w // self.config.patch_size]).repeat(bs, 1)
        return self.vision_model(pixel_values=pixel_values, interpolate_pos_encoding=interpolate_pos_encoding,
                                 spatial_shapes=spatial_shapes)


class SiglipTextConfig:
    r"""
    This is the configuration class to store the configuration of a [`SiglipTextModel`]. It is used to instantiate a
    Siglip text encoder according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the text encoder of the Siglip
    [google/siglip-base-patch16-224](https://huggingface.co/google/siglip-base-patch16-224) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 256000):
            Vocabulary size of the Siglip text model. Defines the number of different tokens that can be represented by
            the `inputs_ids` passed when calling [`SiglipModel`].
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        max_position_embeddings (`int`, *optional*, defaults to 64):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu_pytorch_tanh"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` `"quick_gelu"` are supported.
        layer_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the layer normalization layers.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        pad_token_id (`int`, *optional*, defaults to 1):
            The id of the padding token in the vocabulary.
        bos_token_id (`int`, *optional*, defaults to 49406):
            The id of the beginning-of-sequence token in the vocabulary.
        eos_token_id (`int`, *optional*, defaults to 49407):
            The id of the end-of-sequence token in the vocabulary.
        projection_size (`int`, *optional*, defaults to `hidden_size`):
            The size of the projection head.
    ```"""

    def __init__(
            self,
            vocab_size=256000,
            hidden_size=768,
            intermediate_size=3072,
            num_hidden_layers=12,
            num_attention_heads=12,
            max_position_embeddings=64,
            hidden_act="gelu_pytorch_tanh",
            layer_norm_eps=1e-6,
            attention_dropout=0.0,
            # This differs from `CLIPTokenizer`'s default and from openai/siglip
            # See https://github.com/huggingface/transformers/pull/24773#issuecomment-1632287538
            pad_token_id=1,
            bos_token_id=49406,
            eos_token_id=49407,
            projection_size=None,
            **kwargs,
    ):
        super().__init__(**kwargs)

        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.layer_norm_eps = layer_norm_eps
        self.hidden_act = hidden_act
        self.attention_dropout = attention_dropout
        self.projection_size = projection_size if projection_size is not None else hidden_size
    
    def get(self, key, default=None):
        return getattr(self, key, default)


class SiglipTextEmbeddings(nn.Module):
    def __init__(self, config: SiglipTextConfig):
        super().__init__()
        embed_dim = config.hidden_size

        self.token_embedding = nn.Embedding(config.vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(config.max_position_embeddings, embed_dim)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )

    def forward(
            self,
            input_ids,
    ) -> torch.Tensor:
        seq_length = input_ids.shape[-1]
        max_position_embedding = self.position_embedding.weight.shape[0]

        if seq_length > max_position_embedding:
            raise ValueError(
                f"Sequence length must be less than max_position_embeddings (got `sequence length`: "
                f"{seq_length} and max_position_embeddings: {max_position_embedding}"
            )

        position_ids = self.position_ids[:, :seq_length]
        inputs_embeds = self.token_embedding(input_ids)

        position_embeddings = self.position_embedding(position_ids)
        embeddings = inputs_embeds + position_embeddings

        return embeddings


class Siglip2TextTransformer(nn.Module):
    def __init__(self, config: SiglipTextConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size
        self.embeddings = SiglipTextEmbeddings(config)
        self.encoder = Siglip2Encoder(config)
        self.final_layer_norm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

        self.head = nn.Linear(embed_dim, config.projection_size)
        self._use_flash_attention_2 = False

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            use_pooled_output: bool = True,
    ):
        r"""
        Returns:

        """

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            # Default path: get embeddings from token IDs
            hidden_states = self.embeddings(input_ids=input_ids)
        elif inputs_embeds is not None:
            # Prompt learning path: use the provided embeddings directly
            hidden_states = inputs_embeds
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        # note: SigLIP's text model does not use a causal mask, unlike the original CLIP model.
        # expand attention_mask
        if attention_mask is not None and not self._use_flash_attention_2:
            # [batch_size, seq_len] -> [batch_size, 1, tgt_seq_len, src_seq_len]
            attention_mask = _prepare_4d_attention_mask(attention_mask, hidden_states.dtype)

        last_hidden_state = self.encoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
        )

        last_hidden_state = self.final_layer_norm(last_hidden_state)

        # Assuming "sticky" EOS tokenization, last token is always EOS.
        pooled_output = last_hidden_state[:, -1, :]
        pooled_output = self.head(pooled_output)

        return pooled_output if use_pooled_output else last_hidden_state


class SiglipTextModel(nn.Module):
    def __init__(self, config: SiglipTextConfig):
        super().__init__()
        self.text_model = Siglip2TextTransformer(config)

    def get_input_embeddings(self) -> nn.Module:
        return self.text_model.embeddings.token_embedding

    def set_input_embeddings(self, value):
        self.text_model.embeddings.token_embedding = value

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            use_pooled_output: bool = True,
    ):
        r"""
        Returns:

        ```"""

        return self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            use_pooled_output=use_pooled_output
        )


class SiglipConfig(nn.Module):
    r"""
    [`SiglipConfig`] is the configuration class to store the configuration of a [`SiglipModel`]. It is used to
    instantiate a Siglip model according to the specified arguments, defining the text model and vision model configs.
    Instantiating a configuration with the defaults will yield a similar configuration to that of the Siglip
    [google/siglip-base-patch16-224](https://huggingface.co/google/siglip-base-patch16-224) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        text_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`SiglipTextConfig`].
        vision_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`SiglipVisionConfig`].
        kwargs (*optional*):
            Dictionary of keyword arguments.
    ```"""

    model_type = "siglip2"
    sub_configs = {"text_config": SiglipTextConfig, "vision_config": SiglipvisionConfig}

    def __init__(self, text_config=None, vision_config=None, **kwargs):
        super().__init__()

        self.text_config = SiglipTextConfig(**text_config)
        self.vision_config = SiglipvisionConfig(**vision_config)

        self.initializer_factor = 1.0

    @classmethod
    def from_text_vision_configs(cls, text_config: SiglipTextConfig, vision_config: SiglipvisionConfig, **kwargs):
        r"""
        Instantiate a [`SiglipConfig`] (or a derived class) from siglip text model configuration and siglip vision
        model configuration.

        Returns:
            [`SiglipConfig`]: An instance of a configuration object
        """

        return cls(text_config=text_config, vision_config=vision_config, **kwargs)
    
    def get(self, key, default=None):
        return getattr(self, key, default)


class SiglipModel(nn.Module):

    def __init__(self, config: SiglipConfig):
        super().__init__()

        text_config = config.text_config
        vision_config = config.vision_config

        # First, initialize the text and vision models with proper attention implementation
        text_model = SiglipTextModel(text_config)
        vision_model = SiglipVisionModel(vision_config)

        # Second, get the text and vision submodules (for backward compatibility)
        self.text_model = text_model.text_model
        self.vision_model = vision_model.vision_model

        self.logit_scale = nn.Parameter(torch.randn(1))
        self.logit_bias = nn.Parameter(torch.randn(1))

        self.config = config

    def get_text_features(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:
        r"""
        Returns:
            text_features (`torch.FloatTensor` of shape `(batch_size, output_dim`): The text embeddings obtained by
            applying the projection layer to the pooled output of [`SiglipTextModel`].

        ```"""
        # Use SigLIP model's config for some fields (if specified) instead of those of vision & text components.

        pooled_output = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        return pooled_output

    def get_image_features(
            self,
            pixel_values: Optional[torch.FloatTensor] = None,
            interpolate_pos_encoding: bool = False,
    ) -> torch.FloatTensor:
        r"""
        Returns:
            image_features (`torch.FloatTensor` of shape `(batch_size, output_dim`): The image embeddings obtained by
            applying the projection layer to the pooled output of [`SiglipVisionModel`].

        ```"""
        # Use SiglipModel's config for some fields (if specified) instead of those of vision & text components.

        pooled_output = self.vision_model(
            pixel_values=pixel_values,
            interpolate_pos_encoding=interpolate_pos_encoding,
        )

        return pooled_output

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            pixel_values: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            interpolate_pos_encoding: bool = False,
    ):
        # Use SigLIP model's config for some fields (if specified) instead of those of vision & text components.

        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            interpolate_pos_encoding=interpolate_pos_encoding,
        )

        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        image_embeds = vision_outputs[1]
        text_embeds = text_outputs[1]

        # normalized features
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

        # cosine similarity as logits
        logits_per_text = torch.matmul(text_embeds, image_embeds.t().to(text_embeds.device))

        logit_scale, logit_bias = self.logit_scale.to(text_embeds.device), self.logit_bias.to(text_embeds.device)
        logits_per_text = logits_per_text * logit_scale.exp() + logit_bias

        logits_per_image = logits_per_text.t()

        output = (logits_per_image, logits_per_text, text_embeds, image_embeds, text_outputs, vision_outputs)
        return output
