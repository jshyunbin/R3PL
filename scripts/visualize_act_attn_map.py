"""
visualize_act_attention.py
──────────────────────────
Rolls out a pretrained ACT model in a Robomimic environment and produces a
side-by-side video with three panels:

  [agentview rollout] | [agentview attention heatmap] | [eye-in-hand attention heatmap]

Attention is extracted from the FIRST cross-attention layer of the
TransformerDecoder (decoder_dec).  For every forward pass we hook the
cross-attention module, recompute the softmax attention weights from the
projected Q and K tensors (since scaled_dot_product_attention does not expose
them), and then average the weights over all action-query tokens and all heads
to get a single spatial importance map per camera.

Usage
-----
uv run python scripts/visualize_act_attn_map.py \
    --checkpoint  path/to/epoch_xxx.ckpt \
    --dataset     data/robomimic/can/ph/image_abs.hdf5 \
    --output      attention_rollout.mp4 \
    --seed        42 \
    --max_steps   400 \
    --fps         10 \
    --abs_action              # pass if the model was trained with absolute actions
"""

import argparse
import collections
import math
import os
from pathlib import Path

import cv2
import einops
import numpy as np
import src.robothink.envs
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils
import torch
import torch.nn.functional as F
from einops import rearrange

from src.nn.common.pytorch_util import dict_apply
from src.nn.common.rotation_transformer import RotationTransformer
from src.nn.env.robomimic_image_wrapper import RobomimicImageWrapper
from src.nn.gym_util.multistep_wrapper import MultiStepWrapper
from src.nn.models.act import ACTModule

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def create_env(env_meta, shape_meta, enable_render=True):
    modality_mapping = collections.defaultdict(list)
    for key, attr in shape_meta["obs"].items():
        modality_mapping[attr.get("type", "low_dim")].append(key)
    ObsUtils.initialize_obs_modality_mapping_from_dict(modality_mapping)
    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False,
        render_offscreen=enable_render,
        use_image_obs=enable_render,
    )
    return env


def undo_transform_action(action: np.ndarray, rotation_transformer) -> np.ndarray:
    """Convert rotation_6d actions back to axis-angle for the env."""
    raw_shape = action.shape
    if raw_shape[-1] == 20:
        action = action.reshape(-1, 2, 10)
    d_rot = action.shape[-1] - 4
    pos     = action[..., :3]
    rot     = action[..., 3: 3 + d_rot]
    gripper = action[..., [-1]]
    rot     = rotation_transformer.inverse(rot)
    uaction = np.concatenate([pos, rot, gripper], axis=-1)
    if raw_shape[-1] == 20:
        uaction = uaction.reshape(*raw_shape[:-1], 14)
    return uaction


def apply_colormap(heatmap: np.ndarray) -> np.ndarray:
    """heatmap in [0,1], float → uint8 RGB using JET colormap."""
    heatmap_u8 = (np.clip(heatmap, 0, 1) * 255).astype(np.uint8)
    colored = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)   # BGR
    return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)


def overlay_heatmap(image_rgb: np.ndarray, heatmap: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    Overlay a [0,1] heatmap on top of an HxWx3 uint8 image.
    Both image and heatmap must already be the same spatial size.
    """
    colored = apply_colormap(heatmap)
    return (image_rgb.astype(np.float32) * (1 - alpha) + colored.astype(np.float32) * alpha).astype(np.uint8)



# ──────────────────────────────────────────────────────────────────────────────
# Attention-map extraction via forward hooks
# ──────────────────────────────────────────────────────────────────────────────

class CrossAttentionHook:
    """
    Attaches a forward hook to the cross_attention module inside the FIRST
    DecoderTransformerBlock of a TransformerDecoder.

    The hook intercepts the projected Q and K tensors (after view but before
    scaled_dot_product_attention) and manually computes softmax attention
    weights, which are stored in `self.weights`.

    Shape of `self.weights` after a forward pass:
        (batch, n_heads, n_action_queries, n_source_tokens)
    """

    def __init__(self, cross_attn_module):
        self.weights: torch.Tensor | None = None
        self._q: torch.Tensor | None = None
        self._k: torch.Tensor | None = None
        self._handles = []

        # Hook q_proj and k_proj outputs so we can reconstruct attention weights
        self._handles.append(
            cross_attn_module.q_proj.register_forward_hook(self._save_q)
        )
        self._handles.append(
            cross_attn_module.k_proj.register_forward_hook(self._save_k)
        )
        # After the full cross-attention forward we compute weights
        self._handles.append(
            cross_attn_module.register_forward_hook(self._compute_weights)
        )
        self._cross_attn = cross_attn_module

    def _save_q(self, module, input, output):
        self._q = output  # (B, n_queries, n_heads*head_dim)

    def _save_k(self, module, input, output):
        self._k = output  # (B, n_keys, n_heads*head_dim)

    def _compute_weights(self, module, input, output):
        if self._q is None or self._k is None:
            return
        n_heads  = module.num_heads
        head_dim = module.head_dim
        B        = self._q.shape[0]
        n_q      = self._q.shape[1]
        n_k      = self._k.shape[1]

        q = self._q.view(B, n_q, n_heads, head_dim)
        k = self._k.view(B, n_k, n_heads, head_dim)

        # (B, H, n_q, head_dim)  x  (B, H, head_dim, n_k)  →  (B, H, n_q, n_k)
        q = rearrange(q, "B S H D -> B H S D").float()
        k = rearrange(k, "B S H D -> B H S D").float()
        scale = math.sqrt(head_dim)
        attn_logits = torch.matmul(q, k.transpose(-1, -2)) / scale   # (B, H, n_q, n_k)
        self.weights = attn_logits.softmax(dim=-1).detach().cpu()

        self._q = None
        self._k = None

    def remove(self):
        for h in self._handles:
            h.remove()


def attach_cross_attention_hook(model: ACTModule) -> CrossAttentionHook:
    """
    Returns a hook attached to the cross_attention in the FIRST
    DecoderTransformerBlock of model.decoder_dec.
    """
    first_dec_block = model.decoder_dec.decoder[0]
    return CrossAttentionHook(first_dec_block.cross_attention)


# ──────────────────────────────────────────────────────────────────────────────
# Token-layout helper
# ──────────────────────────────────────────────────────────────────────────────

def compute_token_layout(model: ACTModule, image_h: int, image_w: int):
    """
    Returns a dict describing how source tokens in decoder_enc output are laid
    out, so we can slice the attention weights back to per-camera spatial maps.

    Layout (matching decode_action / predict_action):
        [ cam0_tokens (fmap_h*fmap_w), cam1_tokens (fmap_h*fmap_w), ..., joint_token, latent_token ]

    Derives the actual feature map size via a dummy forward pass so that non-
    power-of-2 input resolutions (e.g. 84×84 → 3×3, not 84//32=2) are handled
    correctly.
    """
    dummy = torch.zeros(
        1, 3, image_h, image_w,
        device=next(model.parameters()).device,
        dtype=next(model.parameters()).dtype,
    )
    with torch.no_grad():
        feat = model.vision_encoder(dummy)['0']  # (1, 512, fH, fW)
    fmap_h, fmap_w = feat.shape[2], feat.shape[3]
    tokens_per_cam = fmap_h * fmap_w

    layout = {}
    offset = 0
    for key in model.vision_key:
        layout[key] = (offset, offset + tokens_per_cam, fmap_h, fmap_w)
        offset += tokens_per_cam

    layout["_joint"]  = (offset, offset + 1, 1, 1);  offset += 1
    layout["_latent"] = (offset, offset + 1, 1, 1);  offset += 1
    return layout


def extract_cam_heatmap(
    attn_weights: torch.Tensor,   # (B, H, n_queries, n_source_tokens)
    token_start: int,
    token_end: int,
    fmap_h: int,
    fmap_w: int,
    target_h: int,
    target_w: int,
) -> np.ndarray:
    """
    Average attention over heads and action queries, slice out the camera
    tokens, reshape to (fmap_h, fmap_w), bilinearly upsample to (target_h, target_w).

    Returns float32 ndarray in [0, 1].
    """
    # Mean over heads and action queries → (B, n_source_tokens)
    cam_weights = attn_weights[0].mean(dim=0).mean(dim=0)           # (n_source_tokens,)
    cam_weights = cam_weights[token_start:token_end]                 # (tokens_per_cam,)
    heatmap     = cam_weights.reshape(fmap_h, fmap_w).numpy()       # (fmap_h, fmap_w)

    # Normalize to [0, 1]
    heatmap -= heatmap.min()
    if heatmap.max() > 1e-8:
        heatmap /= heatmap.max()

    # Upsample
    heatmap_t = torch.from_numpy(heatmap)[None, None]               # (1,1,fH,fW)
    heatmap_t = F.interpolate(heatmap_t, size=(target_h, target_w), mode="bilinear", align_corners=False)
    return heatmap_t[0, 0].numpy()                                  # (target_h, target_w)


# ──────────────────────────────────────────────────────────────────────────────
# Main rollout + visualization
# ──────────────────────────────────────────────────────────────────────────────

def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"Loading checkpoint: {args.checkpoint}")
    model: ACTModule = ACTModule.load_from_checkpoint(
        args.checkpoint, map_location=device, weights_only=False
    )
    model.eval().to(device)

    shape_meta = model.hparams.shape_meta

    # ── Attach attention hook ─────────────────────────────────────────────────
    hook = attach_cross_attention_hook(model)

    # Identify the two camera keys we want to visualize
    # Expecting 'agentview_image' and 'robot0_eye_in_hand_image'
    agentview_key   = "agentview_image"
    eye_in_hand_key = "robot0_eye_in_hand_image"

    # ── Token layout ──────────────────────────────────────────────────────────
    # Derive image resolution from shape_meta (not a guaranteed top-level key).
    image_h = shape_meta['obs'][agentview_key]['shape'][1]
    image_w = shape_meta['obs'][agentview_key]['shape'][2]
    token_layout = compute_token_layout(model, image_h, image_w)

    assert agentview_key   in token_layout, f"{agentview_key} not in model's vision_key list: {model.vision_key}"
    assert eye_in_hand_key in token_layout, f"{eye_in_hand_key} not in model's vision_key list: {model.vision_key}"

    # ── Build environment ─────────────────────────────────────────────────────
    dataset_path = os.path.expanduser(args.dataset)
    env_meta     = FileUtils.get_env_metadata_from_dataset(dataset_path)
    env_meta["env_kwargs"]["use_object_obs"] = False

    rotation_transformer = None
    if args.abs_action:
        env_meta["env_kwargs"]["controller_configs"]["control_delta"] = False
        rotation_transformer = RotationTransformer("axis_angle", "rotation_6d")

    robomimic_env = create_env(env_meta=env_meta, shape_meta=shape_meta)
    robomimic_env.env.hard_reset = False

    env = MultiStepWrapper(
        RobomimicImageWrapper(
            env=robomimic_env,
            shape_meta=shape_meta,
            init_state=None,
            render_obs_key=agentview_key,
        ),
        n_obs_steps=1,
        n_action_steps=1,
        max_episode_steps=args.max_steps,
    )

    # ── Rollout ───────────────────────────────────────────────────────────────
    obs, _ = env.reset(seed=args.seed)

    panel_h = shape_meta['obs'][agentview_key]['shape'][1]          # final height of each panel
    panel_w = shape_meta['obs'][agentview_key]['shape'][2]          # final width  of each panel
    out_w   = panel_w * 3
    out_h   = panel_h

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.fps,
        (out_w, out_h),
    )

    step = 0
    done = False
    pred_buf: collections.deque = collections.deque(maxlen=model.horizon)
    print(f"Starting rollout (max {args.max_steps} steps)…")

    while not done and step < args.max_steps:

        # ── Transfer obs to device ────────────────────────────────────────────
        obs_dict = dict_apply(dict(obs), lambda x: torch.from_numpy(x[None, ...]).to(device=device))

        with torch.no_grad():
            action_hat = model.predict_action({"obs": obs_dict})['pred_action']

        # ── Temporal aggregation (ACT paper §3.3) ─────────────────────────────
        # Append this step's full H-length prediction to the buffer, then
        # compute the exponentially-weighted average of overlapping predictions.
        pred_buf.append(action_hat.cpu().numpy()[0])   # (horizon, action_dim)
        buf_len = len(pred_buf)
        # buf[-1] = most recent (j=0), buf[-j-1] = j steps ago
        agg_weights = np.array([np.exp(-args.k_aggregation * j) for j in range(buf_len)])
        agg_actions = np.stack([pred_buf[-(j + 1)][j] for j in range(buf_len)], axis=0)  # (buf_len, action_dim)
        env_action_single = np.average(agg_actions, axis=0, weights=agg_weights)          # (action_dim,)

        # ── Extract attention weights from hook ───────────────────────────────
        attn_weights = hook.weights          # (1, n_heads, n_queries, n_source_tokens)

        # ── Build raw agentview render (uint8 HxWx3) ─────────────────────────
        # obs[agentview_key] has shape (n_obs_steps, C, H, W), values in [0,1]
        agent_img_chw = obs[agentview_key][-1]                    # (C, H, W)
        agent_img     = (np.moveaxis(agent_img_chw, 0, -1) * 255).astype(np.uint8)

        # ── eye-in-hand raw image ─────────────────────────────────────────────
        eih_img_chw = obs[eye_in_hand_key][-1]                    # (C, H, W)
        eih_img     = (np.moveaxis(eih_img_chw, 0, -1) * 255).astype(np.uint8)

        # ── Heatmaps ──────────────────────────────────────────────────────────
        if attn_weights is not None:
            av_start, av_end, av_fh, av_fw = token_layout[agentview_key]
            agent_heat = extract_cam_heatmap(
                attn_weights, av_start, av_end, av_fh, av_fw, panel_h, panel_w
            )
            agent_overlay = overlay_heatmap(agent_img, agent_heat, alpha=0.55)

            eih_start, eih_end, eih_fh, eih_fw = token_layout[eye_in_hand_key]
            eih_heat = extract_cam_heatmap(
                attn_weights, eih_start, eih_end, eih_fh, eih_fw, panel_h, panel_w
            )
            eih_overlay = overlay_heatmap(eih_img, eih_heat, alpha=0.55)
        else:
            # No attention yet (shouldn't happen, but be safe)
            agent_overlay = agent_img.copy()
            eih_overlay   = eih_img.copy()

        # ── Label panels ──────────────────────────────────────────────────────
        def put_label(img, text):
            return cv2.putText(
                img.copy(), text, (4, 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA
            )

        agent_img     = put_label(agent_img,     "agentview")
        agent_overlay = put_label(agent_overlay, "agentview attn")
        eih_overlay   = put_label(eih_overlay,   "eye-in-hand attn")

        # ── Compose frame ─────────────────────────────────────────────────────
        frame = np.concatenate([agent_img, agent_overlay, eih_overlay], axis=1)   # (H, 3W, 3)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(frame_bgr)

        # ── Step environment ──────────────────────────────────────────────────
        env_action = env_action_single[np.newaxis, ...]            # (1, action_dim)
        if args.abs_action and rotation_transformer is not None:
            env_action = undo_transform_action(env_action, rotation_transformer)

        obs, reward, terminated, truncated, info = env.step(env_action)
        done  = reward == 1.0
        step += 1

        if step % 20 == 0:
            print(f"  step {step:4d}  reward={reward:.3f}")

    writer.release()
    hook.remove()
    print(f"\nDone. Video saved to: {output_path}  ({step} steps, done={done})")


# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="ACT attention-map rollout visualizer")
    p.add_argument("--checkpoint",        required=True,          help="Path to .ckpt file")
    p.add_argument("--dataset",           required=True,          help="Path to robomimic .hdf5 dataset (for env_meta)")
    p.add_argument("--output",            default="attention_rollout.mp4")
    p.add_argument("--seed",              type=int, default=42)
    p.add_argument("--max_steps",         type=int, default=400)
    p.add_argument("--fps",               type=int, default=20)
    p.add_argument("--abs_action",        action="store_true",    help="Model was trained with absolute actions")
    p.add_argument("--k_aggregation",     type=float, default=0.01, help="Temporal aggregation decay constant (ACT paper)")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())