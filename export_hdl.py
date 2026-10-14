#!/usr/bin/env python3
"""
export_model.py

Load a Stable-Baselines3 .zip file and print the `student` sub-module
of the actor network:
    python export_model.py --path=/path/to/your/model.zip
"""

import argparse
from thermometer import *
import math
import torch
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from typing import Tuple, List, Optional
import numpy as np
from wnn_models import WNN
import re
from pathlib import Path
import struct
from pathlib import Path
import numpy as np
import torch


def float32_to_hex(f: float) -> str:
    """Return IEEE-754 float32 bit pattern as Verilog hex literal string body."""
    return struct.pack(">f", np.float32(f)).hex().upper()


def export_fp32_inputs(obs: torch.Tensor, out_path: str) -> None:
    """
    Save one observation vector as decimal float32 values, one line.
    """
    obs_np = obs.detach().cpu().to(torch.float32).numpy().reshape(-1)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(out_path, obs_np[None, :], fmt="%.8f")
    print(f"Wrote FP32 inputs to {out_path}")


def export_fp32_pkg(
    scales: torch.Tensor,
    out_path: str,
    package_name: str = "fp32_pkg",
    mul_latency: int = 9,
    to_int_latency: int = 7,
    doc: str = "Auto-generated FP32 quantization scales.",
) -> None:
    """
    Export per-observation FP32 scales to a SystemVerilog package.

    scales: shape [obs_dim]
    """
    scales = scales.detach().cpu().to(torch.float32).reshape(-1)
    obs_dim = int(scales.numel())

    lines = []
    lines.append("`timescale 1ns/1ps")
    lines.append("")
    lines.append(f"package {package_name};")
    lines.append("")
    lines.append(f"  localparam int OBS = {obs_dim};")
    lines.append(f"  // {doc}")
    lines.append(f"  localparam int FP32_MUL_LATENCY    = {mul_latency};")
    lines.append(f"  localparam int FP32_TO_INT_LATENCY = {to_int_latency};")
    lines.append("")
    lines.append("  localparam logic [31:0] OBS_SCALES [0:OBS-1] = '{")

    for i, s in enumerate(scales.tolist()):
        hex32 = float32_to_hex(s)
        comma = "," if i < obs_dim - 1 else ""
        lines.append(f"      32'h{hex32}{comma} // obs {i} : {s:.8g}")

    lines.append("  };")
    lines.append("")
    lines.append(f"endpackage : {package_name}")
    lines.append("")

    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text("\n".join(lines))
    print(f"Wrote FP32 scale package to {out_path}")
    
@torch.no_grad()
def evaluate_actor_obs(
    actor: nn.Module,
    env_id: str,
    normalizer: nn.Module,
    device: torch.device,
    episodes: int = 10,
    seed: int = 123,
    deterministic: bool = True,
    record_stats: bool = True,
    global_step: Optional[int] = None,
    writer=None,
    # histogram options
    track_action_hist: bool = True,
    num_bins: int = 101,                 # evenly spaced bins on [-1, 1]
    log_action_hist_to_tb: bool = False, # logs raw actions per dim to TensorBoard
    std_dev = 0.1,                     # std dev of normal noise added to obs for eval (to test robustness
    quantizer = None,
):
    """
    Returns:
        ret_mean, ret_std, returns, lengths,
        (action_hist_counts, bin_edges) if track_action_hist else (None, None)

        action_hist_counts: np.ndarray shape [act_dim, num_bins]
        bin_edges: np.ndarray shape [num_bins + 1]
    """
    def update_absmax(x, absmax):
        x_abs = np.abs(x)
        if absmax is None:
            return x_abs.copy()
        return np.maximum(absmax, x_abs)
    obs_absmax = None
    eval_env = gym.make(env_id)
    eval_env = gym.wrappers.RecordEpisodeStatistics(eval_env)

    actor.eval()
    returns: List[float] = []
    lengths: List[int] = []

    # Collect all actions here for histogramming
    collected_actions: List[np.ndarray] = [] if track_action_hist else None

    # Precompute histogram bin edges (evenly spaced on [-1, 1])
    if track_action_hist:
        bin_edges = np.linspace(-1.0, 1.0, num_bins + 1, dtype=np.float64)
    else:
        bin_edges = None
    first_observation = None
    first_action = None
    for ep in range(episodes):
        obs, _ = eval_env.reset(seed=seed + ep)
        obs_absmax = update_absmax(obs, obs_absmax)

        done = False
        truncated = False

        while not (done or truncated):
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            #print("before", obs_t)
            if first_observation is None:
                first_observation = obs_t
            if quantizer is not None:
                obs_t = quantizer.quantize_dequantize(obs_t)
            #print("after", obs_t)
            obs_t = normalizer.norm(obs_t)
            # add normal distributed noise
            obs_t += torch.randn_like(obs_t) * std_dev
            if deterministic:
                # For SAC, use mean action for evaluation
                _, _, mean = actor.get_action(obs_t)
                action = mean[0].detach().cpu().numpy()
            else:
                action, _, _ = actor.get_action(obs_t)
                action = action[0].detach().cpu().numpy()
            if first_action is None:
                first_action = action
            # Track actions for histogram
            if track_action_hist:
                collected_actions.append(action)

            obs, r, done, truncated, infos = eval_env.step(action)
            obs_absmax = update_absmax(obs, obs_absmax)

            # episode stats are already tracked by wrapper; just exit when episode ends
            break
            if done or truncated:
                returns.append(infos["episode"]["r"][0])
                lengths.append(infos["episode"]["l"][0])

    eval_env.close()
    actor.train()

    ret_mean = float(np.mean(returns)) if returns else 0.0
    ret_std  = float(np.std(returns)) if returns else 0.0
    len_mean = float(np.mean(lengths)) if lengths else 0.0

    if record_stats and writer is not None:
        step = int(global_step or 0)
        writer.add_scalar("eval/episodic_return_mean", ret_mean, step)
        writer.add_scalar("eval/episodic_return_std",  ret_std,  step)
        writer.add_scalar("eval/episodic_length_mean", len_mean, step)

    # Build histograms
    action_hist_counts = None
    if track_action_hist and len(collected_actions) > 0:
        actions_all = np.stack(collected_actions, axis=0)  # [T, act_dim]
        act_dim = actions_all.shape[1]
        counts = []
        for d in range(act_dim):
            c, _ = np.histogram(actions_all[:, d], bins=bin_edges)
            counts.append(c.astype(np.int64))
            # Optional: log raw actions for TB’s histogram visual (not the binned counts)
            if log_action_hist_to_tb and writer is not None:
                import torch as _torch
                writer.add_histogram(f"eval/actions_dim{d}", _torch.from_numpy(actions_all[:, d]), int(global_step or 0))
        action_hist_counts = np.stack(counts, axis=0)  # [act_dim, num_bins]

    return ret_mean, ret_std, returns, lengths, action_hist_counts, bin_edges, obs_absmax, first_observation, first_action

def export_to_verilog(layers, obs_dim, act_dim, bits, input_bits, filename="lut_init.sv"):
    """
    Export model[0].luts (shape [LUTS, 2**k]) into a SystemVerilog package.
    
    Parameters
    ----------
    model : torch.nn.Sequential
        Your network whose first layer is dwn.LUTLayer.
    filename : str or Path
        Path of the .sv file to create.
    """
    # 1. grab tensor on CPU, make sure it's ints 0 / 1
    # get first luts
    
    luts = layers[0].luts.detach().cpu().round().to(torch.int)
    print(luts)
    luts_layers, cols = luts.shape
    k = int(math.log2(cols))
    assert cols == (1 << k), "Second dimension must be a power of two"

    # 2. start building the SV source as a list of strings
    lines = []
    lines.append("`ifndef CORE_PKG_SVH")
    lines.append("`define CORE_PKG_SVH\n")
    lines.append("`timescale 1ns/1ps")
    lines.append("package core_pkg;\n")
    lines.append(f"  parameter int OBS = {obs_dim};")
    lines.append(f"  parameter int LUTS       = {luts_layers};")
    lines.append(f"  parameter int INPUTS_LUT = {k};\n")
    lines.append(f"  parameter int BITS_PER_OBS = {bits};")
    lines.append(f"  parameter int RAW_OBS_BIT_WIDTH = {input_bits};")
    lines.append(f"  parameter int N_BITS = INPUTS_LUT * LUTS;")
    lines.append(f"  parameter int ACTIONS = {act_dim};")
    luts_last_layer = layers[-1].luts.shape[0]
    lines.append(f"  parameter int LAST_LUTS = {luts_last_layer};")
    lines.append(f"  parameter int ACTION_BITWIDTH = $clog2(LAST_LUTS/ACTIONS+1); // bitwidth of action outputs")
    lines.append(f"  parameter int ENCODE_OBS_BIT_WIDTH = $clog2(BITS_PER_OBS+1); // bitwidth for encoding observation inputs")
    # Last layer LUTS


    for l, layer in enumerate(layers):
        lines.append("  // Mapping array")
        if l == len(layers) -1:
            luts_n = luts_last_layer
            luts_str = 'LAST_LUTS'
        else:
            luts_n = luts_layers
            luts_str = 'LUTS'
            #lines.append(f"  parameter int unsigned MAPPING_{l} [0:LAST_LUTS* INPUTS_LUT-1] = '")
        lines.append(f"  parameter int unsigned MAPPING_{l} [0:{luts_str}* INPUTS_LUT-1] = '")
        lines[-1] += "{"
        luts = (layer.luts.detach().cpu() > 0.0).to(torch.int)
        if hasattr(layer.mapping, 'weights'):
            mapping = layer.mapping.weights.argmax(dim=0).cpu().detach().numpy().reshape(-1, k)
        else:
            print(layer.mapping.cpu().detach().numpy())
            print(layer.mapping.cpu().detach().numpy().shape)
            mapping = layer.mapping.cpu().detach().numpy().reshape(-1, k)
        for i in range(luts_n):
        # do it from top to bottom to have the right indexing starting with top
        # for i in range(LUTS-1, -1, -1):
            mapping_row = mapping[i] #[::-1] # have to invert the mapping to comforme 
            
            bits = ", ".join(f"{b.item()}" for b in mapping_row)
            comma = "," if i < luts_n - 1 else ""  # no trailing comma
            #comma = "," if i > 0 else ""        # no trailing comma on last
            lines.append(f"    {bits}{comma}   // LUT {i}")
        lines.append("  };")
        
        
        lines.append(f"  parameter logic [0:{luts_str}-1][0:(1<<INPUTS_LUT)-1] LUT_INIT_{l} = '")
        lines[-1] += "{"
        # 3. one sub‑array per LUT        #for idx, row in enumerate(luts):
        # reverse the order of rows to have LUT 0 at the top
        print(luts.shape)
        lut_lines = []
        for idx, row in enumerate(luts):
            # reverse tensor
            row = reversed(row)  # reverse the order of bits in each LUT
            bits = ", ".join(f"1'b{b.item()}" for b in row)
            comma = "," if idx < luts_n - 1 else ""        # no trailing comma on last
            #comma = "," if idx > 0 else ""        # no trailing comma on first
            lut_lines.append(f"    '{{{bits}}}{comma}   // LUT {idx}")
        lut_lines = lut_lines #[::-1]
        lines.extend(lut_lines)
        lines.append("  };")
        lines.append("\n")  
    lines.append("\nendpackage\n`endif\n")

    # 4. write to disk
    Path(filename).write_text("\n".join(lines))
    print(f"Wrote {filename} with {luts_n} LUTs × 2^{k} entries each.")

def make_group_sum_hook(tag: str = "GROUPSUM", 
                         store_attr: str = "_dbg_last_groupsum_out"):
    """
    Hook that prints the output of a GroupSum layer and stores it on the module
    as `module.<store_attr>` (CPU tensor) for later programmatic checks.
    """
    def _hook(module, inp, out=None):
        x = inp[0] if out is None else out  # pre-hook vs forward-hook
        out = x.detach()

        # Print a compact summary
        batch_like = out.view(-1, out.size(-1))  # flatten leading dims for printing
        to_show = min(4, batch_like.size(0))
        print(f"\n[{tag}] groupsum output  shape={tuple(out.shape)}  device={x.device}")
        print(batch_like[:to_show].cpu())

        # Optional: keep a copy for later checks
        if store_attr:
            try:
                setattr(module, store_attr, out.to("cpu"))
            except Exception as e:
                print(f"[{tag}] WARNING: could not store attr '{store_attr}': {e}")
        return

    return _hook
def make_thermo_values_hook(obs_dim: int, bits_per_obs: int, tag: str = "THERMO", 
                            store_attr: str = "_dbg_last_thermo_vals"):
    """
    Reconstructs obs_dim integers from a flattened thermometer code of length obs_dim*bits_per_obs
    along the last dimension. Works as forward pre-hook or forward hook.

    - Assumes grouping like: for i in [0..obs_dim-1], block i is j in [0..bits_per_obs-1],
      i.e., contiguous blocks per observation (matches thermo_out[i*BITS_PER_OBS + j]).
    - Coerces to {0,1} before summation: (>0.5) for float, (!=0) for integer/bool.

    The hook prints the recovered integers and also stores them on the module as
    `module.<store_attr>` (CPU tensor) for later programmatic checks.
    """
    def _hook(module, inp, out=None):
        x = inp[0] if out is None else out  # pre-hook vs forward-hook
        flat = x.detach()

        # Basic shape checks
        if flat.size(-1) != obs_dim * bits_per_obs:
            print(f"[{tag}] ERROR: last dim {flat.size(-1)} != obs_dim*bits_per_obs "
                  f"({obs_dim}*{bits_per_obs}={obs_dim*bits_per_obs})")
            return  # do not modify output

        # Reshape to (..., obs_dim, bits_per_obs)
        resh = flat.view(*flat.shape[:-1], obs_dim, bits_per_obs)

        # Force binary (thermometer should be 0/1)
        if resh.is_floating_point():
            bits01 = (resh > 0.5)          # float safety
        else:
            bits01 = (resh != 0)           # int/bool safety

        # Sum along thermometer axis -> integers per observation
        vals = bits01.sum(dim=-1)          # shape (..., obs_dim), dtype=torch.int64 (bool->long)

        # Optional: keep a copy for later checks
        if store_attr:
            try:
                setattr(module, store_attr, vals.to("cpu"))
            except Exception as e:
                print(f"[{tag}] WARNING: could not store attr '{store_attr}': {e}")

        # Print a compact summary
        batch_like = vals.view(-1, obs_dim)  # flatten leading dims for printing
        to_show = min(4, batch_like.size(0))
        print(f"\n[{tag}] thermo→values  shape={tuple(vals.shape)}  device={x.device}")
        print(batch_like[:to_show].cpu())

        # Do NOT return anything—returning from a forward hook would replace the module's output.
        return

    return _hook



def _needed_signed_width(x: torch.Tensor) -> int:
    x = x.to(torch.int64)
    min_val = int(x.min().item())
    max_val = int(x.max().item())

    # Smallest signed two's-complement width covering [min_val, max_val]
    for bits in range(1, 65):
        lo = -(1 << (bits - 1))
        hi = (1 << (bits - 1)) - 1
        if lo <= min_val and max_val <= hi:
            return bits
    raise ValueError("Values do not fit in 64-bit signed integers.")


def export_thresholds_to_svh(
    new_thresholds: torch.Tensor,
    out_path: str,
    package_name: str = "threshold_pkg",
    param_name: str = "NORMALIZATION_THRESHOLDS",
    threshold_width = None,
):
    """
    Export thresholds to a SystemVerilog package.

    Args:
        new_thresholds: torch.Tensor of shape [obs_dim, num_thresholds].
                       Must contain integer-valued thresholds.
        out_path: output .svh file path
        package_name: SystemVerilog package name
        param_name: parameter array name
        threshold_width: signed literal width; if None, computed automatically
    """
    if new_thresholds.ndim != 2:
        raise ValueError(f"Expected 2D tensor [obs_dim, num_thresholds], got {tuple(new_thresholds.shape)}")

    if not torch.is_floating_point(new_thresholds):
        thresholds_i64 = new_thresholds.to(torch.int64)
    else:
        # Guard against accidental non-integer floats
        rounded = torch.round(new_thresholds)
        if not torch.allclose(new_thresholds, rounded):
            raise ValueError("new_thresholds contains non-integer floating-point values.")
        thresholds_i64 = rounded.to(torch.int64)

    obs_dim, num_thresholds = thresholds_i64.shape

    if threshold_width is None:
        threshold_width = _needed_signed_width(thresholds_i64)

    def sv_literal(v: int) -> str:
        if v < 0:
            return f"-{threshold_width}'sd{abs(v)}"
        else:
            return f"{threshold_width}'sd{v}"

    lines = []
    lines.append(f"package {package_name};")
    lines.append("")
    lines.append(f"  localparam int OBS = {obs_dim};")
    lines.append(f"  localparam int NUM_THRESHOLDS = {num_thresholds};")
    lines.append(f"  localparam int THRESH_WIDTH = {threshold_width};")
    lines.append("")
    lines.append(f"  localparam logic signed [THRESH_WIDTH-1:0] {param_name} [0:OBS-1][0:NUM_THRESHOLDS-1] = '{{")

    for obs_idx in range(obs_dim):
        row = thresholds_i64[obs_idx].tolist()
        row_txt = ", ".join(sv_literal(int(v)) for v in row)
        comma = "," if obs_idx < obs_dim - 1 else ""
        lines.append(f"    '{{{row_txt}}}{comma}")

    lines.append("  };")
    lines.append("")
    lines.append(f"endpackage : {package_name}")
    lines.append("")

    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text("\n".join(lines))

    print(f"Wrote {out_file}")
    print(f"OBS={obs_dim}, NUM_THRESHOLDS={num_thresholds}, THRESH_WIDTH={threshold_width}")

def write_identity_mem(out_path: str, addr_width: int, data_width= None) -> None:
    """
    Generate a Xilinx-style .mem file for an identity ROM.

    Line i contains the value i, zero-padded in hex to match data_width.

    Args:
        out_path: output .mem file path
        addr_width: address width of the ROM
        data_width: data width of the ROM; defaults to addr_width
    """
    if data_width is None:
        data_width = addr_width

    depth = 1 << addr_width
    hex_digits = (data_width + 3) // 4
    mask = (1 << data_width) - 1

    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    with out_file.open("w") as f:
        for i in range(depth):
            value = i & mask
            f.write(f"{value:0{hex_digits}X}\n")

    print(f"Wrote {out_file}")
    print(f"depth={depth}, addr_width={addr_width}, data_width={data_width}, hex_digits={hex_digits}")



from run_evals import LGNActor
def main():
    parser = argparse.ArgumentParser(
        description="Load a model and write hdl file"
    )
    parser.add_argument(
        "--path",
        required=True,
        help="Path to the model file",
    )
    parser.add_argument(
        "--out",
        required=False,
        default="hdl/verilog/pkg",
        help="Path to the output .sv file",
    )
    parser.add_argument(
        "--bits",
        type=int,
        default=63,
        help="Number of bits for thermometer encoding",
    )
    parser.add_argument(
        "--random_interconnect",
        action='store_true',
        help="Use random interconnect mapping instead of learned mapping for SECOND layer",
    )
    parser.add_argument(
        "--env",
        type=str,
    )
    parser.add_argument("--run_episode",
                        action='store_true')
    parser.add_argument("--input_quantization_bits",
                        type=int,
                        default=8)
    args = parser.parse_args()
    # create the out path if it does not exist

    Path(args.out).mkdir(parents=True, exist_ok=True)
    #extract the name of the model from the path
    env_name = re.search(r'_([A-Za-z0-9]+)-v4', args.path).group(1)
    Path(f"{args.out}/{env_name}").mkdir(parents=True, exist_ok=True)
 
    # torch load path 
    print(f"Loading model from {args.path}")
    model = torch.load(args.path, map_location="cpu")
    mean = model['obs_norm']['mean']
    var = model['obs_norm']['var']
    quantization_info = f"data/abs_max_{args.env}.npy"
    print(f"Loading input quantization information from: {quantization_info}")
    
    print(model['fc_mean'].keys())
    obs_dim = model['obs_norm']['mean'].shape[0]
    
    
    
    beta_key = next(k for k in model['fc_mean'].keys() if 'beta' in k)
    act_dim = model['fc_mean'][beta_key].shape[0]
    print()
    group_sum_bits = np.ceil(np.log2(model['fc_mean']['net.2.mapping.weights'].shape[1]/6/act_dim))
    print(act_dim)
    size=model['fc_mean']['net.1.luts'].shape[0]
    Path(f"{args.out}/{env_name}/{size}").mkdir(parents=True, exist_ok=True)
 
    final_path=f"{args.out}/{env_name}/{size}"
    # hard coded layer number
    l=2
    bits = args.bits
    thermo = ThermometerGaussian(n_bits=bits, device='cuda')
    min_values = torch.ones((obs_dim,)) * -10
    max_values = torch.ones((obs_dim,)) * 10
    thermo.fit(torch.zeros((1, obs_dim)), min_value=min_values, max_value=max_values)

    model =WNN(obs_dim=obs_dim, 
                act_dim=act_dim, 
                sizes=[size] * l, 
                thermometer=thermo, bits=bits,
                later_learnable=not(args.random_interconnect)) 
   
    raw_dict = torch.load(args.path, map_location="cpu")['fc_mean']
    print(raw_dict.keys())
    model.load_state_dict(raw_dict, strict=False)
    model_net = model.net
    export_to_verilog([model_net[1],model_net[2]], obs_dim, act_dim, bits,args.input_quantization_bits, filename=f"{final_path}/core_pkg.svh")


    model_net[1].register_forward_pre_hook(make_thermo_values_hook(obs_dim, bits))
    model_net[3].register_forward_hook(make_group_sum_hook())
    
    # pass random randn through
    x = torch.randn((1, obs_dim)).cuda()
    model = model.cuda()
    out = model(x)
    
    
    inputs = model_net[1]._dbg_last_thermo_vals
    outputs = model_net[3]._dbg_last_groupsum_out.type(torch.int32)
    inp_np = inputs.to(torch.int64).cpu().numpy()
    out_np = outputs.to(torch.int64).cpu().numpy()
    print(inp_np)
    print(out_np)
    np.savetxt(f"{final_path}/inputs.txt",  inp_np, fmt="%d")
    np.savetxt(f"{final_path}/outputs.txt", out_np, fmt="%d")

    # recompute the thresholds
    quant_bits = args.input_quantization_bits
    qmax = 2 ** (quant_bits - 1) - 1 
    #print(quantizer)
    from run_evals import SymmetricNBitQuantizer
    quantizer = SymmetricNBitQuantizer(quantization_info,args.input_quantization_bits)
    scale = quantizer.scale
    new_thresholds = (thermo.thresholds.cpu()  * torch.sqrt(var[:,None] + 1e-8) + mean[:,None])/scale[:,None]
    print(new_thresholds)
    new_thresholds = torch.floor(new_thresholds)
    new_thresholds[new_thresholds < -qmax] = -qmax
    new_thresholds[new_thresholds > qmax] = qmax
    # cast new_thresholds to integer
    new_thresholds = new_thresholds.type(torch.int32)
    print(new_thresholds)
    export_thresholds_to_svh(
        new_thresholds,
        out_path=f"hdl/verilog/pkg/{env_name}/256/threshold_pkg.svh",
        package_name="threshold_pkg",
        param_name="NORMALIZATION_THRESHOLDS",
    )
    if args.run_episode:
        from run_evals import evaluate_actor_histogram, ObsNorm
        import gymnasium as gym
        from argparse import Namespace
        args = Namespace(
            size=256,
            l=2,
            return_type='raw',
            m=3.0,
            n=6,
            bits=63,
            network_type='lgn',
            env=args.env,
        )
        from run_evals import make_env
        envs = gym.vector.SyncVectorEnv(
            [make_env(args.env, 0, i, False, 'placeholder') for i in range(1)]
        )
        actor = LGNActor(envs, args)
        actor.fc_mean.load_state_dict(raw_dict, strict=False)
        actor.eval()
        device = 'cuda'
        actor = actor.to('cuda')
        
        obs_norm = ObsNorm(obs_dim, device=device)
        
        obs_norm.mean = mean
        obs_norm.var = var
        obs_norm.count = None
        obs_norm = obs_norm.to(device)
        quantizer.scale = quantizer.scale.to(device)
        ret_mean, ret_std, returns, lengths, action_hist_counts, bin_edges, obs_absmax, first_observation, first_action = evaluate_actor_obs(actor, args.env, obs_norm, device=device, writer=None, global_step=0, episodes=1, seed=1, std_dev=0.0,quantizer=quantizer)
        print(first_observation)
        print(first_action)
    
        print(ret_mean, ret_std)
        
        obs_t = quantizer.quantize_dequantize(first_observation)
        print("after quantizer", obs_t)
        print("Integer representation", quantizer.quantize(first_observation))
        obs_t = obs_norm.norm(obs_t)
        print("after Normalizer", obs_t)
        actor.fc_mean.net[3].register_forward_hook(make_group_sum_hook())
        _, _, mean = actor.get_action(obs_t)

        action = mean[0].detach().cpu().numpy()
        outputs = actor.fc_mean.net[3]._dbg_last_groupsum_out.type(torch.int32)
        print("final output", action)
        inputs = quantizer.quantize(first_observation)
        print("Integer ins", )
        # print the betas
        print(actor.fc_mean.net[4].beta)
        print(outputs)
        inp_np = inputs.to(torch.int64).cpu().numpy()
        out_np = outputs.to(torch.int64).cpu().numpy()
        print(inp_np)
        print(out_np)
        
        
        np.savetxt(f"{final_path}/inputs_0.txt",  inp_np, fmt="%d")
        np.savetxt(f"{final_path}/outputs_0.txt", out_np, fmt="%d")
        write_identity_mem("/nfs/scistore16/tomgrp/fkresse/difflogic/release_dwc/hdl/verilog/identity_8b.mem",
                           int(group_sum_bits),
                           8)
        print(group_sum_bits)
        export_fp32_inputs(
            first_observation,
            f"{final_path}/inputs_fp32_0.txt",
        )

        # New FP32 scale package export
        export_fp32_pkg(
            1.0/quantizer.scale,
            f"{final_path}/fp32_pkg.svh",
            package_name="fp32_pkg",
            mul_latency=9,
            to_int_latency=7,
            doc=f"Auto-generated FP32 quantization scales for {args.env}.",
        )

if __name__ == "__main__":
    main()
