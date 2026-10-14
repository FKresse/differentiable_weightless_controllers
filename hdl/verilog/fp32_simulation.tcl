#!/usr/bin/env bash
set -euo pipefail

ENV=HalfCheetah
SIZE=256
TOP=tb_in_out_fp32

rm -rf xsim.dir .Xil

# Verilog / SystemVerilog
xvlog --sv \
  -i . \
  -i pkg/${ENV}/${SIZE} \
  --work work \
  dwn_lut_layer.sv \
  dwn_group_sum_pipelined.sv \
  mapping_layer.sv \
  thermo_encode_thresholded.sv \
  identity_lookup_bram.sv \
  dwn_top_ooc.sv \
  dwn_top_fp32.sv \
  tb_in_out_fp32.sv \
  managed_ip/managed_ip.gen/sources_1/ip/fp32_mul/sim/fp32_mul.v \
  managed_ip/managed_ip.gen/sources_1/ip/fp32_to_int/sim/fp32_to_int.v \
  managed_ip/managed_ip.gen/sources_1/ip/fp32_mul/hdl/floating_point_v7_1_rfs.v

# Shared VHDL support for the floating-point IP
xvhdl \
  --work work \
  managed_ip/managed_ip.gen/sources_1/ip/fp32_mul/hdl/axi_utils_v2_0_vh_rfs.vhd \
  managed_ip/managed_ip.gen/sources_1/ip/fp32_mul/hdl/mult_gen_v12_0_vh_rfs.vhd \
  managed_ip/managed_ip.gen/sources_1/ip/fp32_mul/hdl/xbip_bram18k_v3_0_vh_rfs.vhd \
  managed_ip/managed_ip.gen/sources_1/ip/fp32_mul/hdl/xbip_dsp48_addsub_v3_0_vh_rfs.vhd \
  managed_ip/managed_ip.gen/sources_1/ip/fp32_mul/hdl/xbip_dsp48_multadd_v3_0_vh_rfs.vhd \
  managed_ip/managed_ip.gen/sources_1/ip/fp32_mul/hdl/xbip_dsp48_wrapper_v3_0_vh_rfs.vhd \
  managed_ip/managed_ip.gen/sources_1/ip/fp32_mul/hdl/xbip_pipe_v3_0_vh_rfs.vhd \
  managed_ip/managed_ip.gen/sources_1/ip/fp32_mul/hdl/xbip_utils_v3_0_vh_rfs.vhd \
  managed_ip/managed_ip.gen/sources_1/ip/fp32_mul/hdl/floating_point_v7_1_rfs.vhd

# Elaborate
xelab --debug typical --timescale 1ns/1ps \
  -L xpm \
  -L unisims_ver \
  -L unimacro_ver \
  -L secureip \
  work.${TOP} \
  -s ${TOP}_sim

# Run
xsim ${TOP}_sim -runall