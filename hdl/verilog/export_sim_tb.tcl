#!/usr/bin/env bash
set -euo pipefail

WORKDIR=xsim_work
TOP=tb_fp32_mul_to_int

rm -rf xsim.dir .Xil
# Compile Verilog/SystemVerilog
xvlog --sv \
  --work work \
  tb_fp32_mul_to_int.sv \
  managed_ip/managed_ip.gen/sources_1/ip/fp32_mul/sim/fp32_mul.v \
  managed_ip/managed_ip.gen/sources_1/ip/fp32_to_int/sim/fp32_to_int.v \
  managed_ip/managed_ip.gen/sources_1/ip/fp32_mul/hdl/floating_point_v7_1_rfs.v \
  managed_ip/managed_ip.gen/sources_1/ip/fp32_to_int/hdl/floating_point_v7_1_rfs.v

# Compile VHDL support files used by the floating-point IP
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
  managed_ip/managed_ip.gen/sources_1/ip/fp32_mul/hdl/floating_point_v7_1_rfs.vhd \
  managed_ip/managed_ip.gen/sources_1/ip/fp32_to_int/hdl/axi_utils_v2_0_vh_rfs.vhd \
  managed_ip/managed_ip.gen/sources_1/ip/fp32_to_int/hdl/mult_gen_v12_0_vh_rfs.vhd \
  managed_ip/managed_ip.gen/sources_1/ip/fp32_to_int/hdl/xbip_bram18k_v3_0_vh_rfs.vhd \
  managed_ip/managed_ip.gen/sources_1/ip/fp32_to_int/hdl/xbip_dsp48_addsub_v3_0_vh_rfs.vhd \
  managed_ip/managed_ip.gen/sources_1/ip/fp32_to_int/hdl/xbip_dsp48_multadd_v3_0_vh_rfs.vhd \
  managed_ip/managed_ip.gen/sources_1/ip/fp32_to_int/hdl/xbip_dsp48_wrapper_v3_0_vh_rfs.vhd \
  managed_ip/managed_ip.gen/sources_1/ip/fp32_to_int/hdl/xbip_pipe_v3_0_vh_rfs.vhd \
  managed_ip/managed_ip.gen/sources_1/ip/fp32_to_int/hdl/xbip_utils_v3_0_vh_rfs.vhd \
  managed_ip/managed_ip.gen/sources_1/ip/fp32_to_int/hdl/floating_point_v7_1_rfs.vhd

# Elaborate
xelab --debug typical --timescale 1ns/1ps \
  -L unisims_ver \
  -L unimacro_ver \
  -L secureip \
  work.${TOP} \
  -s ${TOP}_sim
# Run and print to terminal
xsim ${TOP}_sim -runall