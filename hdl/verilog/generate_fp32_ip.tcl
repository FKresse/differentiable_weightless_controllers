# gen_fp_ip.tcl
set part xc7a15tfgg484-1

create_project managed_ip ./managed_ip -part $part -ip
set_property target_language Verilog [current_project]
set_property simulator_language Mixed [current_project]

# FP32 multiply
create_ip -name floating_point -vendor xilinx.com -library ip -module_name fp32_mul
set_property CONFIG.Operation_Type {Multiply} [get_ips fp32_mul]
set_property -dict [list \
    CONFIG.A_Precision_Type {Single} \
    CONFIG.Result_Precision_Type {Single} \
    CONFIG.C_A_Exponent_Width {8} \
    CONFIG.C_A_Fraction_Width {24} \
    CONFIG.C_Result_Exponent_Width {8} \
    CONFIG.C_Result_Fraction_Width {24} \
] [get_ips fp32_mul]

# FP32 to signed integer / fixed-point
create_ip -name floating_point -vendor xilinx.com -library ip -module_name fp32_to_int
set_property CONFIG.Operation_Type {Float_to_fixed} [get_ips fp32_to_int]
set_property -dict [list \
    CONFIG.A_Precision_Type {Single} \
    CONFIG.Result_Precision_Type {Custom} \
    CONFIG.C_A_Exponent_Width {8} \
    CONFIG.C_A_Fraction_Width {24} \
    CONFIG.C_Result_Exponent_Width {16} \
    CONFIG.C_Result_Fraction_Width {0} \
] [get_ips fp32_to_int]

generate_target all [get_ips fp32_mul]
generate_target all [get_ips fp32_to_int]

create_ip_run [get_ips fp32_mul]
create_ip_run [get_ips fp32_to_int]
launch_runs fp32_mul_synth_1 fp32_to_int_synth_1
wait_on_run fp32_mul_synth_1
wait_on_run fp32_to_int_synth_1