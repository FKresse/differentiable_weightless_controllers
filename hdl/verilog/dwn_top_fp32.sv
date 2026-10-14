`include "core_pkg.svh"
`include "fp32_pkg.svh"
`timescale 1ns/1ps

module dwn_top_fp32 #(
    parameter int PIPE_STAGES = 4,
    parameter int HARDWARE_ACTION_BITWIDTH = 8
)(
    input  logic clk,
    input  logic rstn,
    // one FP32 input per observation
    input  logic [31:0] in_bus_fp [0:core_pkg::OBS-1],

    output logic [0:HARDWARE_ACTION_BITWIDTH-1] group_sum_bram [0:core_pkg::ACTIONS-1]
);
    import core_pkg::*;

    // Quantized signed integer observations that feed the original core
    logic signed [0:RAW_OBS_BIT_WIDTH-1] quant_obs [0:OBS-1];

    // Per-observation FP multiplier outputs
    logic        mul_valid   [0:OBS-1];
    logic [31:0] mul_out     [0:OBS-1];

    // Per-observation float->int outputs
    logic        conv_valid  [0:OBS-1];
    logic [RAW_OBS_BIT_WIDTH-1:0] conv_out [0:OBS-1];

    genvar i;
    generate
        for (i = 0; i < OBS; i++) begin : G_FP_FRONTEND

            // FP32 multiply: input * scale[i]
            fp32_mul u_mul (
                .aclk                 (clk),
                .s_axis_a_tvalid      (1'b1),
                .s_axis_a_tready      (),
                .s_axis_a_tdata       (in_bus_fp[i]),
                .s_axis_b_tvalid      (1'b1),
                .s_axis_b_tready      (),
                .s_axis_b_tdata       (fp32_pkg::OBS_SCALES[i]),
                .m_axis_result_tvalid (mul_valid[i]),
                .m_axis_result_tready (1'b1),
                .m_axis_result_tdata  (mul_out[i])
            );

            // FP32 -> signed integer
            fp32_to_int u_conv (
                .aclk                 (clk),
                .s_axis_a_tvalid      (mul_valid[i]),
                .s_axis_a_tready      (),
                .s_axis_a_tdata       (mul_out[i]),
                .m_axis_result_tvalid (conv_valid[i]),
                .m_axis_result_tready (1'b1),
                .m_axis_result_tdata  (conv_out[i])
            );

            always_ff @(posedge clk or negedge rstn) begin
                if (!rstn)
                    quant_obs[i] <= '0;
                else if (conv_valid[i])
                    quant_obs[i] <= conv_out[i];
            end
        end
    endgenerate

    // Reuse the existing integer-input core
    dwn_top #(
        .PIPE_STAGES(PIPE_STAGES),
        .HARDWARE_ACTION_BITWIDTH(HARDWARE_ACTION_BITWIDTH)
    ) u_core (
        .clk            (clk),
        .rstn           (rstn),
        .in_bus         (quant_obs),
        .group_sum_bram (group_sum_bram)
    );

endmodule