`timescale 1ns/1ps
`include "core_pkg.svh"

module tb_in_out_fp32;
  import core_pkg::*;

  localparam int HARDWARE_ACTION_BITWIDTH = 8;

  // Old integer core latency was about 20.
  // FP frontend adds about 12 (mul) + 7 (float->int) + 1 register = 20.
  localparam int PIPE_LAT = 40;

  logic clk = 0;
  always #5 clk = ~clk;

  // ------------------------------------------------------------------//
  // FP32 inputs to / outputs from DUT
  // ------------------------------------------------------------------//
  logic [31:0] obs_fp [0:core_pkg::OBS-1];
  logic [0:HARDWARE_ACTION_BITWIDTH-1] act [0:ACTIONS-1];

  // ------------------------------------------------------------------//
  // Device Under Test
  // ------------------------------------------------------------------//
  dwn_top_fp32 #(
    .PIPE_STAGES(4),
    .HARDWARE_ACTION_BITWIDTH(HARDWARE_ACTION_BITWIDTH)
  ) dut (
    .clk            (clk),
    .rstn           (1'b1),
    .in_bus_fp      (obs_fp),
    .group_sum_bram (act)
  );

  // ------------------------------------------------------------------//
  // Debug
  // ------------------------------------------------------------------//
  shortreal obs_dbg;

  always @(posedge clk) begin
    #1;
    $display("\n[%0t] ---- cycle debug ----", $time);

    // Show first two FP inputs
    obs_dbg = $bitstoshortreal(obs_fp[0]);
    $display("obs_fp[0] = %f (0x%08h)", obs_dbg, obs_fp[0]);
    obs_dbg = $bitstoshortreal(obs_fp[1]);
    $display("obs_fp[1] = %f (0x%08h)", obs_dbg, obs_fp[1]);

    // Show first two quantized integers inside wrapper if available
    $display("quant_obs[0] = %0d", $signed(dut.quant_obs[0]));
    $display("quant_obs[1] = %0d", $signed(dut.quant_obs[1]));

    // Show first two registered integers inside old core if available
    $display("core in_reg[0] = %0d", $signed(dut.u_core.in_reg[0]));
    $display("core in_reg[1] = %0d", $signed(dut.u_core.in_reg[1]));
  end

  // ------------------------------------------------------------------//
  // Files / vars
  // ------------------------------------------------------------------//
  string  IN_FILE  = "/nfs/scistore16/tomgrp/fkresse/difflogic/release_dwc/hdl/verilog/pkg/HalfCheetah/256/inputs_fp32_0.txt";
  string  OUT_FILE = "/nfs/scistore16/tomgrp/fkresse/difflogic/release_dwc/hdl/verilog/pkg/HalfCheetah/256/outputs_0.txt";

  integer fin, fout;
  integer eof_in, eof_out;
  integer line = 0;

  int rv;
  int exp [0:ACTIONS-1];
  int errors;
  int actual;

  // ------------------------------------------------------------------//
  // Stimulus + checker
  // ------------------------------------------------------------------//
  initial begin
    fin  = $fopen(IN_FILE, "r");
    if (fin == 0) $fatal(1, "Could not open %s", IN_FILE);

    fout = $fopen(OUT_FILE, "r");
    if (fout == 0) $fatal(1, "Could not open %s", OUT_FILE);

    forever begin
      // 1) read one line of OBS floating-point values into obs_fp[]
      eof_in = 0;
      for (int oi = 0; oi < core_pkg::OBS; oi++) begin
        real tmp;
        rv = $fscanf(fin, "%f", tmp);
        if (rv != 1) begin
          eof_in = 1;
          break;
        end
        obs_fp[oi] = $shortrealtobits(shortreal'(tmp));
      end
      if (eof_in) break;

      // Echo what we read
      $write("[READ FP32] ");
      for (int oi = 0; oi < core_pkg::OBS; oi++) begin
        shortreal t;
        t = $bitstoshortreal(obs_fp[oi]);
        $write("%f ", t);
      end
      $write("\n");

      // 2) wait pipeline latency
      @(posedge clk);
      repeat (PIPE_LAT) @(posedge clk);
      #1;

      // 3) read expected ACTIONS integers
      eof_out = 0;
      for (int a = 0; a < ACTIONS; a++) begin
        rv = $fscanf(fout, "%d", exp[a]);
        if (rv != 1) begin
          eof_out = 1;
          break;
        end
      end
      if (eof_out) break;

      // 4) compare / report
      errors = 0;
      for (int a = 0; a < ACTIONS; a++) begin
        actual = act[a];
        if (actual !== exp[a]) begin
          $display("[%0t] line %0d: action[%0d] MISMATCH  got=%0d  exp=%0d",
                   $time, line, a, actual, exp[a]);
          errors++;
        end
      end

      if (errors == 0)
        $display("[%0t] line %0d: PASS  outputs=%p", $time, line, act);

      line++;
    end

    $fclose(fin);
    $fclose(fout);
    $display("Completed %0d vectors.", line);
    $finish;
  end

endmodule