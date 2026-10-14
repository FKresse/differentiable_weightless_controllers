`timescale 1ns/1ps

module tb_fp32_mul_to_int;

  logic clk = 0;
  always #5 clk = ~clk;

  // -----------------------------
  // Multiplier input interface
  // -----------------------------
  logic        mul_a_valid;
  logic        mul_a_ready;
  logic [31:0] mul_a_data;

  logic        mul_b_valid;
  logic        mul_b_ready;
  logic [31:0] mul_b_data;

  // -----------------------------
  // Multiplier output interface
  // -----------------------------
  logic        mul_out_valid;
  logic        mul_out_ready;
  logic [31:0] mul_out_data;

  // -----------------------------
  // Converter interface
  // -----------------------------
  logic        conv_in_ready;
  logic        conv_out_valid;
  logic        conv_out_ready;
  logic [15:0] conv_out_data;  // adjust if your IP output width differs

  shortreal a_sr, b_sr, mul_sr;
  integer expected_q;

  // -----------------------------
  // DUTs
  // -----------------------------
  fp32_mul u_mul (
    .aclk                 (clk),
    .s_axis_a_tvalid      (mul_a_valid),
    .s_axis_a_tready      (mul_a_ready),
    .s_axis_a_tdata       (mul_a_data),
    .s_axis_b_tvalid      (mul_b_valid),
    .s_axis_b_tready      (mul_b_ready),
    .s_axis_b_tdata       (mul_b_data),
    .m_axis_result_tvalid (mul_out_valid),
    .m_axis_result_tready (mul_out_ready),
    .m_axis_result_tdata  (mul_out_data)
  );

  fp32_to_int u_conv (
    .aclk                 (clk),
    .s_axis_a_tvalid      (mul_out_valid),
    .s_axis_a_tready      (conv_in_ready),
    .s_axis_a_tdata       (mul_out_data),
    .m_axis_result_tvalid (conv_out_valid),
    .m_axis_result_tready (conv_out_ready),
    .m_axis_result_tdata  (conv_out_data)
  );

  // -----------------------------
  // Expected quantization
  // round to nearest, ties away from zero
  // -----------------------------
  function automatic integer quantize_expected(input shortreal x);
    integer q;
    begin
      if (x >= 0.0)
        q = $rtoi(x + 0.5);
      else
        q = $rtoi(x - 0.5);

      if (q > 32767)
        q = 32767;
      else if (q < -32768)
        q = -32768;

      quantize_expected = q;
    end
  endfunction

  // -----------------------------
  // Send one transaction
  // -----------------------------
  task automatic send_case(input shortreal a, input shortreal b);
    shortreal prod;
    integer exp;
    begin
      prod = a * b;
      exp  = quantize_expected(prod);

      mul_a_data  <= $shortrealtobits(a);
      mul_b_data  <= $shortrealtobits(b);
      mul_a_valid <= 1'b1;
      mul_b_valid <= 1'b1;

      // wait until both inputs are accepted
      while (!(mul_a_valid && mul_a_ready && mul_b_valid && mul_b_ready)) begin
        @(posedge clk);
      end

      $display("[%0t] INPUT accepted: a=%f  b=%f  expected_product=%f  expected_int=%0d",
               $time, a, b, prod, exp);

      @(posedge clk);
      mul_a_valid <= 1'b0;
      mul_b_valid <= 1'b0;
      mul_a_data  <= '0;
      mul_b_data  <= '0;
    end
  endtask

  // -----------------------------
  // Print multiplier output
  // -----------------------------
  always @(posedge clk) begin
    if (mul_out_valid && mul_out_ready && conv_in_ready) begin
      mul_sr = $bitstoshortreal(mul_out_data);
      $display("[%0t] MUL output: bits=0x%08h  float=%f",
               $time, mul_out_data, mul_sr);
    end
  end

  // -----------------------------
  // Print final integer output
  // -----------------------------
  always @(posedge clk) begin
    if (conv_out_valid && conv_out_ready) begin
      $display("[%0t] INT output: signed=%0d  hex=0x%04h",
               $time, $signed(conv_out_data), conv_out_data);
    end
  end

  // -----------------------------
  // Optional cycle trace
  // -----------------------------
  integer cycle = 0;
  always @(posedge clk) begin
    cycle <= cycle + 1;
    $display("[cycle %0d] a_v=%0b a_r=%0b b_v=%0b b_r=%0b mul_v=%0b conv_r=%0b out_v=%0b",
             cycle, mul_a_valid, mul_a_ready, mul_b_valid, mul_b_ready,
             mul_out_valid, conv_in_ready, conv_out_valid);
  end

  // -----------------------------
  // Stimulus
  // -----------------------------
  initial begin
    mul_a_valid    = 0;
    mul_b_valid    = 0;
    mul_a_data     = '0;
    mul_b_data     = '0;
    mul_out_ready  = 1'b1;
    conv_out_ready = 1'b1;

    $display("Starting tb_fp32_mul_to_int");
    $display("--------------------------------------------------");

    repeat (5) @(posedge clk);

    send_case(1.0,   12.7);   // expect 13
    send_case(-1.0,  12.7);   // expect -13
    send_case(0.5,   12.7);   // expect 6
    send_case(3.2,   4.0);    // expect 13
    send_case(-3.2,  4.0);    // expect -13

    repeat (80) @(posedge clk);

    $display("--------------------------------------------------");
    $display("Finished tb_fp32_mul_to_int");
    $finish;
  end

endmodule