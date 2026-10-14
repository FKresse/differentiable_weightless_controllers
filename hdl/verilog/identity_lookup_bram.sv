module identity_lookup_bram #(
    parameter int ADDR_W = 8,
    parameter int DATA_W = 8,
    parameter int DEPTH  = (1 << ADDR_W)
)(
    input  logic              clk,
    input  logic [0:ADDR_W-1] addr,
    output logic [0:DATA_W-1] dout
);

  // Vivado XPM single-port ROM
  xpm_memory_sprom #(
    .ADDR_WIDTH_A        (ADDR_W),
    .AUTO_SLEEP_TIME     (0),
    .CASCADE_HEIGHT      (0),
    .ECC_MODE            ("no_ecc"),
    .MEMORY_INIT_FILE("identity_8b.mem"),
    .READ_DATA_WIDTH_A(DATA_W),
    .MEMORY_OPTIMIZATION ("true"),
    .MEMORY_PRIMITIVE    ("block"),     // force BRAM-style implementation
    .MEMORY_SIZE         (DEPTH * DATA_W),
    .MESSAGE_CONTROL     (0),
    .READ_LATENCY_A      (1),
    .READ_RESET_VALUE_A  ("0"),
    .RST_MODE_A          ("SYNC"),
    .SIM_ASSERT_CHK      (0),
    .USE_MEM_INIT        (1),
    .WAKEUP_TIME         ("disable_sleep")
  ) u_rom (
    .addra   (addr),
    .clka    (clk),
    .douta   (dout),
    .ena     (1'b1),
    .injectdbiterra(1'b0),
    .injectsbiterra(1'b0),
    .regcea  (1'b1),
    .rsta    (1'b0),
    .sleep   (1'b0)
  );

endmodule