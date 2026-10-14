from pathlib import Path


def write_identity_mem(out_path: str, addr_width: int, data_width: int | None = None) -> None:
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


if __name__ == "__main__":
    # Example: 8-bit address -> 256 entries, each entry stores its own address
    write_identity_mem(
        out_path="identity_8b.mem",
        addr_width=6,
        data_width=8,
    )