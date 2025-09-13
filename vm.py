# vm.py
import sys

DIGITS = "0123456789ab"

def from_base12(s: str) -> int:
    value = 0
    for ch in s:
        value = value * 12 + DIGITS.index(ch)
    return value

def run_dgm(dgm_code: str):
    tokens = dgm_code.strip().split()
    ip = 0
    memory = {}

    while ip < len(tokens):
        opcode = from_base12(tokens[ip])
        ip += 1

        if opcode == 0xA6:  # print string
            chars = []
            while ip < len(tokens):
                val = from_base12(tokens[ip]); ip += 1
                if val == 0: break
                chars.append(chr(val))
            print("".join(chars))

        elif opcode == 0x03:  # store addr value
            addr = from_base12(tokens[ip]); ip += 1
            val = from_base12(tokens[ip]); ip += 1
            memory[addr] = val

        elif opcode == 0x17:  # add addr value
            addr = from_base12(tokens[ip]); ip += 1
            val = from_base12(tokens[ip]); ip += 1
            memory[addr] = memory.get(addr, 0) + val

        elif opcode == 0x18:  # sub addr value
            addr = from_base12(tokens[ip]); ip += 1
            val = from_base12(tokens[ip]); ip += 1
            memory[addr] = memory.get(addr, 0) - val

        elif opcode == 0x33:  # ret
            return

        else:
            raise ValueError(f"Unknown opcode: {opcode}")

def main():
    if len(sys.argv) != 2:
        print("Usage: python vm.py <program.dgm>")
        sys.exit(1)

    infile = sys.argv[1]
    with open(infile, "r", encoding="utf-8") as f:
        dgm_code = f.read()

    run_dgm(dgm_code)

if __name__ == "__main__":
    main()
