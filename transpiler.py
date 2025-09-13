# transpiler.py
import sys, re

DIGITS = "0123456789ab"

def to_base12(num: int) -> str:
    """Convert integer to base-12 string."""
    if num == 0:
        return "0"
    result = []
    while num > 0:
        result.append(DIGITS[num % 12])
        num //= 12
    return "".join(reversed(result))

# NEWS → DGM opcode map
OPCODES = {
    "print": 0xA6,
    "ret":   0x33,
    "let":   0x01,
    "store": 0x03,
    "load":  0x02,
    "add":   0x17,
    "sub":   0x18,
    "icmp":  0x15,
    "br":    0x30,
}

def transpile(src: str) -> str:
    tokens = []
    variables = {}  # map var -> memory index
    mem_index = 1

    lines = [l.strip() for l in src.splitlines() if l.strip()]
    for line in lines:
        if line.startswith("print("):
            # print string
            text = re.match(r'print\("(.*)"\)', line).group(1)
            tokens.append(to_base12(OPCODES["print"]))
            for ch in text:
                tokens.append(to_base12(ord(ch)))
            tokens.append("0")  # null terminator

        elif line.startswith("let "):
            # let x = N
            match = re.match(r'let (\w+) *= *(\d+)', line)
            var, num = match.groups()
            if var not in variables:
                variables[var] = mem_index
                mem_index += 1
            addr = variables[var]
            tokens.append(to_base12(OPCODES["store"]))
            tokens.append(to_base12(addr))
            tokens.append(to_base12(int(num)))

        elif line.startswith("add "):
            # add x, 5
            match = re.match(r'add (\w+), *(\d+)', line)
            var, num = match.groups()
            addr = variables[var]
            tokens.append(to_base12(OPCODES["add"]))
            tokens.append(to_base12(addr))
            tokens.append(to_base12(int(num)))

        elif line.startswith("sub "):
            match = re.match(r'sub (\w+), *(\d+)', line)
            var, num = match.groups()
            addr = variables[var]
            tokens.append(to_base12(OPCODES["sub"]))
            tokens.append(to_base12(addr))
            tokens.append(to_base12(int(num)))

        elif line.startswith("end()"):
            tokens.append(to_base12(OPCODES["ret"]))

        else:
            raise SyntaxError(f"Unrecognized statement: {line}")

    return " ".join(tokens)

def main():
    if len(sys.argv) != 3:
        print("Usage: python transpiler.py <input.news> <output.dgm>")
        sys.exit(1)

    infile, outfile = sys.argv[1], sys.argv[2]
    with open(infile, "r", encoding="utf-8") as f:
        src = f.read()

    dgm_code = transpile(src)
    with open(outfile, "w", encoding="utf-8") as f:
        f.write(dgm_code)

    print(f"Transpiled {infile} → {outfile}")

if __name__ == "__main__":
    main()
