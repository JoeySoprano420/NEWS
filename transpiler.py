# transpiler.py
import sys
import re

# Base-12 encoding table (0-11 → 0-9, a, b)
DIGITS = "0123456789ab"

def to_base12(num: int) -> str:
    """Convert integer opcode into base-12 representation."""
    if num == 0:
        return "0"
    result = []
    while num > 0:
        result.append(DIGITS[num % 12])
        num //= 12
    return "".join(reversed(result))

# NEWS → DGM opcode mapping (minimal for now)
NEWS_OPCODES = {
    "print": 0xA6,     # language.echo
    "end":   0x33,     # ret
}

def transpile(news_code: str) -> str:
    """Transpile NEWS source code into DGM opcodes (base-12)."""
    dgm_tokens = []

    # Match print("...") lines
    for line in news_code.splitlines():
        line = line.strip()
        if line.startswith("print("):
            # Extract string literal
            match = re.match(r'print\s*\("(.*)"\)', line)
            if not match:
                raise SyntaxError(f"Invalid print syntax: {line}")
            text = match.group(1)
            # Opcode for print
            dgm_tokens.append(to_base12(NEWS_OPCODES["print"]))
            # Encode string as ASCII codes
            for ch in text:
                dgm_tokens.append(to_base12(ord(ch)))
            # Null terminator
            dgm_tokens.append("0")
        elif line.startswith("end()"):
            dgm_tokens.append(to_base12(NEWS_OPCODES["end"]))

    return " ".join(dgm_tokens)

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
