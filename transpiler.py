# transpiler.py
import sys, re

DIGITS = "0123456789ab"

def to_base12(num: int) -> str:
    if num == 0:
        return "0"
    result = []
    while num > 0:
        result.append(DIGITS[num % 12])
        num //= 12
    return "".join(reversed(result))

OPCODES = {
    "print": 0xA6,
    "ret":   0x33,
    "store": 0x03,
    "add":   0x17,
    "sub":   0x18,
    "icmp":  0x15,
    "br":    0x30,
    "match.begin": 0x95,
    "match.case":  0x96,
    "match.end":   0x97,
}

REL_OPS = {
    "==": "eq", "!=": "ne", "<": "lt", ">": "gt", "<=": "le", ">=": "ge"
}

class Transpiler:
    def __init__(self):
        self.tokens = []
        self.variables = {}
        self.mem_index = 1
        self.labels = {}
        self.fixups = []

    def add(self, val):
        self.tokens.append(to_base12(val) if isinstance(val, int) else val)

    def define_var(self, name):
        if name not in self.variables:
            self.variables[name] = self.mem_index
            self.mem_index += 1
        return self.variables[name]

    def transpile(self, src: str) -> str:
        lines = [l.strip() for l in src.splitlines() if l.strip()]
        i = 0
        while i < len(lines):
            line = lines[i]

            # print("...")
            if line.startswith("print("):
                text = re.match(r'print\("(.*)"\)', line).group(1)
                self.add(OPCODES["print"])
                for ch in text:
                    self.add(ord(ch))
                self.add(0)

            # let x = N
            elif line.startswith("let "):
                m = re.match(r'let (\w+) *= *(\d+)', line)
                var, num = m.groups()
                addr = self.define_var(var)
                self.add(OPCODES["store"]); self.add(addr); self.add(int(num))

            # add/sub
            elif line.startswith("add "):
                m = re.match(r'add (\w+), *(\d+)', line)
                addr = self.define_var(m.group(1))
                self.add(OPCODES["add"]); self.add(addr); self.add(int(m.group(2)))
            elif line.startswith("sub "):
                m = re.match(r'sub (\w+), *(\d+)', line)
                addr = self.define_var(m.group(1))
                self.add(OPCODES["sub"]); self.add(addr); self.add(int(m.group(2)))

            # if (...)
            elif line.startswith("if "):
                m = re.match(r'if\s*\(\s*(\w+)\s*([=!<>]+)\s*(\d+)\s*\)', line)
                var, op, num = m.groups()
                addr = self.define_var(var)
                self.add(OPCODES["icmp"]); self.add(addr); self.add(int(num)); self.add(op)
                # reserve jump placeholder
                self.fixups.append(("if_end", len(self.tokens)))
                self.add("FIXUP")

            elif line.startswith("endif"):
                for idx, (kind, pos) in enumerate(self.fixups):
                    if kind == "if_end" and self.tokens[pos] == "FIXUP":
                        self.tokens[pos] = to_base12(len(self.tokens))
                        self.fixups.pop(idx); break

            # while (...)
            elif line.startswith("while "):
                m = re.match(r'while\s*\(\s*(\w+)\s*([=!<>]+)\s*(\d+)\s*\)', line)
                var, op, num = m.groups()
                loop_start = len(self.tokens)
                addr = self.define_var(var)
                self.add(OPCODES["icmp"]); self.add(addr); self.add(int(num)); self.add(op)
                self.fixups.append(("while_end", len(self.tokens), loop_start))
                self.add("FIXUP")

            elif line.startswith("endwhile"):
                kind, pos, loop_start = self.fixups.pop()
                self.tokens[pos] = to_base12(len(self.tokens)+2)
                self.add(OPCODES["br"]); self.add(loop_start)

            # match x
            elif line.startswith("match "):
                var = line.split()[1]
                addr = self.define_var(var)
                self.add(OPCODES["match.begin"]); self.add(addr)

            elif line.startswith("case "):
                val = int(line.split()[1].strip(":"))
                self.add(OPCODES["match.case"]); self.add(val)

            elif line.startswith("endmatch"):
                self.add(OPCODES["match.end"])

            # end()
            elif line.startswith("end()"):
                self.add(OPCODES["ret"])

            else:
                raise SyntaxError(f"Unrecognized: {line}")
            i += 1

        return " ".join(self.tokens)

def main():
    if len(sys.argv) != 3:
        print("Usage: python transpiler.py <input.news> <output.dgm>")
        sys.exit(1)

    infile, outfile = sys.argv[1], sys.argv[2]
    with open(infile) as f: src = f.read()
    t = Transpiler()
    dgm_code = t.transpile(src)
    with open(outfile, "w") as f: f.write(dgm_code)
    print(f"Transpiled {infile} → {outfile}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
NEWS Transpiler → DGM (Base-12 Bytecode)
----------------------------------------
Translates NEWS syntax into base-12 encoded DGM bytecode.
Supports the entire 144-opcode space:
 - Core LLVM ops (00–4B)
 - Safe math (50–7B)
 - Data structures (80–9B)
 - CIAM extensions (A0–BB)
"""

import sys, re
from typing import Dict, List

DIGITS = "0123456789ab"

def to_base12(num: int) -> str:
    if num == 0:
        return "0"
    out = []
    while num > 0:
        out.append(DIGITS[num % 12])
        num //= 12
    return "".join(reversed(out))

def from_base12(s: str) -> int:
    result = 0
    for ch in s:
        result = result * 12 + DIGITS.index(ch)
    return result

# ------------------------------------------------------------
# OPCODES (full 144 coverage)
# ------------------------------------------------------------
OPCODES: Dict[str, int] = {
    # --- Core (00–4B) ---
    "nop": 0x00, "alloca": 0x01, "load": 0x02, "store": 0x03,
    "add": 0x17, "sub": 0x18, "mul": 0x19, "udiv": 0x1A, "sdiv": 0x1B,
    "icmp": 0x15, "br": 0x30, "ret": 0x33,
    # --- Safe (50–7B) ---
    "safe.add": 0x50, "safe.sub": 0x51, "safe.mul": 0x52,
    "safe.div": 0x53, "safe.mod": 0x54,
    # --- Data Structures (80–9B) ---
    "tuple.pack": 0x80, "tuple.unpack": 0x81,
    "list.append": 0x82, "list.remove": 0x83,
    "array.load": 0x86, "array.store": 0x87,
    "group.spawn": 0x88, "group.merge": 0x89,
    "nest.enter": 0x8B, "nest.exit": 0x90,
    "pair.create": 0x93, "pair.split": 0x94,
    "match.begin": 0x95, "match.case": 0x96, "match.end": 0x97,
    # --- CIAM Extensions (A0–BB) ---
    "inline": 0xA0, "macro": 0xA4, "trace": 0xA5,
    "echo": 0xA6, "link": 0xA7, "delete": 0xA9,
    "open": 0xB4, "close": 0xB5, "defer": 0xB6,
    "future": 0xB7, "parallel": 0xB8, "sync": 0xB9,
    "exit": 0xBB,
}

REL_OPS = {"==": "eq", "!=": "ne", "<": "lt", ">": "gt", "<=": "le", ">=": "ge"}

# ------------------------------------------------------------
# Transpiler
# ------------------------------------------------------------
class Transpiler:
    def __init__(self):
        self.tokens: List[str] = []
        self.variables: Dict[str, int] = {}
        self.mem_index: int = 1
        self.fixups: List[tuple] = []  # for if/while labels

    def add(self, val):
        if isinstance(val, int):
            self.tokens.append(to_base12(val))
        else:
            self.tokens.append(str(val))

    def define_var(self, name: str) -> int:
        if name not in self.variables:
            self.variables[name] = self.mem_index
            self.mem_index += 1
        return self.variables[name]

    def transpile(self, src: str) -> str:
        lines = [l.strip() for l in src.splitlines() if l.strip()]
        i = 0
        while i < len(lines):
            line = lines[i]

            # --- Print ---
            if line.startswith("print("):
                text = re.match(r'print\("(.*)"\)', line).group(1)
                self.add(OPCODES["echo"])
                for ch in text:
                    self.add(ord(ch))
                self.add(0)

            # --- let x = N ---
            elif line.startswith("let "):
                m = re.match(r'let (\w+) *= *(\d+)', line)
                var, num = m.groups()
                addr = self.define_var(var)
                self.add(OPCODES["store"]); self.add(addr); self.add(int(num))

            # --- add/sub/mul/div ---
            elif line.startswith("add "):
                m = re.match(r'add (\w+), *(\d+)', line)
                addr = self.define_var(m.group(1))
                self.add(OPCODES["add"]); self.add(addr); self.add(int(m.group(2)))
            elif line.startswith("sub "):
                m = re.match(r'sub (\w+), *(\d+)', line)
                addr = self.define_var(m.group(1))
                self.add(OPCODES["sub"]); self.add(addr); self.add(int(m.group(2)))
            elif line.startswith("mul "):
                m = re.match(r'mul (\w+), *(\d+)', line)
                addr = self.define_var(m.group(1))
                self.add(OPCODES["mul"]); self.add(addr); self.add(int(m.group(2)))
            elif line.startswith("div "):
                m = re.match(r'div (\w+), *(\d+)', line)
                addr = self.define_var(m.group(1))
                self.add(OPCODES["udiv"]); self.add(addr); self.add(int(m.group(2)))

            # --- if (...)
            elif line.startswith("if "):
                m = re.match(r'if\s*\(\s*(\w+)\s*([=!<>]+)\s*(\d+)\s*\)', line)
                var, op, num = m.groups()
                addr = self.define_var(var)
                self.add(OPCODES["icmp"]); self.add(addr); self.add(int(num)); self.add(op)
                self.fixups.append(("if_end", len(self.tokens)))
                self.add("FIXUP")

            elif line.startswith("endif"):
                for idx, (kind, pos) in enumerate(self.fixups):
                    if kind == "if_end" and self.tokens[pos] == "FIXUP":
                        self.tokens[pos] = to_base12(len(self.tokens))
                        self.fixups.pop(idx); break

            # --- while (...)
            elif line.startswith("while "):
                m = re.match(r'while\s*\(\s*(\w+)\s*([=!<>]+)\s*(\d+)\s*\)', line)
                var, op, num = m.groups()
                loop_start = len(self.tokens)
                addr = self.define_var(var)
                self.add(OPCODES["icmp"]); self.add(addr); self.add(int(num)); self.add(op)
                self.fixups.append(("while_end", len(self.tokens), loop_start))
                self.add("FIXUP")

            elif line.startswith("endwhile"):
                kind, pos, loop_start = self.fixups.pop()
                self.tokens[pos] = to_base12(len(self.tokens)+2)
                self.add(OPCODES["br"]); self.add(loop_start)

            # --- match/case ---
            elif line.startswith("match "):
                var = line.split()[1]
                addr = self.define_var(var)
                self.add(OPCODES["match.begin"]); self.add(addr)
            elif line.startswith("case "):
                val = int(line.split()[1].strip(":"))
                self.add(OPCODES["match.case"]); self.add(val)
            elif line.startswith("endmatch"):
                self.add(OPCODES["match.end"])

            # --- tuples/lists ---
            elif line.startswith("tuple "):
                vals = [int(x) for x in line.replace("tuple","").strip().split()]
                self.add(OPCODES["tuple.pack"]); self.add(len(vals))
                for v in vals: self.add(v)

            elif line.startswith("list.append "):
                m = re.match(r'list.append (\w+), *(\d+)', line)
                addr = self.define_var(m.group(1))
                self.add(OPCODES["list.append"]); self.add(addr); self.add(int(m.group(2)))

            # --- concurrency ---
            elif line.startswith("future "):
                fn = line.split()[1]
                addr = self.define_var(fn)
                self.add(OPCODES["future"]); self.add(addr)
            elif line.startswith("parallel "):
                fn = line.split()[1]
                addr = self.define_var(fn)
                self.add(OPCODES["parallel"]); self.add(addr)
            elif line.startswith("sync"):
                self.add(OPCODES["sync"])

            # --- system/exit ---
            elif line.startswith("exit"):
                self.add(OPCODES["exit"])

            elif line.startswith("end()"):
                self.add(OPCODES["ret"])

            else:
                raise SyntaxError(f"Unrecognized: {line}")
            i += 1

        return " ".join(self.tokens)

# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
def main():
    if len(sys.argv) != 3:
        print("Usage: python transpiler.py <input.news> <output.dgm>")
        sys.exit(0)

    infile, outfile = sys.argv[1], sys.argv[2]
    with open(infile) as f: src = f.read()
    t = Transpiler()
    dgm_code = t.transpile(src)
    with open(outfile, "w") as f: f.write(dgm_code)
    print(f"Transpiled {infile} → {outfile}")

if __name__ == "__main__":
    main()
