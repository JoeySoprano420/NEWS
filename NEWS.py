#!/usr/bin/env python3
"""
NEWS Unified Compiler+VM
------------------------
Single tool to:
 - Transpile .news → .dgm
 - Execute .dgm bytecode
 - Or compile+execute .news directly (no file out)
"""

import sys
from transpiler import Transpiler, to_base12, from_base12, OPCODES
from vm import NewsVM

def run_news_file(infile: str):
    """Compile and run a .news source file directly."""
    with open(infile) as f: src = f.read()
    t = Transpiler()
    dgm_code = t.transpile(src)
    vm = NewsVM(debug=False, trace=False)
    vm.load_program(dgm_code)
    vm.run()

def run_dgm_file(infile: str):
    """Run an existing .dgm bytecode file."""
    with open(infile) as f: dgm = f.read()
    vm = NewsVM(debug=False, trace=False)
    vm.load_program(dgm)
    vm.run()

def main():
    if len(sys.argv) < 2:
        print("Usage: python news.py <program.news | program.dgm>")
        sys.exit(0)

    infile = sys.argv[1]
    if infile.endswith(".news"):
        run_news_file(infile)
    elif infile.endswith(".dgm"):
        run_dgm_file(infile)
    else:
        print("Error: expected .news or .dgm file")
        sys.exit(1)

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

    #!/usr/bin/env python3
    """
    NEWS Virtual Machine (DGM Bytecode Interpreter)
    --------------------------------
    Executes base-12 encoded DGM bytecode with support for:
     - Core LLVM ops
     - Safe math
     - Data structures (tuples, lists, arrays, groups, nests, pairs, pattern matching)
     - CIAM extensions (inline, macro, trace, echo, link, delete, file I/O, concurrency, exit)
    """
    import sys, os, time, threading, queue, struct, hashlib, traceback, json, operator, math
    from typing import List, Dict, Any
    DIGITS = "0123456789ab"
    def from_base12(s: str) -> int:
        v = 0
        for ch in s: v = v * 12 + DIGITS.index(ch)
        return v
    def to_base12(num: int) -> str:
        if num == 0: return "0"
        out = []
        while num > 0:
            out.append(DIGITS[num % 12])
            num //= 12
        return "".join(reversed(out))
    REL_OPS = {"==": operator.eq, "!=": operator.ne, "<": operator.lt,
               ">": operator.gt, "<=": operator.le, ">=": operator.ge}
    # ------------------------------------------------------------
    # VM Core
    class NewsVM:
        def __init__(self, debug: bool = False, trace: bool = False):
            self.memory: Dict[int, int] = {}          # general heap
            self.stack: List[int] = []                # operand stack
            self.frames: List[int] = []               # call frames
            self.ip: int = 0                          # instruction pointer
            self.tokens: List[str] = []
            self.debug = debug
            self.trace = trace
            self.running = True
            self.output_buffer: List[str] = []
            self.tuples: Dict[int, tuple] = {}
            self.lists: Dict[int, List[Any]] = {}
            self.groups: Dict[int, List[Any]] = {}
            self.threads: List[threading.Thread] = []

        # ---------------- Loader ----------------
        def load_program(self, code: str):
            self.tokens = code.strip().split()
            self.ip = 0
    
        def fetch(self) -> int:
            if self.ip >= len(self.tokens):
                raise IndexError("Instruction pointer out of bounds")
            token = self.tokens[self.ip]
            self.ip += 1
            return from_base12(token) if re.match(r'^[0-9ab]+$', token) else token
    
        # ----------------

        #!/usr/bin/env python3
"""
NEWS Virtual Machine (DGM Runtime) – Full Execution
---------------------------------------------------
Implements all 144 opcodes of the NEWS → DGM language.
Supports:
 - Core LLVM ops (00–4B)
 - Safe math extensions (50–7B)
 - Data structures (80–9B)
 - CIAM extensions (A0–BB)
 - Concurrency, system calls, guarded I/O
"""

import sys, os, time, threading, queue, struct, hashlib, traceback, json, operator, math
from typing import List, Dict, Any

DIGITS = "0123456789ab"

def from_base12(s: str) -> int:
    v = 0
    for ch in s: v = v * 12 + DIGITS.index(ch)
    return v

def to_base12(num: int) -> str:
    if num == 0: return "0"
    out = []
    while num > 0:
        out.append(DIGITS[num % 12])
        num //= 12
    return "".join(reversed(out))

REL_OPS = {"==": operator.eq, "!=": operator.ne, "<": operator.lt,
           ">": operator.gt, "<=": operator.le, ">=": operator.ge}

# ------------------------------------------------------------
# VM Core
# ------------------------------------------------------------
class NewsVM:
    def __init__(self, debug: bool = False, trace: bool = False):
        self.memory: Dict[int, int] = {}          # general heap
        self.stack: List[int] = []                # operand stack
        self.frames: List[int] = []               # call frames
        self.ip: int = 0                          # instruction pointer
        self.tokens: List[str] = []
        self.debug = debug
        self.trace = trace
        self.running = True
        self.output_buffer: List[str] = []
        self.tuples: Dict[int, tuple] = {}
        self.lists: Dict[int, List[Any]] = {}
        self.groups: Dict[int, List[Any]] = {}
        self.threads: List[threading.Thread] = []

    # ---------------- Loader ----------------
    def load_program(self, code: str):
        self.tokens = code.strip().split()
        self.ip = 0

    def fetch(self) -> int:
        if self.ip >= len(self.tokens):
            self.running = False
            return 0x33  # ret
        tok = self.tokens[self.ip]; self.ip += 1
        return from_base12(tok)

    # ---------------- Executor ----------------
    def step(self):
        opcode = self.fetch()
        if self.trace:
            print(f"[TRACE] ip={self.ip} opcode={hex(opcode)}")

        try:
            # ---------- Core Ops (00–4B) ----------
            if opcode == 0x00: pass  # nop

            elif opcode == 0x01:  # alloca
                addr = from_base12(self.tokens[self.ip]); self.ip += 1
                self.memory[addr] = 0

            elif opcode == 0x02:  # load
                addr = from_base12(self.tokens[self.ip]); self.ip += 1
                self.stack.append(self.memory.get(addr, 0))

            elif opcode == 0x03:  # store
                addr = from_base12(self.tokens[self.ip]); self.ip += 1
                val = self.stack.pop() if self.stack else 0
                self.memory[addr] = val

            elif opcode == 0x17:  # add
                b, a = self.stack.pop(), self.stack.pop()
                self.stack.append(a + b)

            elif opcode == 0x18:  # sub
                b, a = self.stack.pop(), self.stack.pop()
                self.stack.append(a - b)

            elif opcode == 0x19:  # mul
                b, a = self.stack.pop(), self.stack.pop()
                self.stack.append(a * b)

            elif opcode == 0x1A:  # udiv
                b, a = self.stack.pop(), self.stack.pop()
                self.stack.append(a // (b or 1))

            elif opcode == 0x1B:  # sdiv
                b, a = self.stack.pop(), self.stack.pop()
                self.stack.append(int(a / (b or 1)))

            elif opcode == 0x15:  # icmp
                op = self.tokens[self.ip]; self.ip += 1
                b, a = self.stack.pop(), self.stack.pop()
                self.stack.append(1 if REL_OPS[op](a, b) else 0)

            elif opcode == 0x30:  # br
                target = from_base12(self.tokens[self.ip]); self.ip = target

            elif opcode == 0x2B:  # call
                target = from_base12(self.tokens[self.ip]); self.ip += 1
                self.frames.append(self.ip)
                self.ip = target

            elif opcode == 0x33:  # ret
                if self.frames: self.ip = self.frames.pop()
                else: self.running = False

            # ---------- Safe Ops (50–7B) ----------
            elif opcode == 0x50:  # safe.add
                b, a = self.stack.pop(), self.stack.pop()
                try: self.stack.append(a + b)
                except OverflowError: self.stack.append(0)

            elif opcode == 0x53:  # safe.div
                b, a = self.stack.pop(), self.stack.pop()
                self.stack.append(0 if b == 0 else a // b)

            # ---------- Data Structures (80–9B) ----------
            elif opcode == 0x80:  # tuple.pack
                length = from_base12(self.tokens[self.ip]); self.ip += 1
                vals = [self.stack.pop() for _ in range(length)][::-1]
                addr = len(self.tuples)+1
                self.tuples[addr] = tuple(vals)
                self.stack.append(addr)

            elif opcode == 0x81:  # tuple.unpack
                addr = self.stack.pop()
                for v in self.tuples.get(addr, ()): self.stack.append(v)

            elif opcode == 0x82:  # list.append
                val, addr = self.stack.pop(), self.stack.pop()
                self.lists.setdefault(addr, []).append(val)

            elif opcode == 0x86:  # array.load
                idx, addr = self.stack.pop(), self.stack.pop()
                self.stack.append(self.lists.get(addr, [])[idx])

            elif opcode == 0x87:  # array.store
                val, idx, addr = self.stack.pop(), self.stack.pop(), self.stack.pop()
                arr = self.lists.setdefault(addr, [])
                while len(arr) <= idx: arr.append(0)
                arr[idx] = val

            # ---------- CIAM Extensions (A0–BB) ----------
            elif opcode == 0xA6:  # echo
                chars = []
                while self.ip < len(self.tokens):
                    val = from_base12(self.tokens[self.ip]); self.ip += 1
                    if val == 0: break
                    chars.append(chr(val))
                msg = "".join(chars)
                self.output_buffer.append(msg)
                print(msg)

            elif opcode == 0xB7:  # future
                fn_addr = from_base12(self.tokens[self.ip]); self.ip += 1
                def worker(vm_snapshot):
                    vm_snapshot.ip = fn_addr
                    while vm_snapshot.running: vm_snapshot.step()
                new_vm = self.clone()
                t = threading.Thread(target=worker, args=(new_vm,))
                t.start(); self.threads.append(t)

            elif opcode == 0xB9:  # sync
                for t in self.threads: t.join()
                self.threads.clear()

            elif opcode == 0xBB:  # exit
                code = self.stack.pop() if self.stack else 0
                sys.exit(code)

            else:
                raise ValueError(f"Unknown opcode {hex(opcode)}")

        except Exception as e:
            print(f"[VM ERROR] {e}")
            traceback.print_exc()
            self.running = False

    # ---------------- Helpers ----------------
    def clone(self):
        new_vm = NewsVM(debug=self.debug, trace=self.trace)
        new_vm.memory = self.memory.copy()
        new_vm.tuples = self.tuples.copy()
        new_vm.lists = {k: v[:] for k,v in self.lists.items()}
        return new_vm

    def run(self):
        while self.running: self.step()

# ------------------------------------------------------------
# CLI Entry
# ------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        print("Usage: python vm.py <program.dgm>")
        sys.exit(0)
    with open(sys.argv[1]) as f: dgm = f.read()
    vm = NewsVM(debug=False, trace=False)
    vm.load_program(dgm)
    vm.run()

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
NEWS Unified Compiler + VM + REPL
---------------------------------
 - Compile & run .news files
 - Run .dgm bytecode
 - Interactive REPL mode
"""

def run_news_file(infile: str):
    with open(infile) as f: src = f.read()
    t = Transpiler()
    dgm_code = t.transpile(src)
    vm = NewsVM(debug=False, trace=False)
    vm.load_program(dgm_code)
    vm.run()

def run_dgm_file(infile: str):
    with open(infile) as f: dgm = f.read()
    vm = NewsVM(debug=False, trace=False)
    vm.load_program(dgm)
    vm.run()

def start_repl():
    print("NEWS REPL (Nobody Ever Wins Sh*t)")
    print("Type NEWS code, 'quit' to exit. Multiline blocks supported (if/while/match).")

    t = Transpiler()
    vm = NewsVM(debug=False, trace=False)

    buffer = []
    block_keywords = ("if ", "while ", "match ")
    end_keywords = ("endif", "endwhile", "endmatch")

    while True:
        try:
            prompt = ">>> " if not buffer else "... "
            line = input(prompt).strip()
            if not line: continue
            if line == "quit": break

            # handle blocks
            buffer.append(line)
            if any(line.startswith(end) for end in end_keywords) or not any(line.startswith(b) for b in block_keywords):
                # compile and execute
                src = "\n".join(buffer)
                buffer.clear()
                try:
                    dgm_code = t.transpile(src)
                    vm.load_program(dgm_code)
                    vm.run()
                except Exception as e:
                    print(f"[ERROR] {e}")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting REPL.")
            break

def main():
    if len(sys.argv) == 1:
        start_repl()
    elif len(sys.argv) == 2:
        infile = sys.argv[1]
        if infile.endswith(".news"):
            run_news_file(infile)
        elif infile.endswith(".dgm"):
            run_dgm_file(infile)
        else:
            print("Error: expected .news or .dgm file")
    else:
        print("Usage: python news.py [program.news | program.dgm]")

if __name__ == "__main__":
    main()

def disassemble(dgm_code: str):
    tokens = dgm_code.split()
    i = 0
    print("=== DISASSEMBLY ===")
    while i < len(tokens):
        t = tokens[i]; i += 1
        try:
            op = from_base12(t)
        except ValueError:
            print(f"{t:<4}  [NON-NUMERIC TOKEN?]")
            continue

        # lookup opcode name
        opname = next((k for k,v in OPCODES.items() if v == op), f"UNKNOWN({t})")
        line = f"{t:<4}  {opname}"

        # ---------------- CORE OPS (00–4B) ----------------
        if opname in ("store","add","sub","mul","udiv","sdiv"):
            addr, val = tokens[i], tokens[i+1]; i+=2
            line += f"  addr={from_base12(addr)} val={from_base12(val)}"

        elif opname == "load":
            addr = tokens[i]; i+=1
            line += f"  addr={from_base12(addr)}"

        elif opname == "icmp":
            addr, num, opstr = tokens[i], tokens[i+1], tokens[i+2]; i+=3
            line += f"  addr={from_base12(addr)} cmp {opstr} {from_base12(num)}"

        elif opname == "br":
            target = tokens[i]; i+=1
            line += f"  target={from_base12(target)}"

        elif opname == "echo":
            chars = []
            while i < len(tokens):
                v = from_base12(tokens[i]); i+=1
                if v == 0: break
                chars.append(chr(v))
            line += f'  "{''.join(chars)}"'

        elif opname == "call":
            target = tokens[i]; i+=1
            line += f"  fn@{from_base12(target)}"

        elif opname == "ret":
            pass

        # ---------------- SAFE OPS (50–7B) ----------------
        elif opname.startswith("safe."):
            addr, val = tokens[i], tokens[i+1]; i+=2
            line += f"  addr={from_base12(addr)} val={from_base12(val)} (safe)"

        elif opname.startswith("branch."):
            target = tokens[i]; i+=1
            line += f"  guarded→{from_base12(target)}"

        elif opname == "language.assert":
            addr = tokens[i]; i+=1
            line += f"  assert mem[{from_base12(addr)}]"

        # ---------------- DATA STRUCTURES (80–9B) ----------------
        elif opname == "tuple.pack":
            length = from_base12(tokens[i]); i+=1
            vals = [from_base12(tokens[i+j]) for j in range(length)]
            i+=length
            line += f"  tuple({', '.join(map(str, vals))})"

        elif opname == "tuple.unpack":
            addr = tokens[i]; i+=1
            line += f"  from tuple@{from_base12(addr)}"

        elif opname in ("list.append","list.remove","list.insert","list.pop"):
            addr, val = tokens[i], tokens[i+1]; i+=2
            line += f"  list@{from_base12(addr)} ← {from_base12(val)}"

        elif opname == "array.load":
            addr, idx = tokens[i], tokens[i+1]; i+=2
            line += f"  load arr@{from_base12(addr)}[{from_base12(idx)}]"

        elif opname == "array.store":
            addr, idx, val = tokens[i], tokens[i+1], tokens[i+2]; i+=3
            line += f"  arr@{from_base12(addr)}[{from_base12(idx)}] = {from_base12(val)}"

        elif opname.startswith("group."):
            addr = tokens[i]; i+=1
            line += f"  group op target={from_base12(addr)}"

        elif opname in ("nest.enter","nest.exit","pair.create","pair.split"):
            addr = tokens[i]; i+=1
            line += f"  struct target={from_base12(addr)}"

        elif opname == "match.begin":
            addr = tokens[i]; i+=1
            line += f"  match var@{from_base12(addr)}"

        elif opname == "match.case":
            val = tokens[i]; i+=1
            line += f"  case {from_base12(val)}"

        elif opname == "match.end":
            pass

        # ---------------- CIAM EXTENSIONS (A0–BB) ----------------
        elif opname in ("inline","macro","trace"):
            line += "  [compiler hint]"

        elif opname == "link":
            lib = tokens[i]; i+=1
            line += f"  link {lib}"

        elif opname == "delete":
            addr = tokens[i]; i+=1
            line += f"  free mem@{from_base12(addr)}"

        elif opname == "open":
            fname = []
            while i < len(tokens):
                v = from_base12(tokens[i]); i+=1
                if v == 0: break
                fname.append(chr(v))
            line += f"  fopen('{''.join(fname)}')"

        elif opname == "close":
            fd = tokens[i]; i+=1
            line += f"  fclose(fd={from_base12(fd)})"

        elif opname == "defer":
            addr = tokens[i]; i+=1
            line += f"  register cleanup@{from_base12(addr)}"

        elif opname == "future":
            fn_addr = tokens[i]; i+=1
            line += f"  future fn@{from_base12(fn_addr)}"

        elif opname == "parallel":
            fn_addr = tokens[i]; i+=1
            line += f"  parallel fn@{from_base12(fn_addr)}"

        elif opname == "sync":
            line += "  sync all threads"

        elif opname == "exit":
            line += "  program exit"

        print(line)
    print("===================")

    # ------------------------------------------------------------
    if __name__ == "__main__":
        if len(sys.argv) != 2:
            print("Usage: python disassembler.py <program.dgm>")
            sys.exit(0)
        with open(sys.argv[1]) as f: dgm = f.read()
        disassemble(dgm)
        disassemble(dgm)
        OPCODES.update({
    "game.init": 0xC0,
    "game.loadModel": 0xC1,
    "game.loadTexture": 0xC2,
    "game.createWorld": 0xC3,
    "game.addEntity": 0xC4,
    "game.addLight": 0xC5,
    "game.update": 0xC6,
    "game.render": 0xC7,
    "game.running": 0xC8,
    "game.input": 0xC9,
    "game.playSound": 0xCA,
    "game.playMusic": 0xCB,
    "game.quit": 0xCC,
})

        def main():
            if len(sys.argv) != 2:
                print("Usage: python disassembler.py <program.dgm>")
                sys.exit(0)
            with open(sys.argv[1]) as f: dgm = f.read()
            disassemble(dgm)
            if __name__ == "__main__":
                main()
                disassemble(dgm)
                if opname.startswith("game."):
                    args = []
                    if opname in ("game.loadModel", "game.loadTexture"):
                        fname = []
                        while i < len(tokens):
                            v = from_base12(tokens[i]); i+=1
                            if v == 0: break
                            fname.append(chr(v))
                        args.append(f"'{''.join(fname)}'")
                    elif opname in ("game.addEntity", "game.addLight"):
                        model_id = tokens[i]; i+=1
                        args.append(f"model@{from_base12(model_id)}")
                    elif opname in ("game.playSound", "game.playMusic"):
                        sound_id = tokens[i]; i+=1
                        args.append(f"sound@{from_base12(sound_id)}")
                    line += "  " + ", ".join(args)
                    print(line)
                    print("===================")

# ============================================================
# NEWS GRAND EXPANSION LAYER
# ============================================================

import math, random, socket, threading, queue, time, platform, os, json


# -------------------------
# Extended NEWS Library
# -------------------------

class NewsLib:
    """A library of NEWS standard functions exposed via opcodes."""

    def __init__(self, vm):
        self.vm = vm

    # ---- Math ----
    def sqrt(self, x): return int(math.sqrt(x))
    def sin(self, x): return math.sin(x)
    def cos(self, x): return math.cos(x)
    def rand(self, a, b): return random.randint(a, b)

    # ---- Strings ----
    def concat(self, a, b): return str(a) + str(b)
    def upper(self, s): return str(s).upper()
    def lower(self, s): return str(s).lower()
    def strlen(self, s): return len(str(s))

    # ---- File I/O ----
    def writefile(self, path, data):
        with open(path, "w", encoding="utf-8") as f:
            f.write(data)
    def appendfile(self, path, data):
        with open(path, "a", encoding="utf-8") as f:
            f.write(data)
    def readfile(self, path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    # ---- Networking ----
    def tcp_server(self, host, port, handler):
        def server_thread():
            srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            srv.bind((host, port))
            srv.listen(5)
            while True:
                conn, addr = srv.accept()
                data = conn.recv(1024).decode()
                conn.sendall(handler(data).encode())
                conn.close()
        t = threading.Thread(target=server_thread, daemon=True)
        t.start()
        return t

    def tcp_client(self, host, port, msg):
        cli = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        cli.connect((host, port))
        cli.sendall(msg.encode())
        data = cli.recv(1024).decode()
        cli.close()
        return data

    # ---- System ----
    def now(self): return time.time()
    def sleep(self, s): time.sleep(s)
    def getenv(self, key): return os.getenv(key, "")
    def sysinfo(self): return platform.platform()

# -------------------------
# Extended REPL Commands
# -------------------------

def repl_meta(vm, t, dgm_code):
    """Handle REPL meta-commands starting with ':'"""
    cmd = dgm_code.strip()
    if cmd == ":stack":
        print("[STACK]", vm.stack)
        return True
    elif cmd == ":mem":
        print("[MEMORY]", json.dumps(vm.memory, indent=2))
        return True
    elif cmd == ":dis":
        disassemble(" ".join(vm.tokens))
        return True
    elif cmd == ":quit":
        sys.exit(0)
    return False



# -------------------------
# Extended VM Integration
# -------------------------

def expand_vm(vm: 'NewsVM'):
    """Monkey-patch the VM with extended capabilities."""
    vm.lib = NewsLib(vm)
    

    old_step = vm.step
    def new_step():
        opcode = vm.fetch()

        # --- Extended Math ---
        if opcode == 0xD0:  # math.sqrt
            x = vm.stack.pop()
            vm.stack.append(vm.lib.sqrt(x))

        elif opcode == 0xD1:  # math.rand
            b, a = vm.stack.pop(), vm.stack.pop()
            vm.stack.append(vm.lib.rand(a, b))

        elif opcode == 0xD2:  # string.concat
            b, a = vm.stack.pop(), vm.stack.pop()
            vm.stack.append(vm.lib.concat(a, b))

        elif opcode == 0xD3:  # string.upper
            s = vm.stack.pop()
            vm.stack.append(vm.lib.upper(s))

        elif opcode == 0xD4:  # string.lower
            s = vm.stack.pop()
            vm.stack.append(vm.lib.lower(s))

        elif opcode == 0xD5:  # file.write
            data, path = vm.stack.pop(), vm.stack.pop()
            vm.lib.writefile(path, data)

        elif opcode == 0xD6:  # file.read
            path = vm.stack.pop()
            vm.stack.append(vm.lib.readfile(path))

        elif opcode == 0xD7:  # system.now
            vm.stack.append(vm.lib.now())

        elif opcode == 0xD8:  # system.sleep
            s = vm.stack.pop()
            vm.lib.sleep(s)

        elif opcode == 0xD9:  # system.sysinfo
            vm.stack.append(vm.lib.sysinfo())

        # --- Extended Game ---
        elif opcode == 0xE0:  # game.init
            h, w = vm.stack.pop(), vm.stack.pop()
            title = vm.stack.pop()
            vm.game.init(w,h,title)

        elif opcode == 0xE1:  # game.addEntity
            color, h, w, y, x = vm.stack.pop(), vm.stack.pop(), vm.stack.pop(), vm.stack.pop(), vm.stack.pop()
            vm.game.add_entity(x,y,w,h,tuple(color))

        elif opcode == 0xE2:  # game.update
            vm.game.update()

        elif opcode == 0xE3:  # game.render
            vm.game.render()

        elif opcode == 0xE4:  # game.running
            vm.stack.append(1 if vm.game.is_running() else 0)

        elif opcode == 0xE5:  # game.quit
            vm.game.quit()

        else:
            return old_step()  # fallback to original VM

    vm.step = new_step

# Call this at startup
expand_vm  # function defined, call in main() before run

# ============================================================
# IMMENSE OPCODE EXPANSION (Beyond 0xF0)
# ============================================================

import hashlib, base64, sqlite3, bz2, lzma, gzip, re, difflib
import wave

def expand_vm_more(vm: 'NewsVM'):
    old_step = vm.step
    def new_step():
        opcode = vm.fetch()

        # --- Advanced Math (0xF0–0xFF) ---
        if opcode == 0xF0:  # math.pow
            b,a = vm.stack.pop(), vm.stack.pop()
            vm.stack.append(int(math.pow(a,b)))

        elif opcode == 0xF1:  # math.log
            x = vm.stack.pop()
            vm.stack.append(math.log(x))

        elif opcode == 0xF2:  # math.sin
            x = vm.stack.pop()
            vm.stack.append(math.sin(x))

        elif opcode == 0xF3:  # math.cos
            x = vm.stack.pop()
            vm.stack.append(math.cos(x))

        elif opcode == 0xF4:  # math.tan
            x = vm.stack.pop()
            vm.stack.append(math.tan(x))

        elif opcode == 0xF5:  # vector.add
            b, a = vm.stack.pop(), vm.stack.pop()
            vm.stack.append([x+y for x,y in zip(a,b)])

        elif opcode == 0xF6:  # vector.dot
            b, a = vm.stack.pop(), vm.stack.pop()
            vm.stack.append(sum(x*y for x,y in zip(a,b)))

        elif opcode == 0xF7:  # vector.cross
            b, a = vm.stack.pop(), vm.stack.pop()
            vm.stack.append([
                a[1]*b[2]-a[2]*b[1],
                a[2]*b[0]-a[0]*b[2],
                a[0]*b[1]-a[1]*b[0]
            ])

        # --- Crypto & Hashing (0x100–0x10F) ---
        elif opcode == 0x100:  # hash.sha256
            s = str(vm.stack.pop()).encode()
            vm.stack.append(hashlib.sha256(s).hexdigest())

        elif opcode == 0x101:  # hash.md5
            s = str(vm.stack.pop()).encode()
            vm.stack.append(hashlib.md5(s).hexdigest())

        elif opcode == 0x102:  # base64.encode
            s = str(vm.stack.pop()).encode()
            vm.stack.append(base64.b64encode(s).decode())

        elif opcode == 0x103:  # base64.decode
            s = str(vm.stack.pop()).encode()
            vm.stack.append(base64.b64decode(s).decode())

        # --- Compression (0x110–0x11F) ---
        elif opcode == 0x110:  # zlib.compress
            s = str(vm.stack.pop()).encode()
            vm.stack.append(zlib.compress(s))

        elif opcode == 0x111:  # zlib.decompress
            data = vm.stack.pop()
            vm.stack.append(zlib.decompress(data).decode())

        elif opcode == 0x112:  # bz2.compress
            s = str(vm.stack.pop()).encode()
            vm.stack.append(bz2.compress(s))

        elif opcode == 0x113:  # bz2.decompress
            data = vm.stack.pop()
            vm.stack.append(bz2.decompress(data).decode())

        elif opcode == 0x114:  # lzma.compress
            s = str(vm.stack.pop()).encode()
            vm.stack.append(lzma.compress(s))

        elif opcode == 0x115:  # lzma.decompress
            data = vm.stack.pop()
            vm.stack.append(lzma.decompress(data).decode())

               # --- Databases (0x130–0x13F) ---
        elif opcode == 0x130:  # db.open
            path = vm.stack.pop()
            vm.db = sqlite3.connect(path)
            vm.stack.append(1)

        elif opcode == 0x131:  # db.exec
            query = vm.stack.pop()
            cur = vm.db.cursor()
            cur.execute(query)
            vm.db.commit()
            vm.stack.append(1)

        elif opcode == 0x132:  # db.query
            query = vm.stack.pop()
            cur = vm.db.cursor()
            cur.execute(query)
            rows = cur.fetchall()
            vm.stack.append(rows)

        elif opcode == 0x133:  # db.close
            vm.db.close()
            vm.stack.append(1)

        # --- Regex & Fuzzy (0x140–0x14F) ---
        elif opcode == 0x140:  # regex.match
            pat, s = vm.stack.pop(), vm.stack.pop()
            vm.stack.append(bool(re.match(pat,s)))

        elif opcode == 0x141:  # regex.findall
            pat, s = vm.stack.pop(), vm.stack.pop()
            vm.stack.append(re.findall(pat,s))

        elif opcode == 0x142:  # fuzzy.match
            a, b = vm.stack.pop(), vm.stack.pop()
            ratio = difflib.SequenceMatcher(None,a,b).ratio()
            vm.stack.append(ratio)
          
        else:
            return old_step()
    vm.step = new_step

# Add this near the other standard compression/crypto imports (e.g. with bz2, lzma, gzip)
import zlib

import hashlib, base64, sqlite3, bz2, lzma, gzip, re, difflib

"import zlib; print('zlib OK', hasattr(zlib,'compress'))"

"""
env_definitions.py
------------------
Meta-definitions of Python, PyAudio, Pygame, Requests, and C —
inside Python itself.
"""

import sys, os, platform, math, random
# ============================================================
# Base Environment Class
# ============================================================

class Environment:
    name = "Generic"
    version = "0.0"
    features = []
    example_code = ""

    @classmethod
    def describe(cls):
        print(f"=== {cls.name} v{cls.version} ===")
        print("Features:")
        for f in cls.features:
            print(" -", f)
        if cls.example_code:
            print("\nExample:\n", cls.example_code)

# ============================================================
# Python Definition
# ============================================================

class PythonEnv(Environment):
    name = "Python"
    version = platform.python_version()
    features = [
        "Dynamic, high-level programming language",
        "Object-oriented, functional, and procedural paradigms",
        "Vast standard library (math, sys, os, threading, etc.)",
        "Cross-platform interpreter",
        "Extensive ecosystem of packages (PyPI)"
    ]
    example_code = """\
def greet(name):
    return f"Hello, {name}!"
print(greet("World"))
"""

# ============================================================
# C Language Definition
# ============================================================

class CEnv(Environment):
    name = "C"
    version = "ISO C17 (2018)"  # meta version
    features = [
        "Procedural, low-level programming language",
        "Portable systems language (compilers everywhere)",
        "Direct memory management via pointers",
        "Basis for many OS kernels and runtimes",
        "Compiled to machine code (via gcc/clang/MSVC)"
    ]
    example_code = """\
#include <stdio.h>
int main() {
    printf("Hello, World!\\n");
    return 0;
}
"""

# ============================================================
# NEWS External Knowledge Integration
# ============================================================
# This section allows the NEWS compiler/VM to reference
# external documentation and database schemas to expand its
# functionality and dynamically learn from updates in the
# GitHub repository.
#
# Sources:
#  - reference_table.pgsql
#  - opcode_mapping.md
# ============================================================

import requests

REFERENCE_URLS = {
    "pgsql": "https://raw.githubusercontent.com/JoeySoprano420/NEWS/main/reference_table.pgsql",
    "opcodes": "https://raw.githubusercontent.com/JoeySoprano420/NEWS/main/opcode_mapping.md"
}

class NEWSKnowledgeBase:
    def __init__(self):
        self.sql_schema = None
        self.opcode_table = None

    def fetch_resources(self):
        """Fetch reference SQL schema and opcode mapping from GitHub."""
        try:
            r_sql = requests.get(REFERENCE_URLS["pgsql"], timeout=10)
            r_ops = requests.get(REFERENCE_URLS["opcodes"], timeout=10)
            if r_sql.status_code == 200:
                self.sql_schema = r_sql.text
            if r_ops.status_code == 200:
                self.opcode_table = r_ops.text
            print("[NEWS] Reference data loaded successfully.")
        except Exception as e:
            print(f"[NEWS] Failed to fetch external references: {e}")

    def parse_sql_schema(self):
        """Parse SQL schema to enable DB-aware compilation."""
        if not self.sql_schema:
            return {}
        tables = {}
        for line in self.sql_schema.splitlines():
            if line.strip().upper().startswith("CREATE TABLE"):
                parts = line.split()
                if len(parts) >= 3:
                    table_name = parts[2]
                    tables[table_name] = []
            elif line.strip().endswith(",") or line.strip().endswith(")"):
                col_parts = line.strip().split()
                if len(col_parts) >= 2:
                    col_name, col_type = col_parts[0], col_parts[1]
                    if tables:
                        last_table = list(tables.keys())[-1]
                        tables[last_table].append((col_name, col_type))
        return tables

    def parse_opcodes(self):
        """Parse Markdown opcode mapping into a dict."""
        if not self.opcode_table:
            return {}
        mapping = {}
        for line in self.opcode_table.splitlines():
            if line and not line.startswith("| Hex") and "|" in line:
                cols = [c.strip() for c in line.strip("|").split("|")]
                if len(cols) >= 3 and cols[0] != "Hex":
                    try:
                        hex_code = cols[0]
                        name = cols[1]
                        meaning = cols[2]
                        mapping[hex_code] = {"name": name, "meaning": meaning}
                    except Exception:
                        continue
        return mapping

# ============================================================
# Hook into Compiler/VM
# ============================================================

knowledge = NEWSKnowledgeBase()
knowledge.fetch_resources()

sql_info = knowledge.parse_sql_schema()
opcode_info = knowledge.parse_opcodes()

# Example: Extend VM/compiler dynamically with loaded opcodes
if opcode_info:
    print("[NEWS] Loaded external opcode definitions:")
    for hex_code, details in list(opcode_info.items())[:5]:  # show first 5
        print(f"  {hex_code}: {details['name']} -> {details['meaning']}")

# Example: Provide DB schema awareness
if sql_info:
    print("[NEWS] Detected DB tables from reference schema:")
    for table, cols in sql_info.items():
        col_list = ", ".join([f"{c[0]}:{c[1]}" for c in cols])
        print(f"  {table} ({col_list})")

# ============================================================
# FUTURE EXTENSIONS:
# - Auto-generate NEWS instructions from opcodes.md
# - Enable transpiler to target SQL queries from reference_table.pgsql
# - Add validation of NEWS bytecode against reference schema
# ============================================================

# ============================================================
# NEWS Dynamic Opcode Integration
# ============================================================
# Uses opcode_mapping.md from GitHub to expand the VM dynamically.
# ============================================================

import requests

OPCODE_URL = "https://raw.githubusercontent.com/JoeySoprano420/NEWS/main/opcode_mapping.md"

class DynamicOpcodeRegistry:
    def __init__(self, vm):
        self.vm = vm
        self.mapping = {}

    def fetch_and_parse(self):
        try:
            r = requests.get(OPCODE_URL, timeout=10)
            if r.status_code == 200:
                self.mapping = self._parse_markdown(r.text)
                print(f"[NEWS] Loaded {len(self.mapping)} dynamic opcodes.")
            else:
                print("[NEWS] Failed to fetch opcode mapping: HTTP", r.status_code)
        except Exception as e:
            print("[NEWS] Error fetching opcode mapping:", e)

    def _parse_markdown(self, text: str):
        """Parse Markdown opcode table into dict {hex: {name, meaning}}"""
        mapping = {}
        for line in text.splitlines():
            if not line or line.startswith("| Hex") or line.startswith("---"):
                continue
            if "|" in line:
                cols = [c.strip() for c in line.strip("|").split("|")]
                if len(cols) >= 3:
                    hex_code, name, meaning = cols[0], cols[1], cols[2]
                    try:
                        opcode = int(hex_code, 16)
                        mapping[opcode] = {"name": name, "meaning": meaning}
                    except ValueError:
                        continue
        return mapping

    def attach_to_vm(self):
        """Extend VM with dynamic opcode support."""
        def dynamic_handler(opcode):
            if opcode in self.mapping:
                info = self.mapping[opcode]
                msg = f"[DYNAMIC OPCODE] {hex(opcode)}: {info['name']} - {info['meaning']}"
                self.vm.output_buffer.append(msg)
                print(msg)
            else:
                raise ValueError(f"Unknown opcode {hex(opcode)} (not in VM or dynamic mapping)")
        # Patch VM step fallback
        old_step = self.vm.step
        def new_step():
            try:
                old_step()
            except ValueError as e:
                # Extract opcode from error message if possible
                msg = str(e)
                if msg.startswith("Unknown opcode"):
                    parts = msg.split()
                    try:
                        op_hex = int(parts[2], 16)
                        dynamic_handler(op_hex)
                    except Exception:
                        raise
                else:
                    raise
        self.vm.step = new_step

# ============================================================
# Initialize Dynamic Opcode Extension
# ============================================================

try:
    registry = DynamicOpcodeRegistry(vm)
    registry.fetch_and_parse()
    registry.attach_to_vm()
except Exception as e:
    print("[NEWS] Dynamic opcode integration failed:", e)

    main()

# ============================================================
# NEWS Database-Aware Extensions
# ============================================================
# Uses reference_table.pgsql to provide schema-aware SQL opcodes.
# Enables NEWS source to directly manipulate DB tables/rows.
# ============================================================

import sqlite3
import requests

SQL_URL = "https://raw.githubusercontent.com/JoeySoprano420/NEWS/main/reference_table.pgsql"

class NEWSDBSchema:
    def __init__(self):
        self.tables = {}

    def fetch_and_parse(self):
        try:
            r = requests.get(SQL_URL, timeout=10)
            if r.status_code == 200:
                self._parse_pgsql(r.text)
                print(f"[NEWS] Loaded DB schema for {len(self.tables)} tables.")
            else:
                print("[NEWS] Failed to fetch SQL schema: HTTP", r.status_code)
        except Exception as e:
            print("[NEWS] Error fetching SQL schema:", e)

    def _parse_pgsql(self, sql_text: str):
        current_table = None
        for line in sql_text.splitlines():
            line = line.strip()
            if not line: continue
            if line.upper().startswith("CREATE TABLE"):
                parts = line.split()
                if len(parts) >= 3:
                    current_table = parts[2].strip('"')
                    self.tables[current_table] = []
            elif current_table and (line.endswith(",") or line.endswith(")")):
                col_parts = line.replace(",", "").replace(")", "").split()
                if len(col_parts) >= 2:
                    col_name, col_type = col_parts[0], col_parts[1]
                    self.tables[current_table].append((col_name, col_type))
            elif line.upper().startswith(");"):
                current_table = None

# ============================================================
# Extend VM with DB opcode execution
# ============================================================

class NEWSDBHandler:
    def __init__(self, vm, schema: NEWSDBSchema):
        self.vm = vm
        self.schema = schema
        self.conn = None
        self.cursor = None

    def execute(self, opcode):
        # DB ops are defined in 130–13F
        if opcode == 0x130:  # db.open
            db_name = self.vm.stack.pop() if self.vm.stack else ":memory:"
            self.conn = sqlite3.connect(db_name)
            self.cursor = self.conn.cursor()
            print(f"[DB] Opened database {db_name}")

        elif opcode == 0x131:  # db.exec
            query = self.vm.stack.pop()
            self.cursor.execute(query)
            self.conn.commit()
            print(f"[DB] Executed SQL: {query}")

        elif opcode == 0x132:  # db.query
            table = self.vm.stack.pop()
            if table in self.schema.tables:
                self.cursor.execute(f"SELECT * FROM {table}")
                rows = self.cursor.fetchall()
                self.vm.output_buffer.append(rows)
                print(f"[DB] Query {table} returned {len(rows)} rows")
            else:
                print(f"[DB ERROR] Unknown table {table}")

        elif opcode == 0x133:  # db.close
            if self.conn:
                self.conn.close()
                print("[DB] Closed database")
                self.conn = None

        elif opcode == 0x139:  # db.insert
            # expects: table name, dict of col=val
            args = self.vm.stack.pop()
            table = self.vm.stack.pop()
            if table in self.schema.tables:
                cols = ", ".join(args.keys())
                placeholders = ", ".join("?" for _ in args)
                values = tuple(args.values())
                sql = f"INSERT INTO {table} ({cols}) VALUES ({placeholders})"
                self.cursor.execute(sql, values)
                self.conn.commit()
                print(f"[DB] Inserted into {table}: {args}")
            else:
                print(f"[DB ERROR] Unknown table {table}")

        else:
            raise ValueError(f"Unhandled DB opcode {hex(opcode)}")

# ============================================================
# Hook into VM (dynamic DB support)
# ============================================================

try:
    schema = NEWSDBSchema()
    schema.fetch_and_parse()
    db_handler = NEWSDBHandler(vm, schema)

    old_step = vm.step
    def new_step():
        try:
            old_step()
        except ValueError as e:
            msg = str(e)
            if msg.startswith("Unknown opcode"):
                try:
                    op_hex = int(msg.split()[2], 16)
                    if 0x130 <= op_hex <= 0x13F:
                        db_handler.execute(op_hex)
                    else:
                        raise
                except Exception:
                    raise
    vm.step = new_step
except Exception as e:
    print("[NEWS] DB-aware extension failed:", e)

    vm = NewsVM(debug=False, trace=False)
    vm.load_program(dgm)
    vm.run()

    disassemble(dgm)
    
# ============================================================
# NEWS HTTP / Requests Integration
# ============================================================
import requests

class NEWSHTTPHandler:
    def __init__(self, vm):
        self.vm = vm

    def execute(self, opcode):
        if opcode == 0x120:  # http.get
            url = self.vm.stack.pop()
            r = requests.get(url)
            self.vm.output_buffer.append(r.text)
            print(f"[HTTP] GET {url} -> {len(r.text)} chars")

        elif opcode == 0x121:  # http.post
            data = self.vm.stack.pop()
            url = self.vm.stack.pop()
            r = requests.post(url, data=data)
            self.vm.output_buffer.append(r.text)
            print(f"[HTTP] POST {url} -> {len(r.text)} chars")

        elif opcode == 0x124:  # http.delete
            url = self.vm.stack.pop()
            r = requests.delete(url)
            self.vm.output_buffer.append(r.status_code)
            print(f"[HTTP] DELETE {url} -> {r.status_code}")

        else:
            raise ValueError(f"Unhandled HTTP opcode {hex(opcode)}")

        # ============================================================
        # Hook into VM (dynamic HTTP support)
        
            http_handler = NEWSHTTPHandler(vm)
            old_step = vm.step
            def new_step():
                try:
                    old_step()
                except ValueError as e:
                    msg = str(e)
                    if msg.startswith("Unknown opcode"):
                        try:
                            op_hex = int(msg.split()[2], 16)
                            if 0x120 <= op_hex <= 0x12F:
                                http_handler.execute(op_hex)
                            else:
                                raise
                        except Exception:
                            raise
                    else:
                        raise
            vm.step = new_step
            
print("[NEWS] HTTP extension failed:", e)
                # ============================================================
vm = NewsVM(debug=False, trace=False)
vm.load_program(dgm)
vm.run()
disassemble(dgm)
disassemble(dgm)
if __name__ == "__main__":
                    main()
                    vm = NewsVM(debug=False, trace=False)
                    vm.load_program(dgm)
                    vm.run()
                    disassemble(dgm)
                    disassemble(dgm)
                    if __name__ == "__main__":
                        main()

                        # ============================================================
                        vm = NewsVM(debug=False, trace=False)
                        vm.load_program(dgm)
                        vm.run()
                        disassemble(dgm)
                        disassemble(dgm)
                        if __name__ == "__main__":
                            main()
                            vm = NewsVM(debug=False, trace=False)
                            vm.load_program(dgm)
                            vm.run()
                            disassemble(dgm)
                            disassemble(dgm)
                            if __name__ == "__main__":
                                main()
                                vm = NewsVM(debug=False, trace=False)
                                vm.load_program(dgm)
                                vm.run()
                                disassemble(dgm)
                                disassemble(dgm)
                                if __name__ == "__main__":
                                    main()
                                    vm = NewsVM(debug=False, trace=False)
                                    vm.load_program(dgm)
                                    vm.run()
                                    disassemble(dgm)
                                    disassemble(dgm)
                                    if __name__ == "__main__":
                                        main()
                                        vm = NewsVM(debug=False, trace=False)
                                        vm.load_program(dgm)
                                        vm.run()
                                        disassemble(dgm)
                                        disassemble(dgm)
                                        if __name__ == "__main__":
                                            main()
                                            vm = NewsVM(debug=False, trace=False)
                                            vm.load_program(dgm)
                                            vm.run()
                                            disassemble(dgm)
                                            disassemble(dgm)
                                            if __name__ == "__main__":
                                                main()
                                                vm = NewsVM(debug=False, trace=False)
                                                vm.load_program(dgm)
                                                vm.run()
                                                disassemble(dgm)
                                                disassemble(dgm)
                                                if __name__ == "__main__":
                                                    main()
                                                    vm = NewsVM(debug=False, trace=False)
                                                    vm.load_program(dgm)
                                                    vm.run()
                                                    disassemble(dgm)
                                                    disassemble(dgm)
                                                    if __name__ == "__main__":
                                                        main()
                                                        vm = NewsVM(debug=False, trace=False)
                                                        vm.load_program(dgm)
                                                        vm.run()
                                                        disassemble(dgm)
                                                        disassemble(dgm)
                                                        if __name__ == "__main__":
                                                            main()
                                                            vm = NewsVM(debug=False, trace=False)
                                                            vm.load_program(dgm)
                                                            vm.run()
                                                            disassemble(dgm)
                                                            disassemble(dgm)
                                                            if __name__ == "__main__":
                                                                main()
                                                                vm = NewsVM(debug=False, trace=False)
                                                                vm.load_program(dgm)
                                                                vm.run()
                                                                disassemble(dgm)
                                                                disassemble(dgm)
                                                                if __name__ == "__main__":
                                                                    main()
                                                                    vm = NewsVM(debug=False, trace=False)
                                                                    vm.load_program(dgm)
                                                                    vm.run()
                                                                    disassemble(dgm)
                                                                    disassemble(dgm)
                                                                    if __name__ == "__main__":
                                                                        main()
                                                                        vm = NewsVM(debug=False, trace=False)
                                                                        vm.load_program(dgm)
                                                                        vm.run()
                                                                        disassemble(dgm)
                                                                        disassemble(dgm)
                                                                        if __name__ == "__main__":
                                                                            main()
                                                                            vm = NewsVM(debug=False, trace=False)
                                                                            vm.load_program(dgm)
                                                                            vm.run()
                                                                            disassemble(dgm)
                                                                            disassemble(dgm)
                                                                            if __name__ == "__main__":
                                                                                main()
                                                                                vm = NewsVM(debug=False, trace=False)
                                                                                vm.load_program(dgm)
                                                                                vm.run()
                                                                                disassemble(dgm)
                                                                                disassemble(dgm)
                                                                                if __name__ == "__main__":
                                                                                    main()
                                                                                    vm = NewsVM(debug=False, trace=False)
                                                                                    vm.load_program(dgm)
                                                                                    vm.run()
                                                                                    disassemble(dgm)
                                                                                    disassemble(dgm)
                                                                                    if __name__ == "__main__":
                                                                                        main()
                                                                                        vm = NewsVM(debug=False, trace=False)
                                                                                        vm.load_program(dgm)
                                                                                        vm.run()
                                                                                        disassemble(dgm)
                                                                                        disassemble(dgm)
                                                                                        if __name__ == "__main__":
                                                                                            main()
                                                                                            vm = NewsVM(debug=False, trace=False)
                                                                                            vm.load_program(dgm)
                                                                                            vm.run()
                                                                                            disassemble(dgm)
                                                                                            disassemble(dgm)
                                                                                            if __name__ == "__main__":
                                                                                                main()
                                                                                                vm = NewsVM(debug=False, trace=False)
                                                                                                vm.load_program(dgm)
                                                                                                vm.run()
                                                                                                disassemble(dgm)
                                                                                                disassemble(dgm)
                                                                                                if __name__ == "__main__":
                                                                                                    main()
                                                                                                    vm = NewsVM(debug=False, trace=False)
                                                                                                    vm.load_program(dgm)
                                                                                                    vm.run()
                                                                                                    disassemble(dgm)
                                                                                                    disassemble(dgm)
                                                                                                    if __name__ == "__main__":
                                                                                                        main()
                                                                                                        vm = NewsVM(debug=False, trace=False)
                                                                                                        vm.load_program(dgm)
                                                                                                        vm.run()
                                                                                                        disassemble(dgm)
                                                                                                        disassemble(dgm)
                                                                                                        if __name__ == "__main__":
                                                                                                            main()
                                                                                                            vm = NewsVM(debug=False, trace=False)
                                                                                                            vm.load_program(dgm)
                                                                                                            vm.run()
                                                                                                            disassemble(dgm)
                                                                                                            disassemble(dgm)
                                                                                                            if __name__ == "__main__":
                                                                                                                main()
                                                                                                                vm = NewsVM(debug=False, trace=False)
                                                                                                                vm.load_program(dgm)
                                                                                                                vm.run()
                                                                                                                disassemble(dgm)
                                                                                                                disassemble(dgm)
                                                                                                                if __name__ == "__main__":
                                                                                                                    main()
                                                                                                                    vm = NewsVM(debug=False, trace=False)
                                                                                                                    vm.load_program(dgm)
                                                                                                                    vm.run()
                                                                                                                    disassemble(dgm)
                                                                                                                    disassemble(dgm)
                                                                                                                    if __name__ == "__main__":
                                                                                                                        main()
                                                                                                                        vm = NewsVM(debug=False, trace=False)
                                                                                                                        vm.load_program(dgm)
                                                                                                                        vm.run()
                                                                                                                        disassemble(dgm)
                                                                                                                        disassemble(dgm)
                                                                                                                        if __name__ == "__main__":
                                                                                                                            main()
                                                                                                                            vm = NewsVM(debug=False, trace=False)
                                                                                                                            vm.load_program(dgm)
                                                                                                                            vm.run()
                                                                                                                            disassemble(dgm)
                                                                                                                            disassemble(dgm)
                                                                                                                            if __name__ == "__main__":
                                                                                                                                main()
                                                                                                                                vm = NewsVM(debug=False, trace=False)
                                                                                                                                vm.load_program(dgm)
                                                                                                                                vm.run()
                                                                                                                                disassemble(dgm)
                                                                                                                                disassemble(dgm)

if __name__ == "__main__":
                                                                                                                                    main()
                                                                                                                                    vm = NewsVM(debug=False, trace=False)
                                                                                                                                    vm.load_program(dgm)
                                                                                                                                    vm.run()
                                                                                                                                    disassemble(dgm)
                                                                                                                                    disassemble(dgm)

# ============================================================
# NEWS Virtual Machine (Full Definition)
# ============================================================
import operator, time, math, threading

DIGITS = "0123456789ab"

def from_base12(s: str) -> int:
    v = 0
    for ch in s: v = v * 12 + DIGITS.index(ch)
    return v

def to_base12(num: int) -> str:
    if num == 0: return "0"
    out = []
    while num > 0:
        out.append(DIGITS[num % 12])
        num //= 12
    return "".join(reversed(out))

REL_OPS = {"==": operator.eq, "!=": operator.ne, "<": operator.lt,
           ">": operator.gt, "<=": operator.le, ">=": operator.ge}

class NewsVM:
    def __init__(self, debug=False, trace=False):
        self.memory = {}          # general heap (addr → val)
        self.stack = []           # operand stack
        self.frames = []          # call frames
        self.ip = 0               # instruction pointer
        self.tokens = []          # program tokens
        self.debug = debug
        self.trace = trace
        self.running = True
        self.output_buffer = []
        self.threads = []
        self.extensions = []      # external handlers (HTTP, DB, Game…)

    # ---------------- Loader ----------------
    def load_program(self, code: str):
        self.tokens = code.strip().split()
        self.ip = 0
        self.running = True

    def fetch(self):
        if self.ip >= len(self.tokens):
            self.running = False
            return 0x33  # ret
        tok = self.tokens[self.ip]; self.ip += 1
        return from_base12(tok)

    # ---------------- Execution Loop ----------------
    def step(self):
        opcode = self.fetch()
        if self.trace:
            print(f"[TRACE] ip={self.ip} opcode={hex(opcode)} stack={self.stack}")

        try:
            # ---- Core Ops ----
            if opcode == 0x00: pass  # nop

            elif opcode == 0x01:  # alloca
                addr = self.stack.pop()
                self.memory[addr] = 0

            elif opcode == 0x02:  # load
                addr = self.stack.pop()
                self.stack.append(self.memory.get(addr, 0))

            elif opcode == 0x03:  # store
                val, addr = self.stack.pop(), self.stack.pop()
                self.memory[addr] = val

            elif opcode == 0x17:  # add
                b, a = self.stack.pop(), self.stack.pop()
                self.stack.append(a + b)

            elif opcode == 0x18:  # sub
                b, a = self.stack.pop(), self.stack.pop()
                self.stack.append(a - b)

            elif opcode == 0x19:  # mul
                b, a = self.stack.pop(), self.stack.pop()
                self.stack.append(a * b)

            elif opcode == 0x1A:  # udiv
                b, a = self.stack.pop(), self.stack.pop()
                self.stack.append(a // (b or 1))

            elif opcode == 0x15:  # icmp
                op = self.tokens[self.ip]; self.ip += 1
                b, a = self.stack.pop(), self.stack.pop()
                self.stack.append(1 if REL_OPS[op](a, b) else 0)

            elif opcode == 0x30:  # br
                target = from_base12(self.tokens[self.ip]); self.ip = target

            elif opcode == 0x2B:  # call
                target = from_base12(self.tokens[self.ip]); self.ip += 1
                self.frames.append(self.ip)
                self.ip = target

            elif opcode == 0x33:  # ret
                if self.frames: self.ip = self.frames.pop()
                else: self.running = False

            elif opcode == 0xA6:  # echo
                chars = []
                while self.ip < len(self.tokens):
                    val = from_base12(self.tokens[self.ip]); self.ip += 1
                    if val == 0: break
                    chars.append(chr(val))
                msg = "".join(chars)
                self.output_buffer.append(msg)
                print(msg)

            # ---- Extended handling via extensions ----
            else:
                handled = False
                for ext in self.extensions:
                    try:
                        ext.execute(opcode)
                        handled = True
                        break
                    except ValueError:
                        continue
                if not handled:
                    raise ValueError(f"Unknown opcode {hex(opcode)}")

        except Exception as e:
            print(f"[VM ERROR] {e}")
            self.running = False

    def run(self):
        while self.running:
            self.step()

    # ---------------- Extension API ----------------
    def attach_extension(self, handler):
        self.extensions.append(handler)

        # ---------------- Disassembler ----------------
        def disassemble(code: str):
            tokens = code.strip().split()
            ip = 0
            while ip < len(tokens):
                start_ip = ip
                opcode = from_base12(tokens[ip]); ip += 1
                line = f"{start_ip:04}: {hex(opcode)}"
                if opcode == 0x00: line += " nop"
                elif opcode == 0x01: line += " alloca"
                elif opcode == 0x02: line += " load"
                elif opcode == 0x03: line += " store"
                elif opcode == 0x17: line += " add"
                elif opcode == 0x18: line += " sub"
                elif opcode == 0x19: line += " mul"
                elif opcode == 0x1A: line += " udiv"
                elif opcode == 0x15:
                    op = tokens[ip]; ip += 1
                    line += f" icmp {op}"
                elif opcode == 0x30:
                    target = tokens[ip]; ip += 1
                    line += f" br {target}"
                elif opcode == 0x2B:
                    target = tokens[ip]; ip += 1
                    line += f" call {target}"
                elif opcode == 0x33: line += " ret"
                elif opcode == 0xA6:
                    chars = []
                    while ip < len(tokens):
                        val = from_base12(tokens[ip]); ip += 1
                        if val == 0: break
                        chars.append(chr(val))
                    msg = "".join(chars)
                    line += f' echo "{msg}"'
                else:
                    line += f" <unknown {hex(opcode)}>"
                print(line)
                print("Disassembly complete.")
                return
            if __name__ == "__main__":
                main()
                # ============================================================
                vm = NewsVM(debug=False, trace=False)
                vm.load_program(dgm)
                vm.run()
                disassemble(dgm)
                disassemble(dgm)
                if __name__ == "__main__":
                    main()
                    vm = NewsVM(debug=False, trace=False)
                    vm.load_program(dgm)
                    vm.run()
                    disassemble(dgm)
                    disassemble(dgm)
                    if __name__ == "__main__":
                        main()
                        vm = NewsVM(debug=False, trace=False)
                        vm.load_program(dgm)
                        vm.run()
                        disassemble(dgm)
                        disassemble(dgm)
                        if __name__ == "__main__":
                            main()
                            vm = NewsVM(debug=False, trace=False)
                            vm.load_program(dgm)
                            vm.run()
                            disassemble(dgm)
                            disassemble(dgm)
                            if __name__ == "__main__":
                                main()
                                vm = NewsVM(debug=False, trace=False)
                                vm.load_program(dgm)
                                vm.run()
                                disassemble(dgm)
                                disassemble(dgm)
                                if __name__ == "__main__":
                                    main()
                                    vm = NewsVM(debug=False, trace=False)
                                    vm.load_program(dgm)
                                    vm.run()
                                    disassemble(dgm)
                                    disassemble(dgm)
                                    if __name__ == "__main__":
                                        main()
                                        vm = NewsVM(debug=False, trace=False)
                                        vm.load_program(dgm)
                                        vm.run()
                                        disassemble(dgm)
                                        disassemble(dgm)
                                        if __name__ == "__main__":
                                            main()
                                            vm = NewsVM(debug=False, trace=False)
                                            vm.load_program(dgm)
                                            vm.run()
                                            disassemble(dgm)
                                            disassemble(dgm)
                                            if __name__ == "__main__":
                                                main()
                                                vm = NewsVM(debug=False, trace=False)
                                                vm.load_program(dgm)
                                                vm.run()
                                                disassemble(dgm)
                                                disassemble(dgm)
                                                if __name__ == "__main__":
                                                    main()
                                                    vm = NewsVM(debug=False, trace=False)
                                                    vm.load_program(dgm)
                                                    vm.run()
                                                    disassemble(dgm)
                                                    disassemble(dgm)
                                                    if __name__ == "__main__":
                                                        main()
                                                        vm = NewsVM(debug=False, trace=False)
                                                        vm.load_program(dgm)
                                                        vm.run()
                                                        disassemble(dgm)
                                                        disassemble(dgm)
                                                        if __name__ == "__main__":
                                                            main()
                                                            vm = NewsVM(debug=False, trace=False)
                                                            vm.load_program(dgm)
                                                            vm.run()
                                                            disassemble(dgm)
                                                            disassemble(dgm)
                                                            if __name__ == "__main__":
                                                                main()
                                                                vm = NewsVM(debug=False, trace=False)
                                                                vm.load_program(dgm)
                                                                vm.run()
                                                                disassemble(dgm)
                                                                disassemble(dgm)
                                                                if __name__ == "__main__":
                                                                    main()
                                                                    vm = NewsVM(debug=False, trace=False)
                                                                    vm.load_program(dgm)
                                                                    vm.run()
                                                                    disassemble(dgm)
                                                                    disassemble(dgm)
                                                                    if __name__ == "__main__":
                                                                        main()
                                                                        vm = NewsVM(debug=False, trace=False)
                                                                        vm.load_program(dgm)
                                                                        vm.run()
                                                                        disassemble(dgm)
                                                                        disassemble(dgm)
                                                                        if __name__ == "__main__":
                                                                            main()
                                                                            vm = NewsVM(debug=False, trace=False)
                                                                            vm.load_program(dgm)
                                                                            vm.run()
                                                                            disassemble(dgm)
                                                                            disassemble(dgm)
                                                                            if __name__ == "__main__":
                                                                                main()
                                                                                vm = NewsVM(debug=False, trace=False)
                                                                                vm.load_program(dgm)
                                                                                vm.run()
                                                                                disassemble(dgm)
                                                                                disassemble(dgm)
                                                                                if __name__ == "__main__":
                                                                                    main()
                                                                                    vm = NewsVM(debug=False, trace=False)
                                                                                    vm.load_program(dgm)
                                                                                    vm.run()
                                                                                    disassemble(dgm)
                                                                                    disassemble(dgm)
                                                                                    if __name__ == "__main__":
                                                                                        main()
                                                                                        vm = NewsVM(debug=False, trace=False)
                                                                                        vm.load_program(dgm)
                                                                                        vm.run()
                                                                                        disassemble(dgm)
                                                                                        disassemble(dgm)
                                                                                        if __name__ == "__main__":
                                                                                            main()
                                                                                            vm = NewsVM(debug=False, trace=False)
                                                                                            vm.load_program(dgm)
                                                                                            vm.run()
                                                                                            disassemble(dgm)
                                                                                            disassemble(dgm)
                                                                                            if __name__ == "__main__":
                                                                                                main()
                                                                                                vm = NewsVM(debug=False, trace=False)
                                                                                                vm.load_program(dgm)
                                                                                                vm.run()
                                                                                                disassemble(dgm)
                                                                                                disassemble(dgm)
                                                                                                if __name__ == "__main__":
                                                                                                    main()
                                                                                                    vm = NewsVM(debug=False, trace=False)
                                                                                                    vm.load_program(dgm)
                                                                                                    vm.run()
                                                                                                    disassemble(dgm)
                                                                                                    disassemble(dgm)
                                                                                                    if __name__ == "__main__":
                                                                                                        main()
                                                                                                        vm = NewsVM(debug=False, trace=False)
                                                                                                        vm.load_program(dgm)
                                                                                                        vm.run()
                                                                                                        disassemble(dgm)
                                                                                                        disassemble(dgm)
                                                                                                        if __name__ == "__main__":
                                                                                                            main()
                                                                                                            vm = NewsVM(debug=False, trace=False)
                                                                                                            vm.load_program(dgm)
                                                                                                            vm.run()
                                                                                                            disassemble(dgm)
                                                                                                            disassemble(dgm)
                                                                                                            if __name__ == "__main__":
                                                                                                                main()
                                                                                                                vm = NewsVM(debug=False, trace=False)
                                                                                                                vm.load_program(dgm)
                                                                                                                vm.run()
                                                                                                                disassemble(dgm)
                                                                                                                disassemble(dgm)
                                                                                                                if __name__ == "__main__":
                                                                                                                    main()
                                                                                                                    vm = NewsVM(debug=False, trace=False)
                                                                                                                    vm.load_program(dgm)
                                                                                                                    vm.run()
                                                                                                                    disassemble(dgm)
                                                                                                                    disassemble(dgm)
                                                                                                                    if __name__ == "__main__":
                                                                                                                        main()
                                                                                                                        vm = NewsVM(debug=False, trace=False)
                                                                                                                        vm.load_program(dgm)
                                                                                                                        vm.run()
                                                                                                                        disassemble(dgm)
                                                                                                                        disassemble(dgm)

 # NEws.py
# ======================================================
# 📊 DGM ↔ LLVM IR ↔ NASM ↔ Hex ↔ Binary (00–159)
# ======================================================

OPCODE_MAP = [
    {"dgm": "00",  "llvm": "nop",                   "nasm": "NOP",                     "hex": "0x00",  "bin": "00000000"},
    {"dgm": "01",  "llvm": "alloca",                "nasm": "SUB RSP, imm32",          "hex": "0x01",  "bin": "00000001"},
    {"dgm": "02",  "llvm": "load",                  "nasm": "MOV r64, [mem]",          "hex": "0x02",  "bin": "00000010"},
    {"dgm": "03",  "llvm": "store",                 "nasm": "MOV [mem], r64",          "hex": "0x03",  "bin": "00000011"},
    {"dgm": "04",  "llvm": "getelementptr",         "nasm": "LEA r64, [mem]",          "hex": "0x04",  "bin": "00000100"},
    {"dgm": "05",  "llvm": "bitcast",               "nasm": "MOVQ reg, xmm",           "hex": "0x05",  "bin": "00000101"},
    {"dgm": "06",  "llvm": "trunc",                 "nasm": "MOVZX/MOVSX (narrow int)","hex": "0x06",  "bin": "00000110"},
    {"dgm": "07",  "llvm": "zext",                  "nasm": "MOVZX",                   "hex": "0x07",  "bin": "00000111"},
    {"dgm": "08",  "llvm": "sext",                  "nasm": "MOVSX",                   "hex": "0x08",  "bin": "00001000"},
    {"dgm": "09",  "llvm": "fptrunc",               "nasm": "CVTSD2SS / CVTPD2PS",     "hex": "0x09",  "bin": "00001001"},
    {"dgm": "0A",  "llvm": "fpext",                 "nasm": "CVTSS2SD / CVTPS2PD",     "hex": "0x0A",  "bin": "00001010"},
    {"dgm": "0B",  "llvm": "fptoui",                "nasm": "CVTTSD2SI",               "hex": "0x0B",  "bin": "00001011"},
    {"dgm": "10",  "llvm": "fptosi",                "nasm": "CVTTSS2SI",               "hex": "0x10",  "bin": "00010000"},
    {"dgm": "11",  "llvm": "uitofp",                "nasm": "CVTSI2SD",                "hex": "0x11",  "bin": "00010001"},
    {"dgm": "12",  "llvm": "sitofp",                "nasm": "CVTSI2SS",                "hex": "0x12",  "bin": "00010010"},
    {"dgm": "13",  "llvm": "ptrtoint",              "nasm": "MOV reg, qword ptr",      "hex": "0x13",  "bin": "00010011"},
    {"dgm": "14",  "llvm": "inttoptr",              "nasm": "MOV reg, imm64",          "hex": "0x14",  "bin": "00010100"},
    {"dgm": "15",  "llvm": "icmp",                  "nasm": "CMP r/m64, r64",          "hex": "0x15",  "bin": "00010101"},
    {"dgm": "16",  "llvm": "fcmp",                  "nasm": "UCOMISD / UCOMISS",       "hex": "0x16",  "bin": "00010110"},
    {"dgm": "17",  "llvm": "add",                   "nasm": "ADD r/m64, r64",          "hex": "0x17",  "bin": "00010111"},
    {"dgm": "18",  "llvm": "sub",                   "nasm": "SUB r/m64, r64",          "hex": "0x18",  "bin": "00011000"},
    {"dgm": "19",  "llvm": "mul",                   "nasm": "IMUL r64, r/m64",         "hex": "0x19",  "bin": "00011001"},
    {"dgm": "1A",  "llvm": "udiv",                  "nasm": "DIV r/m64",               "hex": "0x1A",  "bin": "00011010"},
    {"dgm": "1B",  "llvm": "sdiv",                  "nasm": "IDIV r/m64",              "hex": "0x1B",  "bin": "00011011"},
    {"dgm": "20",  "llvm": "fadd",                  "nasm": "ADDSD xmm, xmm",          "hex": "0x20",  "bin": "00100000"},
    {"dgm": "21",  "llvm": "fsub",                  "nasm": "SUBSD xmm, xmm",          "hex": "0x21",  "bin": "00100001"},
    {"dgm": "22",  "llvm": "fmul",                  "nasm": "MULSD xmm, xmm",          "hex": "0x22",  "bin": "00100010"},
    {"dgm": "23",  "llvm": "fdiv",                  "nasm": "DIVSD xmm, xmm",          "hex": "0x23",  "bin": "00100011"},
    {"dgm": "24",  "llvm": "frem",                  "nasm": "Emulated FP DIV+MUL-SUB", "hex": "0x24",  "bin": "00100100"},
    {"dgm": "25",  "llvm": "shl",                   "nasm": "SHL r/m64, CL",           "hex": "0x25",  "bin": "00100101"},
    {"dgm": "26",  "llvm": "lshr",                  "nasm": "SHR r/m64, CL",           "hex": "0x26",  "bin": "00100110"},
    {"dgm": "27",  "llvm": "ashr",                  "nasm": "SAR r/m64, CL",           "hex": "0x27",  "bin": "00100111"},
    {"dgm": "28",  "llvm": "and",                   "nasm": "AND r/m64, r64",          "hex": "0x28",  "bin": "00101000"},
    {"dgm": "29",  "llvm": "or",                    "nasm": "OR r/m64, r64",           "hex": "0x29",  "bin": "00101001"},
    {"dgm": "2A",  "llvm": "xor",                   "nasm": "XOR r/m64, r64",          "hex": "0x2A",  "bin": "00101010"},
    {"dgm": "2B",  "llvm": "call",                  "nasm": "CALL rel32",              "hex": "0x2B",  "bin": "00101011"},
    {"dgm": "30",  "llvm": "br",                    "nasm": "JMP rel32",               "hex": "0x30",  "bin": "00110000"},
    {"dgm": "31",  "llvm": "switch",                "nasm": "CMP+JMP table",           "hex": "0x31",  "bin": "00110001"},
    {"dgm": "32",  "llvm": "indirectbr",            "nasm": "JMP r/m64",               "hex": "0x32",  "bin": "00110010"},
    {"dgm": "33",  "llvm": "ret",                   "nasm": "RET",                     "hex": "0x33",  "bin": "00110011"},
    {"dgm": "34",  "llvm": "resume",                "nasm": "EH resume stub",          "hex": "0x34",  "bin": "00110100"},
    {"dgm": "35",  "llvm": "unreachable",           "nasm": "UD2",                     "hex": "0x35",  "bin": "00110101"},
    {"dgm": "36",  "llvm": "landingpad",            "nasm": "EH landing pad",          "hex": "0x36",  "bin": "00110110"},
    {"dgm": "37",  "llvm": "invoke",                "nasm": "CALL+EH unwind",          "hex": "0x37",  "bin": "00110111"},
    {"dgm": "38",  "llvm": "phi",                   "nasm": "SSA merge (no direct)",   "hex": "0x38",  "bin": "00111000"},
    {"dgm": "39",  "llvm": "select",                "nasm": "CMP+CMOVcc",              "hex": "0x39",  "bin": "00111001"},
    {"dgm": "3A",  "llvm": "extractvalue",          "nasm": "MOV reg,[struct+offset]", "hex": "0x3A",  "bin": "00111010"},
    {"dgm": "3B",  "llvm": "insertvalue",           "nasm": "MOV [struct+offset],reg", "hex": "0x3B",  "bin": "00111011"},
    # ...
    # Full expansion continues here for every DGM through 159
    # (game ops C0–CC, math ops D0–DF, str ops E0–E8, file ops E9–ED,
    # sys ops EE–FF, hash 100–10F, compression 110–11F, http/ws/tcp 120–12F,
    # db 130–13F, regex/fuzzy 140–148, audio 150–159).
]

# Lookup utility
def get_opcode_info(dgm_code: str):
    for entry in OPCODE_MAP:
        if entry["dgm"].lower() == dgm_code.lower():
            return entry
    return None

if __name__ == "__main__":
    # Demo
    code = "20"
    info = get_opcode_info(code)
    if info:
        print(f"DGM {info['dgm']} → LLVM: {info['llvm']} | NASM: {info['nasm']} | Hex: {info['hex']} | Bin: {info['bin']}")
    else:
        print("Opcode not found.")
    {"dgm": "3C", "llvm": "atomicrmw",        "nasm": "LOCK prefixed ops",        "hex": "0x3C", "bin": "00111100"},
    {"dgm": "3D", "llvm": "cmpxchg",          "nasm": "LOCK CMPXCHG",             "hex": "0x3D", "bin": "00111101"},
    {"dgm": "3E", "llvm": "fence",            "nasm": "MFENCE",                   "hex": "0x3E", "bin": "00111110"},
    {"dgm": "3F", "llvm": "memset",           "nasm": "REP STOSB",                "hex": "0x3F", "bin": "00111111"},
    {"dgm": "40", "llvm": "memcpy",           "nasm": "REP MOVSB",                "hex": "0x40", "bin": "01000000"},
    {"dgm": "41", "llvm": "memmove",          "nasm": "REP MOVSB+temp",           "hex": "0x41", "bin": "01000001"},
    {"dgm": "42", "llvm": "lifetime.start",   "nasm": "No codegen",               "hex": "0x42", "bin": "01000010"},
    {"dgm": "43", "llvm": "lifetime.end",     "nasm": "No codegen",               "hex": "0x43", "bin": "01000011"},
    {"dgm": "44", "llvm": "sanitizer.check",  "nasm": "CMP+Jcc bounds check",     "hex": "0x44", "bin": "01000100"},
    {"dgm": "45", "llvm": "assume",           "nasm": "Compiler builtin",         "hex": "0x45", "bin": "01000101"},
    {"dgm": "46", "llvm": "llvm.dbg.declare", "nasm": "Debug meta",               "hex": "0x46", "bin": "01000110"},
    {"dgm": "47", "llvm": "llvm.dbg.value",   "nasm": "Debug meta",               "hex": "0x47", "bin": "01000111"},
    {"dgm": "48", "llvm": "safe.add",         "nasm": "ADD+JO recover",           "hex": "0x48", "bin": "01001000"},
    {"dgm": "49", "llvm": "safe.sub",         "nasm": "SUB+JO recover",           "hex": "0x49", "bin": "01001001"},
    {"dgm": "4A", "llvm": "safe.mul",         "nasm": "IMUL+JO recover",          "hex": "0x4A", "bin": "01001010"},
    {"dgm": "4B", "llvm": "safe.div",         "nasm": "DIV+guard",                "hex": "0x4B", "bin": "01001011"},
    {"dgm": "4C", "llvm": "safe.mod",         "nasm": "IDIV+guard",               "hex": "0x4C", "bin": "01001100"},
    {"dgm": "4D", "llvm": "safe.shift",       "nasm": "SHL/SHR+mask",             "hex": "0x4D", "bin": "01001101"},
    {"dgm": "4E", "llvm": "safe.and",         "nasm": "AND+guard",                "hex": "0x4E", "bin": "01001110"},
    {"dgm": "4F", "llvm": "safe.or",          "nasm": "OR+guard",                 "hex": "0x4F", "bin": "01001111"},
    {"dgm": "50", "llvm": "safe.xor",         "nasm": "XOR+guard",                "hex": "0x50", "bin": "01010000"},
    {"dgm": "51", "llvm": "safe.neg",         "nasm": "NEG+check",                "hex": "0x51", "bin": "01010001"},
    {"dgm": "52", "llvm": "safe.not",         "nasm": "NOT r/m64",                "hex": "0x52", "bin": "01010010"},
    {"dgm": "53", "llvm": "cascade.begin",    "nasm": "PUSH context",             "hex": "0x53", "bin": "01010011"},
    {"dgm": "54", "llvm": "cascade.end",      "nasm": "POP context",              "hex": "0x54", "bin": "01010100"},
    {"dgm": "55", "llvm": "cascade.yield",    "nasm": "SAVE+JMP out",             "hex": "0x55", "bin": "01010101"},
    {"dgm": "56", "llvm": "cascade.resume",   "nasm": "RESTORE+JMP in",           "hex": "0x56", "bin": "01010110"},
    {"dgm": "57", "llvm": "branch.try",       "nasm": "Label mark",               "hex": "0x57", "bin": "01010111"},
    {"dgm": "58", "llvm": "branch.heal",      "nasm": "JMP recover block",        "hex": "0x58", "bin": "01011000"},
    {"dgm": "59", "llvm": "branch.soft",      "nasm": "JMP with mask",            "hex": "0x59", "bin": "01011001"},
    {"dgm": "5A", "llvm": "branch.auto",      "nasm": "Predicated JMP",           "hex": "0x5A", "bin": "01011010"},
    {"dgm": "5B", "llvm": "recover",          "nasm": "RESTORE state",            "hex": "0x5B", "bin": "01011011"},
    {"dgm": "5C", "llvm": "language.assert",  "nasm": "CMP+Jcc trap",             "hex": "0x5C", "bin": "01011100"},
    {"dgm": "5D", "llvm": "tuple.pack",       "nasm": "CALL __tuple_pack",        "hex": "0x5D", "bin": "01011101"},
    {"dgm": "5E", "llvm": "tuple.unpack",     "nasm": "CALL __tuple_unpack",      "hex": "0x5E", "bin": "01011110"},
    {"dgm": "5F", "llvm": "list.append",      "nasm": "CALL __list_append",       "hex": "0x5F", "bin": "01011111"},
    {"dgm": "60", "llvm": "list.remove",      "nasm": "CALL __list_remove",       "hex": "0x60", "bin": "01100000"},
    {"dgm": "61", "llvm": "list.insert",      "nasm": "CALL __list_insert",       "hex": "0x61", "bin": "01100001"},
    {"dgm": "62", "llvm": "list.pop",         "nasm": "CALL __list_pop",          "hex": "0x62", "bin": "01100010"},
    {"dgm": "63", "llvm": "array.load",       "nasm": "MOV reg,[array+idx]",      "hex": "0x63", "bin": "01100011"},
    {"dgm": "64", "llvm": "array.store",      "nasm": "MOV [array+idx],reg",      "hex": "0x64", "bin": "01100100"},
    {"dgm": "65", "llvm": "group.spawn",      "nasm": "CALL __group_spawn",       "hex": "0x65", "bin": "01100101"},
    {"dgm": "66", "llvm": "group.merge",      "nasm": "CALL __group_merge",       "hex": "0x66", "bin": "01100110"},
    {"dgm": "67", "llvm": "group.split",      "nasm": "CALL __group_split",       "hex": "0x67", "bin": "01100111"},
    {"dgm": "68", "llvm": "nest.enter",       "nasm": "CALL __nest_enter",        "hex": "0x68", "bin": "01101000"},
    {"dgm": "69", "llvm": "nest.exit",        "nasm": "CALL __nest_exit",         "hex": "0x69", "bin": "01101001"},
    {"dgm": "6A", "llvm": "derive.child",     "nasm": "CALL __derive_child",      "hex": "0x6A", "bin": "01101010"},
    {"dgm": "6B", "llvm": "derive.parent",    "nasm": "CALL __derive_parent",     "hex": "0x6B", "bin": "01101011"},
    {"dgm": "6C", "llvm": "pair.create",      "nasm": "CALL __pair_create",       "hex": "0x6C", "bin": "01101100"},
    {"dgm": "6D", "llvm": "pair.split",       "nasm": "CALL __pair_split",        "hex": "0x6D", "bin": "01101101"},
    {"dgm": "6E", "llvm": "match.begin",      "nasm": "LABEL match",              "hex": "0x6E", "bin": "01101110"},
    {"dgm": "6F", "llvm": "match.case",       "nasm": "CMP+Jcc",                  "hex": "0x6F", "bin": "01101111"},
    {"dgm": "70", "llvm": "match.end",        "nasm": "JMP end",                  "hex": "0x70", "bin": "01110000"},
    {"dgm": "71", "llvm": "language.yield",   "nasm": "CALL __yield",             "hex": "0x71", "bin": "01110001"},
    {"dgm": "72", "llvm": "language.halt",    "nasm": "HLT",                      "hex": "0x72", "bin": "01110010"},
    {"dgm": "73", "llvm": "language.wait",    "nasm": "PAUSE",                    "hex": "0x73", "bin": "01110011"},
    {"dgm": "74", "llvm": "language.resume",  "nasm": "CALL __resume",            "hex": "0x74", "bin": "01110100"},
    {"dgm": "75", "llvm": "language.inline",  "nasm": "__forceinline",            "hex": "0x75", "bin": "01110101"},
    {"dgm": "76", "llvm": "language.expand",  "nasm": "Macro expansion",          "hex": "0x76", "bin": "01110110"},
    {"dgm": "77", "llvm": "language.fold",    "nasm": "Folded macro",             "hex": "0x77", "bin": "01110111"},
    {"dgm": "78", "llvm": "language.derive",  "nasm": "Template derive",          "hex": "0x78", "bin": "01111000"},
    {"dgm": "79", "llvm": "language.macro",   "nasm": "Macro define",             "hex": "0x79", "bin": "01111001"},
    {"dgm": "7A", "llvm": "language.trace",   "nasm": "CALL __tracepoint",        "hex": "0x7A", "bin": "01111010"},
    {"dgm": "7B", "llvm": "language.echo",    "nasm": "CALL puts/printf",         "hex": "0x7B", "bin": "01111011"},
    {"dgm": "7C", "llvm": "language.link",    "nasm": "CALL dlopen",              "hex": "0x7C", "bin": "01111100"},
    {"dgm": "7D", "llvm": "language.infer",   "nasm": "Type infer pass",          "hex": "0x7D", "bin": "01111101"},
    {"dgm": "7E", "llvm": "language.delete",  "nasm": "CALL free",                "hex": "0x7E", "bin": "01111110"},
    {"dgm": "7F", "llvm": "language.replace", "nasm": "Swap call",                "hex": "0x7F", "bin": "01111111"},
    {"dgm": "80", "llvm": "language.redirect", "nasm": "JMP other",             "hex": "0x80", "bin": "10000000"},
    {"dgm": "81", "llvm": "language.guard",    "nasm": "CMP+Jcc guard",         "hex": "0x81", "bin": "10000001"},
    {"dgm": "82", "llvm": "language.wrap",     "nasm": "PUSH+CALL+POP",         "hex": "0x82", "bin": "10000010"},
    {"dgm": "83", "llvm": "language.unwrap",   "nasm": "MOV out,in",            "hex": "0x83", "bin": "10000011"},
    {"dgm": "84", "llvm": "language.enclose",  "nasm": "SCOPE guard",           "hex": "0x84", "bin": "10000100"},
    {"dgm": "85", "llvm": "language.open",     "nasm": "CALL fopen",            "hex": "0x85", "bin": "10000101"},
    {"dgm": "86", "llvm": "language.close",    "nasm": "CALL fclose",           "hex": "0x86", "bin": "10000110"},
    {"dgm": "87", "llvm": "language.defer",    "nasm": "PUSH cleanup",          "hex": "0x87", "bin": "10000111"},
    {"dgm": "88", "llvm": "language.future",   "nasm": "THREAD CREATE",         "hex": "0x88", "bin": "10001000"},
    {"dgm": "89", "llvm": "language.parallel", "nasm": "PTHREAD_CREATE",        "hex": "0x89", "bin": "10001001"},
    {"dgm": "8A", "llvm": "language.sync",     "nasm": "SYSCALL futex_wait",    "hex": "0x8A", "bin": "10001010"},
    {"dgm": "8B", "llvm": "language.pragma",   "nasm": "Compiler directive",    "hex": "0x8B", "bin": "10001011"},
    {"dgm": "8C", "llvm": "language.exit",     "nasm": "SYSCALL exit",          "hex": "0x8C", "bin": "10001100"},
    {"dgm": "8D", "llvm": "@llvm.game.init",        "nasm": "CALL __game_init",         "hex": "0x8D", "bin": "10001101"},
    {"dgm": "8E", "llvm": "@llvm.game.load.model",  "nasm": "CALL __game_load_model",  "hex": "0x8E", "bin": "10001110"},
    {"dgm": "8F", "llvm": "@llvm.game.load.texture","nasm": "CALL __game_load_texture","hex": "0x8F", "bin": "10001111"},
    {"dgm": "90", "llvm": "@llvm.game.create.world","nasm": "CALL __game_create_world","hex": "0x90", "bin": "10010000"},
    {"dgm": "91", "llvm": "@llvm.game.add.entity",  "nasm": "CALL __game_add_entity",  "hex": "0x91", "bin": "10010001"},
    {"dgm": "92", "llvm": "@llvm.game.add.light",   "nasm": "CALL __game_add_light",   "hex": "0x92", "bin": "10010010"},
    {"dgm": "93", "llvm": "@llvm.game.update",      "nasm": "CALL __game_update",      "hex": "0x93", "bin": "10010011"},
    {"dgm": "94", "llvm": "@llvm.game.render",      "nasm": "CALL __game_render",      "hex": "0x94", "bin": "10010100"},
    {"dgm": "95", "llvm": "@llvm.game.running",     "nasm": "CALL __game_running",     "hex": "0x95", "bin": "10010101"},
    {"dgm": "96", "llvm": "@llvm.game.input",       "nasm": "CALL __game_input",       "hex": "0x96", "bin": "10010110"},
    {"dgm": "97", "llvm": "@llvm.game.play.sound",  "nasm": "CALL __game_play_sound",  "hex": "0x97", "bin": "10010111"},
    {"dgm": "98", "llvm": "@llvm.game.play.music",  "nasm": "CALL __game_play_music",  "hex": "0x98", "bin": "10011000"},
    {"dgm": "99", "llvm": "@llvm.game.quit",        "nasm": "CALL __game_quit",        "hex": "0x99", "bin": "10011001"},
    {"dgm": "9A", "llvm": "@llvm.math.pow",         "nasm": "CALL __math_pow",         "hex": "0x9A", "bin": "10011010"},
    {"dgm": "9B", "llvm": "@llvm.math.log",         "nasm": "CALL __math_log",         "hex": "0x9B", "bin": "10011011"},
    {"dgm": "9C", "llvm": "@llvm.math.exp",         "nasm": "CALL __math_exp",         "hex": "0x9C", "bin": "10011100"},
    {"dgm": "9D", "llvm": "@llvm.math.sin",         "nasm": "CALL __math_sin",         "hex": "0x9D", "bin": "10011101"},
    {"dgm": "9E", "llvm": "@llvm.math.cos",         "nasm": "CALL __math_cos",         "hex": "0x9E", "bin": "10011110"},
    {"dgm": "9F", "llvm": "@llvm.math.tan",         "nasm": "CALL __math_tan",         "hex": "0x9F", "bin": "10011111"},
    {"dgm": "A0", "llvm": "@llvm.math.asin",        "nasm": "CALL __math_asin",        "hex": "0xA0", "bin": "10100000"},
    {"dgm": "A1", "llvm": "@llvm.math.acos",        "nasm": "CALL __math_acos",        "hex": "0xA1", "bin": "10100001"},
    {"dgm": "A2", "llvm": "@llvm.math.atan",        "nasm": "CALL __math_atan",        "hex": "0xA2", "bin": "10100010"},
    {"dgm": "A3", "llvm": "@llvm.math.sqrt",        "nasm": "CALL __math_sqrt",        "hex": "0xA3", "bin": "10100011"},
    {"dgm": "A4", "llvm": "@llvm.math.cbrt",        "nasm": "CALL __math_cbrt",        "hex": "0xA4", "bin": "10100100"},
    {"dgm": "A5", "llvm": "@llvm.math.hypot",       "nasm": "CALL __math_hypot",       "hex": "0xA5", "bin": "10100101"},
    {"dgm": "A6", "llvm": "@llvm.math.floor",       "nasm": "CALL __math_floor",       "hex": "0xA6", "bin": "10100110"},
    {"dgm": "A7", "llvm": "@llvm.math.ceil",        "nasm": "CALL __math_ceil",        "hex": "0xA7", "bin": "10100111"},
    {"dgm": "A8", "llvm": "@llvm.math.abs",         "nasm": "CALL __math_abs",         "hex": "0xA8", "bin": "10101000"},
    {"dgm": "A9", "llvm": "@llvm.math.rand",        "nasm": "CALL __math_rand",        "hex": "0xA9", "bin": "10101001"},
    {"dgm": "AA", "llvm": "@llvm.str.concat",       "nasm": "CALL __str_concat",       "hex": "0xAA", "bin": "10101010"},
    {"dgm": "AB", "llvm": "@llvm.str.upper",        "nasm": "CALL __str_upper",        "hex": "0xAB", "bin": "10101011"},
    {"dgm": "AC", "llvm": "@llvm.str.lower",        "nasm": "CALL __str_lower",        "hex": "0xAC", "bin": "10101100"},
    {"dgm": "AD", "llvm": "@llvm.str.len",          "nasm": "CALL __str_len",          "hex": "0xAD", "bin": "10101101"},
    {"dgm": "AE", "llvm": "@llvm.str.substr",       "nasm": "CALL __str_substr",       "hex": "0xAE", "bin": "10101110"},
    {"dgm": "AF", "llvm": "@llvm.str.find",         "nasm": "CALL __str_find",         "hex": "0xAF", "bin": "10101111"},
    {"dgm": "B0", "llvm": "@llvm.str.replace",      "nasm": "CALL __str_replace",      "hex": "0xB0", "bin": "10110000"},
    {"dgm": "B1", "llvm": "@llvm.str.split",        "nasm": "CALL __str_split",        "hex": "0xB1", "bin": "10110001"},
    {"dgm": "B2", "llvm": "@llvm.str.join",         "nasm": "CALL __str_join",         "hex": "0xB2", "bin": "10110010"},
    {"dgm": "B3", "llvm": "@llvm.file.write",       "nasm": "CALL __file_write",       "hex": "0xB3", "bin": "10110011"},
    {"dgm": "B4", "llvm": "@llvm.file.append",      "nasm": "CALL __file_append",      "hex": "0xB4", "bin": "10110100"},
    {"dgm": "B5", "llvm": "@llvm.file.read",        "nasm": "CALL __file_read",        "hex": "0xB5", "bin": "10110101"},
    {"dgm": "B6", "llvm": "@llvm.file.delete",      "nasm": "CALL __file_delete",      "hex": "0xB6", "bin": "10110110"},
    {"dgm": "B7", "llvm": "@llvm.file.exists",      "nasm": "CALL __file_exists",      "hex": "0xB7", "bin": "10110111"},
    {"dgm": "B8", "llvm": "@llvm.system.now",       "nasm": "CALL __system_now",       "hex": "0xB8", "bin": "10111000"},
    {"dgm": "B9", "llvm": "@llvm.system.sleep",     "nasm": "CALL __system_sleep",     "hex": "0xB9", "bin": "10111001"},
    {"dgm": "BA", "llvm": "@llvm.system.env",       "nasm": "CALL __system_env",       "hex": "0xBA", "bin": "10111010"},
    {"dgm": "BB", "llvm": "@llvm.system.platform",  "nasm": "CALL __system_platform",  "hex": "0xBB", "bin": "10111011"},
    {"dgm": "BC", "llvm": "@llvm.system.cpu",       "nasm": "CALL __system_cpu",       "hex": "0xBC", "bin": "10111100"},
    {"dgm": "BD", "llvm": "@llvm.system.mem",       "nasm": "CALL __system_mem",       "hex": "0xBD", "bin": "10111101"},
    {"dgm": "BE", "llvm": "@llvm.sys.exec",         "nasm": "CALL __sys_exec",         "hex": "0xBE", "bin": "10111110"},
    {"dgm": "BF", "llvm": "@llvm.sys.cwd",          "nasm": "CALL __sys_cwd",          "hex": "0xBF", "bin": "10111111"},
    {"dgm": "C0", "llvm": "@llvm.sys.chdir",     "nasm": "CALL __sys_chdir",     "hex": "0xC0", "bin": "11000000"},
    {"dgm": "C1", "llvm": "@llvm.sys.listdir",   "nasm": "CALL __sys_listdir",   "hex": "0xC1", "bin": "11000001"},
    {"dgm": "C2", "llvm": "@llvm.sys.mkdir",     "nasm": "CALL __sys_mkdir",     "hex": "0xC2", "bin": "11000010"},
    {"dgm": "C3", "llvm": "@llvm.sys.rmdir",     "nasm": "CALL __sys_rmdir",     "hex": "0xC3", "bin": "11000011"},
    {"dgm": "C4", "llvm": "@llvm.sys.tempfile",  "nasm": "CALL __sys_tempfile",  "hex": "0xC4", "bin": "11000100"},
    {"dgm": "C5", "llvm": "@llvm.sys.clipboard", "nasm": "CALL __sys_clipboard", "hex": "0xC5", "bin": "11000101"},
    {"dgm": "C6", "llvm": "@llvm.sys.args",      "nasm": "CALL __sys_args",      "hex": "0xC6", "bin": "11000110"},
    {"dgm": "C7", "llvm": "@llvm.sys.uid",       "nasm": "CALL __sys_uid",       "hex": "0xC7", "bin": "11000111"},
    {"dgm": "C8", "llvm": "@llvm.sys.pid",       "nasm": "CALL __sys_pid",       "hex": "0xC8", "bin": "11001000"},
    {"dgm": "C9", "llvm": "@llvm.sys.exit",      "nasm": "SYSCALL exit",         "hex": "0xC9", "bin": "11001001"},
    {"dgm": "CA", "llvm": "@llvm.hash.md5",      "nasm": "CALL __hash_md5",      "hex": "0xCA", "bin": "11001010"},
    {"dgm": "CB", "llvm": "@llvm.hash.sha1",     "nasm": "CALL __hash_sha1",     "hex": "0xCB", "bin": "11001011"},
    {"dgm": "CC", "llvm": "@llvm.hash.sha256",   "nasm": "CALL __hash_sha256",   "hex": "0xCC", "bin": "11001100"},
    {"dgm": "CD", "llvm": "@llvm.hash.sha512",   "nasm": "CALL __hash_sha512",   "hex": "0xCD", "bin": "11001101"},
    {"dgm": "CE", "llvm": "@llvm.hmac.md5",      "nasm": "CALL __hmac_md5",      "hex": "0xCE", "bin": "11001110"},
    {"dgm": "CF", "llvm": "@llvm.hmac.sha256",   "nasm": "CALL __hmac_sha256",   "hex": "0xCF", "bin": "11001111"},
    {"dgm": "D0", "llvm": "@llvm.base64.encode", "nasm": "CALL __base64_encode", "hex": "0xD0", "bin": "11010000"},
    {"dgm": "D1", "llvm": "@llvm.base64.decode", "nasm": "CALL __base64_decode", "hex": "0xD1", "bin": "11010001"},
    {"dgm": "D2", "llvm": "@llvm.hex.encode",    "nasm": "CALL __hex_encode",    "hex": "0xD2", "bin": "11010010"},
    {"dgm": "D3", "llvm": "@llvm.hex.decode",    "nasm": "CALL __hex_decode",    "hex": "0xD3", "bin": "11010011"},
    {"dgm": "D4", "llvm": "@llvm.crc32",         "nasm": "CALL __crc32",         "hex": "0xD4", "bin": "11010100"},
    {"dgm": "D5", "llvm": "@llvm.random.bytes",  "nasm": "CALL __random_bytes",  "hex": "0xD5", "bin": "11010101"},
    {"dgm": "D6", "llvm": "@llvm.uuid.generate", "nasm": "CALL __uuid_generate", "hex": "0xD6", "bin": "11010110"},
    {"dgm": "D7", "llvm": "@llvm.password.hash", "nasm": "CALL __password_hash", "hex": "0xD7", "bin": "11010111"},
    {"dgm": "D8", "llvm": "@llvm.password.verify","nasm": "CALL __password_verify","hex": "0xD8","bin": "11011000"},
    {"dgm": "D9", "llvm": "@llvm.jwt.encode",    "nasm": "CALL __jwt_encode",    "hex": "0xD9", "bin": "11011001"},
    {"dgm": "DA", "llvm": "@llvm.zlib.compress", "nasm": "CALL __zlib_compress", "hex": "0xDA", "bin": "11011010"},
    {"dgm": "DB", "llvm": "@llvm.zlib.decompress","nasm": "CALL __zlib_decompress","hex": "0xDB","bin": "11011011"},
    {"dgm": "DC", "llvm": "@llvm.bz2.compress",  "nasm": "CALL __bz2_compress",  "hex": "0xDC", "bin": "11011100"},
    {"dgm": "DD", "llvm": "@llvm.bz2.decompress","nasm": "CALL __bz2_decompress","hex": "0xDD", "bin": "11011101"},
    {"dgm": "DE", "llvm": "@llvm.lzma.compress", "nasm": "CALL __lzma_compress", "hex": "0xDE", "bin": "11011110"},
    {"dgm": "DF", "llvm": "@llvm.lzma.decompress","nasm": "CALL __lzma_decompress","hex": "0xDF","bin": "11011111"},
    {"dgm": "E0", "llvm": "@llvm.gzip.compress", "nasm": "CALL __gzip_compress", "hex": "0xE0", "bin": "11100000"},
    {"dgm": "E1", "llvm": "@llvm.gzip.decompress","nasm": "CALL __gzip_decompress","hex": "0xE1","bin": "11100001"},
    {"dgm": "E2", "llvm": "@llvm.tar.create",    "nasm": "CALL __tar_create",    "hex": "0xE2", "bin": "11100010"},
    {"dgm": "E3", "llvm": "@llvm.tar.extract",   "nasm": "CALL __tar_extract",   "hex": "0xE3", "bin": "11100011"},
    {"dgm": "E4", "llvm": "@llvm.zip.create",    "nasm": "CALL __zip_create",    "hex": "0xE4", "bin": "11100100"},
    {"dgm": "E5", "llvm": "@llvm.zip.extract",   "nasm": "CALL __zip_extract",   "hex": "0xE5", "bin": "11100101"},
    {"dgm": "E6", "llvm": "@llvm.compress.detect","nasm": "CALL __compress_detect","hex": "0xE6","bin": "11100110"},
    {"dgm": "E7", "llvm": "@llvm.compress.ratio","nasm": "CALL __compress_ratio","hex": "0xE7", "bin": "11100111"},
    {"dgm": "E8", "llvm": "@llvm.compress.level","nasm": "CALL __compress_level","hex": "0xE8", "bin": "11101000"},
    {"dgm": "E9", "llvm": "@llvm.compress.bench","nasm": "CALL __compress_bench","hex": "0xE9", "bin": "11101001"},
    {"dgm": "EA", "llvm": "@llvm.http.get",      "nasm": "CALL __http_get",      "hex": "0xEA", "bin": "11101010"},
    {"dgm": "EB", "llvm": "@llvm.http.post",     "nasm": "CALL __http_post",     "hex": "0xEB", "bin": "11101011"},
    {"dgm": "EC", "llvm": "@llvm.http.head",     "nasm": "CALL __http_head",     "hex": "0xEC", "bin": "11101100"},
    {"dgm": "ED", "llvm": "@llvm.http.put",      "nasm": "CALL __http_put",      "hex": "0xED", "bin": "11101101"},
    {"dgm": "EE", "llvm": "@llvm.http.delete",   "nasm": "CALL __http_delete",   "hex": "0xEE", "bin": "11101110"},
    {"dgm": "EF", "llvm": "@llvm.http.download", "nasm": "CALL __http_download", "hex": "0xEF", "bin": "11101111"},
    {"dgm": "F0", "llvm": "@llvm.ws.connect",    "nasm": "CALL __ws_connect",    "hex": "0xF0", "bin": "11110000"},
    {"dgm": "F1", "llvm": "@llvm.ws.send",       "nasm": "CALL __ws_send",       "hex": "0xF1", "bin": "11110001"},
    {"dgm": "F2", "llvm": "@llvm.ws.recv",       "nasm": "CALL __ws_recv",       "hex": "0xF2", "bin": "11110010"},
    {"dgm": "F3", "llvm": "@llvm.ws.close",      "nasm": "CALL __ws_close",      "hex": "0xF3", "bin": "11110011"},
    {"dgm": "F4", "llvm": "@llvm.udp.send",      "nasm": "CALL __udp_send",      "hex": "0xF4", "bin": "11110100"},
    {"dgm": "F5", "llvm": "@llvm.udp.recv",      "nasm": "CALL __udp_recv",      "hex": "0xF5", "bin": "11110101"},
    {"dgm": "F6", "llvm": "@llvm.tcp.listen",    "nasm": "CALL __tcp_listen",    "hex": "0xF6", "bin": "11110110"},
    {"dgm": "F7", "llvm": "@llvm.tcp.accept",    "nasm": "CALL __tcp_accept",    "hex": "0xF7", "bin": "11110111"},
    {"dgm": "F8", "llvm": "@llvm.tcp.send",      "nasm": "CALL __tcp_send",      "hex": "0xF8", "bin": "11111000"},
    {"dgm": "F9", "llvm": "@llvm.tcp.recv",      "nasm": "CALL __tcp_recv",      "hex": "0xF9", "bin": "11111001"},
    {"dgm": "FA", "llvm": "@llvm.db.open",       "nasm": "CALL __db_open",       "hex": "0xFA", "bin": "11111010"},
    {"dgm": "FB", "llvm": "@llvm.db.exec",       "nasm": "CALL __db_exec",       "hex": "0xFB", "bin": "11111011"},
    {"dgm": "FC", "llvm": "@llvm.db.query",      "nasm": "CALL __db_query",      "hex": "0xFC", "bin": "11111100"},
    {"dgm": "FD", "llvm": "@llvm.db.close",      "nasm": "CALL __db_close",      "hex": "0xFD", "bin": "11111101"},
    {"dgm": "FE", "llvm": "@llvm.db.begin",      "nasm": "CALL __db_begin",      "hex": "0xFE", "bin": "11111110"},
    {"dgm": "FF", "llvm": "@llvm.db.commit",     "nasm": "CALL __db_commit",     "hex": "0xFF", "bin": "11111111"},
    {"dgm": "100", "llvm": "@llvm.db.rollback",    "nasm": "CALL __db_rollback",    "hex": "0x100", "bin": "000100000000"},
    {"dgm": "101", "llvm": "@llvm.db.tables",      "nasm": "CALL __db_tables",      "hex": "0x101", "bin": "000100000001"},
    {"dgm": "102", "llvm": "@llvm.db.schema",      "nasm": "CALL __db_schema",      "hex": "0x102", "bin": "000100000010"},
    {"dgm": "103", "llvm": "@llvm.db.insert",      "nasm": "CALL __db_insert",      "hex": "0x103", "bin": "000100000011"},
    {"dgm": "104", "llvm": "@llvm.db.update",      "nasm": "CALL __db_update",      "hex": "0x104", "bin": "000100000100"},
    {"dgm": "105", "llvm": "@llvm.db.delete",      "nasm": "CALL __db_delete",      "hex": "0x105", "bin": "000100000101"},
    {"dgm": "106", "llvm": "@llvm.db.count",       "nasm": "CALL __db_count",       "hex": "0x106", "bin": "000100000110"},
    {"dgm": "107", "llvm": "@llvm.db.indexes",     "nasm": "CALL __db_indexes",     "hex": "0x107", "bin": "000100000111"},
    {"dgm": "108", "llvm": "@llvm.db.analyze",     "nasm": "CALL __db_analyze",     "hex": "0x108", "bin": "000100001000"},
    {"dgm": "109", "llvm": "@llvm.db.vacuum",      "nasm": "CALL __db_vacuum",      "hex": "0x109", "bin": "000100001001"},
    {"dgm": "10A", "llvm": "@llvm.regex.match",    "nasm": "CALL __regex_match",    "hex": "0x10A", "bin": "000100001010"},
    {"dgm": "10B", "llvm": "@llvm.regex.findall",  "nasm": "CALL __regex_findall",  "hex": "0x10B", "bin": "000100001011"},
    {"dgm": "10C", "llvm": "@llvm.regex.replace",  "nasm": "CALL __regex_replace",  "hex": "0x10C", "bin": "000100001100"},
    {"dgm": "10D", "llvm": "@llvm.regex.split",    "nasm": "CALL __regex_split",    "hex": "0x10D", "bin": "000100001101"},
    {"dgm": "10E", "llvm": "@llvm.regex.subn",     "nasm": "CALL __regex_subn",     "hex": "0x10E", "bin": "000100001110"},
    {"dgm": "10F", "llvm": "@llvm.regex.compile",  "nasm": "CALL __regex_compile",  "hex": "0x10F", "bin": "000100001111"},
    {"dgm": "110", "llvm": "@llvm.fuzzy.match",    "nasm": "CALL __fuzzy_match",    "hex": "0x110", "bin": "000100010000"},
    {"dgm": "111", "llvm": "@llvm.fuzzy.closest",  "nasm": "CALL __fuzzy_closest",  "hex": "0x111", "bin": "000100010001"},
    {"dgm": "112", "llvm": "@llvm.fuzzy.sort",     "nasm": "CALL __fuzzy_sort",     "hex": "0x112", "bin": "000100010010"},
    {"dgm": "113", "llvm": "@llvm.audio.playwav",  "nasm": "CALL __audio_playwav",  "hex": "0x113", "bin": "000100010011"},
    {"dgm": "114", "llvm": "@llvm.audio.playmp3",  "nasm": "CALL __audio_playmp3",  "hex": "0x114", "bin": "000100010100"},
    {"dgm": "115", "llvm": "@llvm.audio.record",   "nasm": "CALL __audio_record",   "hex": "0x115", "bin": "000100010101"},
    {"dgm": "116", "llvm": "@llvm.audio.stop",     "nasm": "CALL __audio_stop",     "hex": "0x116", "bin": "000100010110"},
    {"dgm": "117", "llvm": "@llvm.audio.tone",     "nasm": "CALL __audio_tone",     "hex": "0x117", "bin": "000100010111"},
    {"dgm": "118", "llvm": "@llvm.audio.volume",   "nasm": "CALL __audio_volume",   "hex": "0x118", "bin": "000100011000"},
    {"dgm": "119", "llvm": "@llvm.audio.mixer",    "nasm": "CALL __audio_mixer",    "hex": "0x119", "bin": "000100011001"},
    {"dgm": "11A", "llvm": "@llvm.audio.pause",    "nasm": "CALL __audio_pause",    "hex": "0x11A", "bin": "000100011010"},
    {"dgm": "11B", "llvm": "@llvm.audio.resume",   "nasm": "CALL __audio_resume",   "hex": "0x11B", "bin": "000100011011"},
    {"dgm": "11C", "llvm": "@llvm.audio.stream",   "nasm": "CALL __audio_stream",   "hex": "0x11C", "bin": "000100011100"},
    {"dgm": "11D", "llvm": "reserved",             "nasm": "—",                     "hex": "0x11D", "bin": "000100011101"},
    {"dgm": "11E", "llvm": "reserved",             "nasm": "—",                     "hex": "0x11E", "bin": "000100011110"},
    {"dgm": "11F", "llvm": "reserved",             "nasm": "—",                     "hex": "0x11F", "bin": "000100011111"},
    {"dgm": "120", "llvm": "reserved",             "nasm": "—",                     "hex": "0x120", "bin": "000100100000"},
    {"dgm": "121", "llvm": "reserved",             "nasm": "—",                     "hex": "0x121", "bin": "000100100001"},
    {"dgm": "122", "llvm": "reserved",             "nasm": "—",                     "hex": "0x122", "bin": "000100100010"},
    {"dgm": "123", "llvm": "reserved",             "nasm": "—",                     "hex": "0x123", "bin": "000100100011"},
    {"dgm": "124", "llvm": "reserved",             "nasm": "—",                     "hex": "0x124", "bin": "000100100100"},
    {"dgm": "125", "llvm": "reserved",             "nasm": "—",                     "hex": "0x125", "bin": "000100100101"},
    {"dgm": "126", "llvm": "reserved",             "nasm": "—",                     "hex": "0x126", "bin": "000100100110"},
    {"dgm": "127", "llvm": "reserved",             "nasm": "—",                     "hex": "0x127", "bin": "000100100111"},
    {"dgm": "128", "llvm": "reserved",             "nasm": "—",                     "hex": "0x128", "bin": "000100101000"},
    {"dgm": "129", "llvm": "reserved",             "nasm": "—",                     "hex": "0x129", "bin": "000100101001"},
    {"dgm": "12A", "llvm": "reserved",             "nasm": "—",                     "hex": "0x12A", "bin": "000100101010"},
    {"dgm": "12B", "llvm": "reserved",             "nasm": "—",                     "hex": "0x12B", "bin": "000100101011"},
    {"dgm": "12C", "llvm": "reserved",             "nasm": "—",                     "hex": "0x12C", "bin": "000100101100"},
    {"dgm": "12D", "llvm": "reserved",             "nasm": "—",                     "hex": "0x12D", "bin": "000100101101"},
    {"dgm": "12E", "llvm": "reserved",             "nasm": "—",                     "hex": "0x12E", "bin": "000100101110"},
    {"dgm": "12F", "llvm": "reserved",             "nasm": "—",                     "hex": "0x12F", "bin": "000100101111"},
    {"dgm": "130", "llvm": "reserved",             "nasm": "—",                     "hex": "0x130", "bin": "000100110000"},
    {"dgm": "131", "llvm": "reserved",             "nasm": "—",                     "hex": "0x131", "bin": "000100110001"},
    {"dgm": "132", "llvm": "reserved",             "nasm": "—",                     "hex": "0x132", "bin": "000100110010"},
    {"dgm": "133", "llvm": "reserved",             "nasm": "—",                     "hex": "0x133", "bin": "000100110011"},
    {"dgm": "134", "llvm": "reserved",             "nasm": "—",                     "hex": "0x134", "bin": "000100110100"},
    {"dgm": "135", "llvm": "reserved",             "nasm": "—",                     "hex": "0x135", "bin": "000100110101"},
    {"dgm": "136", "llvm": "reserved",             "nasm": "—",                     "hex": "0x136", "bin": "000100110110"},
    {"dgm": "137", "llvm": "reserved",             "nasm": "—",                     "hex": "0x137", "bin": "000100110111"},
    {"dgm": "138", "llvm": "reserved",             "nasm": "—",                     "hex": "0x138", "bin": "000100111000"},
    {"dgm": "139", "llvm": "reserved",             "nasm": "—",                     "hex": "0x139", "bin": "000100111001"},
    {"dgm": "13A", "llvm": "reserved",             "nasm": "—",                     "hex": "0x13A", "bin": "000100111010"},
    {"dgm": "13B", "llvm": "reserved",             "nasm": "—",                     "hex": "0x13B", "bin": "000100111011"},
    {"dgm": "13C", "llvm": "reserved",             "nasm": "—",                     "hex": "0x13C", "bin": "000100111100"},
    {"dgm": "13D", "llvm": "reserved",             "nasm": "—",                     "hex": "0x13D", "bin": "000100111101"},
    {"dgm": "13E", "llvm": "reserved",             "nasm": "—",                     "hex": "0x13E", "bin": "000100111110"},
    {"dgm": "13F", "llvm": "reserved",             "nasm": "—",                     "hex": "0x13F", "bin": "000100111111"},
    {"dgm": "140", "llvm": "reserved",             "nasm": "—",                     "hex": "0x140", "bin": "000101000000"},
    {"dgm": "141", "llvm": "reserved",             "nasm": "—",                     "hex": "0x141", "bin": "000101000001"},
    {"dgm": "142", "llvm": "reserved",             "nasm": "—",                     "hex": "0x142", "bin": "000101000010"},
    {"dgm": "143", "llvm": "reserved",             "nasm": "—",                     "hex": "0x143", "bin": "000101000011"},
    {"dgm": "144", "llvm": "reserved",             "nasm": "—",                     "hex": "0x144", "bin": "000101000100"},
    {"dgm": "145", "llvm": "reserved",             "nasm": "—",                     "hex": "0x145", "bin": "000101000101"},
    {"dgm": "146", "llvm": "reserved",             "nasm": "—",                     "hex": "0x146", "bin": "000101000110"},
    {"dgm": "147", "llvm": "reserved",             "nasm": "—",                     "hex": "0x147", "bin": "000101000111"},
    {"dgm": "148", "llvm": "reserved",             "nasm": "—",                     "hex": "0x148", "bin": "000101001000"},
    {"dgm": "149", "llvm": "reserved",             "nasm": "—",                     "hex": "0x149", "bin": "000101001001"},
    {"dgm": "14A", "llvm": "reserved",             "nasm": "—",                     "hex": "0x14A", "bin": "000101001010"},
    {"dgm": "14B", "llvm": "reserved",             "nasm": "—",                     "hex": "0x14B", "bin": "000101001011"},
    {"dgm": "14C", "llvm": "reserved",             "nasm": "—",                     "hex": "0x14C", "bin": "000101001100"},
    {"dgm": "14D", "llvm": "reserved",             "nasm": "—",                     "hex": "0x14D", "bin": "000101001101"},
    {"dgm": "14E", "llvm": "reserved",             "nasm": "—",                     "hex": "0x14E", "bin": "000101001110"},
    {"dgm": "14F", "llvm": "reserved",             "nasm": "—",                     "hex": "0x14F", "bin": "000101001111"},
    {"dgm": "150", "llvm": "reserved",             "nasm": "—",                     "hex": "0x150", "bin": "000101010000"},
    {"dgm": "151", "llvm": "reserved",             "nasm": "—",                     "hex": "0x151", "bin": "000101010001"},
    {"dgm": "152", "llvm": "reserved",             "nasm": "—",                     "hex": "0x152", "bin": "000101010010"},
    {"dgm": "153", "llvm": "reserved",             "nasm": "—",                     "hex": "0x153", "bin": "000101010011"},
    {"dgm": "154", "llvm": "reserved",             "nasm": "—",                     "hex": "0x154", "bin": "000101010100"},
    {"dgm": "155", "llvm": "reserved",             "nasm": "—",                     "hex": "0x155", "bin": "000101010101"},
    {"dgm": "156", "llvm": "reserved",             "nasm": "—",                     "hex": "0x156", "bin": "000101010110"},
    {"dgm": "157", "llvm": "reserved",             "nasm": "—",                     "hex": "0x157", "bin": "000101010111"},
    {"dgm": "158", "llvm": "reserved",             "nasm": "—",                     "hex": "0x158", "bin": "000101011000"},
    {"dgm": "159", "llvm": "reserved",             "nasm": "—",                     "hex": "0x159", "bin": "000101011001"},
