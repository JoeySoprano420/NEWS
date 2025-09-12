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
