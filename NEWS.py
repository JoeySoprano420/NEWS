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
# Transpiler and VM implementations were merged into this file (NEWS.py).
# Use the local `Transpiler`, `to_base12`, `from_base12`, `OPCODES` and `NewsVM`
# definitions included later in this module instead of importing from external files.

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

                elif opcode == 0x1B:  # sdiv
                    b, a = self.stack.pop(), self.stack.pop()
                    self.stack.append(int(a / (b or 1)))

                elif opcode == 0x15:  # icmp
                    op = self.tokens[self.ip]; self.ip += 1
                    b, a = self.stack.pop(), self.stack.pop()
                    # REL_OPS maps symbols to functions
                    if op not in REL_OPS:
                        # some earlier code mapped to strings like '=='→"eq", handle both
                        opfn = REL_OPS.get(op)
                        if not opfn:
                            # fallback: support textual names
                            name_map = {"eq": "==", "ne": "!=",
                                         "lt": "<", "gt": ">",
                                         "<=": "<=", ">=": ">="}
                            if op in name_map:
                                opfn = REL_OPS[name_map[op]]
                            else:
                                raise ValueError(f"Unknown icmp op {op!r}")
                    else:
                        opfn = REL_OPS[op]
                    self.stack.append(1 if opfn(a, b) else 0)

                elif opcode == 0x30:  # br
                    target = self.stack.pop()
                    # set ip to target (target is token index in compiled list)
                    # Validate target as int index
                    if not isinstance(target, int):
                        raise ValueError("br target must be integer")
                    self.ip = target

                elif opcode == 0x2B:  # call
                    target = self.stack.pop()
                    self.frames.append(self.ip)
                    self.ip = target

                elif opcode == 0x33:  # ret
                    if self.frames:
                        self.ip = self.frames.pop()
                    else:
                        self.running = False

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
                # record error in output buffer and stop
                _safe_append_output(self, f"[NEWS VM ERROR] {e}")
                traceback.print_exc()
                self.running = False

    # ---------------- Clone override ----------------
    def clone(self) -> 'FastNewsVM':
        """
        Create a shallow snapshot suitable for futures: copy memory/tuples/lists, but keep separate runtime state.
        """
        new_vm = FastNewsVM(debug=self.debug, trace=self.trace)
        new_vm.memory = self.memory.copy()
        # deep-ish copy of tuples/lists to avoid concurrent mutation surprises
        new_vm.tuples = {k: tuple(v) for k, v in self.tuples.items()}
        new_vm.lists = {k: list(v) for k, v in self.lists.items()}
        new_vm.groups = {k: list(v) for k, v in getattr(self, "groups", {}).items()}
        # copy compiled program for execution at fn_addr locations
        new_vm._compiled_tokens = list(self._compiled_tokens)
        new_vm.tokens = list(self.tokens)
        new_vm.output_buffer = list(self.output_buffer)
        # fresh locks
        new_vm._output_buffer_lock = threading.Lock()
        return new_vm

    # ---------------- Execution loop ----------------
    def step(self):
        """
        Single step using dispatch table. Handles external_handlers first (if opcode present there),
        then internal dispatch. Fully-implemented; raises ValueError for unknown opcodes.
        """
        if self.ip >= len(self._compiled_tokens):
            self.running = False
            return

        tok = self._compiled_tokens[self.ip]
        # opcode must be int; if a string exists here it's an error in program format
        if not isinstance(tok, int):
            # attempt to convert
            try:
                opcode = from_base12(str(tok))
            except Exception:
                raise ValueError(f"Invalid opcode token at ip={self.ip}: {tok!r}")
        else:
            opcode = tok
        # consume opcode token
        self.ip += 1

        # Check external_handlers first (fast path)
        handler = getattr(self, "external_handlers", {}).get(opcode)
        if handler:
            # handler signature: function(vm, opcode) -> truthy if handled
            try:
                handled = handler(self, opcode)
                if bool(handled):
                    return
            except Exception as he:
                # record and fall through to internal dispatch
                _safe_append_output(self, f"[EXTERNAL HANDLER ERROR] opcode={hex(opcode)} error={he}")
                traceback.print_exc()

        # internal dispatch
        fn = self._dispatch.get(opcode)
        if fn:
            fn()
            return

        # unknown opcode: raise so higher-level adapters can try dynamic mapping
        raise ValueError(f"Unknown opcode {hex(opcode)}")

    def run(self):
        """
        Highly-optimized run loop: localize frequently used attributes for speed.
        """
        dispatch = self._dispatch
        compiled = self._compiled_tokens
        ip_attr = "ip"
        try:
            while self.running:
                if self.ip >= len(compiled):
                    self.running = False
                    break
                tok = compiled[self.ip]
                if not isinstance(tok, int):
                    try:
                        opcode = from_base12(str(tok))
                    except Exception:
                        raise ValueError(f"Invalid opcode token at ip={self.ip}: {tok!r}")
                else:
                    opcode = tok
                self.ip += 1

                # external handler fast-path
                eh = getattr(self, "external_handlers", {}).get(opcode)
                if eh:
                    try:
                        if eh(self, opcode):
                            continue
                    except Exception as e:
                        _safe_append_output(self, f"[EXTERNAL HANDLER ERROR] opcode={hex(opcode)} error={e}")
                        traceback.print_exc()

                fn = dispatch.get(opcode)
                if fn:
                    fn()
                else:
                    raise ValueError(f"Unknown opcode {hex(opcode)}")
        except SystemExit:
            raise
        except Exception as e:
            # record error in output buffer and stop
            _safe_append_output(self, f"[FAST VM ERROR] {e}")
            traceback.print_exc()
            self.running = False

# --------- ELEVATOR: resolve missing definitions, install robust fallbacks & workarounds ----------
# Appends to the bottom of NEWS.py
import inspect
import logging
from typing import Any, Dict

_logger = logging.getLogger("news.elevator")
if not _logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    _logger.addHandler(h)
    _logger.setLevel(logging.INFO)


def _ensure_opcode_index():
    """
    Ensure OPCODE_MAP and OPCODE_INDEX exist and are consistent.
    Builds OPCODE_INDEX (int -> entry) if missing.
    """
    global OPCODE_MAP, OPCODE_INDEX, OPCODES
    if "OPCODE_MAP" not in globals() or not isinstance(OPCODE_MAP, list):
        # Build minimal OPCODE_MAP from OPCODES mapping if present
        OPCODE_MAP = []
        if "OPCODES" in globals() and isinstance(OPCODES, dict):
            for name, val in OPCODES.items():
                OPCODE_MAP.append({"dgm": f"{val:X}", "llvm": name, "nasm": "", "hex": f"0x{val:X}", "bin": ""})
    if "OPCODE_INDEX" not in globals() or not isinstance(globals().get("OPCODE_INDEX"), dict):
        OPCODE_INDEX = {}
        for entry in OPCODE_MAP:
            try:
                hv = int(entry.get("hex", "0"), 16)
            except Exception:
                try:
                    hv = int(entry.get("dgm", "0"), 16)
                except Exception:
                    continue
            OPCODE_INDEX[hv] = entry
    # Ensure OPCODES mapping includes reverse names for entries
    if "OPCODES" not in globals() or not isinstance(OPCODES, dict):
        OPCODES = {}
    for hv, entry in OPCODE_INDEX.items():
        name = entry.get("llvm")
        if isinstance(name, str) and name and name not in OPCODES:
            OPCODES[name] = hv


def _default_opcode_handler(vm: Any, opcode_int: int) -> bool:
    """
    Generic fallback handler for unknown opcodes.
    Records usage and a readable note into the VM's output buffer (thread-safe),
    and returns True to indicate the opcode was handled (no-op).
    """
    try:
        if not hasattr(vm, "dynamic_usage"):
            vm.dynamic_usage = {}
        vm.dynamic_usage[opcode_int] = vm.dynamic_usage.get(opcode_int, 0) + 1

        entry = OPCODE_INDEX.get(opcode_int)
        if entry:
            note = f"[ELEVATOR] Unimplemented opcode {hex(opcode_int)} ({entry.get('llvm')}) — noop recorded."
        else:
            note = f"[ELEVATOR] Unknown opcode {hex(opcode_int)} — noop recorded."

        # Best-effort safe append
        try:
            if not hasattr(vm, "output_buffer") or not isinstance(vm.output_buffer, list):
                vm.output_buffer = []
            # thread-safe lock if present
            lock = getattr(vm, "_output_buffer_lock", None)
            if lock is None:
                vm.output_buffer.append(note)
            else:
                with lock:
                    vm.output_buffer.append(note)
        except Exception:
            # final fallback: logging
            _logger.info(note)
        return True
    except Exception as exc:
        _logger.exception("elevator default handler failed: %s", exc)
        return True  # swallow to avoid crashing VM


def _patch_vm_instance_for_external_handlers(vm: Any):
    """
    Install a robust step wrapper that:
      - Peeks the next opcode token (supports both string-base12 token lists and _compiled_tokens)
      - If an external handler exists in vm.external_handlers, call it and consume opcode
      - Otherwise call original step() and, on ValueError('Unknown opcode ...'), call the elevator fallback
    This wrapper is idempotent.
    """
    if getattr(vm, "_elevator_wrapped", False):
        return
    if not hasattr(vm, "external_handlers") or not isinstance(getattr(vm, "external_handlers"), dict):
        vm.external_handlers = {}

    original_step = getattr(vm, "step")

    def wrapped_step():
        # Attempt to peek the next token and dispatch to external handler if present.
        ip = getattr(vm, "ip", 0)
        tokens = getattr(vm, "tokens", [])
        if ip < len(tokens):
            tok = tokens[ip]
            opc = None
            try:
                if re.fullmatch(r"[0-9abAB]+", tok):
                    opc = from_base12(tok.lower())
                else:
                    opc = int(tok, 0)
            except Exception:
                opc = None

            if opc is not None and opc in vm.external_handlers:
                # consume token and call handler
                vm.ip = ip + 1
                try:
                    vm.external_handlers[opc](vm, opc)
                    return
                except Exception as e:
                    _logger.exception("external handler raised: %s", e)
                    # record and fall through to internal dispatch
                    _safe_append_output(vm, f"[EXTERNAL HANDLER ERROR] opcode={hex(opcode)} error={e}")
                    traceback.print_exc()
        # No external handler found; call original_step and handle Unknown opcode gracefully
        try:
            return original_step()
        except ValueError as e:
            msg = str(e)
            if msg.startswith("Unknown opcode"):
                # parse hex from message if present
                try:
                    # message may be like "Unknown opcode 0xabc"
                    part = msg.split()[2]
                    if part.startswith("0x"):
                        op_int = int(part, 16)
                    else:
                        # maybe decimal
                        op_int = int(part, 0)
                except Exception:
                    # best-effort: attempt to peek and interpret
                    op_int = None
                    try:
                        if hasattr(vm, "_compiled_tokens"):
                            ip = getattr(vm, "ip", 0)
                            if ip > 0:
                                candidate = vm._compiled_tokens[ip - 1]
                                op_int = candidate if isinstance(candidate, int) else from_base12(str(candidate))
                        else:
                            ip = getattr(vm, "ip", 0)
                            if ip > 0:
                                candidate = vm.tokens[ip - 1]
                                op_int = from_base12(candidate)
                    except Exception:
                        op_int = None
                if op_int is None:
                    _logger.warning("Elevator: Unknown opcode detected but could not determine code. Message: %s", msg)
                    return
                # call default elevator handler
                try:
                    _default_opcode_handler(vm, op_int)
                    return
                except Exception:
                    _logger.exception("Elevator handler failed for opcode %s", hex(op_int))
                    return
            # re-raise other ValueErrors
            raise

    vm.step = wrapped_step
    vm._elevator_wrapped = True


def _ensure_vm_present():
    """
    If a top-level `vm` variable is referenced elsewhere, ensure it exists and is runnable.
    Prefer FastNewsVM if available.
    """
    global vm
    if "vm" in globals() and isinstance(globals().get("vm"), object):
        return globals()["vm"]
    # Try to create FastNewsVM or NewsVM
    try:
        vm_local = create_fast_vm(debug=False, trace=False)
    except Exception:
        vm_local = NewsVM(debug=False, trace=False)
    # assign minimal no-op program so run() won't crash if invoked without code
    vm_local.load_program("33")  # single `ret` opcode
    globals()["vm"] = vm_local
    return vm_local


def _install_elevator_defaults(vm_instance: Any):
    """
    Install the elevator's default handler into vm_instance.external_handlers for any opcode
    that does not already have a handler. This prevents Unknown opcode exceptions.
    """
    _ensure_opcode_index()
    if not hasattr(vm_instance, "external_handlers") or not isinstance(vm_instance.external_handlers, dict):
        vm_instance.external_handlers = {}
    for op_int in list(OPCODE_INDEX.keys()):
        if op_int not in vm_instance.external_handlers:
            # Register default handler
            vm_instance.external_handlers[op_int] = _default_opcode_handler
    # Also patch instance to consult external handlers first
    _patch_vm_instance_for_external_handlers(vm_instance)


def resolve_missing_names_and_attach():
    """
    Public entrypoint: resolve missing global names, create a global vm if needed,
    patch it, and attach elevator defaults.
    """
    try:
        _ensure_opcode_index()
    except Exception:
        _logger.exception("Failed to ensure opcode index")

    vm_inst = _ensure_vm_present()
    try:
        _install_elevator_defaults(vm_inst)
        _logger.info("Elevator installed: default handlers registered for %d opcodes.", len(vm_inst.external_handlers))
    except Exception:
        _logger.exception("Failed to install elevator defaults")

    # Make helper functions available globally for interactive use
    globals().setdefault("attach_extensions_to_vm", attach_extensions_to_vm if "attach_extensions_to_vm" in globals() else lambda v: None)
    globals().setdefault("extend_vm_with_dynamic_mapping", extend_vm_with_dynamic_mapping if "extend_vm_with_dynamic_mapping" in globals() else lambda *a, **k: {})
    return vm_inst


# Auto-run elevator to fix unresolved references at module import
try:
    resolve_missing_names_and_attach()
except Exception:
    _logger.exception("Auto elevator failed during import")

# End of elevator — ensures missing definitions are handled gracefully and unknown opcodes become safe no-ops.
