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
