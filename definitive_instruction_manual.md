# 📰 NEWS Programming Language

**Nobody Ever Wins Sh\*t — but everyone gets to run it**

---

## 📖 Table of Contents

1. **Introduction**

   * What is NEWS?
   * Why the name?
   * The philosophy of bytecode, futility, and execution

2. **Ecosystem Overview**

   * The transpiler (`.news → .dgm`)
   * The VM (`.dgm` → execution)
   * Unified runtime (`news.py`)
   * REPL mode

3. **Download, Install & Setup**

   * Requirements
   * Installation
   * Running NEWS files
   * Using the REPL

4. **The NEWS Language**

   * Syntax basics
   * Variables and memory
   * Control flow (if/while/match)
   * Arithmetic (core + safe)
   * Data structures (tuples, lists, arrays, groups, nests, pairs)
   * Functions & calls
   * Concurrency (future, parallel, sync)
   * System integration (I/O, files, exit codes)

5. **The Bytecode (DGM)**

   * Dodecagram base-12 format
   * Opcodes (00–BB)
   * Disassembly

6. **Execution Model**

   * Stack and heap
   * Call frames
   * Threads
   * Error handling

7. **Learning & Understanding**

   * Beginner’s path
   * Intermediate patterns
   * Advanced usage
   * Debugging & tracing

8. **Practical Applications**

   * Small scripts
   * Algorithm testing
   * Educational value
   * Systems research
   * Concurrency experiments

9. **NEWS vs. Other Languages**

   * Comparisons
   * Strengths
   * Weaknesses

10. **Future Directions**

    * Libraries & stdlib
    * Native binary compilation
    * Optimizations
    * Ecosystem growth

---

## 1. Introduction

**NEWS** stands for **Nobody Ever Wins Sh\*t**, a tongue-in-cheek acknowledgement of the absurdity of programming languages: every time we think we’ve “won,” another layer of complexity appears.

NEWS is a **minimal yet massive** programming ecosystem:

* A **high-level source syntax** (`.news`)
* A **Dodecagram base-12 bytecode** format (`.dgm`)
* A **virtual machine interpreter** (the NEWS VM)
* A **unified runtime** with **REPL**

It is both a **serious technical exercise** and a **parody of language design**.
It runs. It works. It compiles. But it doesn’t promise you’ll win.

---

## 2. Ecosystem Overview

### The Transpiler

* Input: `.news`
* Output: `.dgm` (base-12 tokens)
* Converts high-level syntax into bytecode

### The VM

* Input: `.dgm`
* Executes instructions
* Implements **all 144 opcodes**

### The Unified Runtime

* Input: `.news` → compiles & runs instantly
* Input: `.dgm` → executes bytecode
* No difference to the user, just works

### The REPL

* Interactive prompt
* Type NEWS commands and see them execute
* Persistent memory & variables
* Tab completion, history, multiline blocks

---

## 3. Download, Install & Setup

### Requirements

* Python 3.9+
* Works on Linux, macOS, Windows

### Installation

```bash
git clone https://github.com/yourname/news-lang.git
cd news-lang
python3 news.py --help
```

### Running NEWS files

```bash
python3 news.py hello.news
```

### Running DGM files

```bash
python3 news.py hello.dgm
```

### Starting the REPL

```bash
python3 news.py
```

Example:

```
NEWS REPL (Nobody Ever Wins Sh*t)
>>> let x = 5
>>> add x, 3
>>> print("x is now:")
>>> print("done")
```

---

## 4. The NEWS Language

### Syntax Basics

* Variables: `let x = 10`
* Arithmetic: `add x, 3`
* Printing: `print("hello")`
* Ending: `end()`

### Control Flow

```news
if(x == 5)
    print("x is five")
endif

while(x < 10)
    add x, 1
endwhile

match x
    case 7:
        print("x is seven")
    case 9:
        print("x is nine")
endmatch
```

### Arithmetic

* `add`, `sub`, `mul`, `div`
* Safe versions: `safe.add`, `safe.div`

### Data Structures

```news
tuple 1 2 3
list.append myList, 42
```

### Concurrency

```news
future myFunc
parallel workerFunc
sync
```

### System Integration

```news
print("Opening file")
open "data.txt"
close
exit
```

---

## 5. The Bytecode (DGM)

### Dodecagram Base-12

* Digits: `0 1 2 3 4 5 6 7 8 9 a b`
* Example: `3 1 5 17 1 3 a6 68 69 0 33`

### Opcodes

* 00–4B: Core
* 50–7B: Safe
* 80–9B: Data structures
* A0–BB: CIAM extensions

Full disassembler maps tokens → human-readable instructions.

---

## 6. Execution Model

* **Stack** for operands
* **Heap** for variables
* **Frames** for function calls
* **Threads** for futures/parallel
* **Errors** bubble up but don’t crash REPL

---

## 7. Learning & Understanding

### Beginner’s Path

1. Learn `let`, `add`, `print`, `end()`
2. Try `if`, `while`, `match`
3. Experiment with `tuple` and `list`

### Intermediate

* Safe math
* Functions & calls
* File I/O

### Advanced

* Concurrency (future/parallel)
* System calls
* Debugging with disassembler

---

## 8. Practical Applications

* **Educational**: teaches compilation, bytecode, VM execution
* **Systems research**: experiment with concurrency, safe ops
* **Scripting**: quick tasks, toy programs
* **Debugging**: visualize compilation pipelines

---

## 9. NEWS vs Other Languages

* Like Python: interactive, dynamic
* Like C: explicit low-level ops
* Like LLVM: full opcode space
* Unlike all of them: **base-12 bytecode** + parody

Strengths: transparent, hackable, fun.
Weaknesses: not optimized, not mainstream.

---

## 10. Future Directions

* Build a **standard library** (math, strings, fs, net)
* Add **bytecode caching** (`.dgc` like `.pyc`)
* Native compilation to `.exe`
* Debugger integration
* NEWS-on-the-web (WebAssembly backend)

---

# ⚡ Conclusion

NEWS is:

* A language
* A compiler
* A VM
* A REPL
* A teaching tool
* A joke that works

Nobody ever wins sh\*t, but everybody gets to run their code.

---

