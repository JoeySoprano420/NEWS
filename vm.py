# vm.py
import sys
from transpiler import Transpiler
from transpiler import from_base12
from transpiler import to_base12
from transpiler import OPCODES
from transpiler import DIGITS
from transpiler import Transpiler
import re
import sys
from typing import List, Dict
from io import StringIO
from collections import defaultdict
import random
import string
import time
import os
import math
import json
import base64
import struct
import hashlib
import itertools
import functools
import operator
import threading
import queue
import subprocess
import shlex
import traceback
import ast
import inspect
import contextlib
import copy
import types
import pickle
import zlib
import lzma
import bz2
import gzip
import sqlite3
import http.client
import urllib.request
import urllib.parse
import socket
import ssl
import email
import xml.etree.ElementTree as ET
import csv
import configparser
import difflib
import enum
import locale
import platform
import shutil
import tempfile
import unittest
import warnings
import webbrowser
import zipfile
import tarfile
import fnmatch
import importlib
import importlib.util
import importlib.machinery
import pkgutil
import site
import venv
import ensurepip
import pip
import sys
from transpiler import Transpiler

DIGITS = "0123456789ab"
def from_base12(s: str) -> int:
    v = 0
    for ch in s: v = v*12 + DIGITS.index(ch)
    return v

def run_dgm(dgm_code: str):
    tokens = dgm_code.strip().split()
    ip, memory = 0, {}
    while ip < len(tokens):
        tok = tokens[ip]; ip += 1
        try: opcode = from_base12(tok)
        except: continue

        if opcode == 0xA6:  # print string
            chars=[]
            while ip<len(tokens):
                v=from_base12(tokens[ip]); ip+=1
                if v==0: break
                chars.append(chr(v))
            print("".join(chars))

        elif opcode == 0x03:  # store addr val
            addr, val = from_base12(tokens[ip]), from_base12(tokens[ip+1])
            memory[addr]=val; ip+=2

        elif opcode == 0x17:  # add
            addr,val = from_base12(tokens[ip]), from_base12(tokens[ip+1])
            memory[addr]=memory.get(addr,0)+val; ip+=2

        elif opcode == 0x18:  # sub
            addr,val = from_base12(tokens[ip]), from_base12(tokens[ip+1])
            memory[addr]=memory.get(addr,0)-val; ip+=2

        elif opcode == 0x15:  # icmp
            addr,num = from_base12(tokens[ip]), from_base12(tokens[ip+1])
            op=tokens[ip+2]; ip+=3
            a=memory.get(addr,0); cond=False
            if op=="==": cond=(a==num)
            elif op=="!=": cond=(a!=num)
            elif op=="<": cond=(a<num)
            elif op==">": cond=(a>num)
            elif op=="<=": cond=(a<=num)
            elif op==">=": cond=(a>=num)
            jump=int(tokens[ip-1],12) if not cond else None
            if jump: ip=jump

        elif opcode == 0x30:  # br
            target = from_base12(tokens[ip]); ip=target

        elif opcode == 0x95:  # match.begin
            addr=from_base12(tokens[ip]); ip+=1
            match_val=memory.get(addr,0)
            active=False
            while ip<len(tokens):
                op2=from_base12(tokens[ip]); ip+=1
                if op2==0x96:  # case
                    case_val=from_base12(tokens[ip]); ip+=1
                    active=(case_val==match_val)
                elif op2==0x97:  # endmatch
                    break
                elif active:
                    continue  # exec case body
                else:
                    ip+=1

        elif opcode == 0x33: return

        else: raise ValueError(f"Unknown opcode {opcode}")

def main():
    if len(sys.argv) == 3:
        infile, outfile = sys.argv[1], sys.argv[2]

        # src is the source file contents
        with open(infile, "r", encoding="utf-8") as fh:
            src = fh.read()

        # transpiler is an instance of the Transpiler class
        transpiler = Transpiler()
        dgm_code = transpiler.transpile(src)

        # write the generated .dgm file
        with open(outfile, "w", encoding="utf-8") as fh:
            fh.write(dgm_code)
    else:
        # existing VM behavior (run a .dgm file)
        if len(sys.argv) != 2:
            print("Usage: python vm.py <file.dgm>  OR  python vm.py <input.news> <output.dgm>")
            sys.exit(0)
        with open(sys.argv[1]) as f: dgm=f.read()
        run_dgm(dgm)

if __name__=="__main__": main()
# vm.py

DIGITS = "0123456789ab"
def from_base12(s: str) -> int:
    v = 0
    for ch in s: v = v*12 + DIGITS.index(ch)
    return v
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

