# vm.py
import sys

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
    if len(sys.argv)!=2:
        print("Usage: python vm.py <file.dgm>"); return
    with open(sys.argv[1]) as f: dgm=f.read()
    run_dgm(dgm)

if __name__=="__main__": main()
