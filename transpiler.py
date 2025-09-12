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

elif opcode == 0xC0:  # game.init
    width = from_base12(tokens[self.ip]); self.ip+=1
    height = from_base12(tokens[self.ip]); self.ip+=1
    title_chars = []
    while self.ip < len(tokens):
        v = from_base12(tokens[self.ip]); self.ip+=1
        if v==0: break
        title_chars.append(chr(v))
    title = "".join(title_chars)
    import pygame
    pygame.init()
    self.screen = pygame.display.set_mode((width,height))
    pygame.display.set_caption(title)

elif opcode == 0xC1:  # game.loadModel
    path = []
    while self.ip < len(tokens):
        v = from_base12(tokens[self.ip]); self.ip+=1
        if v==0: break
        path.append(chr(v))
    model_path = "".join(path)
    # placeholder: load stub model
    model_id = len(self.memory)+1000
    self.memory[model_id] = {"type":"model","path":model_path}
    self.stack.append(model_id)

elif opcode == 0xC6:  # game.update
    import pygame
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            self.running = False

elif opcode == 0xC7:  # game.render
    self.screen.fill((0,0,0))
    # TODO: render 3D entities here (OpenGL binding)
    import pygame
    pygame.display.flip()

elif opcode == 0xC8:  # game.running
    self.stack.append(1 if self.running else 0)

elif opcode == 0xCC:  # game.quit
    import pygame
    pygame.quit()
    self.running = False

# put this at the top of vm.py
import pygame

class NewsVM:
    def __init__(self, debug: bool = False, trace: bool = False):
        # ... your other state ...
        self.screen = None
        self.clock = None
        self.entities = []  # placeholder scene graph

    def step(self):
        opcode = self.fetch()

        # ---------------- GAME EXTENSIONS ----------------
        if opcode == 0xC0:  # game.init
            width = from_base12(self.tokens[self.ip]); self.ip += 1
            height = from_base12(self.tokens[self.ip]); self.ip += 1
            title_chars = []
            while self.ip < len(self.tokens):
                v = from_base12(self.tokens[self.ip]); self.ip += 1
                if v == 0: break
                title_chars.append(chr(v))
            title = "".join(title_chars)

            pygame.init()
            self.screen = pygame.display.set_mode((width, height))
            pygame.display.set_caption(title)
            self.clock = pygame.time.Clock()

        elif opcode == 0xC1:  # game.loadModel
            path_chars = []
            while self.ip < len(self.tokens):
                v = from_base12(self.tokens[self.ip]); self.ip += 1
                if v == 0: break
                path_chars.append(chr(v))
            model_path = "".join(path_chars)

            # placeholder: just store as an object in memory
            model_id = len(self.memory) + 1000
            self.memory[model_id] = {"type": "model", "path": model_path}
            self.stack.append(model_id)

        elif opcode == 0xC6:  # game.update
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
            if self.clock:
                self.clock.tick(60)  # cap at 60 FPS

        elif opcode == 0xC7:  # game.render
            if self.screen:
                self.screen.fill((0, 0, 0))
                # TODO: render entities here (OpenGL/Pygame draw)
                pygame.display.flip()

        elif opcode == 0xC8:  # game.running
            self.stack.append(1 if self.running else 0)

        elif opcode == 0xCC:  # game.quit
            pygame.quit()
            self.running = False
