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

