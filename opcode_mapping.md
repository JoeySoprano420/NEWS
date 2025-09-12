| Opcode (hex) | Name             | Meaning                                    |
| ------------ | ---------------- | ------------------------------------------ |
| C0           | game.init        | Initialize graphics subsystem (res, title) |
| C1           | game.loadModel   | Load 3D model                              |
| C2           | game.loadTexture | Load texture                               |
| C3           | game.createWorld | Create a scene graph / world               |
| C4           | game.addEntity   | Add model to world                         |
| C5           | game.addLight    | Add light source                           |
| C6           | game.update      | Update logic/physics                       |
| C7           | game.render      | Render frame                               |
| C8           | game.running     | Return 1 if engine is alive                |
| C9           | game.input       | Poll keyboard/mouse/gamepad                |
| CA           | game.playSound   | Play sound effect                          |
| CB           | game.playMusic   | Stream music                               |
| CC           | game.quit        | Shutdown game engine                       |
