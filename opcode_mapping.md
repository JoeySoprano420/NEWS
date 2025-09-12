| Hex    | Name                 | Meaning                                                 |
| ------ | -------------------- | ------------------------------------------------------- |
| 00     | nop                  | No operation                                            |
| 01     | alloca               | Allocate memory/register space                          |
| 02     | load                 | Load from memory                                        |
| 03     | store                | Store to memory                                         |
| 04     | getelementptr        | Pointer arithmetic                                      |
| 05     | bitcast              | Type reinterpret                                        |
| 06     | trunc                | Truncate integer                                        |
| 07     | zext                 | Zero-extend integer                                     |
| 08     | sext                 | Sign-extend integer                                     |
| 09     | fptrunc              | FP truncate                                             |
| 0A     | fpext                | FP extend                                               |
| 0B     | fptoui               | FP → unsigned int                                       |
| 10     | fptosi               | FP → signed int                                         |
| 11     | uitofp               | Unsigned int → FP                                       |
| 12     | sitofp               | Signed int → FP                                         |
| 13     | ptrtoint             | Pointer → integer                                       |
| 14     | inttoptr             | Integer → pointer                                       |
| 15     | icmp                 | Integer compare                                         |
| 16     | fcmp                 | Floating compare                                        |
| 17     | add                  | Integer add                                             |
| 18     | sub                  | Integer subtract                                        |
| 19     | mul                  | Integer multiply                                        |
| 1A     | udiv                 | Unsigned divide                                         |
| 1B     | sdiv                 | Signed divide                                           |
| 20     | fadd                 | Floating add                                            |
| 21     | fsub                 | Floating subtract                                       |
| 22     | fmul                 | Floating multiply                                       |
| 23     | fdiv                 | Floating divide                                         |
| 24     | frem                 | Floating remainder                                      |
| 25     | shl                  | Shift left                                              |
| 26     | lshr                 | Logical shift right                                     |
| 27     | ashr                 | Arithmetic shift right                                  |
| 28     | and                  | Bitwise AND                                             |
| 29     | or                   | Bitwise OR                                              |
| 2A     | xor                  | Bitwise XOR                                             |
| 2B     | call                 | Call function                                           |
| 30     | br                   | Branch                                                  |
| 31     | switch               | Switch table                                            |
| 32     | indirectbr           | Indirect branch                                         |
| 33     | ret                  | Return                                                  |
| 35     | unreachable          | Trap                                                    |
| 39     | select               | Conditional move                                        |
| 3A     | extractvalue         | Extract from struct                                     |
| 3B     | insertvalue          | Insert into struct                                      |
| 40     | atomicrmw            | Atomic read-modify-write                                |
| 41     | cmpxchg              | Atomic compare-exchange                                 |
| 42     | fence                | Memory fence                                            |
| 43     | memset               | Memory set                                              |
| 44     | memcpy               | Memory copy                                             |
| 45     | memmove              | Safe memory move                                        |
| 48     | sanitizer.check      | Bounds check                                            |
| 49     | assume               | Compiler assumption                                     |
| 4A     | dbg.declare          | Debug info                                              |
| 4B     | dbg.value            | Debug info                                              |
| 50     | safe.add             | Safe add (with overflow guard)                          |
| 51     | safe.sub             | Safe subtract                                           |
| 52     | safe.mul             | Safe multiply                                           |
| 53     | safe.div             | Safe divide (with zero guard)                           |
| 54     | safe.mod             | Safe modulo                                             |
| 55     | safe.shift           | Guarded shift                                           |
| 56     | safe.and             | Guarded bitwise AND                                     |
| 57     | safe.or              | Guarded bitwise OR                                      |
| 58     | safe.xor             | Guarded bitwise XOR                                     |
| 59     | safe.neg             | Guarded negate                                          |
| 5A     | safe.not             | Guarded bitwise NOT                                     |
| 60     | cascade.begin        | Begin cascading context                                 |
| 61     | cascade.end          | End cascading context                                   |
| 70     | branch.try           | Mark branch try                                         |
| 71     | branch.heal          | Recovery jump                                           |
| 72     | branch.soft          | Soft branch                                             |
| 73     | branch.auto          | Predicated branch                                       |
| 7A     | recover              | State restore                                           |
| 7B     | language.assert      | Assert                                                  |
| 80     | tuple.pack           | Pack tuple                                              |
| 81     | tuple.unpack         | Unpack tuple                                            |
| 82     | list.append          | Append to list                                          |
| 83     | list.remove          | Remove from list                                        |
| 84     | list.insert          | Insert into list                                        |
| 85     | list.pop             | Pop from list                                           |
| 86     | array.load           | Array load                                              |
| 87     | array.store          | Array store                                             |
| 88     | group.spawn          | Spawn group                                             |
| 89     | group.merge          | Merge group                                             |
| 8A     | group.split          | Split group                                             |
| 8B     | nest.enter           | Enter nested scope                                      |
| 90     | nest.exit            | Exit nested scope                                       |
| 91     | derive.child         | Derive child                                            |
| 92     | derive.parent        | Derive parent                                           |
| 93     | pair.create          | Create pair                                             |
| 94     | pair.split           | Split pair                                              |
| 95     | match.begin          | Start match                                             |
| 96     | match.case           | Case match                                              |
| 97     | match.end            | End match                                               |
| 98     | language.yield       | Yield                                                   |
| 99     | language.halt        | Halt                                                    |
| 9A     | language.wait        | Wait                                                    |
| 9B     | language.resume      | Resume                                                  |
| A0     | language.inline      | Inline                                                  |
| A1     | language.expand      | Macro expansion                                         |
| A2     | language.fold        | Folded macro                                            |
| A3     | language.derive      | Template derive                                         |
| A4     | language.macro       | Macro define                                            |
| A5     | language.trace       | Tracepoint                                              |
| A6     | language.echo        | Echo/print                                              |
| A7     | language.link        | Dynamic link                                            |
| A8     | language.infer       | Type inference                                          |
| A9     | language.delete      | Free memory                                             |
| AA     | language.replace     | Replace                                                 |
| AB     | language.redirect    | Redirect                                                |
| B0     | language.guard       | Guard                                                   |
| B1     | language.wrap        | Wrap                                                    |
| B2     | language.unwrap      | Unwrap                                                  |
| B3     | language.enclose     | Enclose                                                 |
| B4     | language.open        | Open file                                               |
| B5     | language.close       | Close file                                              |
| B6     | language.defer       | Defer cleanup                                           |
| B7     | language.future      | Thread future                                           |
| B8     | language.parallel    | Parallel thread                                         |
| B9     | language.sync        | Synchronize                                             |
| BA     | language.pragma      | Compiler directive                                      |
| BB     | language.exit        | Exit                                                    |
| **C0** | **game.init**        | **Initialize graphics subsystem (res, title)**          |
| **C1** | **game.loadModel**   | **Load 3D model**                                       |
| **C2** | **game.loadTexture** | **Load texture**                                        |
| **C3** | **game.createWorld** | **Create a scene graph / world**                        |
| **C4** | **game.addEntity**   | **Add model/entity to world**                           |
| **C5** | **game.addLight**    | **Add light source**                                    |
| **C6** | **game.update**      | **Update logic/physics**                                |
| **C7** | **game.render**      | **Render frame**                                        |
| **C8** | **game.running**     | **Return 1 if engine is alive**                         |
| **C9** | **game.input**       | **Poll keyboard/mouse/gamepad**                         |
| **CA** | **game.playSound**   | **Play sound effect**                                   |
| **CB** | **game.playMusic**   | **Stream music**                                        |
| **CC** | **game.quit**        | **Shutdown game engine**                                |
| D0–FF  | Extended math/sys    | Pow, trig, RNG, strings, file I/O, system calls, etc.   |
| 100+   | Extended features    | Crypto, compression, networking, DB, regex, audio, etc. |
| Hex | Name            | Meaning                      |
| --- | --------------- | ---------------------------- |
| D0  | math.pow        | Raise `a^b`                  |
| D1  | math.log        | Natural logarithm (ln x)     |
| D2  | math.exp        | Exponential (e^x)            |
| D3  | math.sin        | Sine                         |
| D4  | math.cos        | Cosine                       |
| D5  | math.tan        | Tangent                      |
| D6  | math.asin       | Arcsine                      |
| D7  | math.acos       | Arccosine                    |
| D8  | math.atan       | Arctangent                   |
| D9  | math.sqrt       | Square root                  |
| DA  | math.cbrt       | Cube root                    |
| DB  | math.hypot      | Hypotenuse (√(a²+b²))        |
| DC  | math.floor      | Floor                        |
| DD  | math.ceil       | Ceiling                      |
| DE  | math.abs        | Absolute value               |
| DF  | math.rand       | Random integer in range      |
| E0  | str.concat      | Concatenate two strings      |
| E1  | str.upper       | Convert string to uppercase  |
| E2  | str.lower       | Convert string to lowercase  |
| E3  | str.len         | String length                |
| E4  | str.substr      | Substring (start,len)        |
| E5  | str.find        | Find substring               |
| E6  | str.replace     | Replace substring            |
| E7  | str.split       | Split by delimiter           |
| E8  | str.join        | Join list into string        |
| E9  | file.write      | Write string to file         |
| EA  | file.append     | Append string to file        |
| EB  | file.read       | Read file contents           |
| EC  | file.delete     | Delete file                  |
| ED  | file.exists     | Test if file exists          |
| EE  | system.now      | Current time (epoch seconds) |
| EF  | system.sleep    | Sleep for N seconds          |
| F0  | system.env      | Get environment variable     |
| F1  | system.platform | Return OS/platform string    |
| F2  | system.cpu      | Return CPU info              |
| F3  | system.mem      | Return memory usage          |
| F4  | sys.exec        | Execute shell command        |
| F5  | sys.cwd         | Current working directory    |
| F6  | sys.chdir       | Change working directory     |
| F7  | sys.listdir     | List directory contents      |
| F8  | sys.mkdir       | Create directory             |
| F9  | sys.rmdir       | Remove directory             |
| FA  | sys.tempfile    | Create temporary file        |
| FB  | sys.clipboard   | Access system clipboard      |
| FC  | sys.args        | Get program arguments        |
| FD  | sys.uid         | Get user id / login          |
| FE  | sys.pid         | Get process id               |
| FF  | sys.exit        | Exit program with code       |
| Hex | Name            | Meaning                       |
| --- | --------------- | ----------------------------- |
| 100 | hash.md5        | Compute MD5 hash              |
| 101 | hash.sha1       | Compute SHA1 hash             |
| 102 | hash.sha256     | Compute SHA256 hash           |
| 103 | hash.sha512     | Compute SHA512 hash           |
| 104 | hmac.md5        | HMAC with MD5                 |
| 105 | hmac.sha256     | HMAC with SHA256              |
| 106 | base64.encode   | Encode string in base64       |
| 107 | base64.decode   | Decode base64                 |
| 108 | hex.encode      | Encode to hex                 |
| 109 | hex.decode      | Decode from hex               |
| 10A | crc32           | CRC32 checksum                |
| 10B | random.bytes    | Generate random byte string   |
| 10C | uuid.generate   | Generate UUID                 |
| 10D | password.hash   | Secure password hash (PBKDF2) |
| 10E | password.verify | Verify password hash          |
| 10F | jwt.encode      | Encode JSON Web Token         |
| Hex | Name            | Meaning                     |
| --- | --------------- | --------------------------- |
| 110 | zlib.compress   | Compress data with zlib     |
| 111 | zlib.decompress | Decompress zlib             |
| 112 | bz2.compress    | Compress with bzip2         |
| 113 | bz2.decompress  | Decompress bzip2            |
| 114 | lzma.compress   | Compress with lzma          |
| 115 | lzma.decompress | Decompress lzma             |
| 116 | gzip.compress   | Compress with gzip          |
| 117 | gzip.decompress | Decompress gzip             |
| 118 | tar.create      | Create tar archive          |
| 119 | tar.extract     | Extract tar archive         |
| 11A | zip.create      | Create zip archive          |
| 11B | zip.extract     | Extract zip archive         |
| 11C | compress.detect | Detect compression format   |
| 11D | compress.ratio  | Compression ratio check     |
| 11E | compress.level  | Set compression level       |
| 11F | compress.bench  | Benchmark compression speed |
| Hex | Name          | Meaning                |
| --- | ------------- | ---------------------- |
| 120 | http.get      | Perform HTTP GET       |
| 121 | http.post     | Perform HTTP POST      |
| 122 | http.head     | HTTP HEAD request      |
| 123 | http.put      | HTTP PUT request       |
| 124 | http.delete   | HTTP DELETE request    |
| 125 | http.download | Download file via HTTP |
| 126 | ws.connect    | WebSocket connect      |
| 127 | ws.send       | WebSocket send         |
| 128 | ws.recv       | WebSocket receive      |
| 129 | ws.close      | WebSocket close        |
| 12A | udp.send      | Send UDP packet        |
| 12B | udp.recv      | Receive UDP packet     |
| 12C | tcp.listen    | Start TCP server       |
| 12D | tcp.accept    | Accept TCP client      |
| 12E | tcp.send      | Send TCP data          |
| 12F | tcp.recv      | Receive TCP data       |
| Hex | Name        | Meaning              |
| --- | ----------- | -------------------- |
| 130 | db.open     | Open SQLite database |
| 131 | db.exec     | Execute SQL command  |
| 132 | db.query    | Query rows           |
| 133 | db.close    | Close DB connection  |
| 134 | db.begin    | Begin transaction    |
| 135 | db.commit   | Commit transaction   |
| 136 | db.rollback | Rollback transaction |
| 137 | db.tables   | List tables          |
| 138 | db.schema   | Show table schema    |
| 139 | db.insert   | Insert row           |
| 13A | db.update   | Update row           |
| 13B | db.delete   | Delete row           |
| 13C | db.count    | Count rows           |
| 13D | db.indexes  | List indexes         |
| 13E | db.analyze  | Run ANALYZE          |
| 13F | db.vacuum   | Run VACUUM           |
| Hex | Name          | Meaning                        |
| --- | ------------- | ------------------------------ |
| 140 | regex.match   | Regex match                    |
| 141 | regex.findall | Regex find all                 |
| 142 | regex.replace | Regex replace                  |
| 143 | regex.split   | Regex split                    |
| 144 | regex.subn    | Regex substitute with count    |
| 145 | regex.compile | Compile regex pattern          |
| 146 | fuzzy.match   | Fuzzy string ratio (difflib)   |
| 147 | fuzzy.closest | Find closest string match      |
| 148 | fuzzy.sort    | Sort list by similarity        |
| 150 | audio.playwav | Play WAV file                  |
| 151 | audio.playmp3 | Play MP3 file                  |
| 152 | audio.record  | Record microphone input        |
| 153 | audio.stop    | Stop audio playback            |
| 154 | audio.tone    | Generate tone (freq, duration) |
| 155 | audio.volume  | Set playback volume            |
| 156 | audio.mixer   | Mix multiple streams           |
| 157 | audio.pause   | Pause playback                 |
| 158 | audio.resume  | Resume playback                |
| 159 | audio.stream  | Stream audio buffer            |

