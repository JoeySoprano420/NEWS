| DGM | LLVM IR / Meaning           | NASM x64 Equivalent       | Hex   | Binary          |
|-----|-----------------------------|---------------------------|-------|-----------------|
| 00  | nop                         | NOP                       | 0x00  | 00000000        |
| 01  | alloca                      | SUB RSP, imm32            | 0x01  | 00000001        |
| 02  | load                        | MOV r64, [mem]            | 0x02  | 00000010        |
| 03  | store                       | MOV [mem], r64            | 0x03  | 00000011        |
| 04  | getelementptr               | LEA r64, [mem]            | 0x04  | 00000100        |
| 05  | bitcast                     | MOVQ reg, xmm             | 0x05  | 00000101        |
| 06  | trunc                       | MOVZX/MOVSX (narrow int)  | 0x06  | 00000110        |
| 07  | zext                        | MOVZX                     | 0x07  | 00000111        |
| 08  | sext                        | MOVSX                     | 0x08  | 00001000        |
| 09  | fptrunc                     | CVTSD2SS / CVTPD2PS       | 0x09  | 00001001        |
| 0A  | fpext                       | CVTSS2SD / CVTPS2PD       | 0x0A  | 00001010        |
| 0B  | fptoui                      | CVTTSD2SI                 | 0x0B  | 00001011        |
| 10  | fptosi                      | CVTTSS2SI                 | 0x10  | 00010000        |
| 11  | uitofp                      | CVTSI2SD                  | 0x11  | 00010001        |
| 12  | sitofp                      | CVTSI2SS                  | 0x12  | 00010010        |
| 13  | ptrtoint                    | MOV reg, qword ptr        | 0x13  | 00010011        |
| 14  | inttoptr                    | MOV reg, imm64            | 0x14  | 00010100        |
| 15  | icmp                        | CMP r/m64, r64            | 0x15  | 00010101        |
| 16  | fcmp                        | UCOMISD / UCOMISS         | 0x16  | 00010110        |
| 17  | add                         | ADD r/m64, r64            | 0x17  | 00010111        |
| 18  | sub                         | SUB r/m64, r64            | 0x18  | 00011000        |
| 19  | mul                         | IMUL r64, r/m64           | 0x19  | 00011001        |
| 1A  | udiv                        | DIV r/m64                 | 0x1A  | 00011010        |
| 1B  | sdiv                        | IDIV r/m64                | 0x1B  | 00011011        |
| 20  | fadd                        | ADDSD xmm, xmm            | 0x20  | 00100000        |
| 21  | fsub                        | SUBSD xmm, xmm            | 0x21  | 00100001        |
| 22  | fmul                        | MULSD xmm, xmm            | 0x22  | 00100010        |
| 23  | fdiv                        | DIVSD xmm, xmm            | 0x23  | 00100011        |
| 24  | frem                        | Emulated DIV+MUL-SUB      | 0x24  | 00100100        |
| 25  | shl                         | SHL r/m64, CL             | 0x25  | 00100101        |
| 26  | lshr                        | SHR r/m64, CL             | 0x26  | 00100110        |
| 27  | ashr                        | SAR r/m64, CL             | 0x27  | 00100111        |
| 28  | and                         | AND r/m64, r64            | 0x28  | 00101000        |
| 29  | or                          | OR r/m64, r64             | 0x29  | 00101001        |
| 2A  | xor                         | XOR r/m64, r64            | 0x2A  | 00101010        |
| 2B  | call                        | CALL rel32                | 0x2B  | 00101011        |
| 30  | br                          | JMP rel32                 | 0x30  | 00110000        |
| 31  | switch                      | CMP+JMP table             | 0x31  | 00110001        |
| 32  | indirectbr                  | JMP r/m64                 | 0x32  | 00110010        |
| 33  | ret                         | RET                       | 0x33  | 00110011        |
| 34  | resume                      | EH resume stub            | 0x34  | 00110100        |
| 35  | unreachable                 | UD2                       | 0x35  | 00110101        |
| 36  | landingpad                  | EH landing pad            | 0x36  | 00110110        |
| 37  | invoke                      | CALL+EH unwind            | 0x37  | 00110111        |
| 38  | phi                         | SSA merge (no direct)     | 0x38  | 00111000        |
| 39  | select                      | CMP+CMOVcc                | 0x39  | 00111001        |
| 3A  | extractvalue                | MOV reg,[struct+offset]   | 0x3A  | 00111010        |
| 3B  | insertvalue                 | MOV [struct+offset],reg   | 0x3B  | 00111011        |
| 40  | atomicrmw                   | LOCK prefixed ops         | 0x40  | 01000000        |
| 41  | cmpxchg                     | LOCK CMPXCHG              | 0x41  | 01000001        |
| 42  | fence                       | MFENCE                    | 0x42  | 01000010        |
| 43  | memset                      | REP STOSB                 | 0x43  | 01000011        |
| 44  | memcpy                      | REP MOVSB                 | 0x44  | 01000100        |
| 45  | memmove                     | REP MOVSB+temp            | 0x45  | 01000101        |
| 46  | lifetime.start              | No codegen                | 0x46  | 01000110        |
| 47  | lifetime.end                | No codegen                | 0x47  | 01000111        |
| 48  | sanitizer.check             | CMP+Jcc bounds check      | 0x48  | 01001000        |
| 49  | assume                      | Compiler builtin          | 0x49  | 01001001        |
| 4A  | llvm.dbg.declare            | Debug meta                | 0x4A  | 01001010        |
| 4B  | llvm.dbg.value              | Debug meta                | 0x4B  | 01001011        |
| 50  | safe.add                    | ADD+JO recover            | 0x50  | 01010000        |
| 51  | safe.sub                    | SUB+JO recover            | 0x51  | 01010001        |
| 52  | safe.mul                    | IMUL+JO recover           | 0x52  | 01010010        |
| 53  | safe.div                    | DIV+guard                 | 0x53  | 01010011        |
| 54  | safe.mod                    | IDIV+guard                | 0x54  | 01010100        |
| 55  | safe.shift                  | SHL/SHR+mask              | 0x55  | 01010101        |
| 56  | safe.and                    | AND+guard                 | 0x56  | 01010110        |
| 57  | safe.or                     | OR+guard                  | 0x57  | 01010111        |
| 58  | safe.xor                    | XOR+guard                 | 0x58  | 01011000        |
| 59  | safe.neg                    | NEG+check                 | 0x59  | 01011001        |
| 5A  | safe.not                    | NOT r/m64                 | 0x5A  | 01011010        |
| 60  | cascade.begin               | PUSH context              | 0x60  | 01100000        |
| 61  | cascade.end                 | POP context               | 0x61  | 01100001        |
| 62  | cascade.yield               | SAVE+JMP out              | 0x62  | 01100010        |
| 63  | cascade.resume              | RESTORE+JMP in            | 0x63  | 01100011        |
| 70  | branch.try                  | Label mark                | 0x70  | 01110000        |
| 71  | branch.heal                 | JMP recover block         | 0x71  | 01110001        |
| 72  | branch.soft                 | JMP with mask             | 0x72  | 01110010        |
| 73  | branch.auto                 | Predicated JMP            | 0x73  | 01110011        |
| 7A  | recover                     | RESTORE state             | 0x7A  | 01111010        |
| 7B  | language.assert             | CMP+Jcc trap              | 0x7B  | 01111011        |
| 80  | tuple.pack                  | CALL __tuple_pack         | 0x80  | 10000000        |
| 81  | tuple.unpack                | CALL __tuple_unpack       | 0x81  | 10000001        |
| 82  | list.append                 | CALL __list_append        | 0x82  | 10000010        |
| 83  | list.remove                 | CALL __list_remove        | 0x83  | 10000011        |
| 84  | list.insert                 | CALL __list_insert        | 0x84  | 10000100        |
| 85  | list.pop                    | CALL __list_pop           | 0x85  | 10000101        |
| 86  | array.load                  | MOV reg,[array+idx]       | 0x86  | 10000110        |
| 87  | array.store                 | MOV [array+idx],reg       | 0x87  | 10000111        |
| 88  | group.spawn                 | CALL __group_spawn        | 0x88  | 10001000        |
| 89  | group.merge                 | CALL __group_merge        | 0x89  | 10001001        |
| 8A  | group.split                 | CALL __group_split        | 0x8A  | 10001010        |
| 8B  | nest.enter                  | CALL __nest_enter         | 0x8B  | 10001011        |
| 90  | nest.exit                   | CALL __nest_exit          | 0x90  | 10010000        |
| 91  | derive.child                | CALL __derive_child       | 0x91  | 10010001        |
| 92  | derive.parent               | CALL __derive_parent      | 0x92  | 10010010        |
| 93  | pair.create                 | CALL __pair_create        | 0x93  | 10010011        |
| 94  | pair.split                  | CALL __pair_split         | 0x94  | 10010100        |
| 95  | match.begin                 | LABEL match               | 0x95  | 10010101        |
| 96  | match.case                  | CMP+Jcc                   | 0x96  | 10010110        |
| 97  | match.end                   | JMP end                   | 0x97  | 10010111        |
| 98  | language.yield              | CALL __yield              | 0x98  | 10011000        |
| 99  | language.halt               | HLT                       | 0x99  | 10011001        |
| 9A  | language.wait               | PAUSE                     | 0x9A  | 10011010        |
| 9B  | language.resume             | CALL __resume             | 0x9B  | 10011011        |
| A0  | language.inline             | __forceinline             | 0xA0  | 10100000        |
| A1  | language.expand             | Macro expansion           | 0xA1  | 10100001        |
| A2  | language.fold               | Folded macro              | 0xA2  | 10100010        |
| A3  | language.derive             | Template derive           | 0xA3  | 10100011        |
| A4  | language.macro              | Macro define              | 0xA4  | 10100100        |
| A5  | language.trace              | CALL __tracepoint         | 0xA5  | 10100101        |
| A6  | language.echo               | CALL puts/printf          | 0xA6  | 10100110        |
| A7  | language.link               | CALL dlopen               | 0xA7  | 10100111        |
| A8  | language.infer              | Type infer pass           | 0xA8  | 10101000        |
| A9  | language.delete             | CALL free                 | 0xA9  | 10101001        |
| AA  | language.replace            | Swap call                 | 0xAA  | 10101010        |
| AB  | language.redirect           | JMP other                 | 0xAB  | 10101011        |
| B0  | language.guard              | CMP+Jcc guard             | 0xB0  | 10110000        |
| B1  | language.wrap               | PUSH+CALL+POP             | 0xB1  | 10110001        |
| B2  | language.unwrap             | MOV out,in                | 0xB2  | 10110010        |
| B3  | language.enclose            | SCOPE guard               | 0xB3  | 10110011        |
| B4  | language.open               | CALL fopen                | 0xB4  | 10110100        |
| B5  | language.close              | CALL fclose               | 0xB5  | 10110101        |
| B6  | language.defer              | PUSH cleanup              | 0xB6  | 10110110        |
| B7  | language.future             | THREAD CREATE             | 0xB7  | 10110111        |
| B8  | language.parallel           | PTHREAD_CREATE            | 0xB8  | 10111000        |
| B9  | language.sync               | SYSCALL futex_wait        | 0xB9  | 10111001        |
| BA  | language.pragma             | Compiler directive        | 0xBA  | 10111010        |
| BB  | language.exit               | SYSCALL exit              | 0xBB  | 10111011        |
| C0  | @llvm.game.init             | CALL __game_init          | 0xC0  | 11000000        |
| C1  | @llvm.game.load.model       | CALL __game_load_model    | 0xC1  | 11000001        |
| C2  | @llvm.game.load.texture     | CALL __game_load_texture  | 0xC2  | 11000010        |
| C3  | @llvm.game.create.world     | CALL __game_create_world  | 0xC3  | 11000011        |
| C4  | @llvm.game.add.entity       | CALL __game_add_entity    | 0xC4  | 11000100        |
| C5  | @llvm.game.add.light        | CALL __game_add_light     | 0xC5  | 11000101        |
| C6  | @llvm.game.update           | CALL __game_update        | 0xC6  | 11000110        |
| C7  | @llvm.game.render           | CALL __game_render        | 0xC7  | 11000111        |
| C8  | @llvm.game.running          | CALL __game_running       | 0xC8  | 11001000        |
| C9  | @llvm.game.input            | CALL __game_input         | 0xC9  | 11001001        |
| CA  | @llvm.game.play.sound       | CALL __game_play_sound    | 0xCA  | 11001010        |
| CB  | @llvm.game.play.music       | CALL __game_play_music    | 0xCB  | 11001011        |
| CC  | @llvm.game.quit             | CALL __game_quit          | 0xCC  | 11001100        |
| D0  | @llvm.math.pow              | CALL __math_pow           | 0xD0  | 11010000        |
| D1  | @llvm.math.log              | CALL __math_log           | 0xD1  | 11010001        |
| D2  | @llvm.math.exp              | CALL __math_exp           | 0xD2  | 11010010        |
| D3  | @llvm.math.sin              | CALL __math_sin           | 0xD3  | 11010011        |
| D4  | @llvm.math.cos              | CALL __math_cos           | 0xD4  | 11010100        |
| D5  | @llvm.math.tan              | CALL __math_tan           | 0xD5  | 11010101        |
| D6  | @llvm.math.asin             | CALL __math_asin          | 0xD6  | 11010110        |
| D7  | @llvm.math.acos             | CALL __math_acos          | 0xD7  | 11010111        |
| D8  | @llvm.math.atan             | CALL __math_atan          | 0xD8  | 11011000        |
| D9  | @llvm.math.sqrt             | CALL __math_sqrt          | 0xD9  | 11011001        |
| DA  | @llvm.math.cbrt             | CALL __math_cbrt          | 0xDA  | 11011010        |
| DB  | @llvm.math.hypot            | CALL __math_hypot         | 0xDB  | 11011011        |
| DC  | @llvm.math.floor            | CALL __math_floor         | 0xDC  | 11011100        |
| DD  | @llvm.math.ceil             | CALL __math_ceil          | 0xDD  | 11011101        |
| DE  | @llvm.math.abs              | CALL __math_abs           | 0xDE  | 11011110        |
| DF  | @llvm.math.rand             | CALL __math_rand          | 0xDF  | 11011111        |
| E0  | @llvm.str.concat            | CALL __str_concat         | 0xE0  | 11100000        |
| E1  | @llvm.str.upper             | CALL __str_upper          | 0xE1  | 11100001        |
| E2  | @llvm.str.lower             | CALL __str_lower          | 0xE2  | 11100010        |
| E3  | @llvm.str.len               | CALL __str_len            | 0xE3  | 11100011        |
| E4  | @llvm.str.substr            | CALL __str_substr         | 0xE4  | 11100100        |
| E5  | @llvm.str.find              | CALL __str_find           | 0xE5  | 11100101        |
| E6  | @llvm.str.replace           | CALL __str_replace        | 0xE6  | 11100110        |
| E7  | @llvm.str.split             | CALL __str_split          | 0xE7  | 11100111        |
| E8  | @llvm.str.join              | CALL __str_join           | 0xE8  | 11101000        |
| E9  | @llvm.file.write            | CALL __file_write         | 0xE9  | 11101001        |
| EA  | @llvm.file.append           | CALL __file_append        | 0xEA  | 11101010        |
| EB  | @llvm.file.read             | CALL __file_read          | 0xEB  | 11101011        |
| EC  | @llvm.file.delete           | CALL __file_delete        | 0xEC  | 11101100        |
| ED  | @llvm.file.exists           | CALL __file_exists        | 0xED  | 11101101        |
| EE  | @llvm.system.now            | CALL __system_now         | 0xEE  | 11101110        |
| EF  | @llvm.system.sleep          | CALL __system_sleep       | 0xEF  | 11101111        |
| F0  | @llvm.system.env            | CALL __system_env         | 0xF0  | 11110000        |
| F1  | @llvm.system.platform       | CALL __system_platform    | 0xF1  | 11110001        |
| F2  | @llvm.system.cpu            | CALL __system_cpu         | 0xF2  | 11110010        |
| F3  | @llvm.system.mem            | CALL __system_mem         | 0xF3  | 11110011        |
| F4  | @llvm.sys.exec              | CALL __sys_exec           | 0xF4  | 11110100        |
| F5  | @llvm.sys.cwd               | CALL __sys_cwd            | 0xF5  | 11110101        |
| F6  | @llvm.sys.chdir             | CALL __sys_chdir          | 0xF6  | 11110110        |
| F7  | @llvm.sys.listdir           | CALL __sys_listdir        | 0xF7  | 11110111        |
| F8  | @llvm.sys.mkdir             | CALL __sys_mkdir          | 0xF8  | 11111000        |
| F9  | @llvm.sys.rmdir             | CALL __sys_rmdir          | 0xF9  | 11111001        |
| FA  | @llvm.sys.tempfile          | CALL __sys_tempfile       | 0xFA  | 11111010        |
| FB  | @llvm.sys.clipboard         | CALL __sys_clipboard      | 0xFB  | 11111011        |
| FC  | @llvm.sys.args              | CALL __sys_args           | 0xFC  | 11111100        |
| FD  | @llvm.sys.uid               | CALL __sys_uid            | 0xFD  | 11111101        |
| FE  | @llvm.sys.pid               | CALL __sys_pid            | 0xFE  | 11111110        |
| FF  | @llvm.sys.exit              | SYSCALL exit              | 0xFF  | 11111111        |
| 100 | @llvm.hash.md5              | CALL __hash_md5           | 0x100 | 000100000000    |
| 101 | @llvm.hash.sha1             | CALL __hash_sha1          | 0x101 | 000100000001    |
| 102 | @llvm.hash.sha256           | CALL __hash_sha256        | 0x102 | 000100000010    |
| 103 | @llvm.hash.sha512           | CALL __hash_sha512        | 0x103 | 000100000011    |
| 104 | @llvm.hmac.md5              | CALL __hmac_md5           | 0x104 | 000100000100    |
| 105 | @llvm.hmac.sha256           | CALL __hmac_sha256        | 0x105 | 000100000101    |
| 106 | @llvm.base64.encode         | CALL __base64_encode      | 0x106 | 000100000110    |
| 107 | @llvm.base64.decode         | CALL __base64_decode      | 0x107 | 000100000111    |
| 108 | @llvm.hex.encode            | CALL __hex_encode         | 0x108 | 000100001000    |
| 109 | @llvm.hex.decode            | CALL __hex_decode         | 0x109 | 000100001001    |
| 10A | @llvm.crc32                 | CALL __crc32              | 0x10A | 000100001010    |
| 10B | @llvm.random.bytes          | CALL __random_bytes       | 0x10B | 000100001011    |
| 10C | @llvm.uuid.generate         | CALL __uuid_generate      | 0x10C | 000100001100    |
| 10D | @llvm.password.hash         | CALL __password_hash      | 0x10D | 000100001101    |
| 10E | @llvm.password.verify       | CALL __password_verify    | 0x10E | 000100001110    |
| 10F | @llvm.jwt.encode            | CALL __jwt_encode         | 0x10F | 000100001111    |
| 110 | @llvm.zlib.compress         | CALL __zlib_compress      | 0x110 | 000100010000    |
| 111 | @llvm.zlib.decompress       | CALL __zlib_decompress    | 0x111 | 000100010001    |
| 112 | @llvm.bz2.compress          | CALL __bz2_compress       | 0x112 | 000100010010    |
| 113 | @llvm.bz2.decompress        | CALL __bz2_decompress     | 0x113 | 000100010011    |
| 114 | @llvm.lzma.compress         | CALL __lzma_compress      | 0x114 | 000100010100    |
| 115 | @llvm.lzma.decompress       | CALL __lzma_decompress    | 0x115 | 000100010101    |
| 116 | @llvm.gzip.compress         | CALL __gzip_compress      | 0x116 | 000100010110    |
| 117 | @llvm.gzip.decompress       | CALL __gzip_decompress    | 0x117 | 000100010111    |
| 118 | @llvm.tar.create            | CALL __tar_create         | 0x118 | 000100011000    |
| 119 | @llvm.tar.extract           | CALL __tar_extract        | 0x119 | 000100011001    |
| 11A | @llvm.zip.create            | CALL __zip_create         | 0x11A | 000100011010    |
| 11B | @llvm.zip.extract           | CALL __zip_extract        | 0x11B | 000100011011    |
| 11C | @llvm.compress.detect       | CALL __compress_detect    | 0x11C | 000100011100    |
| 11D | @llvm.compress.ratio        | CALL __compress_ratio     | 0x11D | 000100011101    |
| 11E | @llvm.compress.level        | CALL __compress_level     | 0x11E | 000100011110    |
| 11F | @llvm.compress.bench        | CALL __compress_bench     | 0x11F | 000100011111    |
| 120 | @llvm.http.get              | CALL __http_get           | 0x120 | 000100100000    |
| 121 | @llvm.http.post             | CALL __http_post          | 0x121 | 000100100001    |
| 122 | @llvm.http.head             | CALL __http_head          | 0x122 | 000100100010    |
| 123 | @llvm.http.put              | CALL __http_put           | 0x123 | 000100100011    |
| 124 | @llvm.http.delete           | CALL __http_delete        | 0x124 | 000100100100    |
| 125 | @llvm.http.download         | CALL __http_download      | 0x125 | 000100100101    |
| 126 | @llvm.ws.connect            | CALL __ws_connect         | 0x126 | 000100100110    |
| 127 | @llvm.ws.send               | CALL __ws_send            | 0x127 | 000100100111    |
| 128 | @llvm.ws.recv               | CALL __ws_recv            | 0x128 | 000100101000    |
| 129 | @llvm.ws.close              | CALL __ws_close           | 0x129 | 000100101001    |
| 12A | @llvm.udp.send              | CALL __udp_send           | 0x12A | 000100101010    |
| 12B | @llvm.udp.recv              | CALL __udp_recv           | 0x12B | 000100101011    |
| 12C | @llvm.tcp.listen            | CALL __tcp_listen         | 0x12C | 000100101100    |
| 12D | @llvm.tcp.accept            | CALL __tcp_accept         | 0x12D | 000100101101    |
| 12E | @llvm.tcp.send              | CALL __tcp_send           | 0x12E | 000100101110    |
| 12F | @llvm.tcp.recv              | CALL __tcp_recv           | 0x12F | 000100101111    |
| 130 | @llvm.db.open               | CALL __db_open            | 0x130 | 000100110000    |
| 131 | @llvm.db.exec               | CALL __db_exec            | 0x131 | 000100110001    |
| 132 | @llvm.db.query              | CALL __db_query           | 0x132 | 000100110010    |
| 133 | @llvm.db.close              | CALL __db_close           | 0x133 | 000100110011    |
| 134 | @llvm.db.begin              | CALL __db_begin           | 0x134 | 000100110100    |
| 135 | @llvm.db.commit             | CALL __db_commit          | 0x135 | 000100110101    |
| 136 | @llvm.db.rollback           | CALL __db_rollback        | 0x136 | 000100110110    |
| 137 | @llvm.db.tables             | CALL __db_tables          | 0x137 | 000100110111    |
| 138 | @llvm.db.schema             | CALL __db_schema          | 0x138 | 000100111000    |
| 139 | @llvm.db.insert             | CALL __db_insert          | 0x139 | 000100111001    |
| 13A | @llvm.db.update             | CALL __db_update          | 0x13A | 000100111010    |
| 13B | @llvm.db.delete             | CALL __db_delete          | 0x13B | 000100111011    |
| 13C | @llvm.db.count              | CALL __db_count           | 0x13C | 000100111100    |
| 13D | @llvm.db.indexes            | CALL __db_indexes         | 0x13D | 000100111101    |
| 13E | @llvm.db.analyze            | CALL __db_analyze         | 0x13E | 000100111110    |
| 13F | @llvm.db.vacuum             | CALL __db_vacuum          | 0x13F | 000100111111    |
| 140 | @llvm.regex.match           | CALL __regex_match        | 0x140 | 000101000000    |
| 141 | @llvm.regex.findall         | CALL __regex_findall      | 0x141 | 000101000001    |
| 142 | @llvm.regex.replace         | CALL __regex_replace      | 0x142 | 000101000010    |
| 143 | @llvm.regex.split           | CALL __regex_split        | 0x143 | 000101000011    |
| 144 | @llvm.regex.subn            | CALL __regex_subn         | 0x144 | 000101000100    |
| 145 | @llvm.regex.compile         | CALL __regex_compile      | 0x145 | 000101000101    |
| 146 | @llvm.fuzzy.match           | CALL __fuzzy_match        | 0x146 | 000101000110    |
| 147 | @llvm.fuzzy.closest         | CALL __fuzzy_closest      | 0x147 | 000101000111    |
| 148 | @llvm.fuzzy.sort            | CALL __fuzzy_sort         | 0x148 | 000101001000    |
| 150 | @llvm.audio.playwav         | CALL __audio_playwav      | 0x150 | 000101010000    |
| 151 | @llvm.audio.playmp3         | CALL __audio_playmp3      | 0x151 | 000101010001    |
| 152 | @llvm.audio.record          | CALL __audio_record       | 0x152 | 000101010010    |
| 153 | @llvm.audio.stop            | CALL __audio_stop         | 0x153 | 000101010011    |
| 154 | @llvm.audio.tone            | CALL __audio_tone         | 0x154 | 000101010100    |
| 155 | @llvm.audio.volume          | CALL __audio_volume       | 0x155 | 000101010101    |
| 156 | @llvm.audio.mixer           | CALL __audio_mixer        | 0x156 | 000101010110    |
| 157 | @llvm.audio.pause           | CALL __audio_pause        | 0x157 | 000101010111    |
| 158 | @llvm.audio.resume          | CALL __audio_resume       | 0x158 | 000101011000    |
| 159 | @llvm.audio.stream          | CALL __audio_stream       | 0x159 | 000101011001    |
