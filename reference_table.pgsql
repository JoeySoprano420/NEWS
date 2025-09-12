📊 Full 144-Entry DGM ↔ LLVM IR ↔ NASM Mapping (Markdown)
DGM	LLVM IR / Meaning	NASM x64 Equivalent	Hex	Binary
00	nop	NOP	0x90	10010000
01	alloca	SUB RSP, imm32	0x81EC	1000000111101100
02	load	MOV r64, [mem]	0x8B	10001011
03	store	MOV [mem], r64	0x89	10001001
04	getelementptr	LEA r64, [mem]	0x8D	10001101
05	bitcast	MOVQ reg, xmm	0x66	01100110
06	trunc	MOVZX/MOVSX (narrow int)	0x0F B6	0000111110110110
07	zext	MOVZX	0x0F B6	0000111110110110
08	sext	MOVSX	0x0F BE	0000111110111110
09	fptrunc	CVTSD2SS / CVTPD2PS	0xF2 0F 2A	111100100000111100101010
0A	fpext	CVTSS2SD / CVTPS2PD	0xF3 0F 2A	111100110000111100101010
0B	fptoui	CVTTSD2SI	0xF2 0F 2C	111100100000111100101100
10	fptosi	CVTTSS2SI	0xF3 0F 2C	111100110000111100101100
11	uitofp	CVTSI2SD	0xF2 0F 2A	111100100000111100101010
12	sitofp	CVTSI2SS	0xF3 0F 2A	111100110000111100101010
13	ptrtoint	MOV reg, qword ptr	0x8B	10001011
14	inttoptr	MOV reg, imm64	0x48 B8	0100100010111000
15	icmp	CMP r/m64, r64	0x39	00111001
16	fcmp	UCOMISD / UCOMISS	0x66 0F 2E	011001100000111100101110
17	add	ADD r/m64, r64	0x01	00000001
18	sub	SUB r/m64, r64	0x29	00101001
19	mul	IMUL r64, r/m64	0x0F AF	0000111110101111
1A	udiv	DIV r/m64	0xF7 F0	1111011111110000
1B	sdiv	IDIV r/m64	0xF7 F8	1111011111111000
20	fadd	ADDSD xmm, xmm	0xF2 0F 58	111100100000111101011000
21	fsub	SUBSD xmm, xmm	0xF2 0F 5C	111100100000111101011100
22	fmul	MULSD xmm, xmm	0xF2 0F 59	111100100000111101011001
23	fdiv	DIVSD xmm, xmm	0xF2 0F 5E	111100100000111101011110
24	frem	Emulated FP DIV+MUL-SUB	–	–
25	shl	SHL r/m64, CL	0xD3 E0	1101001111100000
26	lshr	SHR r/m64, CL	0xD3 E8	1101001111101000
27	ashr	SAR r/m64, CL	0xD3 F8	1101001111111000
28	and	AND r/m64, r64	0x21	00100001
29	or	OR r/m64, r64	0x09	00001001
2A	xor	XOR r/m64, r64	0x31	00110001
2B	call	CALL rel32	0xE8	11101000
30	br	JMP rel32	0xE9	11101001
31	switch	CMP+JMP table	–	–
32	indirectbr	JMP r/m64	0xFF E0	1111111111100000
33	ret	RET	0xC3	11000011
34	resume	EH resume stub	–	–
35	unreachable	UD2	0x0F 0B	0000111100001011
36	landingpad	EH landing pad	–	–
37	invoke	CALL+EH unwind	–	–
38	phi	SSA merge (no direct)	–	–
39	select	CMP+CMOVcc	0x0F 4x	00001111xxxx
3A	extractvalue	MOV reg,[struct+offset]	0x8B	10001011
3B	insertvalue	MOV [struct+offset],reg	0x89	10001001
40	atomicrmw	LOCK prefixed ops	0xF0 xx	11110000xxxx
41	cmpxchg	LOCK CMPXCHG	0x0F B1	0000111110110001
42	fence	MFENCE	0x0F AE F0	000011111010111011110000
43	memset	REP STOSB	0xF3 AA	1111001110101010
44	memcpy	REP MOVSB	0xF3 A4	1111001110100100
45	memmove	REP MOVSB+temp	–	–
46	lifetime.start	No codegen hint	–	–
47	lifetime.end	No codegen hint	–	–
48	sanitizer.check	Bounds check (CMP+Jcc)	–	–
49	assume	Compiler builtin (no code)	–	–
4A	llvm.dbg.declare	Debug meta	–	–
4B	llvm.dbg.value	Debug meta	–	–
50	safe.add	ADD+JO recover	0x01+0F80	00000001+overflowjmp
51	safe.sub	SUB+JO recover	0x29+0F80	00101001+overflowjmp
52	safe.mul	IMUL+JO recover	0x0F AF+0F80	overflow guarded
53	safe.div	DIV+err guard	0xF7 F0+check	guarded
54	safe.mod	IDIV+remainder	0xF7 F8+guard	guarded
55	safe.shift	SHL/SHR+mask	0xD3 xx	masked
56	safe.and	AND+guard	0x21	00100001
57	safe.or	OR+guard	0x09	00001001
58	safe.xor	XOR+guard	0x31	00110001
59	safe.neg	NEG+check	0xF7 D8	1111011111011000
5A	safe.not	NOT r/m64	0xF7 D0	1111011111010000
60	cascade.begin	PUSH context	0x50+	01010000+
61	cascade.end	POP context	0x58+	01011000+
62	cascade.yield	SAVE+JMP out	–	–
63	cascade.resume	RESTORE+JMP in	–	–
70	branch.try	Label mark	–	–
71	branch.heal	JMP recover block	–	–
72	branch.soft	JMP with mask	–	–
73	branch.auto	Predicated JMP	–	–
7A	recover	RESTORE state	–	–
7B	language.assert	CMP+Jcc trap	0x0F 84	0000111110000100
80	tuple.pack	CALL __tuple_pack	–	–
81	tuple.unpack	CALL __tuple_unpack	–	–
82	list.append	CALL __list_append	–	–
83	list.remove	CALL __list_remove	–	–
84	list.insert	CALL __list_insert	–	–
85	list.pop	CALL __list_pop	–	–
86	array.load	MOV reg,[array+idx]	0x8B	10001011
87	array.store	MOV [array+idx],reg	0x89	10001001
88	group.spawn	CALL __group_spawn	–	–
89	group.merge	CALL __group_merge	–	–
8A	group.split	CALL __group_split	–	–
8B	nest.enter	CALL __nest_enter	–	–
90	nest.exit	CALL __nest_exit	–	–
91	derive.child	CALL __derive_child	–	–
92	derive.parent	CALL __derive_parent	–	–
93	pair.create	CALL __pair_create	–	–
94	pair.split	CALL __pair_split	–	–
95	match.begin	LABEL match	–	–
96	match.case	CMP+Jcc	0x0F 8x	000011111000xxxx
97	match.end	JMP end	0xE9	11101001
98	language.yield	CALL __yield	–	–
99	language.halt	HLT	0xF4	11110100
9A	language.wait	PAUSE	0xF3 90	1111001110010000
9B	language.resume	CALL __resume	–	–
A0	language.inline	__forceinline	–	–
A1	language.expand	Macro expansion	–	–
A2	language.fold	Folded macro	–	–
A3	language.derive	Template derive	–	–
A4	language.macro	Macro define	–	–
A5	language.trace	CALL __tracepoint	–	–
A6	language.echo	CALL puts/printf	0xE8+libc	11101000+call
A7	language.link	CALL dlopen	–	–
A8	language.infer	Type infer pass	–	–
A9	language.delete	CALL free	0xE8+free	–
AA	language.replace	Swap call	–	–
AB	language.redirect	JMP other	0xE9	11101001
B0	language.guard	CMP+Jcc guard	0x0F 8x	000011111000xxxx
B1	language.wrap	PUSH+CALL+POP	–	–
B2	language.unwrap	MOV out,in	–	–
B3	language.enclose	SCOPE guard	–	–
B4	language.open	CALL fopen	0xE8+fopen	–
B5	language.close	CALL fclose	0xE8+fclose	–
B6	language.defer	PUSH cleanup	0x50+	01010000+
B7	language.future	THREAD CREATE	–	–
B8	language.parallel	PTHREAD_CREATE	–	–
B9	language.sync	SYSCALL futex_wait	0F 05	0000111100000101
BA	language.pragma	Compiler directive	–	–
BB	language.exit	CALL exit / SYSCALL 60	0F 05	0000111100000101
________________________________________
