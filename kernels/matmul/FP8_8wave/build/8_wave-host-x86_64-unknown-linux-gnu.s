	.file	"8_wave.cu"
	.section	.rodata.cst8,"aM",@progbits,8
	.p2align	3, 0x0                          # -- Begin function main
.LCPI0_0:
	.quad	0x412e848000000000              # double 1.0E+6
.LCPI0_1:
	.quad	0x46293e5939a08cea              # double 1.0E+30
.LCPI0_2:
	.quad	0x40c0000000000000              # double 8192
.LCPI0_3:
	.quad	0x426d1a94a2000000              # double 1.0E+12
.LCPI0_4:
	.quad	0x408f400000000000              # double 1000
	.section	.rodata.cst16,"aM",@progbits,16
	.p2align	4, 0x0
.LCPI0_5:
	.long	0x7fffffff                      # float NaN
	.long	0x7fffffff                      # float NaN
	.long	0x7fffffff                      # float NaN
	.long	0x7fffffff                      # float NaN
	.section	.rodata.cst4,"aM",@progbits,4
	.p2align	2, 0x0
.LCPI0_6:
	.long	0x3f800000                      # float 1
	.text
	.globl	main
	.p2align	4
	.type	main,@function
main:                                   # @main
.Lfunc_begin0:
	.cfi_startproc
	.cfi_personality 3, __gxx_personality_v0
	.cfi_lsda 3, .Lexception0
# %bb.0:
	pushq	%r15
	.cfi_def_cfa_offset 16
	pushq	%r14
	.cfi_def_cfa_offset 24
	pushq	%r12
	.cfi_def_cfa_offset 32
	pushq	%rbx
	.cfi_def_cfa_offset 40
	subq	$168, %rsp
	.cfi_def_cfa_offset 208
	.cfi_offset %rbx, -40
	.cfi_offset %r12, -32
	.cfi_offset %r14, -24
	.cfi_offset %r15, -16
	.cfi_escape 0x2e, 0x00
	movl	$.L.str, %edi
	movl	$8192, %esi                     # imm = 0x2000
	movl	$8192, %edx                     # imm = 0x2000
	movl	$8192, %ecx                     # imm = 0x2000
	movl	$256, %r8d                      # imm = 0x100
	xorl	%eax, %eax
	callq	printf
	.cfi_escape 0x2e, 0x00
	movl	$.L.str.1, %edi
	movl	$200, %esi
	movl	$1000, %edx                     # imm = 0x3E8
	xorl	%eax, %eax
	callq	printf
	.cfi_escape 0x2e, 0x00
	movl	$67108864, %edi                 # imm = 0x4000000
	callq	_Znwm
	movq	%rax, 32(%rsp)
	leaq	67108864(%rax), %rbx
	movq	%rbx, 48(%rsp)
	.cfi_escape 0x2e, 0x00
	movl	$67108864, %edx                 # imm = 0x4000000
	movq	%rax, %rdi
	xorl	%esi, %esi
	callq	memset@PLT
	movq	%rbx, 40(%rsp)
.Ltmp0:
	.cfi_escape 0x2e, 0x00
	movl	$67108864, %edi                 # imm = 0x4000000
	callq	_Znwm
.Ltmp1:
# %bb.1:
	movq	%rax, 56(%rsp)
	movq	%rax, %rbx
	addq	$67108864, %rbx                 # imm = 0x4000000
	movq	%rbx, 72(%rsp)
	.cfi_escape 0x2e, 0x00
	movl	$67108864, %edx                 # imm = 0x4000000
	movq	%rax, %rdi
	xorl	%esi, %esi
	callq	memset@PLT
	movq	%rbx, 64(%rsp)
.Ltmp3:
	.cfi_escape 0x2e, 0x00
	movl	$134217728, %edi                # imm = 0x8000000
	callq	_Znwm
.Ltmp4:
# %bb.2:
	movq	%rax, 104(%rsp)
	movq	%rax, %rbx
	addq	$134217728, %rbx                # imm = 0x8000000
	movq	%rbx, 120(%rsp)
	.cfi_escape 0x2e, 0x00
	movl	$134217728, %edx                # imm = 0x8000000
	movq	%rax, %rdi
	xorl	%esi, %esi
	callq	memset@PLT
	movq	%rbx, 112(%rsp)
.Ltmp6:
	.cfi_escape 0x2e, 0x00
	movl	$134217728, %edi                # imm = 0x8000000
	callq	_Znwm
.Ltmp7:
# %bb.3:
	movq	%rax, %r14
	movq	%rax, 80(%rsp)
	movq	%rax, %rbx
	addq	$134217728, %rbx                # imm = 0x8000000
	movq	%rbx, 96(%rsp)
	.cfi_escape 0x2e, 0x00
	movl	$134217728, %edx                # imm = 0x8000000
	movq	%rax, %rdi
	xorl	%esi, %esi
	callq	memset@PLT
	movq	%rbx, 88(%rsp)
.Ltmp9:
	.cfi_escape 0x2e, 0x00
	leaq	32(%rsp), %rdi
	leaq	56(%rsp), %rbx
	movq	%rbx, %rsi
	callq	_Z11random_initILi8192ELi8192ELi8192EEvRSt6vectorI14__hip_fp8_e4m3SaIS1_EES4_
.Ltmp10:
# %bb.4:
	.cfi_escape 0x2e, 0x00
	movl	$.Lstr, %edi
	callq	puts@PLT
	movq	40(%rsp), %rax
	subq	32(%rsp), %rax
	shrq	$13, %rax
	movl	%eax, 128(%rsp)
	movq	64(%rsp), %rax
	subq	56(%rsp), %rax
	shrq	$13, %rax
	movl	%eax, 12(%rsp)
	.cfi_escape 0x2e, 0x00
	callq	_ZNSt6chrono3_V212system_clock3nowEv
	movq	%rax, %r14
	.cfi_escape 0x2e, 0x10
	leaq	104(%rsp), %r10
	leaq	128(%rsp), %rcx
	leaq	12(%rsp), %r8
	leaq	32(%rsp), %r15
	movl	$.L__unnamed_1, %edi
	movl	$_ZZ4mainENKUlRKSt6vectorI14__hip_fp8_e4m3SaIS0_EES4_RS_I14__hip_bfloat16SaIS5_EEiE_clES4_S4_S8_i.omp_outlined, %edx
	movl	$5, %esi
	movq	%r15, %r9
	xorl	%eax, %eax
	pushq	%r10
	.cfi_adjust_cfa_offset 8
	pushq	%rbx
	.cfi_adjust_cfa_offset 8
	callq	__kmpc_fork_call@PLT
	addq	$16, %rsp
	.cfi_adjust_cfa_offset -16
	.cfi_escape 0x2e, 0x00
	callq	_ZNSt6chrono3_V212system_clock3nowEv
	movq	%rax, %r12
	cvtsi2sdl	128(%rsp), %xmm0
	movsd	%xmm0, 24(%rsp)                 # 8-byte Spill
	xorps	%xmm0, %xmm0
	cvtsi2sdl	12(%rsp), %xmm0
	movsd	%xmm0, 16(%rsp)                 # 8-byte Spill
	.cfi_escape 0x2e, 0x00
	movl	$.Lstr.1, %edi
	callq	puts@PLT
.Ltmp12:
	.cfi_escape 0x2e, 0x00
	leaq	128(%rsp), %rdi
	leaq	80(%rsp), %rcx
	movq	%r15, %rsi
	movq	%rbx, %rdx
	movl	$200, %r8d
	movl	$1000, %r9d                     # imm = 0x3E8
	callq	_Z11matmul_hostILi8192ELi8192ELi8192ELi256EE12TimingResultRKSt6vectorI14__hip_fp8_e4m3SaIS2_EES6_RS1_I14__hip_bfloat16SaIS7_EEii
.Ltmp13:
# %bb.5:                                # %.preheader74
	subq	%r14, %r12
	xorps	%xmm0, %xmm0
	cvtsi2sd	%r12, %xmm0
	divsd	.LCPI0_0(%rip), %xmm0
	movapd	%xmm0, %xmm1
	minsd	.LCPI0_1(%rip), %xmm1
	xorpd	%xmm2, %xmm2
	movsd	24(%rsp), %xmm5                 # 8-byte Reload
                                        # xmm5 = mem[0],zero
	addsd	%xmm5, %xmm5
	mulsd	16(%rsp), %xmm5                 # 8-byte Folded Reload
	mulsd	.LCPI0_2(%rip), %xmm5
	divsd	.LCPI0_3(%rip), %xmm5
	cvtsd2ss	%xmm1, %xmm3
	movss	%xmm3, 8(%rsp)                  # 4-byte Spill
	addsd	%xmm0, %xmm2
	xorps	%xmm0, %xmm0
	cvtsd2ss	%xmm2, %xmm0
	movss	%xmm0, 16(%rsp)                 # 4-byte Spill
	movsd	.LCPI0_4(%rip), %xmm0           # xmm0 = [1.0E+3,0.0E+0]
	divsd	%xmm0, %xmm1
	movapd	%xmm5, %xmm3
	divsd	%xmm1, %xmm3
	movsd	%xmm3, 160(%rsp)                # 8-byte Spill
	divsd	%xmm0, %xmm2
	divsd	%xmm2, %xmm5
	movq	80(%rsp), %rax
	movq	104(%rsp), %rcx
	addq	$2, %rcx
	addq	$2, %rax
	xorl	%esi, %esi
	movaps	.LCPI0_5(%rip), %xmm0           # xmm0 = [NaN,NaN,NaN,NaN]
	movss	.LCPI0_6(%rip), %xmm1           # xmm1 = [1.0E+0,0.0E+0,0.0E+0,0.0E+0]
	movsd	%xmm5, 24(%rsp)                 # 8-byte Spill
.LBB0_6:                                # %.preheader
                                        # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_7 Depth 2
	xorl	%edx, %edx
	.p2align	4
.LBB0_7:                                #   Parent Loop BB0_6 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	movzwl	-2(%rax,%rdx,2), %edi
	shll	$16, %edi
	movd	%edi, %xmm2
	movzwl	-2(%rcx,%rdx,2), %edi
	shll	$16, %edi
	movd	%edi, %xmm3
	movdqa	%xmm2, %xmm4
	subss	%xmm3, %xmm4
	andps	%xmm0, %xmm4
	ucomiss	%xmm1, %xmm4
	ja	.LBB0_17
# %bb.8:                                #   in Loop: Header=BB0_7 Depth=2
	movzwl	(%rax,%rdx,2), %edi
	shll	$16, %edi
	movd	%edi, %xmm2
	movzwl	(%rcx,%rdx,2), %edi
	shll	$16, %edi
	movd	%edi, %xmm3
	movdqa	%xmm2, %xmm4
	subss	%xmm3, %xmm4
	andps	%xmm0, %xmm4
	ucomiss	%xmm1, %xmm4
	ja	.LBB0_16
# %bb.9:                                #   in Loop: Header=BB0_7 Depth=2
	addq	$2, %rdx
	cmpq	$8192, %rdx                     # imm = 0x2000
	jne	.LBB0_7
# %bb.10:                               #   in Loop: Header=BB0_6 Depth=1
	incq	%rsi
	addq	$16384, %rcx                    # imm = 0x4000
	addq	$16384, %rax                    # imm = 0x4000
	cmpq	$8192, %rsi                     # imm = 0x2000
	jne	.LBB0_6
# %bb.11:
	movl	$.Lstr.6, %ebx
	jmp	.LBB0_18
.LBB0_16:                               # %..thread72_crit_edge
	incq	%rdx
.LBB0_17:                               # %.thread72
	xorps	%xmm0, %xmm0
	cvtss2sd	%xmm2, %xmm0
	xorps	%xmm1, %xmm1
	cvtss2sd	%xmm3, %xmm1
	xorps	%xmm2, %xmm2
	cvtss2sd	%xmm4, %xmm2
	.cfi_escape 0x2e, 0x00
	movl	$.L.str.4, %edi
                                        # kill: def $esi killed $esi killed $rsi
                                        # kill: def $edx killed $edx killed $rdx
	movb	$3, %al
	callq	printf
	movl	$.Lstr.5, %ebx
.LBB0_18:                               # %.loopexit
	.cfi_escape 0x2e, 0x00
	movl	$.Lstr.2, %edi
	callq	puts@PLT
	.cfi_escape 0x2e, 0x00
	movl	$.Lstr.3, %edi
	callq	puts@PLT
	movss	8(%rsp), %xmm0                  # 4-byte Reload
                                        # xmm0 = mem[0],zero,zero,zero
	cvtss2sd	%xmm0, %xmm0
	.cfi_escape 0x2e, 0x00
	movl	$.L.str.7, %edi
	movsd	160(%rsp), %xmm1                # 8-byte Reload
                                        # xmm1 = mem[0],zero
	movb	$2, %al
	callq	printf
	movss	16(%rsp), %xmm0                 # 4-byte Reload
                                        # xmm0 = mem[0],zero,zero,zero
	cvtss2sd	%xmm0, %xmm0
	.cfi_escape 0x2e, 0x00
	movl	$.L.str.8, %edi
	movsd	24(%rsp), %xmm1                 # 8-byte Reload
                                        # xmm1 = mem[0],zero
	movb	$2, %al
	callq	printf
	.cfi_escape 0x2e, 0x00
	movl	$.Lstr.4, %edi
	callq	puts@PLT
	movss	128(%rsp), %xmm0                # xmm0 = mem[0],zero,zero,zero
	cvtss2sd	%xmm0, %xmm0
	movsd	136(%rsp), %xmm1                # xmm1 = mem[0],zero
	.cfi_escape 0x2e, 0x00
	movl	$.L.str.7, %edi
	movb	$2, %al
	callq	printf
	movss	132(%rsp), %xmm0                # xmm0 = mem[0],zero,zero,zero
	cvtss2sd	%xmm0, %xmm0
	movsd	144(%rsp), %xmm1                # xmm1 = mem[0],zero
	.cfi_escape 0x2e, 0x00
	movl	$.L.str.8, %edi
	movb	$2, %al
	callq	printf
	movss	8(%rsp), %xmm0                  # 4-byte Reload
                                        # xmm0 = mem[0],zero,zero,zero
	divss	128(%rsp), %xmm0
	cvtss2sd	%xmm0, %xmm0
	.cfi_escape 0x2e, 0x00
	movl	$.L.str.10, %edi
	movb	$1, %al
	callq	printf
	movss	16(%rsp), %xmm0                 # 4-byte Reload
                                        # xmm0 = mem[0],zero,zero,zero
	divss	132(%rsp), %xmm0
	cvtss2sd	%xmm0, %xmm0
	.cfi_escape 0x2e, 0x00
	movl	$.L.str.11, %edi
	movb	$1, %al
	callq	printf
	.cfi_escape 0x2e, 0x00
	movq	%rbx, %rdi
	callq	puts@PLT
	movq	80(%rsp), %rdi
	testq	%rdi, %rdi
	je	.LBB0_20
# %bb.19:
	movq	96(%rsp), %rsi
	subq	%rdi, %rsi
	.cfi_escape 0x2e, 0x00
	callq	_ZdlPvm
.LBB0_20:                               # %_ZNSt6vectorI14__hip_bfloat16SaIS0_EED2Ev.exit
	movq	104(%rsp), %rdi
	testq	%rdi, %rdi
	je	.LBB0_22
# %bb.21:
	movq	120(%rsp), %rsi
	subq	%rdi, %rsi
	.cfi_escape 0x2e, 0x00
	callq	_ZdlPvm
.LBB0_22:                               # %_ZNSt6vectorI14__hip_bfloat16SaIS0_EED2Ev.exit51
	movq	56(%rsp), %rdi
	testq	%rdi, %rdi
	je	.LBB0_24
# %bb.23:
	movq	72(%rsp), %rsi
	subq	%rdi, %rsi
	.cfi_escape 0x2e, 0x00
	callq	_ZdlPvm
.LBB0_24:                               # %_ZNSt6vectorI14__hip_fp8_e4m3SaIS0_EED2Ev.exit
	movq	32(%rsp), %rdi
	testq	%rdi, %rdi
	je	.LBB0_26
# %bb.25:
	movq	48(%rsp), %rsi
	subq	%rdi, %rsi
	.cfi_escape 0x2e, 0x00
	callq	_ZdlPvm
.LBB0_26:                               # %_ZNSt6vectorI14__hip_fp8_e4m3SaIS0_EED2Ev.exit54
	xorl	%eax, %eax
	addq	$168, %rsp
	.cfi_def_cfa_offset 40
	popq	%rbx
	.cfi_def_cfa_offset 32
	popq	%r12
	.cfi_def_cfa_offset 24
	popq	%r14
	.cfi_def_cfa_offset 16
	popq	%r15
	.cfi_def_cfa_offset 8
	retq
.LBB0_27:
	.cfi_def_cfa_offset 208
.Ltmp14:
	movq	%rax, %rbx
	movq	80(%rsp), %r14
	testq	%r14, %r14
	jne	.LBB0_28
# %bb.29:                               # %_ZNSt6vectorI14__hip_bfloat16SaIS0_EED2Ev.exit56
	movq	104(%rsp), %rdi
	testq	%rdi, %rdi
	jne	.LBB0_30
.LBB0_31:                               # %_ZNSt6vectorI14__hip_bfloat16SaIS0_EED2Ev.exit58
	movq	56(%rsp), %rdi
	testq	%rdi, %rdi
	jne	.LBB0_32
.LBB0_33:                               # %_ZNSt6vectorI14__hip_fp8_e4m3SaIS0_EED2Ev.exit60
	movq	32(%rsp), %rdi
	testq	%rdi, %rdi
	jne	.LBB0_34
.LBB0_35:                               # %_ZNSt6vectorI14__hip_fp8_e4m3SaIS0_EED2Ev.exit62
	.cfi_escape 0x2e, 0x00
	movq	%rbx, %rdi
	callq	_Unwind_Resume@PLT
.LBB0_15:                               # %.thread
.Ltmp11:
	movq	%rax, %rbx
.LBB0_28:
	movq	96(%rsp), %rsi
	subq	%r14, %rsi
	.cfi_escape 0x2e, 0x00
	movq	%r14, %rdi
	callq	_ZdlPvm
	movq	104(%rsp), %rdi
	testq	%rdi, %rdi
	je	.LBB0_31
	jmp	.LBB0_30
.LBB0_14:
.Ltmp8:
	movq	%rax, %rbx
	movq	104(%rsp), %rdi
	testq	%rdi, %rdi
	je	.LBB0_31
.LBB0_30:
	movq	120(%rsp), %rsi
	subq	%rdi, %rsi
	.cfi_escape 0x2e, 0x00
	callq	_ZdlPvm
	movq	56(%rsp), %rdi
	testq	%rdi, %rdi
	je	.LBB0_33
	jmp	.LBB0_32
.LBB0_13:
.Ltmp5:
	movq	%rax, %rbx
	movq	56(%rsp), %rdi
	testq	%rdi, %rdi
	je	.LBB0_33
.LBB0_32:
	movq	72(%rsp), %rsi
	subq	%rdi, %rsi
	.cfi_escape 0x2e, 0x00
	callq	_ZdlPvm
	movq	32(%rsp), %rdi
	testq	%rdi, %rdi
	je	.LBB0_35
	jmp	.LBB0_34
.LBB0_12:
.Ltmp2:
	movq	%rax, %rbx
	movq	32(%rsp), %rdi
	testq	%rdi, %rdi
	je	.LBB0_35
.LBB0_34:
	movq	48(%rsp), %rsi
	subq	%rdi, %rsi
	.cfi_escape 0x2e, 0x00
	callq	_ZdlPvm
	.cfi_escape 0x2e, 0x00
	movq	%rbx, %rdi
	callq	_Unwind_Resume@PLT
.Lfunc_end0:
	.size	main, .Lfunc_end0-main
	.cfi_endproc
	.section	.gcc_except_table,"a",@progbits
	.p2align	2, 0x0
GCC_except_table0:
.Lexception0:
	.byte	255                             # @LPStart Encoding = omit
	.byte	255                             # @TType Encoding = omit
	.byte	1                               # Call site Encoding = uleb128
	.uleb128 .Lcst_end0-.Lcst_begin0
.Lcst_begin0:
	.uleb128 .Lfunc_begin0-.Lfunc_begin0    # >> Call Site 1 <<
	.uleb128 .Ltmp0-.Lfunc_begin0           #   Call between .Lfunc_begin0 and .Ltmp0
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp0-.Lfunc_begin0           # >> Call Site 2 <<
	.uleb128 .Ltmp1-.Ltmp0                  #   Call between .Ltmp0 and .Ltmp1
	.uleb128 .Ltmp2-.Lfunc_begin0           #     jumps to .Ltmp2
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp1-.Lfunc_begin0           # >> Call Site 3 <<
	.uleb128 .Ltmp3-.Ltmp1                  #   Call between .Ltmp1 and .Ltmp3
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp3-.Lfunc_begin0           # >> Call Site 4 <<
	.uleb128 .Ltmp4-.Ltmp3                  #   Call between .Ltmp3 and .Ltmp4
	.uleb128 .Ltmp5-.Lfunc_begin0           #     jumps to .Ltmp5
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp4-.Lfunc_begin0           # >> Call Site 5 <<
	.uleb128 .Ltmp6-.Ltmp4                  #   Call between .Ltmp4 and .Ltmp6
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp6-.Lfunc_begin0           # >> Call Site 6 <<
	.uleb128 .Ltmp7-.Ltmp6                  #   Call between .Ltmp6 and .Ltmp7
	.uleb128 .Ltmp8-.Lfunc_begin0           #     jumps to .Ltmp8
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp7-.Lfunc_begin0           # >> Call Site 7 <<
	.uleb128 .Ltmp9-.Ltmp7                  #   Call between .Ltmp7 and .Ltmp9
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp9-.Lfunc_begin0           # >> Call Site 8 <<
	.uleb128 .Ltmp10-.Ltmp9                 #   Call between .Ltmp9 and .Ltmp10
	.uleb128 .Ltmp11-.Lfunc_begin0          #     jumps to .Ltmp11
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp12-.Lfunc_begin0          # >> Call Site 9 <<
	.uleb128 .Ltmp13-.Ltmp12                #   Call between .Ltmp12 and .Ltmp13
	.uleb128 .Ltmp14-.Lfunc_begin0          #     jumps to .Ltmp14
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp13-.Lfunc_begin0          # >> Call Site 10 <<
	.uleb128 .Lfunc_end0-.Ltmp13            #   Call between .Ltmp13 and .Lfunc_end0
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
.Lcst_end0:
	.p2align	2, 0x0
                                        # -- End function
	.section	.rodata.cst4,"aM",@progbits,4
	.p2align	2, 0x0                          # -- Begin function _Z11random_initILi8192ELi8192ELi8192EEvRSt6vectorI14__hip_fp8_e4m3SaIS1_EES4_
.LCPI1_0:
	.long	0x4f800000                      # float 4.2949673E+9
.LCPI1_1:
	.long	0x40000000                      # float 2
.LCPI1_2:
	.long	0x5f000000                      # float 9.22337203E+18
.LCPI1_3:
	.long	0x3f800000                      # float 1
.LCPI1_7:
	.long	0xbf800000                      # float -1
	.section	.rodata.cst16,"aM",@progbits,16
	.p2align	4, 0x0
.LCPI1_4:
	.quad	-2147483648                     # 0xffffffff80000000
	.quad	-2147483648                     # 0xffffffff80000000
.LCPI1_5:
	.quad	2147483646                      # 0x7ffffffe
	.quad	2147483646                      # 0x7ffffffe
.LCPI1_6:
	.quad	2567483615                      # 0x9908b0df
	.quad	2567483615                      # 0x9908b0df
	.section	.text._Z11random_initILi8192ELi8192ELi8192EEvRSt6vectorI14__hip_fp8_e4m3SaIS1_EES4_,"axG",@progbits,_Z11random_initILi8192ELi8192ELi8192EEvRSt6vectorI14__hip_fp8_e4m3SaIS1_EES4_,comdat
	.weak	_Z11random_initILi8192ELi8192ELi8192EEvRSt6vectorI14__hip_fp8_e4m3SaIS1_EES4_
	.p2align	4
	.type	_Z11random_initILi8192ELi8192ELi8192EEvRSt6vectorI14__hip_fp8_e4m3SaIS1_EES4_,@function
_Z11random_initILi8192ELi8192ELi8192EEvRSt6vectorI14__hip_fp8_e4m3SaIS1_EES4_: # @_Z11random_initILi8192ELi8192ELi8192EEvRSt6vectorI14__hip_fp8_e4m3SaIS1_EES4_
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	pushq	%r15
	.cfi_def_cfa_offset 24
	pushq	%r14
	.cfi_def_cfa_offset 32
	pushq	%r13
	.cfi_def_cfa_offset 40
	pushq	%r12
	.cfi_def_cfa_offset 48
	pushq	%rbx
	.cfi_def_cfa_offset 56
	subq	$5064, %rsp                     # imm = 0x13C8
	.cfi_def_cfa_offset 5120
	.cfi_offset %rbx, -56
	.cfi_offset %r12, -48
	.cfi_offset %r13, -40
	.cfi_offset %r14, -32
	.cfi_offset %r15, -24
	.cfi_offset %rbp, -16
	movq	%rsi, 32(%rsp)                  # 8-byte Spill
	movq	%rdi, 24(%rsp)                  # 8-byte Spill
	movq	$42, 64(%rsp)
	movl	$42, %ecx
	movl	$2, %eax
	.p2align	4
.LBB1_1:                                # =>This Inner Loop Header: Depth=1
	movq	%rcx, %rdx
	shrq	$30, %rdx
	xorq	%rcx, %rdx
	imulq	$1812433253, %rdx, %rcx         # imm = 0x6C078965
	addq	%rax, %rcx
	decq	%rcx
	movl	%ecx, %edx
	movq	%rdx, 56(%rsp,%rax,8)
	cmpq	$624, %rax                      # imm = 0x270
	je	.LBB1_3
# %bb.2:                                #   in Loop: Header=BB1_1 Depth=1
	shrl	$30, %edx
	xorl	%edx, %ecx
	imull	$1812433253, %ecx, %ecx         # imm = 0x6C078965
	addl	%eax, %ecx
	movq	%rcx, 64(%rsp,%rax,8)
	addq	$2, %rax
	jmp	.LBB1_1
.LBB1_3:                                # %_ZNSt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEC2Em.exit
	movq	$624, 5056(%rsp)                # imm = 0x270
	flds	.LCPI1_0(%rip)
	fstpt	(%rsp)
	callq	logl@PLT
	fstpt	52(%rsp)                        # 10-byte Folded Spill
	flds	.LCPI1_1(%rip)
	fstpt	(%rsp)
	callq	logl@PLT
	fldt	52(%rsp)                        # 10-byte Folded Reload
	fdivp	%st, %st(1)
	flds	.LCPI1_2(%rip)
	xorl	%ecx, %ecx
	fxch	%st(1)
	fucomi	%st(1), %st
	fldz
	fcmovnb	%st(2), %st
	fstp	%st(2)
	fsubp	%st, %st(1)
	setae	%cl
	fnstcw	20(%rsp)
	movzwl	20(%rsp), %eax
	orl	$3072, %eax                     # imm = 0xC00
	movw	%ax, 22(%rsp)
	fldcw	22(%rsp)
	fistpll	40(%rsp)
	fldcw	20(%rsp)
	shlq	$63, %rcx
	xorq	40(%rsp), %rcx
	leaq	23(%rcx), %rax
	movq	%rax, %rdx
	orq	%rcx, %rdx
	shrq	$32, %rdx
	je	.LBB1_4
# %bb.5:
	xorl	%edx, %edx
	divq	%rcx
	movq	%rax, %r14
	jmp	.LBB1_6
.LBB1_4:
                                        # kill: def $eax killed $eax killed $rax
	xorl	%edx, %edx
	divl	%ecx
	movl	%eax, %r14d
.LBB1_6:
	cmpq	$1, %r14
	adcq	$0, %r14
	movl	$2567483615, %r13d              # imm = 0x9908B0DF
	movl	$624, %r15d                     # imm = 0x270
	movl	$42, %ebp
	xorl	%ebx, %ebx
	movss	.LCPI1_3(%rip), %xmm10          # xmm10 = [1.0E+0,0.0E+0,0.0E+0,0.0E+0]
	movq	$-2147483648, %r12              # imm = 0x80000000
	movss	.LCPI1_0(%rip), %xmm9           # xmm9 = [4.2949673E+9,0.0E+0,0.0E+0,0.0E+0]
	.p2align	4
.LBB1_7:                                # =>This Loop Header: Depth=1
                                        #     Child Loop BB1_8 Depth 2
                                        #       Child Loop BB1_10 Depth 3
                                        #       Child Loop BB1_12 Depth 3
	movq	%r14, %rax
	movaps	%xmm10, %xmm1
	xorps	%xmm0, %xmm0
	movaps	.LCPI1_4(%rip), %xmm6           # xmm6 = [18446744071562067968,18446744071562067968]
	movaps	.LCPI1_5(%rip), %xmm7           # xmm7 = [2147483646,2147483646]
	movdqa	.LCPI1_6(%rip), %xmm8           # xmm8 = [2567483615,2567483615]
	jmp	.LBB1_8
	.p2align	4
.LBB1_16:                               # %_ZNSt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEclEv.exit
                                        #   in Loop: Header=BB1_8 Depth=2
	xorps	%xmm2, %xmm2
	cvtsi2ss	%rcx, %xmm2
.LBB1_17:                               # %_ZNSt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEclEv.exit
                                        #   in Loop: Header=BB1_8 Depth=2
	mulss	%xmm1, %xmm2
	addss	%xmm2, %xmm0
	mulss	%xmm9, %xmm1
	decq	%rax
	je	.LBB1_18
.LBB1_8:                                # %select.unfold.i.i.i.i
                                        #   Parent Loop BB1_7 Depth=1
                                        # =>  This Loop Header: Depth=2
                                        #       Child Loop BB1_10 Depth 3
                                        #       Child Loop BB1_12 Depth 3
	cmpq	$624, %r15                      # imm = 0x270
	jb	.LBB1_14
# %bb.9:                                # %vector.ph67
                                        #   in Loop: Header=BB1_8 Depth=2
	movq	%rbp, %xmm2
	pshufd	$68, %xmm2, %xmm2               # xmm2 = xmm2[0,1,0,1]
	xorl	%ecx, %ecx
	.p2align	4
.LBB1_10:                               # %vector.body68
                                        #   Parent Loop BB1_7 Depth=1
                                        #     Parent Loop BB1_8 Depth=2
                                        # =>    This Inner Loop Header: Depth=3
	movdqa	%xmm2, %xmm3
	movups	72(%rsp,%rcx,8), %xmm2
	shufps	$78, %xmm2, %xmm3               # xmm3 = xmm3[2,3],xmm2[0,1]
	andps	%xmm6, %xmm3
	movaps	%xmm2, %xmm4
	andps	%xmm7, %xmm4
	orps	%xmm3, %xmm4
	movdqu	3240(%rsp,%rcx,8), %xmm3
	psrlq	$1, %xmm4
	movaps	%xmm2, %xmm5
	pslld	$31, %xmm5
	psrad	$31, %xmm5
	pand	%xmm8, %xmm5
	pxor	%xmm3, %xmm5
	pxor	%xmm4, %xmm5
	movdqu	%xmm5, 64(%rsp,%rcx,8)
	addq	$2, %rcx
	cmpq	$226, %rcx
	jne	.LBB1_10
# %bb.11:                               # %vector.ph
                                        #   in Loop: Header=BB1_8 Depth=2
	pshufd	$238, %xmm2, %xmm2              # xmm2 = xmm2[2,3,2,3]
	movq	%xmm2, %rcx
	andq	$-2147483648, %rcx              # imm = 0x80000000
	movq	1880(%rsp), %rdx
	movl	%edx, %esi
	movl	%edx, %edi
	andl	$2147483646, %edi               # imm = 0x7FFFFFFE
	orq	%rcx, %rdi
	shrq	%rdi
	xorq	5048(%rsp), %rdi
	movq	%rdx, %xmm2
	andl	$1, %esi
	negl	%esi
	andl	%r13d, %esi
	xorq	%rdi, %rsi
	movq	%rsi, 1872(%rsp)
	pshufd	$68, %xmm2, %xmm2               # xmm2 = xmm2[0,1,0,1]
	movl	$228, %ecx
	.p2align	4
.LBB1_12:                               # %vector.body
                                        #   Parent Loop BB1_7 Depth=1
                                        #     Parent Loop BB1_8 Depth=2
                                        # =>    This Inner Loop Header: Depth=3
	movups	64(%rsp,%rcx,8), %xmm3
	shufps	$78, %xmm3, %xmm2               # xmm2 = xmm2[2,3],xmm3[0,1]
	andps	%xmm6, %xmm2
	movaps	%xmm3, %xmm4
	andps	%xmm7, %xmm4
	orps	%xmm2, %xmm4
	movdqu	-1760(%rsp,%rcx,8), %xmm5
	psrlq	$1, %xmm4
	movaps	%xmm3, %xmm2
	pslld	$31, %xmm3
	psrad	$31, %xmm3
	pand	%xmm8, %xmm3
	pxor	%xmm5, %xmm3
	pxor	%xmm4, %xmm3
	movdqu	%xmm3, 56(%rsp,%rcx,8)
	addq	$2, %rcx
	cmpq	$624, %rcx                      # imm = 0x270
	jne	.LBB1_12
# %bb.13:                               # %_ZNSt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EE11_M_gen_randEv.exit.i
                                        #   in Loop: Header=BB1_8 Depth=2
	movq	5048(%rsp), %rcx
	andq	%r12, %rcx
	movq	64(%rsp), %rbp
	movl	%ebp, %edx
	andl	$2147483646, %edx               # imm = 0x7FFFFFFE
	orq	%rcx, %rdx
	shrq	%rdx
	xorq	3232(%rsp), %rdx
	movl	%ebp, %ecx
	andl	$1, %ecx
	negl	%ecx
	andl	%r13d, %ecx
	xorq	%rdx, %rcx
	movq	%rcx, 5048(%rsp)
	xorl	%r15d, %r15d
.LBB1_14:                               # %_ZNSt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEclEv.exit
                                        #   in Loop: Header=BB1_8 Depth=2
	movq	%r15, %rcx
	incq	%r15
	movq	%r15, 5056(%rsp)
	movq	64(%rsp,%rcx,8), %rcx
	movq	%rcx, %rdx
	shrq	$11, %rdx
	movl	%edx, %edx
	xorq	%rcx, %rdx
	movl	%edx, %ecx
	shll	$7, %ecx
	andl	$-1658038656, %ecx              # imm = 0x9D2C5680
	xorq	%rdx, %rcx
	movl	%ecx, %edx
	shll	$15, %edx
	andl	$-272236544, %edx               # imm = 0xEFC60000
	xorq	%rcx, %rdx
	movq	%rdx, %rcx
	shrq	$18, %rcx
	xorq	%rdx, %rcx
	jns	.LBB1_16
# %bb.15:                               #   in Loop: Header=BB1_8 Depth=2
	movq	%rcx, %rdx
	shrq	%rdx
	andl	$1, %ecx
	orq	%rdx, %rcx
	xorps	%xmm2, %xmm2
	cvtsi2ss	%rcx, %xmm2
	addss	%xmm2, %xmm2
	jmp	.LBB1_17
	.p2align	4
.LBB1_18:                               #   in Loop: Header=BB1_7 Depth=1
	divss	%xmm1, %xmm0
	ucomiss	%xmm10, %xmm0
	jae	.LBB1_19
.LBB1_20:                               # %_ZNSt25uniform_real_distributionIfEclISt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEEEfRT_.exit
                                        #   in Loop: Header=BB1_7 Depth=1
	addss	%xmm0, %xmm0
	addss	.LCPI1_7(%rip), %xmm0
	leaq	19(%rsp), %rdi
	callq	_ZN14__hip_fp8_e4m3C2Ef
	movq	24(%rsp), %rax                  # 8-byte Reload
	movq	(%rax), %rax
	movzbl	19(%rsp), %ecx
	movb	%cl, (%rax,%rbx)
	incq	%rbx
	cmpq	$67108864, %rbx                 # imm = 0x4000000
	movss	.LCPI1_0(%rip), %xmm9           # xmm9 = [4.2949673E+9,0.0E+0,0.0E+0,0.0E+0]
	movss	.LCPI1_3(%rip), %xmm10          # xmm10 = [1.0E+0,0.0E+0,0.0E+0,0.0E+0]
	jne	.LBB1_7
	jmp	.LBB1_21
.LBB1_19:                               #   in Loop: Header=BB1_7 Depth=1
	xorps	%xmm1, %xmm1
	movss	.LCPI1_3(%rip), %xmm0           # xmm0 = [1.0E+0,0.0E+0,0.0E+0,0.0E+0]
	callq	nextafterf
	jmp	.LBB1_20
.LBB1_21:                               # %.preheader.preheader
	xorl	%ebx, %ebx
	movss	.LCPI1_3(%rip), %xmm10          # xmm10 = [1.0E+0,0.0E+0,0.0E+0,0.0E+0]
	movq	$-2147483648, %r12              # imm = 0x80000000
	.p2align	4
.LBB1_22:                               # %.preheader
                                        # =>This Loop Header: Depth=1
                                        #     Child Loop BB1_23 Depth 2
                                        #       Child Loop BB1_25 Depth 3
                                        #       Child Loop BB1_27 Depth 3
	movq	%r14, %rax
	movaps	%xmm10, %xmm1
	xorps	%xmm0, %xmm0
	movaps	.LCPI1_4(%rip), %xmm6           # xmm6 = [18446744071562067968,18446744071562067968]
	movaps	.LCPI1_5(%rip), %xmm7           # xmm7 = [2147483646,2147483646]
	movdqa	.LCPI1_6(%rip), %xmm8           # xmm8 = [2567483615,2567483615]
	jmp	.LBB1_23
	.p2align	4
.LBB1_31:                               # %_ZNSt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEclEv.exit30
                                        #   in Loop: Header=BB1_23 Depth=2
	xorps	%xmm2, %xmm2
	cvtsi2ss	%rcx, %xmm2
.LBB1_32:                               # %_ZNSt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEclEv.exit30
                                        #   in Loop: Header=BB1_23 Depth=2
	mulss	%xmm1, %xmm2
	addss	%xmm2, %xmm0
	mulss	%xmm9, %xmm1
	decq	%rax
	je	.LBB1_33
.LBB1_23:                               # %select.unfold.i.i.i.i9
                                        #   Parent Loop BB1_22 Depth=1
                                        # =>  This Loop Header: Depth=2
                                        #       Child Loop BB1_25 Depth 3
                                        #       Child Loop BB1_27 Depth 3
	cmpq	$624, %r15                      # imm = 0x270
	jb	.LBB1_29
# %bb.24:                               # %vector.ph94
                                        #   in Loop: Header=BB1_23 Depth=2
	movq	%rbp, %xmm2
	pshufd	$68, %xmm2, %xmm2               # xmm2 = xmm2[0,1,0,1]
	xorl	%ecx, %ecx
	.p2align	4
.LBB1_25:                               # %vector.body95
                                        #   Parent Loop BB1_22 Depth=1
                                        #     Parent Loop BB1_23 Depth=2
                                        # =>    This Inner Loop Header: Depth=3
	movdqa	%xmm2, %xmm3
	movups	72(%rsp,%rcx,8), %xmm2
	shufps	$78, %xmm2, %xmm3               # xmm3 = xmm3[2,3],xmm2[0,1]
	andps	%xmm6, %xmm3
	movaps	%xmm2, %xmm4
	andps	%xmm7, %xmm4
	orps	%xmm3, %xmm4
	movdqu	3240(%rsp,%rcx,8), %xmm3
	psrlq	$1, %xmm4
	movaps	%xmm2, %xmm5
	pslld	$31, %xmm5
	psrad	$31, %xmm5
	pand	%xmm8, %xmm5
	pxor	%xmm3, %xmm5
	pxor	%xmm4, %xmm5
	movdqu	%xmm5, 64(%rsp,%rcx,8)
	addq	$2, %rcx
	cmpq	$226, %rcx
	jne	.LBB1_25
# %bb.26:                               # %vector.ph80
                                        #   in Loop: Header=BB1_23 Depth=2
	pshufd	$238, %xmm2, %xmm2              # xmm2 = xmm2[2,3,2,3]
	movq	%xmm2, %rcx
	andq	$-2147483648, %rcx              # imm = 0x80000000
	movq	1880(%rsp), %rdx
	movl	%edx, %esi
	movl	%edx, %edi
	andl	$2147483646, %edi               # imm = 0x7FFFFFFE
	orq	%rcx, %rdi
	shrq	%rdi
	xorq	5048(%rsp), %rdi
	movq	%rdx, %xmm2
	andl	$1, %esi
	negl	%esi
	andl	%r13d, %esi
	xorq	%rdi, %rsi
	movq	%rsi, 1872(%rsp)
	pshufd	$68, %xmm2, %xmm2               # xmm2 = xmm2[0,1,0,1]
	movl	$228, %ecx
	.p2align	4
.LBB1_27:                               # %vector.body81
                                        #   Parent Loop BB1_22 Depth=1
                                        #     Parent Loop BB1_23 Depth=2
                                        # =>    This Inner Loop Header: Depth=3
	movups	64(%rsp,%rcx,8), %xmm3
	shufps	$78, %xmm3, %xmm2               # xmm2 = xmm2[2,3],xmm3[0,1]
	andps	%xmm6, %xmm2
	movaps	%xmm3, %xmm4
	andps	%xmm7, %xmm4
	orps	%xmm2, %xmm4
	movdqu	-1760(%rsp,%rcx,8), %xmm5
	psrlq	$1, %xmm4
	movaps	%xmm3, %xmm2
	pslld	$31, %xmm3
	psrad	$31, %xmm3
	pand	%xmm8, %xmm3
	pxor	%xmm5, %xmm3
	pxor	%xmm4, %xmm3
	movdqu	%xmm3, 56(%rsp,%rcx,8)
	addq	$2, %rcx
	cmpq	$624, %rcx                      # imm = 0x270
	jne	.LBB1_27
# %bb.28:                               # %_ZNSt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EE11_M_gen_randEv.exit.i28
                                        #   in Loop: Header=BB1_23 Depth=2
	movq	5048(%rsp), %rcx
	andq	%r12, %rcx
	movq	64(%rsp), %rbp
	movl	%ebp, %edx
	andl	$2147483646, %edx               # imm = 0x7FFFFFFE
	orq	%rcx, %rdx
	shrq	%rdx
	xorq	3232(%rsp), %rdx
	movl	%ebp, %ecx
	andl	$1, %ecx
	negl	%ecx
	andl	%r13d, %ecx
	xorq	%rdx, %rcx
	movq	%rcx, 5048(%rsp)
	xorl	%r15d, %r15d
.LBB1_29:                               # %_ZNSt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEclEv.exit30
                                        #   in Loop: Header=BB1_23 Depth=2
	movq	%r15, %rcx
	incq	%r15
	movq	%r15, 5056(%rsp)
	movq	64(%rsp,%rcx,8), %rcx
	movq	%rcx, %rdx
	shrq	$11, %rdx
	movl	%edx, %edx
	xorq	%rcx, %rdx
	movl	%edx, %ecx
	shll	$7, %ecx
	andl	$-1658038656, %ecx              # imm = 0x9D2C5680
	xorq	%rdx, %rcx
	movl	%ecx, %edx
	shll	$15, %edx
	andl	$-272236544, %edx               # imm = 0xEFC60000
	xorq	%rcx, %rdx
	movq	%rdx, %rcx
	shrq	$18, %rcx
	xorq	%rdx, %rcx
	jns	.LBB1_31
# %bb.30:                               #   in Loop: Header=BB1_23 Depth=2
	movq	%rcx, %rdx
	shrq	%rdx
	andl	$1, %ecx
	orq	%rdx, %rcx
	xorps	%xmm2, %xmm2
	cvtsi2ss	%rcx, %xmm2
	addss	%xmm2, %xmm2
	jmp	.LBB1_32
	.p2align	4
.LBB1_33:                               #   in Loop: Header=BB1_22 Depth=1
	divss	%xmm1, %xmm0
	ucomiss	%xmm10, %xmm0
	jae	.LBB1_34
.LBB1_35:                               # %_ZNSt25uniform_real_distributionIfEclISt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEEEfRT_.exit15
                                        #   in Loop: Header=BB1_22 Depth=1
	addss	%xmm0, %xmm0
	addss	.LCPI1_7(%rip), %xmm0
	leaq	19(%rsp), %rdi
	callq	_ZN14__hip_fp8_e4m3C2Ef
	movq	32(%rsp), %rax                  # 8-byte Reload
	movq	(%rax), %rax
	movzbl	19(%rsp), %ecx
	movb	%cl, (%rax,%rbx)
	incq	%rbx
	cmpq	$67108864, %rbx                 # imm = 0x4000000
	movss	.LCPI1_0(%rip), %xmm9           # xmm9 = [4.2949673E+9,0.0E+0,0.0E+0,0.0E+0]
	movss	.LCPI1_3(%rip), %xmm10          # xmm10 = [1.0E+0,0.0E+0,0.0E+0,0.0E+0]
	jne	.LBB1_22
	jmp	.LBB1_36
.LBB1_34:                               #   in Loop: Header=BB1_22 Depth=1
	xorps	%xmm1, %xmm1
	movss	.LCPI1_3(%rip), %xmm0           # xmm0 = [1.0E+0,0.0E+0,0.0E+0,0.0E+0]
	callq	nextafterf
	jmp	.LBB1_35
.LBB1_36:
	addq	$5064, %rsp                     # imm = 0x13C8
	.cfi_def_cfa_offset 56
	popq	%rbx
	.cfi_def_cfa_offset 48
	popq	%r12
	.cfi_def_cfa_offset 40
	popq	%r13
	.cfi_def_cfa_offset 32
	popq	%r14
	.cfi_def_cfa_offset 24
	popq	%r15
	.cfi_def_cfa_offset 16
	popq	%rbp
	.cfi_def_cfa_offset 8
	retq
.Lfunc_end1:
	.size	_Z11random_initILi8192ELi8192ELi8192EEvRSt6vectorI14__hip_fp8_e4m3SaIS1_EES4_, .Lfunc_end1-_Z11random_initILi8192ELi8192ELi8192EEvRSt6vectorI14__hip_fp8_e4m3SaIS1_EES4_
	.cfi_endproc
                                        # -- End function
	.section	.rodata.cst4,"aM",@progbits,4
	.p2align	2, 0x0                          # -- Begin function _Z11matmul_hostILi8192ELi8192ELi8192ELi256EE12TimingResultRKSt6vectorI14__hip_fp8_e4m3SaIS2_EES6_RS1_I14__hip_bfloat16SaIS7_EEii
.LCPI2_0:
	.long	0x7149f2ca                      # float 1.00000002E+30
	.section	.rodata.cst16,"aM",@progbits,16
	.p2align	4, 0x0
.LCPI2_1:
	.quad	0x3f50624dd2f1a9fc              # double 0.001
	.quad	0x3f50624dd2f1a9fc              # double 0.001
.LCPI2_2:
	.quad	0x4270000000000000              # double 1099511627776
	.quad	0x4270000000000000              # double 1099511627776
.LCPI2_3:
	.quad	0x426d1a94a2000000              # double 1.0E+12
	.quad	0x426d1a94a2000000              # double 1.0E+12
	.section	.text._Z11matmul_hostILi8192ELi8192ELi8192ELi256EE12TimingResultRKSt6vectorI14__hip_fp8_e4m3SaIS2_EES6_RS1_I14__hip_bfloat16SaIS7_EEii,"axG",@progbits,_Z11matmul_hostILi8192ELi8192ELi8192ELi256EE12TimingResultRKSt6vectorI14__hip_fp8_e4m3SaIS2_EES6_RS1_I14__hip_bfloat16SaIS7_EEii,comdat
	.weak	_Z11matmul_hostILi8192ELi8192ELi8192ELi256EE12TimingResultRKSt6vectorI14__hip_fp8_e4m3SaIS2_EES6_RS1_I14__hip_bfloat16SaIS7_EEii
	.p2align	4
	.type	_Z11matmul_hostILi8192ELi8192ELi8192ELi256EE12TimingResultRKSt6vectorI14__hip_fp8_e4m3SaIS2_EES6_RS1_I14__hip_bfloat16SaIS7_EEii,@function
_Z11matmul_hostILi8192ELi8192ELi8192ELi256EE12TimingResultRKSt6vectorI14__hip_fp8_e4m3SaIS2_EES6_RS1_I14__hip_bfloat16SaIS7_EEii: # @_Z11matmul_hostILi8192ELi8192ELi8192ELi256EE12TimingResultRKSt6vectorI14__hip_fp8_e4m3SaIS2_EES6_RS1_I14__hip_bfloat16SaIS7_EEii
.Lfunc_begin1:
	.cfi_startproc
	.cfi_personality 3, __gxx_personality_v0
	.cfi_lsda 3, .Lexception1
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	pushq	%r15
	.cfi_def_cfa_offset 24
	pushq	%r14
	.cfi_def_cfa_offset 32
	pushq	%r13
	.cfi_def_cfa_offset 40
	pushq	%r12
	.cfi_def_cfa_offset 48
	pushq	%rbx
	.cfi_def_cfa_offset 56
	subq	$296, %rsp                      # imm = 0x128
	.cfi_def_cfa_offset 352
	.cfi_offset %rbx, -56
	.cfi_offset %r12, -48
	.cfi_offset %r13, -40
	.cfi_offset %r14, -32
	.cfi_offset %r15, -24
	.cfi_offset %rbp, -16
	movq	%rdx, %r14
	movq	%rdi, %r12
	movq	8(%rsi), %rdx
	subq	(%rsi), %rdx
	cmpq	$67108864, %rdx                 # imm = 0x4000000
	jne	.LBB2_1
# %bb.3:
	movq	8(%r14), %rdx
	subq	(%r14), %rdx
	cmpq	$67108864, %rdx                 # imm = 0x4000000
	jne	.LBB2_4
# %bb.5:
	movl	%r9d, %ebp
	movl	%r8d, %ebx
	movq	%rcx, %r13
	movq	%rsi, %r15
	movq	(%rcx), %rax
	movq	8(%rcx), %rcx
	movq	%rcx, %rdx
	subq	%rax, %rdx
	movq	%rdx, %rdi
	sarq	%rdi
	cmpq	$67108863, %rdi                 # imm = 0x3FFFFFF
	ja	.LBB2_7
# %bb.6:
	movl	$67108864, %esi                 # imm = 0x4000000
	subq	%rdi, %rsi
	.cfi_escape 0x2e, 0x00
	movq	%r13, %rdi
	callq	_ZNSt6vectorI14__hip_bfloat16SaIS0_EE17_M_default_appendEm
	jmp	.LBB2_10
.LBB2_7:
	cmpq	$134217728, %rdx                # imm = 0x8000000
	je	.LBB2_10
# %bb.8:
	addq	$134217728, %rax                # imm = 0x8000000
	cmpq	%rax, %rcx
	je	.LBB2_10
# %bb.9:
	movq	%rax, 8(%r13)
.LBB2_10:                               # %_ZNSt6vectorI14__hip_bfloat16SaIS0_EE6resizeEm.exit
	.cfi_escape 0x2e, 0x00
	leaq	72(%rsp), %rdi
	movl	$67108864, %esi                 # imm = 0x4000000
	callq	hipMalloc
	.cfi_escape 0x2e, 0x00
	leaq	64(%rsp), %rdi
	movl	$67108864, %esi                 # imm = 0x4000000
	callq	hipMalloc
	.cfi_escape 0x2e, 0x00
	leaq	8(%rsp), %rdi
	movl	$134217728, %esi                # imm = 0x8000000
	callq	hipMalloc
	.cfi_escape 0x2e, 0x00
	callq	hipGetLastError
	testl	%eax, %eax
	jne	.LBB2_11
# %bb.14:
	.cfi_escape 0x2e, 0x00
	callq	hipDeviceSynchronize
	testl	%eax, %eax
	jne	.LBB2_15
# %bb.16:                               # %_Z15__hipCheckErrorPKci.exit
	movl	%ebp, 4(%rsp)                   # 4-byte Spill
	movq	72(%rsp), %rdi
	movq	(%r15), %rsi
	.cfi_escape 0x2e, 0x00
	movl	$67108864, %edx                 # imm = 0x4000000
	movl	$1, %ecx
	callq	hipMemcpy
	movq	64(%rsp), %rdi
	movq	(%r14), %rsi
	.cfi_escape 0x2e, 0x00
	movl	$67108864, %edx                 # imm = 0x4000000
	movl	$1, %ecx
	callq	hipMemcpy
	movq	8(%rsp), %rdi
	.cfi_escape 0x2e, 0x00
	movl	$134217728, %edx                # imm = 0x8000000
	xorl	%esi, %esi
	callq	hipMemset
	.cfi_escape 0x2e, 0x00
	callq	hipGetLastError
	testl	%eax, %eax
	jne	.LBB2_17
# %bb.19:
	movq	%r13, 168(%rsp)                 # 8-byte Spill
	movq	%r12, 128(%rsp)                 # 8-byte Spill
	.cfi_escape 0x2e, 0x00
	callq	hipDeviceSynchronize
	testl	%eax, %eax
	jne	.LBB2_20
# %bb.21:                               # %_Z15__hipCheckErrorPKci.exit46
	movq	72(%rsp), %rax
	movq	%rax, 144(%rsp)                 # 8-byte Spill
	movq	64(%rsp), %rax
	movq	%rax, 16(%rsp)                  # 8-byte Spill
	movq	8(%rsp), %rax
	movq	%rax, 136(%rsp)                 # 8-byte Spill
	testl	%ebx, %ebx
	jle	.LBB2_29
# %bb.22:                               # %.lr.ph
	movabsq	$4294967808, %r14               # imm = 0x100000200
	leaq	512(%r14), %r15
	leaq	248(%rsp), %rbp
	leaq	80(%rsp), %r12
	leaq	32(%rsp), %r13
	.p2align	4
.LBB2_23:                               # =>This Inner Loop Header: Depth=1
	movq	8(%rsp), %rdi
	.cfi_escape 0x2e, 0x00
	movl	$134217728, %edx                # imm = 0x8000000
	xorl	%esi, %esi
	callq	hipMemset
	.cfi_escape 0x2e, 0x00
	movq	%r15, %rdi
	movl	$1, %esi
	movq	%r14, %rdx
	movl	$1, %ecx
	xorl	%r8d, %r8d
	xorl	%r9d, %r9d
	callq	__hipPushCallConfiguration
	testl	%eax, %eax
	jne	.LBB2_25
# %bb.24:                               #   in Loop: Header=BB2_23 Depth=1
	movq	144(%rsp), %rax                 # 8-byte Reload
	movq	%rax, 280(%rsp)
	movq	16(%rsp), %rax                  # 8-byte Reload
	movq	%rax, 264(%rsp)
	movq	136(%rsp), %rax                 # 8-byte Reload
	movq	%rax, 248(%rsp)
	leaq	280(%rsp), %rax
	movq	%rax, 32(%rsp)
	leaq	264(%rsp), %rax
	movq	%rax, 40(%rsp)
	movq	%rbp, 48(%rsp)
	.cfi_escape 0x2e, 0x00
	leaq	112(%rsp), %rdi
	leaq	96(%rsp), %rsi
	leaq	88(%rsp), %rdx
	movq	%r12, %rcx
	callq	__hipPopCallConfiguration
	movq	112(%rsp), %rsi
	movl	120(%rsp), %edx
	movq	96(%rsp), %rcx
	movl	104(%rsp), %r8d
	.cfi_escape 0x2e, 0x10
	movl	$_Z13matmul_deviceILi8192ELi8192ELi8192EEvN7kittens2glI14__hip_fp8_e4m3Li1ELi1EXT_EXT1_EJEEENS1_IS2_Li1ELi1EXT0_EXT1_EJEEENS1_I14__hip_bfloat16Li1ELi1EXT_EXT0_EJEEE, %edi
	movq	%r13, %r9
	pushq	80(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	96(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel
	addq	$16, %rsp
	.cfi_adjust_cfa_offset -16
.LBB2_25:                               #   in Loop: Header=BB2_23 Depth=1
	.cfi_escape 0x2e, 0x00
	callq	hipGetLastError
	testl	%eax, %eax
	jne	.LBB2_26
# %bb.34:                               #   in Loop: Header=BB2_23 Depth=1
	.cfi_escape 0x2e, 0x00
	callq	hipDeviceSynchronize
	testl	%eax, %eax
	jne	.LBB2_35
# %bb.28:                               # %_Z15__hipCheckErrorPKci.exit50
                                        #   in Loop: Header=BB2_23 Depth=1
	.cfi_escape 0x2e, 0x00
	callq	hipDeviceSynchronize
	decl	%ebx
	jne	.LBB2_23
.LBB2_29:                               # %._crit_edge
	.cfi_escape 0x2e, 0x00
	leaq	56(%rsp), %rdi
	callq	hipEventCreate
	.cfi_escape 0x2e, 0x00
	leaq	24(%rsp), %rdi
	callq	hipEventCreate
	movl	4(%rsp), %ebp                   # 4-byte Reload
	testl	%ebp, %ebp
	js	.LBB2_121
# %bb.30:
	xorps	%xmm0, %xmm0
	movaps	%xmm0, 176(%rsp)                # 16-byte Spill
	movss	.LCPI2_0(%rip), %xmm2           # xmm2 = [1.00000002E+30,0.0E+0,0.0E+0,0.0E+0]
	je	.LBB2_31
# %bb.39:                               # %.lr.ph165
	movslq	%ebp, %rbx
	leaq	(,%rbx,4), %rdi
	.cfi_escape 0x2e, 0x00
	callq	_Znwm
	movq	%rax, %r13
	leaq	(%rax,%rbx,4), %r12
	movq	%rax, %r15
	.p2align	4
.LBB2_40:                               # =>This Inner Loop Header: Depth=1
	movq	8(%rsp), %rdi
.Ltmp15:
	.cfi_escape 0x2e, 0x00
	movl	$134217728, %edx                # imm = 0x8000000
	xorl	%esi, %esi
	callq	hipMemset
.Ltmp16:
# %bb.41:                               #   in Loop: Header=BB2_40 Depth=1
	movq	56(%rsp), %rdi
.Ltmp17:
	.cfi_escape 0x2e, 0x00
	xorl	%esi, %esi
	callq	hipEventRecord
.Ltmp18:
# %bb.42:                               #   in Loop: Header=BB2_40 Depth=1
.Ltmp19:
	.cfi_escape 0x2e, 0x00
	movabsq	$4294968320, %rdi               # imm = 0x100000400
	movl	$1, %esi
	movabsq	$4294967808, %rdx               # imm = 0x100000200
	movl	$1, %ecx
	xorl	%r8d, %r8d
	xorl	%r9d, %r9d
	callq	__hipPushCallConfiguration
.Ltmp20:
# %bb.43:                               #   in Loop: Header=BB2_40 Depth=1
	testl	%eax, %eax
	jne	.LBB2_46
# %bb.44:                               #   in Loop: Header=BB2_40 Depth=1
	movq	144(%rsp), %rax                 # 8-byte Reload
	movq	%rax, 232(%rsp)
	movq	16(%rsp), %rax                  # 8-byte Reload
	movq	%rax, 216(%rsp)
	movq	136(%rsp), %rax                 # 8-byte Reload
	movq	%rax, 200(%rsp)
	leaq	232(%rsp), %rax
	movq	%rax, 32(%rsp)
	leaq	216(%rsp), %rax
	movq	%rax, 40(%rsp)
	leaq	200(%rsp), %rax
	movq	%rax, 48(%rsp)
.Ltmp21:
	.cfi_escape 0x2e, 0x00
	leaq	112(%rsp), %rdi
	leaq	96(%rsp), %rsi
	leaq	88(%rsp), %rdx
	leaq	80(%rsp), %rcx
	callq	__hipPopCallConfiguration
.Ltmp22:
# %bb.45:                               # %.noexc57
                                        #   in Loop: Header=BB2_40 Depth=1
	movq	112(%rsp), %rsi
	movl	120(%rsp), %edx
	movq	96(%rsp), %rcx
	movl	104(%rsp), %r8d
.Ltmp23:
	.cfi_escape 0x2e, 0x10
	movl	$_Z13matmul_deviceILi8192ELi8192ELi8192EEvN7kittens2glI14__hip_fp8_e4m3Li1ELi1EXT_EXT1_EJEEENS1_IS2_Li1ELi1EXT0_EXT1_EJEEENS1_I14__hip_bfloat16Li1ELi1EXT_EXT0_EJEEE, %edi
	leaq	32(%rsp), %r9
	pushq	80(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	96(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel
	addq	$16, %rsp
	.cfi_adjust_cfa_offset -16
.Ltmp24:
.LBB2_46:                               #   in Loop: Header=BB2_40 Depth=1
	movq	24(%rsp), %rdi
.Ltmp25:
	.cfi_escape 0x2e, 0x00
	xorl	%esi, %esi
	callq	hipEventRecord
.Ltmp26:
# %bb.47:                               #   in Loop: Header=BB2_40 Depth=1
	movq	24(%rsp), %rdi
.Ltmp27:
	.cfi_escape 0x2e, 0x00
	callq	hipEventSynchronize
.Ltmp28:
# %bb.48:                               #   in Loop: Header=BB2_40 Depth=1
	movl	$0, 32(%rsp)
	movq	56(%rsp), %rsi
	movq	24(%rsp), %rdx
.Ltmp30:
	.cfi_escape 0x2e, 0x00
	leaq	32(%rsp), %rdi
	callq	hipEventElapsedTime
.Ltmp31:
# %bb.49:                               #   in Loop: Header=BB2_40 Depth=1
	cmpq	%r12, %r15
	je	.LBB2_51
# %bb.50:                               #   in Loop: Header=BB2_40 Depth=1
	movss	32(%rsp), %xmm0                 # xmm0 = mem[0],zero,zero,zero
	movss	%xmm0, (%r15)
	movq	%r15, %rbx
	jmp	.LBB2_64
	.p2align	4
.LBB2_51:                               #   in Loop: Header=BB2_40 Depth=1
	movq	%r15, %rbx
	subq	%r13, %rbx
	movabsq	$9223372036854775804, %rax      # imm = 0x7FFFFFFFFFFFFFFC
	cmpq	%rax, %rbx
	je	.LBB2_52
# %bb.54:                               # %_ZNKSt6vectorIfSaIfEE12_M_check_lenEmPKc.exit.i.i
                                        #   in Loop: Header=BB2_40 Depth=1
	movq	%rbx, %rax
	sarq	$2, %rax
	cmpq	$1, %rax
	movq	%rax, %rcx
	adcq	$0, %rcx
	leaq	(%rcx,%rax), %rdx
	movabsq	$2305843009213693951, %r14      # imm = 0x1FFFFFFFFFFFFFFF
	cmpq	%r14, %rdx
	jb	.LBB2_56
# %bb.55:                               # %_ZNKSt6vectorIfSaIfEE12_M_check_lenEmPKc.exit.i.i
                                        #   in Loop: Header=BB2_40 Depth=1
	movq	%r14, %rdx
.LBB2_56:                               # %_ZNKSt6vectorIfSaIfEE12_M_check_lenEmPKc.exit.i.i
                                        #   in Loop: Header=BB2_40 Depth=1
	addq	%rax, %rcx
	jb	.LBB2_58
# %bb.57:                               # %_ZNKSt6vectorIfSaIfEE12_M_check_lenEmPKc.exit.i.i
                                        #   in Loop: Header=BB2_40 Depth=1
	movq	%rdx, %r14
.LBB2_58:                               # %_ZNKSt6vectorIfSaIfEE12_M_check_lenEmPKc.exit.i.i
                                        #   in Loop: Header=BB2_40 Depth=1
	leaq	(,%r14,4), %rdi
.Ltmp32:
	.cfi_escape 0x2e, 0x00
	movq	%r15, %r12
	callq	_Znwm
.Ltmp33:
# %bb.59:                               # %.noexc61
                                        #   in Loop: Header=BB2_40 Depth=1
	movq	%rax, %r15
	movss	32(%rsp), %xmm0                 # xmm0 = mem[0],zero,zero,zero
	movss	%xmm0, (%rax,%rbx)
	testq	%rbx, %rbx
	jle	.LBB2_61
# %bb.60:                               #   in Loop: Header=BB2_40 Depth=1
	.cfi_escape 0x2e, 0x00
	movq	%r15, %rdi
	movq	%r13, %rsi
	movq	%rbx, %rdx
	callq	memmove@PLT
.LBB2_61:                               # %_ZNSt6vectorIfSaIfEE11_S_relocateEPfS2_S2_RS0_.exit16.i.i
                                        #   in Loop: Header=BB2_40 Depth=1
	testq	%r13, %r13
	je	.LBB2_63
# %bb.62:                               #   in Loop: Header=BB2_40 Depth=1
	.cfi_escape 0x2e, 0x00
	movq	%r13, %rdi
	movq	%rbx, %rsi
	callq	_ZdlPvm
.LBB2_63:                               # %_ZNSt6vectorIfSaIfEE17_M_realloc_insertIJRKfEEEvN9__gnu_cxx17__normal_iteratorIPfS1_EEDpOT_.exit.i
                                        #   in Loop: Header=BB2_40 Depth=1
	addq	%r15, %rbx
	leaq	(%r15,%r14,4), %r12
	movq	%r15, %r13
.LBB2_64:                               # %_ZNSt6vectorIfSaIfEE9push_backERKf.exit
                                        #   in Loop: Header=BB2_40 Depth=1
.Ltmp34:
	.cfi_escape 0x2e, 0x00
	callq	hipGetLastError
.Ltmp35:
# %bb.65:                               # %.noexc64
                                        #   in Loop: Header=BB2_40 Depth=1
	testl	%eax, %eax
	jne	.LBB2_66
# %bb.69:                               #   in Loop: Header=BB2_40 Depth=1
.Ltmp38:
	.cfi_escape 0x2e, 0x00
	callq	hipDeviceSynchronize
.Ltmp39:
# %bb.70:                               # %.noexc66
                                        #   in Loop: Header=BB2_40 Depth=1
	testl	%eax, %eax
	jne	.LBB2_71
# %bb.36:                               # %_Z15__hipCheckErrorPKci.exit68
                                        #   in Loop: Header=BB2_40 Depth=1
	leaq	4(%rbx), %r15
	decl	%ebp
	jne	.LBB2_40
# %bb.37:                               # %.preheader
	cmpq	%r15, %r13
	movq	128(%rsp), %r14                 # 8-byte Reload
	je	.LBB2_38
# %bb.83:                               # %.lr.ph171.preheader
	leaq	-4(%r13), %rax
	movss	.LCPI2_0(%rip), %xmm2           # xmm2 = [1.00000002E+30,0.0E+0,0.0E+0,0.0E+0]
	xorps	%xmm1, %xmm1
	movl	4(%rsp), %ebp                   # 4-byte Reload
	.p2align	4
.LBB2_84:                               # %.lr.ph171
                                        # =>This Inner Loop Header: Depth=1
	movaps	%xmm2, %xmm0
	movss	4(%rax), %xmm2                  # xmm2 = mem[0],zero,zero,zero
	addq	$4, %rax
	addss	%xmm2, %xmm1
	minss	%xmm0, %xmm2
	cmpq	%rbx, %rax
	jne	.LBB2_84
# %bb.85:
	movaps	%xmm1, 176(%rsp)                # 16-byte Spill
	subq	%r13, %r15
	sarq	$2, %r15
	movaps	%xmm2, 144(%rsp)                # 16-byte Spill
	jns	.LBB2_75
	jmp	.LBB2_33
.LBB2_31:
	xorl	%r13d, %r13d
	xorl	%r15d, %r15d
	xorl	%r12d, %r12d
	movq	128(%rsp), %r14                 # 8-byte Reload
	subq	%r13, %r15
	sarq	$2, %r15
	movaps	%xmm2, 144(%rsp)                # 16-byte Spill
	js	.LBB2_33
.LBB2_75:                               # %._crit_edge172
	xorps	%xmm0, %xmm0
	cvtsi2ss	%r15, %xmm0
	jmp	.LBB2_76
.LBB2_38:
	movl	4(%rsp), %ebp                   # 4-byte Reload
	movss	.LCPI2_0(%rip), %xmm2           # xmm2 = [1.00000002E+30,0.0E+0,0.0E+0,0.0E+0]
	subq	%r13, %r15
	sarq	$2, %r15
	movaps	%xmm2, 144(%rsp)                # 16-byte Spill
	jns	.LBB2_75
.LBB2_33:
	movq	%r15, %rax
	shrq	%rax
	andl	$1, %r15d
	orq	%rax, %r15
	xorps	%xmm0, %xmm0
	cvtsi2ss	%r15, %xmm0
	addss	%xmm0, %xmm0
.LBB2_76:                               # %._crit_edge172
	movss	%xmm0, 16(%rsp)                 # 4-byte Spill
	movq	56(%rsp), %rdi
.Ltmp46:
	.cfi_escape 0x2e, 0x00
	callq	hipEventDestroy
.Ltmp47:
# %bb.77:
	movq	24(%rsp), %rdi
.Ltmp48:
	.cfi_escape 0x2e, 0x00
	callq	hipEventDestroy
.Ltmp49:
# %bb.78:
.Ltmp50:
	.cfi_escape 0x2e, 0x00
	callq	hipGetLastError
.Ltmp51:
# %bb.79:                               # %.noexc71
	testl	%eax, %eax
	jne	.LBB2_80
# %bb.86:
.Ltmp54:
	.cfi_escape 0x2e, 0x00
	callq	hipDeviceSynchronize
.Ltmp55:
# %bb.87:                               # %.noexc73
	testl	%eax, %eax
	jne	.LBB2_88
# %bb.90:                               # %_Z15__hipCheckErrorPKci.exit75
	movq	168(%rsp), %rax                 # 8-byte Reload
	movq	(%rax), %rdi
	movq	8(%rsp), %rsi
.Ltmp58:
	.cfi_escape 0x2e, 0x00
	movl	$134217728, %edx                # imm = 0x8000000
	movl	$2, %ecx
	callq	hipMemcpy
.Ltmp59:
# %bb.91:
.Ltmp60:
	.cfi_escape 0x2e, 0x00
	callq	hipGetLastError
.Ltmp61:
# %bb.92:                               # %.noexc78
	testl	%eax, %eax
	jne	.LBB2_93
# %bb.96:
.Ltmp64:
	.cfi_escape 0x2e, 0x00
	callq	hipDeviceSynchronize
.Ltmp65:
# %bb.97:                               # %.noexc80
	testl	%eax, %eax
	jne	.LBB2_98
# %bb.100:                              # %_Z15__hipCheckErrorPKci.exit82
	movq	72(%rsp), %rdi
.Ltmp68:
	.cfi_escape 0x2e, 0x00
	callq	hipFree
.Ltmp69:
# %bb.101:
	movq	64(%rsp), %rdi
.Ltmp70:
	.cfi_escape 0x2e, 0x00
	callq	hipFree
.Ltmp71:
# %bb.102:
	movq	8(%rsp), %rdi
.Ltmp72:
	.cfi_escape 0x2e, 0x00
	callq	hipFree
.Ltmp73:
# %bb.103:
.Ltmp74:
	.cfi_escape 0x2e, 0x00
	callq	hipGetLastError
.Ltmp75:
# %bb.104:                              # %.noexc85
	testl	%eax, %eax
	jne	.LBB2_105
# %bb.108:
.Ltmp78:
	.cfi_escape 0x2e, 0x00
	callq	hipDeviceSynchronize
.Ltmp79:
# %bb.109:                              # %.noexc87
	testl	%eax, %eax
	jne	.LBB2_110
# %bb.112:                              # %_Z15__hipCheckErrorPKci.exit89
	movaps	176(%rsp), %xmm2                # 16-byte Reload
	divss	16(%rsp), %xmm2                 # 4-byte Folded Reload
	movaps	144(%rsp), %xmm3                # 16-byte Reload
	movaps	%xmm3, %xmm0
	unpcklps	%xmm2, %xmm0                    # xmm0 = xmm0[0],xmm2[0],xmm0[1],xmm2[1]
	cvtps2pd	%xmm0, %xmm0
	mulpd	.LCPI2_1(%rip), %xmm0
	movapd	.LCPI2_2(%rip), %xmm1           # xmm1 = [1.099511627776E+12,1.099511627776E+12]
	divpd	%xmm0, %xmm1
	divpd	.LCPI2_3(%rip), %xmm1
	movss	%xmm3, (%r14)
	movss	%xmm2, 4(%r14)
	movupd	%xmm1, 8(%r14)
	movl	%ebp, 24(%r14)
	testq	%r13, %r13
	je	.LBB2_114
# %bb.113:
	subq	%r13, %r12
	.cfi_escape 0x2e, 0x00
	movq	%r13, %rdi
	movq	%r12, %rsi
	callq	_ZdlPvm
.LBB2_114:                              # %_ZNSt6vectorIfSaIfEED2Ev.exit
	movq	%r14, %rax
.LBB2_115:                              # %_ZNSt6vectorIfSaIfEED2Ev.exit
	addq	$296, %rsp                      # imm = 0x128
	.cfi_def_cfa_offset 56
	popq	%rbx
	.cfi_def_cfa_offset 48
	popq	%r12
	.cfi_def_cfa_offset 40
	popq	%r13
	.cfi_def_cfa_offset 32
	popq	%r14
	.cfi_def_cfa_offset 24
	popq	%r15
	.cfi_def_cfa_offset 16
	popq	%rbp
	.cfi_def_cfa_offset 8
	retq
.LBB2_1:
	.cfi_def_cfa_offset 352
	movq	stderr(%rip), %rdi
	.cfi_escape 0x2e, 0x00
	movl	$.L.str.15, %esi
	jmp	.LBB2_2
.LBB2_4:
	movq	stderr(%rip), %rdi
	.cfi_escape 0x2e, 0x00
	movl	$.L.str.16, %esi
.LBB2_2:
	movl	$67108864, %ecx                 # imm = 0x4000000
	xorl	%eax, %eax
	callq	fprintf
	xorpd	%xmm0, %xmm0
	movupd	%xmm0, (%r12)
	movupd	%xmm0, 16(%r12)
	movq	%r12, %rax
	jmp	.LBB2_115
.LBB2_35:
	movq	stderr(%rip), %rbx
	.cfi_escape 0x2e, 0x00
	movl	%eax, %edi
	callq	hipGetErrorString
	.cfi_escape 0x2e, 0x00
	movl	$.L.str.20, %esi
	jmp	.LBB2_27
.LBB2_26:
	movq	stderr(%rip), %rbx
	.cfi_escape 0x2e, 0x00
	movl	%eax, %edi
	callq	hipGetErrorString
	.cfi_escape 0x2e, 0x00
	movl	$.L.str.19, %esi
.LBB2_27:
	movl	$.L.str.17, %edx
	movq	%rbx, %rdi
	movl	$395, %ecx                      # imm = 0x18B
	jmp	.LBB2_13
.LBB2_71:
	movq	stderr(%rip), %rbx
.Ltmp41:
	.cfi_escape 0x2e, 0x00
	movq	%r12, %r15
	movl	%eax, %edi
	callq	hipGetErrorString
.Ltmp42:
# %bb.72:                               # %.noexc67
	.cfi_escape 0x2e, 0x00
	movl	$.L.str.20, %esi
	jmp	.LBB2_68
.LBB2_66:
	movq	stderr(%rip), %rbx
.Ltmp36:
	.cfi_escape 0x2e, 0x00
	movq	%r12, %r15
	movl	%eax, %edi
	callq	hipGetErrorString
.Ltmp37:
# %bb.67:                               # %.noexc65
	.cfi_escape 0x2e, 0x00
	movl	$.L.str.19, %esi
.LBB2_68:                               # %.noexc65
	movl	$.L.str.17, %edx
	movq	%rbx, %rdi
	movl	$416, %ecx                      # imm = 0x1A0
.LBB2_13:
	movq	%rax, %r8
	xorl	%eax, %eax
	callq	fprintf
	.cfi_escape 0x2e, 0x00
	movl	$-1, %edi
	callq	exit
.LBB2_52:
.Ltmp43:
	.cfi_escape 0x2e, 0x00
	movl	$.L.str.22, %edi
	callq	_ZSt20__throw_length_errorPKc
.Ltmp44:
# %bb.53:                               # %.noexc60
.LBB2_11:
	movq	stderr(%rip), %rbx
	.cfi_escape 0x2e, 0x00
	movl	%eax, %edi
	callq	hipGetErrorString
	.cfi_escape 0x2e, 0x00
	movl	$.L.str.19, %esi
	jmp	.LBB2_12
.LBB2_15:
	movq	stderr(%rip), %rbx
	.cfi_escape 0x2e, 0x00
	movl	%eax, %edi
	callq	hipGetErrorString
	.cfi_escape 0x2e, 0x00
	movl	$.L.str.20, %esi
.LBB2_12:
	movl	$.L.str.17, %edx
	movq	%rbx, %rdi
	movl	$378, %ecx                      # imm = 0x17A
	jmp	.LBB2_13
.LBB2_17:
	movq	stderr(%rip), %rbx
	.cfi_escape 0x2e, 0x00
	movl	%eax, %edi
	callq	hipGetErrorString
	.cfi_escape 0x2e, 0x00
	movl	$.L.str.19, %esi
	jmp	.LBB2_18
.LBB2_20:
	movq	stderr(%rip), %rbx
	.cfi_escape 0x2e, 0x00
	movl	%eax, %edi
	callq	hipGetErrorString
	.cfi_escape 0x2e, 0x00
	movl	$.L.str.20, %esi
.LBB2_18:
	movl	$.L.str.17, %edx
	movq	%rbx, %rdi
	movl	$384, %ecx                      # imm = 0x180
	jmp	.LBB2_13
.LBB2_121:                              # %.noexc
	.cfi_escape 0x2e, 0x00
	movl	$.L.str.21, %edi
	callq	_ZSt20__throw_length_errorPKc
.LBB2_80:
	movq	stderr(%rip), %rbx
.Ltmp52:
	.cfi_escape 0x2e, 0x00
	movl	%eax, %edi
	callq	hipGetErrorString
.Ltmp53:
# %bb.81:                               # %.noexc72
	.cfi_escape 0x2e, 0x00
	movl	$.L.str.19, %esi
	jmp	.LBB2_82
.LBB2_88:
	movq	stderr(%rip), %rbx
.Ltmp56:
	.cfi_escape 0x2e, 0x00
	movl	%eax, %edi
	callq	hipGetErrorString
.Ltmp57:
# %bb.89:                               # %.noexc74
	.cfi_escape 0x2e, 0x00
	movl	$.L.str.20, %esi
.LBB2_82:                               # %.noexc72
	movl	$.L.str.17, %edx
	movq	%rbx, %rdi
	movl	$435, %ecx                      # imm = 0x1B3
	jmp	.LBB2_13
.LBB2_93:
	movq	stderr(%rip), %rbx
.Ltmp62:
	.cfi_escape 0x2e, 0x00
	movl	%eax, %edi
	callq	hipGetErrorString
.Ltmp63:
# %bb.94:                               # %.noexc79
	.cfi_escape 0x2e, 0x00
	movl	$.L.str.19, %esi
	jmp	.LBB2_95
.LBB2_98:
	movq	stderr(%rip), %rbx
.Ltmp66:
	.cfi_escape 0x2e, 0x00
	movl	%eax, %edi
	callq	hipGetErrorString
.Ltmp67:
# %bb.99:                               # %.noexc81
	.cfi_escape 0x2e, 0x00
	movl	$.L.str.20, %esi
.LBB2_95:                               # %.noexc79
	movl	$.L.str.17, %edx
	movq	%rbx, %rdi
	movl	$439, %ecx                      # imm = 0x1B7
	jmp	.LBB2_13
.LBB2_105:
	movq	stderr(%rip), %rbx
.Ltmp76:
	.cfi_escape 0x2e, 0x00
	movl	%eax, %edi
	callq	hipGetErrorString
.Ltmp77:
# %bb.106:                              # %.noexc86
	.cfi_escape 0x2e, 0x00
	movl	$.L.str.19, %esi
	jmp	.LBB2_107
.LBB2_110:
	movq	stderr(%rip), %rbx
.Ltmp80:
	.cfi_escape 0x2e, 0x00
	movl	%eax, %edi
	callq	hipGetErrorString
.Ltmp81:
# %bb.111:                              # %.noexc88
	.cfi_escape 0x2e, 0x00
	movl	$.L.str.20, %esi
.LBB2_107:                              # %.noexc86
	movl	$.L.str.17, %edx
	movq	%rbx, %rdi
	movl	$445, %ecx                      # imm = 0x1BD
	jmp	.LBB2_13
.LBB2_116:
.Ltmp82:
	jmp	.LBB2_117
.LBB2_74:                               # %.loopexit.split-lp
.Ltmp45:
	movq	%rax, %rbx
	movq	%r15, %r12
	jmp	.LBB2_118
.LBB2_73:                               # %.loopexit
.Ltmp40:
	jmp	.LBB2_117
.LBB2_122:
.Ltmp29:
.LBB2_117:
	movq	%rax, %rbx
.LBB2_118:
	testq	%r13, %r13
	je	.LBB2_120
# %bb.119:
	subq	%r13, %r12
	.cfi_escape 0x2e, 0x00
	movq	%r13, %rdi
	movq	%r12, %rsi
	callq	_ZdlPvm
.LBB2_120:                              # %_ZNSt6vectorIfSaIfEED2Ev.exit92
	.cfi_escape 0x2e, 0x00
	movq	%rbx, %rdi
	callq	_Unwind_Resume@PLT
.Lfunc_end2:
	.size	_Z11matmul_hostILi8192ELi8192ELi8192ELi256EE12TimingResultRKSt6vectorI14__hip_fp8_e4m3SaIS2_EES6_RS1_I14__hip_bfloat16SaIS7_EEii, .Lfunc_end2-_Z11matmul_hostILi8192ELi8192ELi8192ELi256EE12TimingResultRKSt6vectorI14__hip_fp8_e4m3SaIS2_EES6_RS1_I14__hip_bfloat16SaIS7_EEii
	.cfi_endproc
	.section	.gcc_except_table._Z11matmul_hostILi8192ELi8192ELi8192ELi256EE12TimingResultRKSt6vectorI14__hip_fp8_e4m3SaIS2_EES6_RS1_I14__hip_bfloat16SaIS7_EEii,"aG",@progbits,_Z11matmul_hostILi8192ELi8192ELi8192ELi256EE12TimingResultRKSt6vectorI14__hip_fp8_e4m3SaIS2_EES6_RS1_I14__hip_bfloat16SaIS7_EEii,comdat
	.p2align	2, 0x0
GCC_except_table2:
.Lexception1:
	.byte	255                             # @LPStart Encoding = omit
	.byte	255                             # @TType Encoding = omit
	.byte	1                               # Call site Encoding = uleb128
	.uleb128 .Lcst_end1-.Lcst_begin1
.Lcst_begin1:
	.uleb128 .Lfunc_begin1-.Lfunc_begin1    # >> Call Site 1 <<
	.uleb128 .Ltmp15-.Lfunc_begin1          #   Call between .Lfunc_begin1 and .Ltmp15
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp15-.Lfunc_begin1          # >> Call Site 2 <<
	.uleb128 .Ltmp28-.Ltmp15                #   Call between .Ltmp15 and .Ltmp28
	.uleb128 .Ltmp29-.Lfunc_begin1          #     jumps to .Ltmp29
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp30-.Lfunc_begin1          # >> Call Site 3 <<
	.uleb128 .Ltmp33-.Ltmp30                #   Call between .Ltmp30 and .Ltmp33
	.uleb128 .Ltmp40-.Lfunc_begin1          #     jumps to .Ltmp40
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp33-.Lfunc_begin1          # >> Call Site 4 <<
	.uleb128 .Ltmp34-.Ltmp33                #   Call between .Ltmp33 and .Ltmp34
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp34-.Lfunc_begin1          # >> Call Site 5 <<
	.uleb128 .Ltmp39-.Ltmp34                #   Call between .Ltmp34 and .Ltmp39
	.uleb128 .Ltmp40-.Lfunc_begin1          #     jumps to .Ltmp40
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp46-.Lfunc_begin1          # >> Call Site 6 <<
	.uleb128 .Ltmp79-.Ltmp46                #   Call between .Ltmp46 and .Ltmp79
	.uleb128 .Ltmp82-.Lfunc_begin1          #     jumps to .Ltmp82
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp79-.Lfunc_begin1          # >> Call Site 7 <<
	.uleb128 .Ltmp41-.Ltmp79                #   Call between .Ltmp79 and .Ltmp41
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp41-.Lfunc_begin1          # >> Call Site 8 <<
	.uleb128 .Ltmp44-.Ltmp41                #   Call between .Ltmp41 and .Ltmp44
	.uleb128 .Ltmp45-.Lfunc_begin1          #     jumps to .Ltmp45
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp44-.Lfunc_begin1          # >> Call Site 9 <<
	.uleb128 .Ltmp52-.Ltmp44                #   Call between .Ltmp44 and .Ltmp52
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp52-.Lfunc_begin1          # >> Call Site 10 <<
	.uleb128 .Ltmp81-.Ltmp52                #   Call between .Ltmp52 and .Ltmp81
	.uleb128 .Ltmp82-.Lfunc_begin1          #     jumps to .Ltmp82
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp81-.Lfunc_begin1          # >> Call Site 11 <<
	.uleb128 .Lfunc_end2-.Ltmp81            #   Call between .Ltmp81 and .Lfunc_end2
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
.Lcst_end1:
	.p2align	2, 0x0
                                        # -- End function
	.section	.text._ZN14__hip_fp8_e4m3C2Ef,"axG",@progbits,_ZN14__hip_fp8_e4m3C2Ef,comdat
	.weak	_ZN14__hip_fp8_e4m3C2Ef         # -- Begin function _ZN14__hip_fp8_e4m3C2Ef
	.p2align	4
	.type	_ZN14__hip_fp8_e4m3C2Ef,@function
_ZN14__hip_fp8_e4m3C2Ef:                # @_ZN14__hip_fp8_e4m3C2Ef
	.cfi_startproc
# %bb.0:
	movd	%xmm0, %eax
	movl	%eax, %edx
	shrl	$24, %edx
	movl	%eax, %ecx
	notl	%ecx
	testl	$2139095040, %ecx               # imm = 0x7F800000
	jne	.LBB3_2
# %bb.1:
	orb	$127, %dl
	movb	%dl, (%rdi)
	retq
.LBB3_2:
	andl	$-128, %edx
	movl	%eax, %ecx
	andl	$2147483647, %ecx               # imm = 0x7FFFFFFF
	cmpl	$1138753537, %ecx               # imm = 0x43E00001
	jb	.LBB3_4
# %bb.3:
	orb	$126, %dl
	movb	%dl, (%rdi)
	retq
.LBB3_4:
	testq	%rax, %rax
	je	.LBB3_5
# %bb.6:
	movl	%eax, %esi
	andl	$8388607, %esi                  # imm = 0x7FFFFF
	shrl	$23, %eax
	movzbl	%al, %ecx
	testl	%ecx, %ecx
	je	.LBB3_7
# %bb.8:
	leal	-127(%rcx), %r8d
	movl	$121, %r9d
	subl	%ecx, %r9d
	orq	$8388608, %rsi                  # imm = 0x800000
	xorl	%eax, %eax
	cmpl	$122, %ecx
	cmovbl	%r9d, %eax
	jmp	.LBB3_9
.LBB3_5:
	xorl	%edx, %edx
	movb	%dl, (%rdi)
	retq
.LBB3_7:
	movl	$120, %eax
	movl	$-126, %r8d
.LBB3_9:                                # %select.unfold.i.i
	leal	20(%rax), %ecx
	movq	$-1, %r10
                                        # kill: def $cl killed $cl killed $ecx
	shlq	%cl, %r10
	notl	%r10d
	andl	%esi, %r10d
	leal	19(%rax), %ecx
	movl	$1, %r11d
                                        # kill: def $cl killed $cl killed $ecx
	shlq	%cl, %r11
	movl	%eax, %ecx
	shrq	%cl, %rsi
	addl	%eax, %r8d
	movl	%esi, %r9d
	shrl	$23, %r9d
	addl	%r8d, %r9d
	xorl	%ecx, %ecx
	btl	$20, %esi
	movl	$0, %eax
	adcl	$1048575, %eax                  # imm = 0xFFFFF
	cmpq	%r11, %r10
	cmovnel	%ecx, %eax
	addl	%esi, %eax
	andl	$1048575, %eax                  # imm = 0xFFFFF
	addq	%rsi, %rax
	movl	%r9d, %ecx
	addl	$6, %ecx
	je	.LBB3_10
# %bb.11:
	testl	$16777216, %eax                 # imm = 0x1000000
	je	.LBB3_13
# %bb.12:
	shrl	%eax
	addl	$7, %r9d
	movl	%r9d, %ecx
	jmp	.LBB3_13
.LBB3_10:
	movl	%eax, %ecx
	shrl	$23, %ecx
	andl	$1, %ecx
.LBB3_13:
	shrl	$20, %eax
	cmpl	$15, %ecx
	movl	$15, %esi
	cmovll	%ecx, %esi
	movl	$7, %r8d
	cmovleq	%rax, %r8
	movl	%r8d, %eax
	andl	$7, %eax
	shll	$3, %esi
	orl	%edx, %esi
	orl	%eax, %esi
	testq	%r8, %r8
	cmovnel	%esi, %edx
	testl	%ecx, %ecx
	cmovnel	%esi, %edx
	movb	%dl, (%rdi)
	retq
.Lfunc_end3:
	.size	_ZN14__hip_fp8_e4m3C2Ef, .Lfunc_end3-_ZN14__hip_fp8_e4m3C2Ef
	.cfi_endproc
                                        # -- End function
	.section	.rodata.cst4,"aM",@progbits,4
	.p2align	2, 0x0                          # -- Begin function _ZZ4mainENKUlRKSt6vectorI14__hip_fp8_e4m3SaIS0_EES4_RS_I14__hip_bfloat16SaIS5_EEiE_clES4_S4_S8_i.omp_outlined
.LCPI4_0:
	.long	0x80000000                      # float -0
.LCPI4_1:
	.long	0x7f800001                      # float NaN
	.text
	.p2align	4
	.type	_ZZ4mainENKUlRKSt6vectorI14__hip_fp8_e4m3SaIS0_EES4_RS_I14__hip_bfloat16SaIS5_EEiE_clES4_S4_S8_i.omp_outlined,@function
_ZZ4mainENKUlRKSt6vectorI14__hip_fp8_e4m3SaIS0_EES4_RS_I14__hip_bfloat16SaIS5_EEiE_clES4_S4_S8_i.omp_outlined: # @_ZZ4mainENKUlRKSt6vectorI14__hip_fp8_e4m3SaIS0_EES4_RS_I14__hip_bfloat16SaIS5_EEiE_clES4_S4_S8_i.omp_outlined
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	pushq	%r15
	.cfi_def_cfa_offset 24
	pushq	%r14
	.cfi_def_cfa_offset 32
	pushq	%r13
	.cfi_def_cfa_offset 40
	pushq	%r12
	.cfi_def_cfa_offset 48
	pushq	%rbx
	.cfi_def_cfa_offset 56
	subq	$40, %rsp
	.cfi_def_cfa_offset 96
	.cfi_offset %rbx, -56
	.cfi_offset %r12, -48
	.cfi_offset %r13, -40
	.cfi_offset %r14, -32
	.cfi_offset %r15, -24
	.cfi_offset %rbp, -16
	movq	%r9, 32(%rsp)                   # 8-byte Spill
	movq	%r8, 24(%rsp)                   # 8-byte Spill
	movl	(%rdx), %ebx
	testl	%ebx, %ebx
	jle	.LBB4_15
# %bb.1:
	movq	%rcx, %r15
	decl	%ebx
	movl	$0, 8(%rsp)
	movl	%ebx, 4(%rsp)
	movl	$1, 20(%rsp)
	movl	$0, 16(%rsp)
	movl	(%rdi), %esi
	subq	$8, %rsp
	.cfi_adjust_cfa_offset 8
	leaq	28(%rsp), %rax
	leaq	24(%rsp), %rcx
	leaq	16(%rsp), %r8
	leaq	12(%rsp), %r9
	movl	$.L__unnamed_2, %edi
	movl	%esi, 20(%rsp)                  # 4-byte Spill
	movl	$34, %edx
	pushq	$1
	.cfi_adjust_cfa_offset 8
	pushq	$1
	.cfi_adjust_cfa_offset 8
	pushq	%rax
	.cfi_adjust_cfa_offset 8
	callq	__kmpc_for_static_init_4@PLT
	addq	$32, %rsp
	.cfi_adjust_cfa_offset -32
	movl	4(%rsp), %eax
	cmpl	%ebx, %eax
	cmovll	%eax, %ebx
	movl	%ebx, 4(%rsp)
	movslq	8(%rsp), %rbp
	cmpl	%ebx, %ebp
	jg	.LBB4_14
# %bb.2:                                # %.preheader37.lr.ph
	movl	(%r15), %r13d
	testl	%r13d, %r13d
	jle	.LBB4_14
# %bb.3:                                # %.preheader37.preheader
	movq	%rbp, %r12
	shlq	$13, %r12
	movd	.LCPI4_0(%rip), %xmm3           # xmm3 = [-0.0E+0,0.0E+0,0.0E+0,0.0E+0]
	movd	.LCPI4_1(%rip), %xmm4           # xmm4 = [NaN,0.0E+0,0.0E+0,0.0E+0]
	jmp	.LBB4_4
	.p2align	4
.LBB4_12:                               # %._crit_edge.loopexit
                                        #   in Loop: Header=BB4_4 Depth=1
	movl	4(%rsp), %ebx
.LBB4_13:                               # %._crit_edge
                                        #   in Loop: Header=BB4_4 Depth=1
	leaq	1(%rbp), %rax
	movslq	%ebx, %rcx
	addq	$8192, %r12                     # imm = 0x2000
	cmpq	%rcx, %rbp
	movq	%rax, %rbp
	jge	.LBB4_14
.LBB4_4:                                # %.preheader37
                                        # =>This Loop Header: Depth=1
                                        #     Child Loop BB4_6 Depth 2
                                        #       Child Loop BB4_7 Depth 3
	testl	%r13d, %r13d
	jle	.LBB4_13
# %bb.5:                                # %.preheader.lr.ph
                                        #   in Loop: Header=BB4_4 Depth=1
	xorl	%ebx, %ebx
	xorl	%r14d, %r14d
	jmp	.LBB4_6
	.p2align	4
.LBB4_27:                               #   in Loop: Header=BB4_6 Depth=2
	callq	__truncsfbf2@PLT
	movd	.LCPI4_1(%rip), %xmm4           # xmm4 = [NaN,0.0E+0,0.0E+0,0.0E+0]
	movd	.LCPI4_0(%rip), %xmm3           # xmm3 = [-0.0E+0,0.0E+0,0.0E+0,0.0E+0]
	movslq	%r13d, %rax
	imulq	%rbp, %rax
	addq	%rax, %rax
	movq	96(%rsp), %rcx
	addq	(%rcx), %rax
	pextrw	$0, %xmm0, %ecx
	movw	%cx, (%rax,%r14,2)
	incq	%r14
	movslq	(%r15), %r13
	addq	$8192, %rbx                     # imm = 0x2000
	cmpq	%r13, %r14
	jge	.LBB4_12
.LBB4_6:                                # %.preheader
                                        #   Parent Loop BB4_4 Depth=1
                                        # =>  This Loop Header: Depth=2
                                        #       Child Loop BB4_7 Depth 3
	movq	32(%rsp), %rax                  # 8-byte Reload
	movq	(%rax), %rax
	addq	%rbx, %rax
	movq	24(%rsp), %rcx                  # 8-byte Reload
	movq	(%rcx), %rdx
	addq	%r12, %rdx
	xorps	%xmm0, %xmm0
	xorl	%esi, %esi
	jmp	.LBB4_7
	.p2align	4
.LBB4_19:                               #   in Loop: Header=BB4_7 Depth=3
	pxor	%xmm2, %xmm2
.LBB4_26:                               # %_ZNK14__hip_fp8_e4m3cvfEv.exit36
                                        #   in Loop: Header=BB4_7 Depth=3
	mulss	%xmm2, %xmm1
	addss	%xmm1, %xmm0
	incq	%rsi
	cmpq	$8192, %rsi                     # imm = 0x2000
	je	.LBB4_27
.LBB4_7:                                #   Parent Loop BB4_4 Depth=1
                                        #     Parent Loop BB4_6 Depth=2
                                        # =>    This Inner Loop Header: Depth=3
	movzbl	(%rdx,%rsi), %edi
	pxor	%xmm1, %xmm1
	testl	%edi, %edi
	je	.LBB4_18
# %bb.8:                                #   in Loop: Header=BB4_7 Depth=3
	movdqa	%xmm3, %xmm1
	cmpq	$128, %rdi
	je	.LBB4_18
# %bb.9:                                #   in Loop: Header=BB4_7 Depth=3
	movl	%edi, %ecx
	andl	$127, %ecx
	movdqa	%xmm4, %xmm1
	cmpl	$127, %ecx
	je	.LBB4_18
# %bb.10:                               #   in Loop: Header=BB4_7 Depth=3
	movl	%edi, %r8d
	andl	$7, %r8d
	cmpl	$7, %ecx
	ja	.LBB4_11
# %bb.16:                               #   in Loop: Header=BB4_7 Depth=3
	bsrl	%r8d, %r9d
	xorl	$31, %r9d
	leal	-28(%r9), %ecx
                                        # kill: def $cl killed $cl killed $ecx
	shlq	%cl, %r8
	movl	$29, %ecx
	subl	%r9d, %ecx
	andl	$7, %r8d
	jmp	.LBB4_17
.LBB4_11:                               #   in Loop: Header=BB4_7 Depth=3
	shrl	$3, %ecx
.LBB4_17:                               #   in Loop: Header=BB4_7 Depth=3
	shll	$20, %r8d
	andl	$-128, %edi
	shll	$24, %edi
	orl	%r8d, %edi
	shll	$23, %ecx
	addl	$1006632960, %ecx               # imm = 0x3C000000
	orl	%edi, %ecx
	movd	%ecx, %xmm1
	.p2align	4
.LBB4_18:                               # %_ZNK14__hip_fp8_e4m3cvfEv.exit
                                        #   in Loop: Header=BB4_7 Depth=3
	movzbl	(%rax,%rsi), %edi
	testl	%edi, %edi
	je	.LBB4_19
# %bb.20:                               #   in Loop: Header=BB4_7 Depth=3
	movdqa	%xmm3, %xmm2
	cmpq	$128, %rdi
	je	.LBB4_26
# %bb.21:                               #   in Loop: Header=BB4_7 Depth=3
	movl	%edi, %ecx
	andl	$127, %ecx
	movdqa	%xmm4, %xmm2
	cmpl	$127, %ecx
	je	.LBB4_26
# %bb.22:                               #   in Loop: Header=BB4_7 Depth=3
	movl	%edi, %r8d
	andl	$7, %r8d
	cmpl	$7, %ecx
	ja	.LBB4_23
# %bb.24:                               #   in Loop: Header=BB4_7 Depth=3
	bsrl	%r8d, %r9d
	xorl	$31, %r9d
	leal	-28(%r9), %ecx
                                        # kill: def $cl killed $cl killed $ecx
	shlq	%cl, %r8
	movl	$29, %ecx
	subl	%r9d, %ecx
	andl	$7, %r8d
	jmp	.LBB4_25
.LBB4_23:                               #   in Loop: Header=BB4_7 Depth=3
	shrl	$3, %ecx
.LBB4_25:                               #   in Loop: Header=BB4_7 Depth=3
	shll	$20, %r8d
	andl	$-128, %edi
	shll	$24, %edi
	orl	%r8d, %edi
	shll	$23, %ecx
	addl	$1006632960, %ecx               # imm = 0x3C000000
	orl	%edi, %ecx
	movd	%ecx, %xmm2
	jmp	.LBB4_26
.LBB4_14:                               # %._crit_edge43
	movl	$.L__unnamed_3, %edi
	movl	12(%rsp), %esi                  # 4-byte Reload
	callq	__kmpc_for_static_fini@PLT
.LBB4_15:
	addq	$40, %rsp
	.cfi_def_cfa_offset 56
	popq	%rbx
	.cfi_def_cfa_offset 48
	popq	%r12
	.cfi_def_cfa_offset 40
	popq	%r13
	.cfi_def_cfa_offset 32
	popq	%r14
	.cfi_def_cfa_offset 24
	popq	%r15
	.cfi_def_cfa_offset 16
	popq	%rbp
	.cfi_def_cfa_offset 8
	retq
.Lfunc_end4:
	.size	_ZZ4mainENKUlRKSt6vectorI14__hip_fp8_e4m3SaIS0_EES4_RS_I14__hip_bfloat16SaIS5_EEiE_clES4_S4_S8_i.omp_outlined, .Lfunc_end4-_ZZ4mainENKUlRKSt6vectorI14__hip_fp8_e4m3SaIS0_EES4_RS_I14__hip_bfloat16SaIS5_EEiE_clES4_S4_S8_i.omp_outlined
	.cfi_endproc
                                        # -- End function
	.section	.text._Z28__device_stub__matmul_deviceILi8192ELi8192ELi8192EEvN7kittens2glI14__hip_fp8_e4m3Li1ELi1EXT_EXT1_EJEEENS1_IS2_Li1ELi1EXT0_EXT1_EJEEENS1_I14__hip_bfloat16Li1ELi1EXT_EXT0_EJEEE,"axG",@progbits,_Z28__device_stub__matmul_deviceILi8192ELi8192ELi8192EEvN7kittens2glI14__hip_fp8_e4m3Li1ELi1EXT_EXT1_EJEEENS1_IS2_Li1ELi1EXT0_EXT1_EJEEENS1_I14__hip_bfloat16Li1ELi1EXT_EXT0_EJEEE,comdat
	.weak	_Z28__device_stub__matmul_deviceILi8192ELi8192ELi8192EEvN7kittens2glI14__hip_fp8_e4m3Li1ELi1EXT_EXT1_EJEEENS1_IS2_Li1ELi1EXT0_EXT1_EJEEENS1_I14__hip_bfloat16Li1ELi1EXT_EXT0_EJEEE # -- Begin function _Z28__device_stub__matmul_deviceILi8192ELi8192ELi8192EEvN7kittens2glI14__hip_fp8_e4m3Li1ELi1EXT_EXT1_EJEEENS1_IS2_Li1ELi1EXT0_EXT1_EJEEENS1_I14__hip_bfloat16Li1ELi1EXT_EXT0_EJEEE
	.p2align	4
	.type	_Z28__device_stub__matmul_deviceILi8192ELi8192ELi8192EEvN7kittens2glI14__hip_fp8_e4m3Li1ELi1EXT_EXT1_EJEEENS1_IS2_Li1ELi1EXT0_EXT1_EJEEENS1_I14__hip_bfloat16Li1ELi1EXT_EXT0_EJEEE,@function
_Z28__device_stub__matmul_deviceILi8192ELi8192ELi8192EEvN7kittens2glI14__hip_fp8_e4m3Li1ELi1EXT_EXT1_EJEEENS1_IS2_Li1ELi1EXT0_EXT1_EJEEENS1_I14__hip_bfloat16Li1ELi1EXT_EXT0_EJEEE: # @_Z28__device_stub__matmul_deviceILi8192ELi8192ELi8192EEvN7kittens2glI14__hip_fp8_e4m3Li1ELi1EXT_EXT1_EJEEENS1_IS2_Li1ELi1EXT0_EXT1_EJEEENS1_I14__hip_bfloat16Li1ELi1EXT_EXT0_EJEEE
	.cfi_startproc
# %bb.0:
	subq	$72, %rsp
	.cfi_def_cfa_offset 80
	movq	%rdi, 48(%rsp)
	movq	%rsi, 56(%rsp)
	movq	%rdx, 64(%rsp)
	leaq	32(%rsp), %rdi
	leaq	16(%rsp), %rsi
	leaq	8(%rsp), %rdx
	movq	%rsp, %rcx
	callq	__hipPopCallConfiguration
	movq	32(%rsp), %rsi
	movl	40(%rsp), %edx
	movq	16(%rsp), %rcx
	movl	24(%rsp), %r8d
	leaq	48(%rsp), %r9
	movl	$_Z13matmul_deviceILi8192ELi8192ELi8192EEvN7kittens2glI14__hip_fp8_e4m3Li1ELi1EXT_EXT1_EJEEENS1_IS2_Li1ELi1EXT0_EXT1_EJEEENS1_I14__hip_bfloat16Li1ELi1EXT_EXT0_EJEEE, %edi
	pushq	(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	16(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel
	addq	$88, %rsp
	.cfi_adjust_cfa_offset -88
	retq
.Lfunc_end5:
	.size	_Z28__device_stub__matmul_deviceILi8192ELi8192ELi8192EEvN7kittens2glI14__hip_fp8_e4m3Li1ELi1EXT_EXT1_EJEEENS1_IS2_Li1ELi1EXT0_EXT1_EJEEENS1_I14__hip_bfloat16Li1ELi1EXT_EXT0_EJEEE, .Lfunc_end5-_Z28__device_stub__matmul_deviceILi8192ELi8192ELi8192EEvN7kittens2glI14__hip_fp8_e4m3Li1ELi1EXT_EXT1_EJEEENS1_IS2_Li1ELi1EXT0_EXT1_EJEEENS1_I14__hip_bfloat16Li1ELi1EXT_EXT0_EJEEE
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNSt6vectorI14__hip_bfloat16SaIS0_EE17_M_default_appendEm,"axG",@progbits,_ZNSt6vectorI14__hip_bfloat16SaIS0_EE17_M_default_appendEm,comdat
	.weak	_ZNSt6vectorI14__hip_bfloat16SaIS0_EE17_M_default_appendEm # -- Begin function _ZNSt6vectorI14__hip_bfloat16SaIS0_EE17_M_default_appendEm
	.p2align	4
	.type	_ZNSt6vectorI14__hip_bfloat16SaIS0_EE17_M_default_appendEm,@function
_ZNSt6vectorI14__hip_bfloat16SaIS0_EE17_M_default_appendEm: # @_ZNSt6vectorI14__hip_bfloat16SaIS0_EE17_M_default_appendEm
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	pushq	%r15
	.cfi_def_cfa_offset 24
	pushq	%r14
	.cfi_def_cfa_offset 32
	pushq	%r13
	.cfi_def_cfa_offset 40
	pushq	%r12
	.cfi_def_cfa_offset 48
	pushq	%rbx
	.cfi_def_cfa_offset 56
	subq	$24, %rsp
	.cfi_def_cfa_offset 80
	.cfi_offset %rbx, -56
	.cfi_offset %r12, -48
	.cfi_offset %r13, -40
	.cfi_offset %r14, -32
	.cfi_offset %r15, -24
	.cfi_offset %rbp, -16
	testq	%rsi, %rsi
	je	.LBB6_13
# %bb.1:
	movq	%rsi, %r14
	movq	%rdi, %rbx
	movq	8(%rdi), %r15
	movq	16(%rdi), %rcx
	movq	%rcx, %rax
	subq	%r15, %rax
	sarq	%rax
	cmpq	%rsi, %rax
	jae	.LBB6_2
# %bb.5:
	movq	%rcx, 16(%rsp)                  # 8-byte Spill
	movq	(%rbx), %rax
	movq	%rax, %r12
	subq	%rax, %r15
	movq	%r15, %rcx
	sarq	%rcx
	movabsq	$4611686018427387903, %rax      # imm = 0x3FFFFFFFFFFFFFFF
	movq	%rcx, %rdx
	xorq	%rax, %rdx
	cmpq	%r14, %rdx
	jb	.LBB6_14
# %bb.6:                                # %_ZNKSt6vectorI14__hip_bfloat16SaIS0_EE12_M_check_lenEmPKc.exit
	cmpq	%r14, %rcx
	movq	%r14, %r13
	cmovaq	%rcx, %r13
	addq	%rcx, %r13
	cmpq	%rax, %r13
	cmovaeq	%rax, %r13
	leaq	(,%r13,2), %rdi
	callq	_Znwm
	movq	%rax, %rbp
	addq	%r15, %rax
	movq	%rax, 8(%rsp)                   # 8-byte Spill
	movw	$0, (%rbp,%r15)
	cmpq	$1, %r14
	je	.LBB6_8
# %bb.7:                                # %.lr.ph.preheader.i.i.i.i.i.i.i30
	movq	8(%rsp), %rax                   # 8-byte Reload
	leaq	2(%rax), %rdi
	leaq	-2(,%r14,2), %rdx
	xorl	%esi, %esi
	callq	memset@PLT
.LBB6_8:                                # %_ZSt27__uninitialized_default_n_aIP14__hip_bfloat16mS0_ET_S2_T0_RSaIT1_E.exit32
	testq	%r15, %r15
	jle	.LBB6_10
# %bb.9:
	movq	%rbp, %rdi
	movq	%r12, %rsi
	movq	%r15, %rdx
	callq	memmove@PLT
.LBB6_10:                               # %_ZNSt6vectorI14__hip_bfloat16SaIS0_EE11_S_relocateEPS0_S3_S3_RS1_.exit
	testq	%r12, %r12
	je	.LBB6_12
# %bb.11:
	movq	16(%rsp), %rsi                  # 8-byte Reload
	subq	%r12, %rsi
	movq	%r12, %rdi
	callq	_ZdlPvm
.LBB6_12:                               # %_ZNSt12_Vector_baseI14__hip_bfloat16SaIS0_EE13_M_deallocateEPS0_m.exit35
	movq	%rbp, (%rbx)
	movq	8(%rsp), %rax                   # 8-byte Reload
	leaq	(%rax,%r14,2), %rax
	movq	%rax, 8(%rbx)
	leaq	(,%r13,2), %rax
	addq	%rbp, %rax
	movq	%rax, 16(%rbx)
	jmp	.LBB6_13
.LBB6_2:
	movw	$0, (%r15)
	leaq	2(%r15), %rdi
	cmpq	$1, %r14
	je	.LBB6_4
# %bb.3:                                # %.lr.ph.preheader.i.i.i.i.i.i.i
	leaq	-2(,%r14,2), %rdx
	xorl	%esi, %esi
	callq	memset@PLT
	leaq	(%r15,%r14,2), %rdi
.LBB6_4:                                # %_ZSt27__uninitialized_default_n_aIP14__hip_bfloat16mS0_ET_S2_T0_RSaIT1_E.exit
	movq	%rdi, 8(%rbx)
.LBB6_13:
	addq	$24, %rsp
	.cfi_def_cfa_offset 56
	popq	%rbx
	.cfi_def_cfa_offset 48
	popq	%r12
	.cfi_def_cfa_offset 40
	popq	%r13
	.cfi_def_cfa_offset 32
	popq	%r14
	.cfi_def_cfa_offset 24
	popq	%r15
	.cfi_def_cfa_offset 16
	popq	%rbp
	.cfi_def_cfa_offset 8
	retq
.LBB6_14:
	.cfi_def_cfa_offset 80
	movl	$.L.str.18, %edi
	callq	_ZSt20__throw_length_errorPKc
.Lfunc_end6:
	.size	_ZNSt6vectorI14__hip_bfloat16SaIS0_EE17_M_default_appendEm, .Lfunc_end6-_ZNSt6vectorI14__hip_bfloat16SaIS0_EE17_M_default_appendEm
	.cfi_endproc
                                        # -- End function
	.section	.text.startup,"ax",@progbits
	.p2align	4                               # -- Begin function _GLOBAL__sub_I_8_wave.cu
	.type	_GLOBAL__sub_I_8_wave.cu,@function
_GLOBAL__sub_I_8_wave.cu:               # @_GLOBAL__sub_I_8_wave.cu
	.cfi_startproc
# %bb.0:
	pushq	%rax
	.cfi_def_cfa_offset 16
	movl	$_ZStL8__ioinit, %edi
	callq	_ZNSt8ios_base4InitC1Ev
	movl	$_ZNSt8ios_base4InitD1Ev, %edi
	movl	$_ZStL8__ioinit, %esi
	movl	$__dso_handle, %edx
	popq	%rax
	.cfi_def_cfa_offset 8
	jmp	__cxa_atexit                    # TAILCALL
.Lfunc_end7:
	.size	_GLOBAL__sub_I_8_wave.cu, .Lfunc_end7-_GLOBAL__sub_I_8_wave.cu
	.cfi_endproc
                                        # -- End function
	.text
	.p2align	4                               # -- Begin function __hip_module_ctor
	.type	__hip_module_ctor,@function
__hip_module_ctor:                      # @__hip_module_ctor
	.cfi_startproc
# %bb.0:
	subq	$40, %rsp
	.cfi_def_cfa_offset 48
	movq	__hip_gpubin_handle_e843e8b1c0f57c70(%rip), %rdi
	testq	%rdi, %rdi
	jne	.LBB8_2
# %bb.1:
	movl	$__hip_fatbin_wrapper, %edi
	callq	__hipRegisterFatBinary
	movq	%rax, %rdi
	movq	%rax, __hip_gpubin_handle_e843e8b1c0f57c70(%rip)
.LBB8_2:
	xorps	%xmm0, %xmm0
	movups	%xmm0, 16(%rsp)
	movups	%xmm0, (%rsp)
	movl	$_Z13matmul_deviceILi8192ELi8192ELi8192EEvN7kittens2glI14__hip_fp8_e4m3Li1ELi1EXT_EXT1_EJEEENS1_IS2_Li1ELi1EXT0_EXT1_EJEEENS1_I14__hip_bfloat16Li1ELi1EXT_EXT0_EJEEE, %esi
	movl	$.L__unnamed_4, %edx
	movl	$.L__unnamed_4, %ecx
	movl	$-1, %r8d
	xorl	%r9d, %r9d
	callq	__hipRegisterFunction
	movl	$__hip_module_dtor, %edi
	addq	$40, %rsp
	.cfi_def_cfa_offset 8
	jmp	atexit                          # TAILCALL
.Lfunc_end8:
	.size	__hip_module_ctor, .Lfunc_end8-__hip_module_ctor
	.cfi_endproc
                                        # -- End function
	.p2align	4                               # -- Begin function __hip_module_dtor
	.type	__hip_module_dtor,@function
__hip_module_dtor:                      # @__hip_module_dtor
	.cfi_startproc
# %bb.0:
	movq	__hip_gpubin_handle_e843e8b1c0f57c70(%rip), %rdi
	testq	%rdi, %rdi
	je	.LBB9_2
# %bb.1:
	pushq	%rax
	.cfi_def_cfa_offset 16
	callq	__hipUnregisterFatBinary
	movq	$0, __hip_gpubin_handle_e843e8b1c0f57c70(%rip)
	addq	$8, %rsp
	.cfi_def_cfa_offset 8
.LBB9_2:
	retq
.Lfunc_end9:
	.size	__hip_module_dtor, .Lfunc_end9-__hip_module_dtor
	.cfi_endproc
                                        # -- End function
	.type	_ZStL8__ioinit,@object          # @_ZStL8__ioinit
	.local	_ZStL8__ioinit
	.comm	_ZStL8__ioinit,1,1
	.hidden	__dso_handle
	.type	.L.str,@object                  # @.str
	.section	.rodata.str1.1,"aMS",@progbits,1
.L.str:
	.asciz	"Matrix dimensions: %dx%dx%d, CUs: %d\n"
	.size	.L.str, 38

	.type	.L.str.1,@object                # @.str.1
.L.str.1:
	.asciz	"Warmup iterations: %d, Timing iterations: %d\n\n"
	.size	.L.str.1, 47

	.type	.L.str.4,@object                # @.str.4
.L.str.4:
	.asciz	"Mismatch at (row=%d, col=%d): c_host = %f, c_ref = %f, diff = %f\n"
	.size	.L.str.4, 66

	.type	.L.str.7,@object                # @.str.7
.L.str.7:
	.asciz	"  Kernel time (best): %.3f ms,  TFLOPS: %.2f\n"
	.size	.L.str.7, 46

	.type	.L.str.8,@object                # @.str.8
.L.str.8:
	.asciz	"  Kernel time (avg ): %.3f ms,  TFLOPS: %.2f\n"
	.size	.L.str.8, 46

	.type	.L.str.10,@object               # @.str.10
.L.str.10:
	.asciz	"\nSpeedup (best): %.2fx\n"
	.size	.L.str.10, 24

	.type	.L.str.11,@object               # @.str.11
.L.str.11:
	.asciz	"Speedup (avg ): %.2fx\n"
	.size	.L.str.11, 23

	.type	.L__unnamed_5,@object           # @0
.L__unnamed_5:
	.asciz	";8_wave.cu;main;486;1;;"
	.size	.L__unnamed_5, 24

	.type	.L__unnamed_2,@object           # @1
	.section	.rodata,"a",@progbits
	.p2align	3, 0x0
.L__unnamed_2:
	.long	0                               # 0x0
	.long	514                             # 0x202
	.long	0                               # 0x0
	.long	23                              # 0x17
	.quad	.L__unnamed_5
	.size	.L__unnamed_2, 24

	.type	.L__unnamed_6,@object           # @2
	.section	.rodata.str1.1,"aMS",@progbits,1
.L__unnamed_6:
	.asciz	";8_wave.cu;main;486;25;;"
	.size	.L__unnamed_6, 25

	.type	.L__unnamed_3,@object           # @3
	.section	.rodata,"a",@progbits
	.p2align	3, 0x0
.L__unnamed_3:
	.long	0                               # 0x0
	.long	514                             # 0x202
	.long	0                               # 0x0
	.long	24                              # 0x18
	.quad	.L__unnamed_6
	.size	.L__unnamed_3, 24

	.type	.L__unnamed_1,@object           # @4
	.p2align	3, 0x0
.L__unnamed_1:
	.long	0                               # 0x0
	.long	2                               # 0x2
	.long	0                               # 0x0
	.long	23                              # 0x17
	.quad	.L__unnamed_5
	.size	.L__unnamed_1, 24

	.type	.L.str.15,@object               # @.str.15
	.section	.rodata.str1.1,"aMS",@progbits,1
.L.str.15:
	.asciz	"Error: Input vector 'a' size %zu does not match expected M*K=%d\n"
	.size	.L.str.15, 65

	.type	.L.str.16,@object               # @.str.16
.L.str.16:
	.asciz	"Error: Input vector 'b' size %zu does not match expected N*K=%d\n"
	.size	.L.str.16, 65

	.type	.L.str.17,@object               # @.str.17
.L.str.17:
	.asciz	"8_wave.cu"
	.size	.L.str.17, 10

	.type	_Z13matmul_deviceILi8192ELi8192ELi8192EEvN7kittens2glI14__hip_fp8_e4m3Li1ELi1EXT_EXT1_EJEEENS1_IS2_Li1ELi1EXT0_EXT1_EJEEENS1_I14__hip_bfloat16Li1ELi1EXT_EXT0_EJEEE,@object # @_Z13matmul_deviceILi8192ELi8192ELi8192EEvN7kittens2glI14__hip_fp8_e4m3Li1ELi1EXT_EXT1_EJEEENS1_IS2_Li1ELi1EXT0_EXT1_EJEEENS1_I14__hip_bfloat16Li1ELi1EXT_EXT0_EJEEE
	.section	.rodata._Z13matmul_deviceILi8192ELi8192ELi8192EEvN7kittens2glI14__hip_fp8_e4m3Li1ELi1EXT_EXT1_EJEEENS1_IS2_Li1ELi1EXT0_EXT1_EJEEENS1_I14__hip_bfloat16Li1ELi1EXT_EXT0_EJEEE,"aG",@progbits,_Z13matmul_deviceILi8192ELi8192ELi8192EEvN7kittens2glI14__hip_fp8_e4m3Li1ELi1EXT_EXT1_EJEEENS1_IS2_Li1ELi1EXT0_EXT1_EJEEENS1_I14__hip_bfloat16Li1ELi1EXT_EXT0_EJEEE,comdat
	.weak	_Z13matmul_deviceILi8192ELi8192ELi8192EEvN7kittens2glI14__hip_fp8_e4m3Li1ELi1EXT_EXT1_EJEEENS1_IS2_Li1ELi1EXT0_EXT1_EJEEENS1_I14__hip_bfloat16Li1ELi1EXT_EXT0_EJEEE
	.p2align	3, 0x0
_Z13matmul_deviceILi8192ELi8192ELi8192EEvN7kittens2glI14__hip_fp8_e4m3Li1ELi1EXT_EXT1_EJEEENS1_IS2_Li1ELi1EXT0_EXT1_EJEEENS1_I14__hip_bfloat16Li1ELi1EXT_EXT0_EJEEE:
	.quad	_Z28__device_stub__matmul_deviceILi8192ELi8192ELi8192EEvN7kittens2glI14__hip_fp8_e4m3Li1ELi1EXT_EXT1_EJEEENS1_IS2_Li1ELi1EXT0_EXT1_EJEEENS1_I14__hip_bfloat16Li1ELi1EXT_EXT0_EJEEE
	.size	_Z13matmul_deviceILi8192ELi8192ELi8192EEvN7kittens2glI14__hip_fp8_e4m3Li1ELi1EXT_EXT1_EJEEENS1_IS2_Li1ELi1EXT0_EXT1_EJEEENS1_I14__hip_bfloat16Li1ELi1EXT_EXT0_EJEEE, 8

	.type	.L.str.18,@object               # @.str.18
	.section	.rodata.str1.1,"aMS",@progbits,1
.L.str.18:
	.asciz	"vector::_M_default_append"
	.size	.L.str.18, 26

	.type	.L.str.19,@object               # @.str.19
.L.str.19:
	.asciz	"hipCheckError() failed at %s:%i : %s\n"
	.size	.L.str.19, 38

	.type	.L.str.20,@object               # @.str.20
.L.str.20:
	.asciz	"hipCheckError() with sync failed at %s:%i : %s\n"
	.size	.L.str.20, 48

	.type	.L.str.21,@object               # @.str.21
.L.str.21:
	.asciz	"vector::reserve"
	.size	.L.str.21, 16

	.type	.L.str.22,@object               # @.str.22
.L.str.22:
	.asciz	"vector::_M_realloc_insert"
	.size	.L.str.22, 26

	.type	.L__unnamed_4,@object           # @5
.L__unnamed_4:
	.asciz	"_Z13matmul_deviceILi8192ELi8192ELi8192EEvN7kittens2glI14__hip_fp8_e4m3Li1ELi1EXT_EXT1_EJEEENS1_IS2_Li1ELi1EXT0_EXT1_EJEEENS1_I14__hip_bfloat16Li1ELi1EXT_EXT0_EJEEE"
	.size	.L__unnamed_4, 164

	.type	.L__unnamed_7,@object           # @6
	.section	.hip_fatbin,"a",@progbits
	.p2align	12, 0x0
.L__unnamed_7:
	.asciz	"__CLANG_OFFLOAD_BUNDLE__\002\000\000\000\000\000\000\000\000\020\000\000\000\000\000\000\000\000\000\000\000\000\000\000\036\000\000\000\000\000\000\000host-x86_64-unknown-linux-gnu-\000\020\000\000\000\000\000\000\2406\000\000\000\000\000\000\037\000\000\000\000\000\000\000hipv4-amdgcn-amd-amdhsa--gfx950\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\177ELF\002\001\001@\004\000\000\000\000\000\000\000\003\000\340\000\001\000\000\000\000\000\000\000\000\000\000\000@\000\000\000\000\000\000\000`2\000\000\000\000\000\000O\005\000\000@\0008\000\t\000@\000\021\000\017\000\006\000\000\000\004\000\000\000@\000\000\000\000\000\000\000@\000\000\000\000\000\000\000@\000\000\000\000\000\000\000\370\001\000\000\000\000\000\000\370\001\000\000\000\000\000\000\b\000\000\000\000\000\000\000\001\000\000\000\004\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000@\b\000\000\000\000\000\000@\b\000\000\000\000\000\000\000\020\000\000\000\000\000\000\001\000\000\000\005\000\000\000\000\t\000\000\000\000\000\000\000\031\000\000\000\000\000\000\000\031\000\000\000\000\000\000\210\035\000\000\000\000\000\000\210\035\000\000\000\000\000\000\000\020\000\000\000\000\000\000\001\000\000\000\006\000\000\000\210&\000\000\000\000\000\000\210F\000\000\000\000\000\000\210F\000\000\000\000\000\000p\000\000\000\000\000\000\000x\t\000\000\000\000\000\000\000\020\000\000\000\000\000\000\001\000\000\000\006\000\000\000\370&\000\000\000\000\000\000\370V\000\000\000\000\000\000\370V\000\000\000\000\000\000\000\000\000\000\000\000\000\000\001\000\000\000\000\000\000\000\000\020\000\000\000\000\000\000\002\000\000\000\006\000\000\000\210&\000\000\000\000\000\000\210F\000\000\000\000\000\000\210F\000\000\000\000\000\000p\000\000\000\000\000\000\000p\000\000\000\000\000\000\000\b\000\000\000\000\000\000\000R\345td\004\000\000\000\210&\000\000\000\000\000\000\210F\000\000\000\000\000\000\210F\000\000\000\000\000\000p\000\000\000\000\000\000\000x\t\000\000\000\000\000\000\001\000\000\000\000\000\000\000Q\345td\006\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\004\000\000\000\004\000\000\0008\002\000\000\000\000\000\0008\002\000\000\000\000\000\0008\002\000\000\000\000\000\000t\003\000\000\000\000\000\000t\003\000\000\000\000\000\000\004\000\000\000\000\000\000\000\007\000\000\000_\003\000\000 \000\000\000AMDGPU\000\000\203\256amdhsa.kernels\221\336\000\022\253.agpr_count\000\245.args\223\203\247.offset\000\245.size\020\253.value_kind\250by_value\203\247.offset\020\245.size\020\253.value_kind\250by_value\203\247.offset \245.size\020\253.value_kind\250by_value\271.group_segment_fixed_size\316\000\002\000\000\266.kernarg_segment_align\b\265.kernarg_segment_size0\251.language\250OpenCL C\261.language_version\222\002\000\270.max_flat_workgroup_size\315\002\000\245.name\331\243_Z13matmul_deviceILi8192ELi8192ELi8192EEvN7kittens2glI14__hip_fp8_e4m3Li1ELi1EXT_EXT1_EJEEENS1_IS2_Li1ELi1EXT0_EXT1_EJEEENS1_I14__hip_bfloat16Li1ELi1EXT_EXT0_EJEEE\273.private_segment_fixed_size\000\253.sgpr_count\037\261.sgpr_spill_count\000\247.symbol\331\246_Z13matmul_deviceILi8192ELi8192ELi8192EEvN7kittens2glI14__hip_fp8_e4m3Li1ELi1EXT_EXT1_EJEEENS1_IS2_Li1ELi1EXT0_EXT1_EJEEENS1_I14__hip_bfloat16Li1ELi1EXT_EXT0_EJEEE.kd\270.uniform_work_group_size\001\263.uses_dynamic_stack\302\253.vgpr_count\314\376\261.vgpr_spill_count\000\257.wavefront_size@\255amdhsa.target\271amdgcn-amd-amdhsa--gfx950\256amdhsa.version\222\001\002\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\001\000\000\000\022\003\007\000\000\035\000\000\000\000\000\000\210\031\000\000\000\000\000\000\245\000\000\000\021\003\006\000\000\b\000\000\000\000\000\000@\000\000\000\000\000\000\000L\001\000\000\021\000\n\000\370V\000\000\000\000\000\000\001\000\000\000\000\000\000\000\001\000\000\000\001\000\000\000\001\000\000\000\032\000\000\000\000\005@@\000@\b\000\001\000\000\000\226\205d{\262w%\272K\355\301 \004\000\000\000\004\000\000\000\003\000\000\000\001\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\002\000\000\000\000_Z13matmul_deviceILi8192ELi8192ELi8192EEvN7kittens2glI14__hip_fp8_e4m3Li1ELi1EXT_EXT1_EJEEENS1_IS2_Li1ELi1EXT0_EXT1_EJEEENS1_I14__hip_bfloat16Li1ELi1EXT_EXT0_EJEEE\000_Z13matmul_deviceILi8192ELi8192ELi8192EEvN7kittens2glI14__hip_fp8_e4m3Li1ELi1EXT_EXT1_EJEEENS1_IS2_Li1ELi1EXT0_EXT1_EJEEENS1_I14__hip_bfloat16Li1ELi1EXT_EXT0_EJEEE.kd\000__hip_cuid_e843e8b1c0f57c70\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\002\000\000\000\000\0000\000\000\000\000\000\000\000\000\025\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000?\000\000\000\037\003\257\000\204\000\000\000\b\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\200\001\006\300\000\000\000\000\000\002\006\300\020\000\000\000\000\001\006\300 \000\000\000\002\237\000\220\000\233\000\217\002\000\000\201\000\205\n\220\000\237\000\211\002\000\213\201\013\225\022\216\204\000\002$\022\237\023\220\377\002\210'\000\034\000\000\177\300\214\277\b\022\020\200\377\210\005)\000\000\001\000\212\000\004$\001\001\002*\377\000\214\276p\340\007\000\t\023\024\202\302\0014\322\001\031\b\034\202\005\030~\377\210\007)\000 \001\000\377\000\203\276\000\000\021\000\377\000\202\276\000\000\020\000\020\000\200\276\024\000\201\276\f\000\374\276\203\005\030~\000\020]\340\302\000\000\200\f\000\374\276\n\225\f\216\f\237\021\220\006\f\025\200\377\204\207)\000\000\b\000\007\021\026\202\304\005\032~\377\210E(\000 \000\000\000\020]\340\303\000\000\200\025\000\200\276\026\000\201\276\r\000\374\276\"\005\032~\000\020]\340\302\000\000\200\r\000\374\276\377\210E)\000@\001\000\000\020]\340\303\000\000\200\022\377\000\207\000\000\020\000\000\237\001\220\b\000\000\200\242\005\032~\377\210G)\000`\001\000\t\001\001\202\r\000\374\276\243\005\032~\000\020]\340\302\000\000\200\r\000\374\276\377\210\213)\000@\000\000\000\020]\340\303\000\000\200\f\377\000\207\000\000\020\000\000\237\001\220\006\000\000\200\305\005\032~\377\210\003(\000`\000\000\007\001\001\202\r\000\374\276\001\005\032~\000\020]\340\302\000\000\200\r\000\374\276\377\210\003(\000\200\000\000\000\020]\340\303\000\000\200\025\377\000\200\200\000\000\000\001\005\032~\377\210\003(\000\240\000\000\026\200\001\202\r\000\374\276\001\005\032~\000\020]\340\302\000\000\200\r\000\374\276\377\210\003(\000\200\001\000\000\020]\340\303\000\000\200\020\377\000\200\200\000\000\000\001\005\032~\377\210\003(\000\240\001\000\024\200\001\202\r\000\374\276\001\005\032~\000\020]\340\302\000\000\200\r\000\374\276\377\210\003(\000\300\001\000\000\020]\340\303\000\000\200\022\377\000\207\200\000\020\000\000\237\001\220\b\000\000\200\001\005\032~\377\210\003(\000\340\001\000\t\001\001\202\r\000\374\276\001\005\032~\000\020]\340\302\000\000\200\r\000\374\276\210\000\224!\000\020]\340\303\000\000\200\001\000\310\321\000\r\t\002\200\000\215\276|\017\214\277\000\000\212\277\214\002\220%\207\000\004$\260\000\006&\200\007\000\260\377\220\223)\000\000\001\000\313\000\001\322\002\001\f\004\311\227\005(\204\004\006 p\000\016\260\n\0054\322\003\005:\210\300\004\004(\016\0054\322\003\005:\210\000\000\376\331\n\000\000\002\000\000\376\331\016\000\000\006\000\b\376\331\n\000\000\n\000\b\376\331\016\000\000\016z\017\214\277\177\300\214\277\000\000\212\277\201\000\217\276\201\224\225}j \200\276\001\000\210\277\000\000\212\277~\000\376\207\215\224\231%\314\227G(\204F$ \307\0054\322\022G:\210\300F$h\204$& \000\000\376\331\307\000\000$\306\0054\322\023%:\210\000\000\376\331\306\000\000(\000\b\376\331\307\000\000,\000\b\376\331\306\000\0000\000\020\376\331\307\000\0004\f\377\000\207\200\000\020\000\000\020\376\331\306\000\0008\000\237\001\220\377\210\241)\000\300\000\000\000\030\376\331\307\000\000\204\006\000\000\200\320\005.~\377\210\243)\000\340\000\000\000\030\376\331\306\000\000\210\007\001\001\202\027\000\374\276\321\005.~\000\020]\340\302\000\000\200\027\000\374\276\000\000\200\277\000\020]\340\303\000\000\200\177\300\214\277z\017\214\277\000\000\212\277\001\000\217\277^\000\255\323$\005\002\002Z\000\255\323$\025\002\002V\000\255\323,\005\002\002R\000\255\323,\025\002\002N\000\255\3234\005\002\002J\000\255\3234\025\002\002F\000\255\323\204\005\002\002B\000\255\323\204\025\002\002\000\000\217\277\000\000\212\277\311\227\233i\377\232%h\000@\000\000\204$& \317\0054\322\023%:\210\377\232%h@@\000\000\204$& \316\0054\322\023%:\210\000\000\376\331\317\000\000\022\000\000\376\331\316\000\000\026\000\b\376\331\317\000\000\032\025\377\000\200\000\001\000\000\304\005*~\000\b\376\331\316\000\000\036\026\200\001\202\025\000\374\276\"\005*~\000\020]\340\302\000\000\200\025\000\374\276\000\000\200\277\000\020]\340\303\000\000\200\177\300\214\277z\017\214\277\000\000\212\277\001\000\217\277~\000\255\323$%\002\002z\000\255\323$5\002\002v\000\255\323,%\002\002r\000\255\323,5\002\002n\000\255\3234%\002\002j\000\255\32345\002\002f\000\255\323\204%\002\002b\000\255\323\2045\002\002\000\000\217\277\000\000\212\277\377FD(\000@\000\000\204\226I \323\0054\322$E:\210\377FDh@@\000\000\204DF \322\0054\322#E:\210\000\000\376\331\323\000\000\"\000\000\376\331\322\000\000&\000\b\376\331\323\000\000*\000\b\376\331\322\000\000.\000\020\376\331\323\000\0002\000\020\376\331\322\000\0006\000\030\376\331\323\000\000:\020\377\000\200\000\001\000\000\202\005 ~\000\030\376\331\322\000\000>\024\200\001\202\020\000\374\276\203\005 ~\000\020]\340\302\000\000\200\020\000\374\276\000\000\200\277\000\020]\340\303\000\000\200\177\300\214\277z\017\214\277\000\000\212\277\001\000\217\277\236\000\255\323\"\005\002\002\232\000\255\323\"\025\002\002\226\000\255\323*\005\002\002\222\000\255\323*\025\002\002\216\000\255\3232\005\002\002\212\000\255\3232\025\002\002\206\000\255\323:\005\002\002\202\000\255\323:\025\002\002\000\000\217\277\000\000\212\277\377\232\005h\000\200\000\000\204\004\006 \325\0054\322\003\005:\210\377\232\005h@\200\000\000\204\004\006 \324\0054\322\003\005:\210\000\000\376\331\325\000\000\002\022\377\000\207\000\001\020\000\000\000\376\331\324\000\000\006\000\237\001\220\000\b\376\331\325\000\000\n\b\000\000\200\242\005 ~\000\b\376\331\324\000\000\016\t\001\001\202\020\000\374\276\243\005 ~\000\020]\340\302\000\000\200\020\000\374\276\006\f\020\200\000\020]\340\303\000\000\200\007\021\021\202\b\022\022\200\200\002D\177\t\023\023\202\200\001\210\276\242\003F\177\242\003H\177\242\003J\177\242\003L\177\242\003N\177\242\003P\177\242\003R\177\242\003T\177\242\003V\177\242\003X\177\242\003Z\177\242\003\\\177\242\003^\177\242\003`\177\242\003b\177\242\003d\177\242\003f\177\242\003h\177\242\003j\177\242\003l\177\242\003n\177\242\003p\177\242\003r\177\242\003t\177\242\003v\177\242\003x\177\242\003z\177\242\003|\177\242\003~\177\242\003\200\177\242\003\202\177z\017\214\277\000\000\212\277\001\000\217\277\276\000\255\323\"%\372\006\272\000\255\323\"5\352\006\266\000\255\323*%\332\006\262\000\255\323*5\312\006\256\000\255\3232%\272\006\252\000\255\32325\252\006\246\000\255\323:%\232\006\242\000\255\323:5\212\006\000\000\217\277\000\000\212\277\017\217\025\216\025\230%(\022\227\255i\204\254% \300\254'h\022\0054\322\022\255;\210\204&( \000\000\376\331\022\000\000\"\023\0054\322\024':\210\000\000\376\331\023\000\000&\000\b\376\331\022\000\000*\020\b\026\200\000\b\376\331\023\000\000.\021\t\027\202\000\020\376\331\022\000\0002\026\377\000\200\000\001\020\000\000\020\376\331\023\000\0006\027\200\001\202\r\217\024\216\000\030\376\331\022\000\000:\024\212%(\000\030\376\331\023\000\000>\000\000\200\277\022\0050~\377$$h\000 \000\000\030\000\374\276\022\0050~\000\020]\340\302\000\000\200\030\000\374\276\000\000\200\277\000\020]\340\303\000\000\200\177\300\214\277z\017\214\277\000\000\212\277\001\000\217\277^\000\255\323\"\005z\005Z\000\255\323\"\025j\005V\000\255\323*\005Z\005R\000\255\323*\025J\005N\000\255\3232\005:\005J\000\255\3232\025*\005F\000\255\323:\005\032\005B\000\255\323:\025\n\005\000\000\217\277\000\000\212\277\025\377\030\201\000\000\001\000\030\220%(\022\227%h\377$&h\000@\000\000\204&( \377$$h@@\000\000\032\0054\322\024':\210\204$& \036\0054\322\023%:\210\000\000\376\331\032\000\000\022\000\000\376\331\036\000\000\026\025\210\257)\000\b\376\331\032\000\000\032\026\377\000\200\200\001\000\000\327\005*~\377\256\257)\000 \000\000\000\b\376\331\036\000\000\036\027\200\001\202\025\000\374\276\327\005*~\000\020]\340\302\000\000\200\025\000\374\276\000\000\200\277\000\020]\340\303\000\000\200\177\300\214\277z\017\214\277\000\000\212\277\001\000\217\277~\000\255\323\"%\372\005z\000\255\323\"5\352\005v\000\255\323*%\332\005r\000\255\323*5\312\005n\000\255\3232%\272\005j\000\255\32325\252\005f\000\255\323:%\232\005b\000\255\323:5\212\005\000\000\217\277\000\000\212\277\377\254Eh\000@\000\000\204DF :\0054\322#E:\210\377\254Eh@@\000\000\204DF >\0054\322#E:\210\000\000\376\331:\000\000\"\000\000\376\331>\000\000&\000\b\376\331:\000\000*\000\b\376\331>\000\000.\000\020\376\331:\000\0002\022\b\025\200\000\020\376\331>\000\0006\023\t\026\202\030\210\255)\000\030\376\331:\000\000:\025\377\000\200\200\001\000\000\326\005.~\377\254\257)\000 \000\000\000\030\376\331>\000\000>\026\200\001\202\027\000\374\276\327\005.~\000\020]\340\302\000\000\200\027\000\374\276\000\000\200\277\000\020]\340\303\000\000\200\177\300\214\277z\017\214\277\000\000\212\277\001\000\217\277\236\000\255\323\"\005z\006\232\000\255\323\"\025j\006\226\000\255\323*\005Z\006\222\000\255\323*\025J\006\216\000\255\3232\005:\006\212\000\255\3232\025*\006\206\000\255\323:\005\032\006\202\000\255\323:\025\n\006\000\000\217\277\000\000\212\277\002\000\377\321\311),\007\204\004\006 \n\0054\322\003\005:\210\300\004\004h\204\004\006 \016\0054\322\003\005:\210\000\000\376\331\n\000\000\002\000\000\376\331\016\000\000\006\377\254\257)\000@\000\000\000\b\376\331\n\000\000\n\025\377\000\200\200\001\020\000\327\005(~\377\254\255)\000`\000\000\000\b\376\331\016\000\000\016\026\200\001\202\024\000\374\276\326\005(~\000\020]\340\302\000\000\200\024\000\374\276\017\201\017\210\000\020]\340\303\000\000\200\r\201\r\210\b\377\b\200\200\000\000\000\t\200\t\202\200\036\b\261\005\377\204\277z\017\214\277\000\000\212\277\001\000\217\277\276\000\255\323\"%\372\006&\000\255\323\"5\352\006\266\000\255\323*%\332\006.\000\255\323*5\312\006\256\000\255\3232%\272\0066\000\255\32325\252\006\022\000\255\323:%\232\006\032\000\255\323:5\212\006\000\000\217\277\000\000\212\277\000\000\376\331\307\000\000:\000\000\376\331\306\000\000>\000\b\376\331\307\000\000\326\000\b\376\331\306\000\000\332\000\020\376\331\307\000\000\336\f\377\000\207\200\037\020\000\000\020\376\331\306\000\000\342\000\237\001\220\000\030\376\331\307\000\000\346\006\000\000\200\320\005\f~\000\030\376\331\306\000\000\352\007\001\001\202\377\000\203\276\000\000\021\000\377\000\202\276\000\000\020\000\006\000\374\276\321\005\f~\000\020]\340\302\000\000\200\006\000\374\276\000\000\200\277\000\020]\340\303\000\000\200\177\300\214\277z\017\214\277\000\000\212\277\001\000\217\277\242\000\255\323:\005z\005V\000\255\323\326\005Z\005\246\000\255\323:\025j\005\252\000\255\323\326\025J\005\262\000\255\323\336\005:\005\272\000\255\323\336\025*\005\302\000\255\323\346\005\032\005\306\000\255\323\346\025\n\005\000\000\217\277\000\000\212\277\000\000\376\331\317\000\000\356\000\000\376\331\316\000\000\362\000\b\376\331\317\000\000\366\000\b\376\331\316\000\000\372\177\300\214\277x\017\214\277\000\000\212\277\001\000\217\277\036\000\255\323:\335\373\005:\000\255\323:\355\353\005J\000\255\323\326\335\333\005R\000\255\323\326\355\313\005Z\000\255\323\336\335\273\005j\000\255\323\336\355\253\005f\000\255\323\346\335\233\005n\000\255\323\346\355\213\005\000\000\217\277\000\000\212\277\000\000\376\331\323\000\000r\000\000\376\331\322\000\000v\000\b\376\331\323\000\000\326\000\b\376\331\322\000\000\332\000\020\376\331\323\000\000\336\000\020\376\331\322\000\000\342\000\030\376\331\323\000\000\346\000\030\376\331\322\000\000\352\177\300\214\277v\017\214\277\000\000\212\277\001\000\217\277\026\000\255\323r\005z\006\"\000\255\323r\025j\006*\000\255\323\326\005Z\0062\000\255\323\326\025J\006>\000\255\323\336\005:\006B\000\255\323\336\025*\006N\000\255\323\346\005\032\006^\000\255\323\346\025\n\006\000\000\217\277\000\000\212\277\000\000\376\331\325\000\000\002\000\000\376\331\324\000\000\006\000\b\376\331\325\000\000\n\000\b\376\331\324\000\000\016p\017\214\277\000\000\212\277\001\000\217\277z\000\255\323r\335\373\006~\000\255\323r\355\233\004\202\000\255\323\326\335\333\006\206\000\255\323\326\355\273\004\212\000\255\323\336\335\273\006\216\000\255\323\336\355\333\004\222\000\255\323\346\335K\004\226\000\255\323\346\355k\004\000\000\217\277\000\000\212\277\313\231mi\002\000\200\277\377l%(\000\200\000\000\204l' p\000\000\260\022\0054\322\023%\002\210\377l'h@\200\000\000\204&( \000\000\376\331\022\000\000\316\023\0054\322\024'\002\210\000\000\376\331\023\000\000\322\000\b\376\331\022\000\000\326\000\b\376\331\023\000\000\332\000\020\376\331\022\000\000\336\000\020\376\331\023\000\000\342\000\030\376\331\022\000\000\346\000\030\376\331\023\000\000\352\177\300\214\277\000\000\212\277\001\000\217\277\242\000\255\323\316\005\212\006\236\000\255\323\316\025\232\006\232\000\255\323\326\005Z\005v\000\255\323\326\025\252\006r\000\255\323\336\005\312\006F\000\255\323\336\025\352\0066\000\255\323\346\005\n\007\032\000\255\323\346\025\032\007\000\000\217\277\000\000\212\277\377\232%h\000\300\000\000\204$& \022\0054\322\023%\002\210\377\232'h@\300\000\000\204&( \000\000\376\331\022\000\000\246\023\0054\322\024'\002\210\000\000\376\331\023\000\000\252\000\b\376\331\022\000\000\256\000\b\376\331\023\000\000\262\177\300\214\277\000\000\212\277\001\000\217\277V\000\255\323\316M{\004b\000\255\323\316]\353\004J\000\255\323\326M+\005:\000\255\323\326]K\005.\000\255\323\336Mk\005&\000\255\323\336]\253\005\036\000\255\323\346M\233\005\022\000\255\323\346]\273\005\000\000\217\277\000\000\212\277\000\000\200\277\377l\245h\000\300\000\000\204\244\246 R\0054\322S\245\002\210\377l\247h@\300\000\000\204\246\250 \000\000\376\331R\000\000\266S\0054\322T\247\002\210\000\000\376\331S\000\000\272\000\b\376\331R\000\000\276\000\b\376\331S\000\000\302\000\020\376\331R\000\000\314\000\020\376\331S\000\000\320\000\030\376\331R\000\000\324\000\030\376\331S\000\000\330\177\300\214\277\000\000\212\277\001\000\217\277n\000\255\323\266\005Z\004f\000\255\323\266\025\212\004j\000\255\323\276\005\252\004R\000\255\323\276\025\312\004Z\000\255\323\314\005\372\004B\000\255\323\314\025\n\005N\000\255\323\324\005:\005>\000\255\323\324\025z\0052\000\255\323\266M\353\005*\000\255\323\266]\373\005\"\000\255\323\276M\013\006\026\000\255\323\276]\033\006\016\000\255\323\314M+\006\n\000\255\323\314];\006\006\000\255\323\324MK\006\002\000\255\323\324][\006\000\000\217\277\000\001\000\260\000\000\230}j \200\276\001\000\210\277\000\000\212\277~\000\376\207\223\224\275$\205\002\002$^\000\000\322\n*y\005\001\000\000\322\013\020\005\004^\003\030i\213\000\002$\377\000\200\276\017\200\001\000\000\0014\322\001\001\000\034\237\030\033#\201\000\000$\200\002\002~\216\000\b\322\214\003\021\000\377\000\274(\000@\000\000\001\003\276~`\000\b\322\216\001y\005\000\200l\334`\243\177\000\377\000\300(\000\200\000\000\001\003\302~z\000\b\322\216\001\201\005\000\200l\334z\244\177\000\377\000\364(\000\300\000\000\001\003\366~\220\000\b\322\216\001\001\004|\000\b\322\216\001\351\005\000\200l\334\220\242\177\000\000\200l\334|\245\177\000 \200l\334\220\236\177\000\377\000\370( @\000\000\001\003\372~~\000\b\322\216\001\361\005\000\200l\334~\237\177\000\377\000\374( \200\000\000\001\003\376~\200\000\b\322\216\001\371\005\000\200l\334\200\240\177\000\377\000\000) \300\000\000\001\003\002\177\202\000\b\322\216\001\001\006\000\200l\334\202\241\177\000\377\000\004)\000\000\004\000\001\003\006\177\204\000\b\322\216\001\t\006\000\200l\334\204\232\177\000\377\000\b)\000@\004\000\001\003\n\177\206\000\b\322\216\001\021\006\000\200l\334\206\233\177\000\377\000\f)\000\200\004\000\001\003\016\177\210\000\b\322\216\001\031\006\000\200l\334\210\234\177\000\377\000\020)\000\300\004\000\001\003\022\177\212\000\b\322\216\001!\006\000\200l\334\212\235\177\000\377\000\024) \000\004\000\001\003\026\177\222\000\b\322\216\001)\006\000\200l\334\222v\177\000\377\000$) @\004\000\001\003&\177\224\000\b\322\216\001I\006\000\200l\334\224w\177\000\377\000\354( \200\004\000\001\003\356~\224\000\b\322\216\001\331\005\000\200l\334\224x\177\000\377\000() \300\004\000\001\003*\177\226\000\b\322\216\001Q\006\000\200l\334\226y\177\000\377\000\360(\000\000\b\000\001\003\362~\226\000\b\322\216\001\341\005\000\200l\334\226r\177\000\377\000,)\000@\b\000\001\003.\177\230\000\b\322\216\001Y\006\000\200l\334\230s\177\000\377\000\344(\000\200\b\000\001\003\346~\230\000\b\322\216\001\311\005\000\200l\334\230t\177\000\377\0000)\000\300\b\000\001\0032\177\232\000\b\322\216\001a\006\000\200l\334\232u\177\000\377\000\350( \000\b\000\001\003\352~\232\000\b\322\216\001\321\005\000\200l\334\232F\177\000\377\0004) @\b\000\001\0036\177\234\000\b\322\216\001i\006\000\200l\334\234G\177\000\377\000\214( \200\b\000\001\003\216~\234\000\b\322\216\001\031\005\000\200l\334\234H\177\000\377\0008) \300\b\000\001\003:\177\236\000\b\322\216\001q\006\000\200l\334\236I\177\000\377\000\220(\000\000\f\000\001\003\222~\236\000\b\322\216\001!\005\000\200l\334\2366\177\000\377\000<)\000@\f\000\001\003>\177\240\000\b\322\216\001y\006\000\200l\334\2407\177\000\377\000l(\000\200\f\000\001\003n~\240\000\b\322\216\001\331\004\000\200l\334\2408\177\000\377\000@)\000\300\f\000\001\003B\177\242\000\b\322\216\001\201\006\000\200l\334\2429\177\000\377\000p( \000\f\000\001\003r~\242\000\b\322\216\001\341\004\000\200l\334\242\032\177\000\377\000D) @\f\000\001\003F\177\244\000\b\322\216\001\211\006\000\200l\334\244\033\177\000\377\0004( \200\f\000\001\0036~\244\000\b\322\216\001i\004\000\200l\334\244\034\177\000\377\000H) \300\f\000\001\003J\177\246\000\b\322\216\001\221\006\377\001\200\276\000\001\000\000\000\200l\334\246\035\177\000\034\000\b\322\216\001\001\000\216\000\b\322\034\001y\005\000\201l\334\220V\177\000\000\200l\334\216W\177\000V\000\b\322\034\001\201\005\000\200l\334VX\177\000V\000\b\322\034\001\351\005\000\200l\334VY\177\000V\000\b\322\034\001\001\004 \200l\334Vb\177\000V\000\b\322\034\001\361\005\000\200l\334Vc\177\000V\000\b\322\034\001\371\005\000\200l\334Vd\177\000V\000\b\322\034\001\001\006\000\200l\334Ve\177\000V\000\b\322\034\001\t\006\000\200l\334VJ\177\000V\000\b\322\034\001\021\006\000\200l\334VK\177\000J\000\b\322\034\001\031\006\000\200l\334JL\177\000J\000\b\322\034\001!\006\000\200l\334JM\177\000J\000\b\322\034\001)\006\000\200l\334J:\177\000J\000\b\322\034\001I\006\000\200l\334J;\177\000:\000\b\322\034\001\331\005\000\200l\334:<\177\000:\000\b\322\034\001Q\006\000\200l\334:=\177\000:\000\b\322\034\001\341\005\000\200l\334:.\177\000:\000\b\322\034\001Y\006\000\200l\334:/\177\000.\000\b\322\034\001\311\005\000\200l\334.0\177\000.\000\b\322\034\001a\006\000\200l\334.1\177\000.\000\b\322\034\001\321\005\000\200l\334.&\177\000.\000\b\322\034\001i\006\000\200l\334.'\177\000&\000\b\322\034\001\031\005\000\200l\334&(\177\000&\000\b\322\034\001q\006\000\200l\334&)\177\000&\000\b\322\034\001!\005\000\200l\334&\036\177\000&\000\b\322\034\001y\006\000\200l\334&\037\177\000\036\000\b\322\034\001\331\004\000\200l\334\036 \177\000\036\000\b\322\034\001\201\006\000\200l\334\036!\177\000\036\000\b\322\034\001\341\004\000\200l\334\036\022\177\000\036\000\b\322\034\001\211\006\000\200l\334\036\023\177\000\022\000\b\322\034\001i\004\000\200l\334\022\024\177\000\022\000\b\322\034\001\221\006\000\200l\334\022\025\177\000\377\030%h\000\000\020\000\237$&\"\022\000\b\322\022\003\021\000\034\000\b\322\022\001y\005\000\200l\334\034o\177\000\034\000\b\322\022\001\201\005\024\000\b\322\022\001\001\004\000\200l\334\034p\177\000\034\000\b\322\022\001\351\005\000\200l\334\024n\177\000\000\200l\334\034q\177\000 \200l\334\024f\177\000\034\000\b\322\022\001\361\005\000\200l\334\034g\177\000\034\000\b\322\022\001\371\005\000\200l\334\034h\177\000\034\000\b\322\022\001\001\006\000\200l\334\034i\177\000\034\000\b\322\022\001\t\006\000\200l\334\034j\177\000\034\000\b\322\022\001\021\006\000\200l\334\034k\177\000\034\000\b\322\022\001\031\006\000\200l\334\034l\177\000\034\000\b\322\022\001!\006\000\200l\334\034m\177\000\034\000\b\322\022\001)\006\000\200l\334\034R\177\000\034\000\b\322\022\001I\006\000\200l\334\034S\177\000\034\000\b\322\022\001\331\005\000\200l\334\034T\177\000\034\000\b\322\022\001Q\006\000\200l\334\034U\177\000\034\000\b\322\022\001\341\005\000\200l\334\034Z\177\000\034\000\b\322\022\001Y\006\000\200l\334\034[\177\000\034\000\b\322\022\001\311\005\000\200l\334\034\\\177\000\034\000\b\322\022\001a\006\000\200l\334\034]\177\000\034\000\b\322\022\001\321\005\000\200l\334\034B\177\000\034\000\b\322\022\001i\006\000\200l\334\034C\177\000\034\000\b\322\022\001\031\005\000\200l\334\034D\177\000\034\000\b\322\022\001q\006\000\200l\334\034E\177\000\034\000\b\322\022\001!\005\000\200l\334\034N\177\000\034\000\b\322\022\001y\006\000\200l\334\034O\177\000\034\000\b\322\022\001\331\004\000\200l\334\034P\177\000\034\000\b\322\022\001\201\006\000\200l\334\034Q\177\000\034\000\b\322\022\001\341\004\000\200l\334\034>\177\000\034\000\b\322\022\001\211\006\000\200l\334\034?\177\000\034\000\b\322\022\001i\004\000\200l\334\034@\177\000\034\000\b\322\022\001\221\006\022\000\b\322\022\001\001\000\000\000\b\322\022\001\001\004 \200l\334\000*\177\000\000\000\b\322\022\001\361\005\000\200l\334\000+\177\000\000\000\b\322\022\001\371\005\000\200l\334\000,\177\000\000\000\b\322\022\001\001\006\000\200l\334\000-\177\000\000\000\b\322\022\001\t\006\000\200l\334\000\"\177\000\000\000\b\322\022\001\021\006\000\200l\334\000#\177\000\000\000\b\322\022\001\031\006\000\200l\334\000$\177\000\000\000\b\322\022\001!\006\000\200l\334\000%\177\000\000\000\b\322\022\001)\006\000\200l\334\000\026\177\000\000\000\b\322\022\001I\006\000\200l\334\000\027\177\000\000\000\b\322\022\001\331\005\000\200l\334\000\030\177\000\000\000\b\322\022\001Q\006\000\200l\334\000\031\177\000\000\000\b\322\022\001\341\005\000\200l\334\000\016\177\000\000\000\b\322\022\001Y\006\000\200l\334\000\017\177\000\000\000\b\322\022\001\311\005\000\200l\334\000\020\177\000\000\000\b\322\022\001a\006\000\200l\334\000\021\177\000\000\000\b\322\022\001\321\005\000\200l\334\000\n\177\000\000\000\b\322\022\001i\006\000\200l\334\000\013\177\000\000\000\b\322\022\001\031\005\000\200l\334\000\f\177\000\000\000\b\322\022\001q\006\000\200l\334\000\r\177\000\000\000\b\322\022\001!\005\000\200l\334\000\006\177\000\000\000\b\322\022\001y\006\000\200l\334\000\007\177\000\000\000\b\322\022\001\331\004\000\200l\334\000\b\177\000\000\000\b\322\022\001\201\006\000\200l\334\000\t\177\000\000\000\b\322\022\001\341\004\000\201l\334\0242\177\000\024\000\b\322\022\001y\005\000\200l\334\000\002\177\000\000\000\b\322\022\001\211\006\000\200l\334\0243\177\000\024\000\b\322\022\001\201\005\000\200l\334\000\003\177\000\000\000\b\322\022\001i\004\000\200l\334\0244\177\000\024\000\b\322\022\001\351\005\000\200l\334\000\004\177\000\000\000\b\322\022\001\221\006\000\200l\334\034A\177\000\000\200l\334\0245\177\000\000\200l\334\000\005\177\000\000\000\201\277\006\000\000\000\000\000\000\000\260\005\000\000\000\000\000\000\013\000\000\000\000\000\000\000\030\000\000\000\000\000\000\000\005\000\000\000\000\000\000\000`\006\000\000\000\000\000\000\n\000\000\000\000\000\000\000h\001\000\000\000\000\000\000\365\376\377o\000\000\000\000\020\006\000\000\000\000\000\000\004\000\000\000\000\000\000\0008\006\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000Linker: AMD LLD 20.0.0 (/longer_pathname_so_that_rpms_can_support_packaging_the_debug_info_for_all_os_profiles/src/llvm-project/llvm 82aed4e69d70bef3c89c38a2ee85c8c41294dfc9)\000AMD clang version 20.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-7.0.0 25304 82aed4e69d70bef3c89c38a2ee85c8c41294dfc9)\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\001\000\000\000\000\000\361\377\376\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\256\000\000\000\000\000\361\377\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000[\001\000\000\000\000\361\377\031\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\r\002\000\000\000\000\361\377\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\302\002\000\000\000\000\361\377\001\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000o\003\000\000\000\000\361\377\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000%\004\000\000\000\000\361\377\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\335\004\000\000\000\000\361\377\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\217\005\000\000\000\000\361\377\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000E\006\000\000\000\000\361\377\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000Y\006\000\000\000\000\361\377\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000m\006\000\000\000\000\361\377\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\350\007\000\000\000\002\b\000\210F\000\000\000\000\000\000\000\000\000\000\000\000\000\000\201\006\000\000\022\003\007\000\000\035\000\000\000\000\000\000\210\031\000\000\000\000\000\000%\007\000\000\021\003\006\000\000\b\000\000\000\000\000\000@\000\000\000\000\000\000\000\314\007\000\000\021\000\n\000\370V\000\000\000\000\000\000\001\000\000\000\000\000\000\000\000.note\000.dynsym\000.gnu.hash\000.hash\000.dynstr\000.rodata\000.text\000.dynamic\000.relro_padding\000.bss\000.AMDGPU.csdata\000.AMDGPU.gpr_maximums\000.comment\000.symtab\000.shstrtab\000.strtab\000\000_Z13matmul_deviceILi8192ELi8192ELi8192EEvN7kittens2glI14__hip_fp8_e4m3Li1ELi1EXT_EXT1_EJEEENS1_IS2_Li1ELi1EXT0_EXT1_EJEEENS1_I14__hip_bfloat16Li1ELi1EXT_EXT0_EJEEE.num_vgpr\000_Z13matmul_deviceILi8192ELi8192ELi8192EEvN7kittens2glI14__hip_fp8_e4m3Li1ELi1EXT_EXT1_EJEEENS1_IS2_Li1ELi1EXT0_EXT1_EJEEENS1_I14__hip_bfloat16Li1ELi1EXT_EXT0_EJEEE.num_agpr\000_Z13matmul_deviceILi8192ELi8192ELi8192EEvN7kittens2glI14__hip_fp8_e4m3Li1ELi1EXT_EXT1_EJEEENS1_IS2_Li1ELi1EXT0_EXT1_EJEEENS1_I14__hip_bfloat16Li1ELi1EXT_EXT0_EJEEE.numbered_sgpr\000_Z13matmul_deviceILi8192ELi8192ELi8192EEvN7kittens2glI14__hip_fp8_e4m3Li1ELi1EXT_EXT1_EJEEENS1_IS2_Li1ELi1EXT0_EXT1_EJEEENS1_I14__hip_bfloat16Li1ELi1EXT_EXT0_EJEEE.private_seg_size\000_Z13matmul_deviceILi8192ELi8192ELi8192EEvN7kittens2glI14__hip_fp8_e4m3Li1ELi1EXT_EXT1_EJEEENS1_IS2_Li1ELi1EXT0_EXT1_EJEEENS1_I14__hip_bfloat16Li1ELi1EXT_EXT0_EJEEE.uses_vcc\000_Z13matmul_deviceILi8192ELi8192ELi8192EEvN7kittens2glI14__hip_fp8_e4m3Li1ELi1EXT_EXT1_EJEEENS1_IS2_Li1ELi1EXT0_EXT1_EJEEENS1_I14__hip_bfloat16Li1ELi1EXT_EXT0_EJEEE.uses_flat_scratch\000_Z13matmul_deviceILi8192ELi8192ELi8192EEvN7kittens2glI14__hip_fp8_e4m3Li1ELi1EXT_EXT1_EJEEENS1_IS2_Li1ELi1EXT0_EXT1_EJEEENS1_I14__hip_bfloat16Li1ELi1EXT_EXT0_EJEEE.has_dyn_sized_stack\000_Z13matmul_deviceILi8192ELi8192ELi8192EEvN7kittens2glI14__hip_fp8_e4m3Li1ELi1EXT_EXT1_EJEEENS1_IS2_Li1ELi1EXT0_EXT1_EJEEENS1_I14__hip_bfloat16Li1ELi1EXT_EXT0_EJEEE.has_recursion\000_Z13matmul_deviceILi8192ELi8192ELi8192EEvN7kittens2glI14__hip_fp8_e4m3Li1ELi1EXT_EXT1_EJEEENS1_IS2_Li1ELi1EXT0_EXT1_EJEEENS1_I14__hip_bfloat16Li1ELi1EXT_EXT0_EJEEE.has_indirect_call\000amdgpu.max_num_vgpr\000amdgpu.max_num_agpr\000amdgpu.max_num_sgpr\000_Z13matmul_deviceILi8192ELi8192ELi8192EEvN7kittens2glI14__hip_fp8_e4m3Li1ELi1EXT_EXT1_EJEEENS1_IS2_Li1ELi1EXT0_EXT1_EJEEENS1_I14__hip_bfloat16Li1ELi1EXT_EXT0_EJEEE\000_Z13matmul_deviceILi8192ELi8192ELi8192EEvN7kittens2glI14__hip_fp8_e4m3Li1ELi1EXT_EXT1_EJEEENS1_IS2_Li1ELi1EXT0_EXT1_EJEEENS1_I14__hip_bfloat16Li1ELi1EXT_EXT0_EJEEE.kd\000__hip_cuid_e843e8b1c0f57c70\000_DYNAMIC\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\001\000\000\000\007\000\000\000\002\000\000\000\000\000\000\0008\002\000\000\000\000\000\0008\002\000\000\000\000\000\000t\003\000\000\000\000\000\000\000\000\000\000\000\000\000\000\004\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\007\000\000\000\013\000\000\000\002\000\000\000\000\000\000\000\260\005\000\000\000\000\000\000\260\005\000\000\000\000\000\000`\000\000\000\000\000\000\000\005\000\000\000\001\000\000\000\b\000\000\000\000\000\000\000\030\000\000\000\000\000\000\000\017\000\000\000\366\377\377o\002\000\000\000\000\000\000\000\020\006\000\000\000\000\000\000\020\006\000\000\000\000\000\000(\000\000\000\000\000\000\000\002\000\000\000\000\000\000\000\b\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\031\000\000\000\005\000\000\000\002\000\000\000\000\000\000\0008\006\000\000\000\000\000\0008\006\000\000\000\000\000\000(\000\000\000\000\000\000\000\002\000\000\000\000\000\000\000\004\000\000\000\000\000\000\000\004\000\000\000\000\000\000\000\037\000\000\000\003\000\000\000\002\000\000\000\000\000\000\000`\006\000\000\000\000\000\000`\006\000\000\000\000\000\000h\001\000\000\000\000\000\000\000\000\000\000\000\000\000\000\001\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000'\000\000\000\001\000\000\000\002\000\000\000\000\000\000\000\000\b\000\000\000\000\000\000\000\b\000\000\000\000\000\000@\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000@\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000/\000\000\000\001\000\000\000\006\000\000\000\000\000\000\000\000\031\000\000\000\000\000\000\000\t\000\000\000\000\000\000\210\035\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\001\000\000\000\000\000\000\000\000\000\000\000\000\000\0005\000\000\000\006\000\000\000\003\000\000\000\000\000\000\000\210F\000\000\000\000\000\000\210&\000\000\000\000\000\000p\000\000\000\000\000\000\000\005\000\000\000\000\000\000\000\b\000\000\000\000\000\000\000\020\000\000\000\000\000\000\000>\000\000\000\b\000\000\000\003\000\000\000\000\000\000\000\370F\000\000\000\000\000\000\370&\000\000\000\000\000\000\b\t\000\000\000\000\000\000\000\000\000\000\000\000\000\000\001\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000M\000\000\000\b\000\000\000\003\000\000\000\000\000\000\000\370V\000\000\000\000\000\000\370&\000\000\000\000\000\000\001\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\001\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000R\000\000\000\001\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\370&\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\001\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000a\000\000\000\001\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\370&\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\001\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000v\000\000\000\001\000\000\0000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\370&\000\000\000\000\000\0009\001\000\000\000\000\000\000\000\000\000\000\000\000\000\000\001\000\000\000\000\000\000\000\001\000\000\000\000\000\000\000\177\000\000\000\002\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\0008(\000\000\000\000\000\000\230\001\000\000\000\000\000\000\020\000\000\000\016\000\000\000\b\000\000\000\000\000\000\000\030\000\000\000\000\000\000\000\207\000\000\000\003\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\320)\000\000\000\000\000\000\231\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\001\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\221\000\000\000\003\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000i*\000\000\000\000\000\000\361\007\000\000\000\000\000\000\000\000\000\000\000\000\000\000\001\000\000\000\000\000\000\000\000\000\000\000\000\000\000"
	.size	.L__unnamed_7, 18080

	.type	__hip_fatbin_wrapper,@object    # @__hip_fatbin_wrapper
	.section	.hipFatBinSegment,"a",@progbits
	.p2align	3, 0x0
__hip_fatbin_wrapper:
	.long	1212764230                      # 0x48495046
	.long	1                               # 0x1
	.quad	.L__unnamed_7
	.quad	0
	.size	__hip_fatbin_wrapper, 24

	.type	__hip_gpubin_handle_e843e8b1c0f57c70,@object # @__hip_gpubin_handle_e843e8b1c0f57c70
	.local	__hip_gpubin_handle_e843e8b1c0f57c70
	.comm	__hip_gpubin_handle_e843e8b1c0f57c70,8,8
	.section	.init_array,"aw",@init_array
	.p2align	3, 0x0
	.quad	_GLOBAL__sub_I_8_wave.cu
	.quad	__hip_module_ctor
	.type	__hip_cuid_e843e8b1c0f57c70,@object # @__hip_cuid_e843e8b1c0f57c70
	.bss
	.globl	__hip_cuid_e843e8b1c0f57c70
__hip_cuid_e843e8b1c0f57c70:
	.byte	0                               # 0x0
	.size	__hip_cuid_e843e8b1c0f57c70, 1

	.type	.Lstr,@object                   # @str
	.section	.rodata.str1.1,"aMS",@progbits,1
.Lstr:
	.asciz	"Running reference kernel (matmul_device_ref)..."
	.size	.Lstr, 48

	.type	.Lstr.1,@object                 # @str.1
.Lstr.1:
	.asciz	"Running optimized kernel (matmul_device)..."
	.size	.Lstr.1, 44

	.type	.Lstr.2,@object                 # @str.2
.Lstr.2:
	.asciz	"\n=== PERFORMANCE RESULTS ==="
	.size	.Lstr.2, 29

	.type	.Lstr.3,@object                 # @str.3
.Lstr.3:
	.asciz	"Reference kernel (matmul_device_ref):"
	.size	.Lstr.3, 38

	.type	.Lstr.4,@object                 # @str.4
.Lstr.4:
	.asciz	"\nOptimized kernel (matmul_device):"
	.size	.Lstr.4, 35

	.type	.Lstr.5,@object                 # @str.5
.Lstr.5:
	.asciz	"\nCorrectness: FAILED"
	.size	.Lstr.5, 21

	.type	.Lstr.6,@object                 # @str.6
.Lstr.6:
	.asciz	"\nCorrectness: PASSED"
	.size	.Lstr.6, 21

	.section	".linker-options","e",@llvm_linker_options
	.ident	"AMD clang version 20.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-7.0.0 25304 82aed4e69d70bef3c89c38a2ee85c8c41294dfc9)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __gxx_personality_v0
	.addrsig_sym _ZZ4mainENKUlRKSt6vectorI14__hip_fp8_e4m3SaIS0_EES4_RS_I14__hip_bfloat16SaIS5_EEiE_clES4_S4_S8_i.omp_outlined
	.addrsig_sym _Z28__device_stub__matmul_deviceILi8192ELi8192ELi8192EEvN7kittens2glI14__hip_fp8_e4m3Li1ELi1EXT_EXT1_EJEEENS1_IS2_Li1ELi1EXT0_EXT1_EJEEENS1_I14__hip_bfloat16Li1ELi1EXT_EXT0_EJEEE
	.addrsig_sym _GLOBAL__sub_I_8_wave.cu
	.addrsig_sym __hip_module_ctor
	.addrsig_sym __hip_module_dtor
	.addrsig_sym _Unwind_Resume
	.addrsig_sym _ZStL8__ioinit
	.addrsig_sym __dso_handle
	.addrsig_sym _Z13matmul_deviceILi8192ELi8192ELi8192EEvN7kittens2glI14__hip_fp8_e4m3Li1ELi1EXT_EXT1_EJEEENS1_IS2_Li1ELi1EXT0_EXT1_EJEEENS1_I14__hip_bfloat16Li1ELi1EXT_EXT0_EJEEE
	.addrsig_sym .L__unnamed_7
	.addrsig_sym __hip_fatbin_wrapper
	.addrsig_sym __hip_cuid_e843e8b1c0f57c70
