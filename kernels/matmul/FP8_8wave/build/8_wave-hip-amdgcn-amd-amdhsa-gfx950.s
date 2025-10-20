	.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
	.amdhsa_code_object_version 6
	.section	.text._Z13matmul_deviceILi8192ELi8192ELi8192EEvN7kittens2glI14__hip_fp8_e4m3Li1ELi1EXT_EXT1_EJEEENS1_IS2_Li1ELi1EXT0_EXT1_EJEEENS1_I14__hip_bfloat16Li1ELi1EXT_EXT0_EJEEE,"axG",@progbits,_Z13matmul_deviceILi8192ELi8192ELi8192EEvN7kittens2glI14__hip_fp8_e4m3Li1ELi1EXT_EXT1_EJEEENS1_IS2_Li1ELi1EXT0_EXT1_EJEEENS1_I14__hip_bfloat16Li1ELi1EXT_EXT0_EJEEE,comdat
	.protected	_Z13matmul_deviceILi8192ELi8192ELi8192EEvN7kittens2glI14__hip_fp8_e4m3Li1ELi1EXT_EXT1_EJEEENS1_IS2_Li1ELi1EXT0_EXT1_EJEEENS1_I14__hip_bfloat16Li1ELi1EXT_EXT0_EJEEE ; -- Begin function _Z13matmul_deviceILi8192ELi8192ELi8192EEvN7kittens2glI14__hip_fp8_e4m3Li1ELi1EXT_EXT1_EJEEENS1_IS2_Li1ELi1EXT0_EXT1_EJEEENS1_I14__hip_bfloat16Li1ELi1EXT_EXT0_EJEEE
	.globl	_Z13matmul_deviceILi8192ELi8192ELi8192EEvN7kittens2glI14__hip_fp8_e4m3Li1ELi1EXT_EXT1_EJEEENS1_IS2_Li1ELi1EXT0_EXT1_EJEEENS1_I14__hip_bfloat16Li1ELi1EXT_EXT0_EJEEE
	.p2align	8
	.type	_Z13matmul_deviceILi8192ELi8192ELi8192EEvN7kittens2glI14__hip_fp8_e4m3Li1ELi1EXT_EXT1_EJEEENS1_IS2_Li1ELi1EXT0_EXT1_EJEEENS1_I14__hip_bfloat16Li1ELi1EXT_EXT0_EJEEE,@function
_Z13matmul_deviceILi8192ELi8192ELi8192EEvN7kittens2glI14__hip_fp8_e4m3Li1ELi1EXT_EXT1_EJEEENS1_IS2_Li1ELi1EXT0_EXT1_EJEEENS1_I14__hip_bfloat16Li1ELi1EXT_EXT0_EJEEE: ; @_Z13matmul_deviceILi8192ELi8192ELi8192EEvN7kittens2glI14__hip_fp8_e4m3Li1ELi1EXT_EXT1_EJEEENS1_IS2_Li1ELi1EXT0_EXT1_EJEEENS1_I14__hip_bfloat16Li1ELi1EXT_EXT0_EJEEE
; %bb.0:
	s_load_dwordx2 s[6:7], s[0:1], 0x0
	s_load_dwordx2 s[8:9], s[0:1], 0x10
	s_load_dwordx2 s[4:5], s[0:1], 0x20
	s_ashr_i32 s0, s2, 31
	s_lshr_b32 s0, s0, 27
	s_add_i32 s0, s2, s0
	s_ashr_i32 s10, s0, 5
	s_andn2_b32 s0, s0, 31
	s_sub_i32 s11, s2, s0
	s_lshl_b32 s18, s11, 21
	v_lshlrev_b32_e32 v1, 4, v0
	s_ashr_i32 s19, s18, 31
	v_and_b32_e32 v196, 0x1c00, v1
	s_waitcnt lgkmcnt(0)
	s_add_u32 s16, s8, s18
	v_or_b32_e32 v130, 0x10000, v196
	v_lshlrev_b32_e32 v2, 10, v0
	v_xor_b32_e32 v1, v1, v0
	s_mov_b32 s12, 0x7e070
	s_addc_u32 s20, s9, s19
	v_bitop3_b32 v194, v1, s12, v2 bitop3:0xc8
	v_readfirstlane_b32 s12, v130
	v_or_b32_e32 v131, 0x12000, v196
	s_mov_b32 s3, 0x110000
	s_mov_b32 s2, 0x100000
	s_mov_b32 s0, s16
	s_mov_b32 s1, s20
	s_mov_b32 m0, s12
	v_readfirstlane_b32 s12, v131
	buffer_load_dwordx4 v194, s[0:3], 0 offen lds
	s_mov_b32 m0, s12
	s_lshl_b32 s12, s10, 21
	s_ashr_i32 s17, s12, 31
	s_add_u32 s21, s6, s12
	v_or_b32_e32 v195, 0x80000, v194
	s_addc_u32 s22, s7, s17
	v_readfirstlane_b32 s13, v196
	v_or_b32_e32 v34, 0x2000, v196
	buffer_load_dwordx4 v195, s[0:3], 0 offen lds
	s_mov_b32 s0, s21
	s_mov_b32 s1, s22
	s_mov_b32 m0, s13
	v_readfirstlane_b32 s13, v34
	buffer_load_dwordx4 v194, s[0:3], 0 offen lds
	s_mov_b32 m0, s13
	v_or_b32_e32 v162, 0x14000, v196
	buffer_load_dwordx4 v195, s[0:3], 0 offen lds
	s_or_b32 s0, s18, 0x100000
	s_ashr_i32 s1, s0, 31
	s_add_u32 s0, s8, s0
	v_readfirstlane_b32 s13, v162
	v_or_b32_e32 v163, 0x16000, v196
	s_addc_u32 s1, s9, s1
	s_mov_b32 m0, s13
	v_readfirstlane_b32 s13, v163
	buffer_load_dwordx4 v194, s[0:3], 0 offen lds
	s_mov_b32 m0, s13
	v_or_b32_e32 v197, 0x4000, v196
	buffer_load_dwordx4 v195, s[0:3], 0 offen lds
	s_or_b32 s0, s12, 0x100000
	s_ashr_i32 s1, s0, 31
	s_add_u32 s0, s6, s0
	v_readfirstlane_b32 s13, v197
	v_or_b32_e32 v1, 0x6000, v196
	s_addc_u32 s1, s7, s1
	s_mov_b32 m0, s13
	v_readfirstlane_b32 s13, v1
	buffer_load_dwordx4 v194, s[0:3], 0 offen lds
	s_mov_b32 m0, s13
	v_or_b32_e32 v1, 0x8000, v196
	buffer_load_dwordx4 v195, s[0:3], 0 offen lds
	s_add_u32 s0, s21, 0x80
	v_readfirstlane_b32 s13, v1
	v_or_b32_e32 v1, 0xa000, v196
	s_addc_u32 s1, s22, 0
	s_mov_b32 m0, s13
	v_readfirstlane_b32 s13, v1
	buffer_load_dwordx4 v194, s[0:3], 0 offen lds
	s_mov_b32 m0, s13
	v_or_b32_e32 v1, 0x18000, v196
	buffer_load_dwordx4 v195, s[0:3], 0 offen lds
	s_add_u32 s0, s16, 0x80
	v_readfirstlane_b32 s13, v1
	v_or_b32_e32 v1, 0x1a000, v196
	s_addc_u32 s1, s20, 0
	s_mov_b32 m0, s13
	v_readfirstlane_b32 s13, v1
	buffer_load_dwordx4 v194, s[0:3], 0 offen lds
	s_mov_b32 m0, s13
	v_or_b32_e32 v1, 0x1c000, v196
	buffer_load_dwordx4 v195, s[0:3], 0 offen lds
	s_or_b32 s0, s18, 0x100080
	s_ashr_i32 s1, s0, 31
	s_add_u32 s0, s8, s0
	v_readfirstlane_b32 s13, v1
	v_or_b32_e32 v1, 0x1e000, v196
	s_addc_u32 s1, s9, s1
	s_mov_b32 m0, s13
	v_readfirstlane_b32 s13, v1
	buffer_load_dwordx4 v194, s[0:3], 0 offen lds
	s_mov_b32 m0, s13
	v_lshrrev_b32_e32 v202, 8, v0
	buffer_load_dwordx4 v195, s[0:3], 0 offen lds
	v_bfe_u32 v1, v0, 6, 2
	s_mov_b32 s13, 0
	;;#ASMSTART
	s_waitcnt vmcnt(12)
	;;#ASMEND
	s_barrier
	; sched_barrier mask(0x00000000)
	v_lshlrev_b32_e32 v200, 12, v1
	v_lshlrev_b32_e32 v2, 7, v0
	v_and_b32_e32 v3, 48, v0
	s_movk_i32 s0, 0x780
	v_or_b32_e32 v201, 0x10000, v200
	v_and_or_b32 v203, v2, s0, v3
	v_or_b32_e32 v2, v201, v203
	v_lshrrev_b32_e32 v3, 4, v2
	s_movk_i32 s14, 0x70
	v_bitop3_b32 v10, v3, v2, s14 bitop3:0x6c
	v_or_b32_e32 v2, 64, v2
	v_bitop3_b32 v14, v3, v2, s14 bitop3:0x6c
	;;#ASMSTART
	ds_read_b128 v[2:5], v10 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[6:9], v14 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[10:13], v10 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[14:17], v14 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(10)
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_barrier
	; sched_barrier mask(0x00000000)
	s_mov_b32 s15, 1
	v_cmp_eq_u32_e32 vcc, 1, v202
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_2
; %bb.1:
	s_barrier
.LBB0_2:
	s_or_b64 exec, exec, s[0:1]
	v_lshlrev_b32_e32 v204, 13, v202
	v_or_b32_e32 v35, v204, v203
	v_lshrrev_b32_e32 v18, 4, v35
	v_bitop3_b32 v199, v18, v35, s14 bitop3:0x6c
	v_add_u32_e32 v18, 64, v35
	v_lshrrev_b32_e32 v19, 4, v18
	;;#ASMSTART
	ds_read_b128 v[36:39], v199 offset:0

	;;#ASMEND
	v_bitop3_b32 v198, v19, v18, s14 bitop3:0x6c
	;;#ASMSTART
	ds_read_b128 v[40:43], v198 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[44:47], v199 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[48:51], v198 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[52:55], v199 offset:0x1000

	;;#ASMEND
	s_or_b32 s0, s12, 0x100080
	;;#ASMSTART
	ds_read_b128 v[56:59], v198 offset:0x1000

	;;#ASMEND
	s_ashr_i32 s1, s0, 31
	v_or_b32_e32 v208, 0xc000, v196
	;;#ASMSTART
	ds_read_b128 v[132:135], v199 offset:0x1800

	;;#ASMEND
	s_add_u32 s0, s6, s0
	v_readfirstlane_b32 s23, v208
	v_or_b32_e32 v209, 0xe000, v196
	;;#ASMSTART
	ds_read_b128 v[136:139], v198 offset:0x1800

	;;#ASMEND
	s_addc_u32 s1, s7, s1
	s_mov_b32 m0, s23
	v_readfirstlane_b32 s23, v209
	buffer_load_dwordx4 v194, s[0:3], 0 offen lds
	s_mov_b32 m0, s23
	s_nop 0
	buffer_load_dwordx4 v195, s[0:3], 0 offen lds
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(10)
	;;#ASMEND
	s_barrier
	; sched_barrier mask(0x00000000)
	s_setprio 1
	v_mfma_f32_16x16x128_f8f6f4 v[94:97], v[36:43], v[2:9], 0
	v_mfma_f32_16x16x128_f8f6f4 v[90:93], v[36:43], v[10:17], 0
	v_mfma_f32_16x16x128_f8f6f4 v[86:89], v[44:51], v[2:9], 0
	v_mfma_f32_16x16x128_f8f6f4 v[82:85], v[44:51], v[10:17], 0
	v_mfma_f32_16x16x128_f8f6f4 v[78:81], v[52:59], v[2:9], 0
	v_mfma_f32_16x16x128_f8f6f4 v[74:77], v[52:59], v[10:17], 0
	v_mfma_f32_16x16x128_f8f6f4 v[70:73], v[132:139], v[2:9], 0
	v_mfma_f32_16x16x128_f8f6f4 v[66:69], v[132:139], v[10:17], 0
	s_setprio 0
	; sched_barrier mask(0x00000000)
	s_barrier
	; sched_barrier mask(0x00000000)
	v_add_u32_e32 v205, v201, v203
	v_add_u32_e32 v18, 0x4000, v205
	v_lshrrev_b32_e32 v19, 4, v18
	v_bitop3_b32 v207, v19, v18, s14 bitop3:0x6c
	v_add_u32_e32 v18, 0x4040, v205
	v_lshrrev_b32_e32 v19, 4, v18
	v_bitop3_b32 v206, v19, v18, s14 bitop3:0x6c
	;;#ASMSTART
	ds_read_b128 v[18:21], v207 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[22:25], v206 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[26:29], v207 offset:0x800

	;;#ASMEND
	s_add_u32 s0, s21, 0x100
	v_readfirstlane_b32 s21, v196
	;;#ASMSTART
	ds_read_b128 v[30:33], v206 offset:0x800

	;;#ASMEND
	s_addc_u32 s1, s22, 0
	s_mov_b32 m0, s21
	v_readfirstlane_b32 s21, v34
	buffer_load_dwordx4 v194, s[0:3], 0 offen lds
	s_mov_b32 m0, s21
	s_nop 0
	buffer_load_dwordx4 v195, s[0:3], 0 offen lds
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(10)
	;;#ASMEND
	s_barrier
	; sched_barrier mask(0x00000000)
	s_setprio 1
	v_mfma_f32_16x16x128_f8f6f4 v[126:129], v[36:43], v[18:25], 0
	v_mfma_f32_16x16x128_f8f6f4 v[122:125], v[36:43], v[26:33], 0
	v_mfma_f32_16x16x128_f8f6f4 v[118:121], v[44:51], v[18:25], 0
	v_mfma_f32_16x16x128_f8f6f4 v[114:117], v[44:51], v[26:33], 0
	v_mfma_f32_16x16x128_f8f6f4 v[110:113], v[52:59], v[18:25], 0
	v_mfma_f32_16x16x128_f8f6f4 v[106:109], v[52:59], v[26:33], 0
	v_mfma_f32_16x16x128_f8f6f4 v[102:105], v[132:139], v[18:25], 0
	v_mfma_f32_16x16x128_f8f6f4 v[98:101], v[132:139], v[26:33], 0
	s_setprio 0
	; sched_barrier mask(0x00000000)
	s_barrier
	; sched_barrier mask(0x00000000)
	v_or_b32_e32 v34, 0x4000, v35
	v_lshrrev_b32_e32 v36, 4, v203
	v_bitop3_b32 v211, v36, v34, s14 bitop3:0x6c
	v_add_u32_e32 v34, 0x4040, v35
	v_lshrrev_b32_e32 v35, 4, v34
	v_bitop3_b32 v210, v35, v34, s14 bitop3:0x6c
	;;#ASMSTART
	ds_read_b128 v[34:37], v211 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[38:41], v210 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[42:45], v211 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[46:49], v210 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[50:53], v211 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[54:57], v210 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[58:61], v211 offset:0x1800

	;;#ASMEND
	s_add_u32 s0, s16, 0x100
	v_readfirstlane_b32 s16, v130
	;;#ASMSTART
	ds_read_b128 v[62:65], v210 offset:0x1800

	;;#ASMEND
	s_addc_u32 s1, s20, 0
	s_mov_b32 m0, s16
	v_readfirstlane_b32 s16, v131
	buffer_load_dwordx4 v194, s[0:3], 0 offen lds
	s_mov_b32 m0, s16
	s_nop 0
	buffer_load_dwordx4 v195, s[0:3], 0 offen lds
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(10)
	;;#ASMEND
	s_barrier
	; sched_barrier mask(0x00000000)
	s_setprio 1
	v_mfma_f32_16x16x128_f8f6f4 v[158:161], v[34:41], v[2:9], 0
	v_mfma_f32_16x16x128_f8f6f4 v[154:157], v[34:41], v[10:17], 0
	v_mfma_f32_16x16x128_f8f6f4 v[150:153], v[42:49], v[2:9], 0
	v_mfma_f32_16x16x128_f8f6f4 v[146:149], v[42:49], v[10:17], 0
	v_mfma_f32_16x16x128_f8f6f4 v[142:145], v[50:57], v[2:9], 0
	v_mfma_f32_16x16x128_f8f6f4 v[138:141], v[50:57], v[10:17], 0
	v_mfma_f32_16x16x128_f8f6f4 v[134:137], v[58:65], v[2:9], 0
	v_mfma_f32_16x16x128_f8f6f4 v[130:133], v[58:65], v[10:17], 0
	s_setprio 0
	; sched_barrier mask(0x00000000)
	s_barrier
	; sched_barrier mask(0x00000000)
	v_add_u32_e32 v2, 0x8000, v205
	v_lshrrev_b32_e32 v3, 4, v2
	v_bitop3_b32 v213, v3, v2, s14 bitop3:0x6c
	v_add_u32_e32 v2, 0x8040, v205
	v_lshrrev_b32_e32 v3, 4, v2
	v_bitop3_b32 v212, v3, v2, s14 bitop3:0x6c
	;;#ASMSTART
	ds_read_b128 v[2:5], v213 offset:0

	;;#ASMEND
	s_or_b32 s0, s18, 0x100100
	;;#ASMSTART
	ds_read_b128 v[6:9], v212 offset:0

	;;#ASMEND
	s_ashr_i32 s1, s0, 31
	;;#ASMSTART
	ds_read_b128 v[10:13], v213 offset:0x800

	;;#ASMEND
	s_add_u32 s0, s8, s0
	v_readfirstlane_b32 s16, v162
	;;#ASMSTART
	ds_read_b128 v[14:17], v212 offset:0x800

	;;#ASMEND
	s_addc_u32 s1, s9, s1
	s_mov_b32 m0, s16
	v_readfirstlane_b32 s16, v163
	buffer_load_dwordx4 v194, s[0:3], 0 offen lds
	s_mov_b32 m0, s16
	s_add_u32 s16, s6, s12
	buffer_load_dwordx4 v195, s[0:3], 0 offen lds
	s_addc_u32 s17, s7, s17
	s_add_u32 s18, s8, s18
	v_mov_b32_e32 v162, 0
	s_addc_u32 s19, s9, s19
	s_mov_b64 s[8:9], 0
	v_mov_b32_e32 v163, v162
	v_mov_b32_e32 v164, v162
	v_mov_b32_e32 v165, v162
	v_mov_b32_e32 v166, v162
	v_mov_b32_e32 v167, v162
	v_mov_b32_e32 v168, v162
	v_mov_b32_e32 v169, v162
	v_mov_b32_e32 v170, v162
	v_mov_b32_e32 v171, v162
	v_mov_b32_e32 v172, v162
	v_mov_b32_e32 v173, v162
	v_mov_b32_e32 v174, v162
	v_mov_b32_e32 v175, v162
	v_mov_b32_e32 v176, v162
	v_mov_b32_e32 v177, v162
	v_mov_b32_e32 v178, v162
	v_mov_b32_e32 v179, v162
	v_mov_b32_e32 v180, v162
	v_mov_b32_e32 v181, v162
	v_mov_b32_e32 v182, v162
	v_mov_b32_e32 v183, v162
	v_mov_b32_e32 v184, v162
	v_mov_b32_e32 v185, v162
	v_mov_b32_e32 v186, v162
	v_mov_b32_e32 v187, v162
	v_mov_b32_e32 v188, v162
	v_mov_b32_e32 v189, v162
	v_mov_b32_e32 v190, v162
	v_mov_b32_e32 v191, v162
	v_mov_b32_e32 v192, v162
	v_mov_b32_e32 v193, v162
.LBB0_3:                                ; =>This Inner Loop Header: Depth=1
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	s_waitcnt vmcnt(10)
	;;#ASMEND
	s_barrier
	; sched_barrier mask(0x00000000)
	s_setprio 1
	v_mfma_f32_16x16x128_f8f6f4 v[190:193], v[34:41], v[18:25], v[190:193]
	v_mfma_f32_16x16x128_f8f6f4 v[186:189], v[34:41], v[26:33], v[186:189]
	v_mfma_f32_16x16x128_f8f6f4 v[182:185], v[42:49], v[18:25], v[182:185]
	v_mfma_f32_16x16x128_f8f6f4 v[178:181], v[42:49], v[26:33], v[178:181]
	v_mfma_f32_16x16x128_f8f6f4 v[174:177], v[50:57], v[18:25], v[174:177]
	v_mfma_f32_16x16x128_f8f6f4 v[170:173], v[50:57], v[26:33], v[170:173]
	v_mfma_f32_16x16x128_f8f6f4 v[166:169], v[58:65], v[18:25], v[166:169]
	v_mfma_f32_16x16x128_f8f6f4 v[162:165], v[58:65], v[26:33], v[162:165]
	s_setprio 0
	; sched_barrier mask(0x00000000)
	s_barrier
	; sched_barrier mask(0x00000000)
	s_lshl_b32 s21, s15, 15
	v_or_b32_e32 v18, s21, v204
	v_add_u32_e32 v214, v18, v203
	v_lshrrev_b32_e32 v18, 4, v214
	v_add_u32_e32 v19, 64, v214
	v_bitop3_b32 v18, v18, v214, s14 bitop3:0x6c
	v_lshrrev_b32_e32 v20, 4, v19
	;;#ASMSTART
	ds_read_b128 v[34:37], v18 offset:0

	;;#ASMEND
	v_bitop3_b32 v19, v20, v19, s14 bitop3:0x6c
	;;#ASMSTART
	ds_read_b128 v[38:41], v19 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[42:45], v18 offset:0x800

	;;#ASMEND
	s_add_u32 s22, s16, s8
	;;#ASMSTART
	ds_read_b128 v[46:49], v19 offset:0x800

	;;#ASMEND
	s_addc_u32 s23, s17, s9
	;;#ASMSTART
	ds_read_b128 v[50:53], v18 offset:0x1000

	;;#ASMEND
	s_add_u32 s0, s22, 0x100100
	;;#ASMSTART
	ds_read_b128 v[54:57], v19 offset:0x1000

	;;#ASMEND
	s_addc_u32 s1, s23, 0
	s_lshl_b32 s20, s13, 15
	;;#ASMSTART
	ds_read_b128 v[58:61], v18 offset:0x1800

	;;#ASMEND
	v_or_b32_e32 v18, s20, v197
	;;#ASMSTART
	ds_read_b128 v[62:65], v19 offset:0x1800

	;;#ASMEND
	s_nop 0
	v_readfirstlane_b32 s24, v18
	v_add_u32_e32 v18, 0x2000, v18
	s_mov_b32 m0, s24
	v_readfirstlane_b32 s24, v18
	buffer_load_dwordx4 v194, s[0:3], 0 offen lds
	s_mov_b32 m0, s24
	s_nop 0
	buffer_load_dwordx4 v195, s[0:3], 0 offen lds
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(10)
	;;#ASMEND
	s_barrier
	; sched_barrier mask(0x00000000)
	s_setprio 1
	v_mfma_f32_16x16x128_f8f6f4 v[94:97], v[34:41], v[2:9], v[94:97]
	v_mfma_f32_16x16x128_f8f6f4 v[90:93], v[34:41], v[10:17], v[90:93]
	v_mfma_f32_16x16x128_f8f6f4 v[86:89], v[42:49], v[2:9], v[86:89]
	v_mfma_f32_16x16x128_f8f6f4 v[82:85], v[42:49], v[10:17], v[82:85]
	v_mfma_f32_16x16x128_f8f6f4 v[78:81], v[50:57], v[2:9], v[78:81]
	v_mfma_f32_16x16x128_f8f6f4 v[74:77], v[50:57], v[10:17], v[74:77]
	v_mfma_f32_16x16x128_f8f6f4 v[70:73], v[58:65], v[2:9], v[70:73]
	v_mfma_f32_16x16x128_f8f6f4 v[66:69], v[58:65], v[10:17], v[66:69]
	s_setprio 0
	; sched_barrier mask(0x00000000)
	s_barrier
	; sched_barrier mask(0x00000000)
	s_add_i32 s24, s21, 0x10000
	v_or_b32_e32 v18, s24, v200
	v_add_u32_e32 v18, v18, v203
	v_add_u32_e32 v19, 0x4000, v18
	v_lshrrev_b32_e32 v20, 4, v19
	v_add_u32_e32 v18, 0x4040, v18
	v_bitop3_b32 v26, v20, v19, s14 bitop3:0x6c
	v_lshrrev_b32_e32 v19, 4, v18
	v_bitop3_b32 v30, v19, v18, s14 bitop3:0x6c
	;;#ASMSTART
	ds_read_b128 v[18:21], v26 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[22:25], v30 offset:0

	;;#ASMEND
	v_or_b32_e32 v215, s21, v196
	;;#ASMSTART
	ds_read_b128 v[26:29], v26 offset:0x800

	;;#ASMEND
	s_add_u32 s0, s22, 0x180
	v_readfirstlane_b32 s21, v215
	v_or_b32_e32 v215, 0x2000, v215
	;;#ASMSTART
	ds_read_b128 v[30:33], v30 offset:0x800

	;;#ASMEND
	s_addc_u32 s1, s23, 0
	s_mov_b32 m0, s21
	v_readfirstlane_b32 s21, v215
	buffer_load_dwordx4 v194, s[0:3], 0 offen lds
	s_mov_b32 m0, s21
	s_nop 0
	buffer_load_dwordx4 v195, s[0:3], 0 offen lds
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(10)
	;;#ASMEND
	s_barrier
	; sched_barrier mask(0x00000000)
	s_setprio 1
	v_mfma_f32_16x16x128_f8f6f4 v[126:129], v[34:41], v[18:25], v[126:129]
	v_mfma_f32_16x16x128_f8f6f4 v[122:125], v[34:41], v[26:33], v[122:125]
	v_mfma_f32_16x16x128_f8f6f4 v[118:121], v[42:49], v[18:25], v[118:121]
	v_mfma_f32_16x16x128_f8f6f4 v[114:117], v[42:49], v[26:33], v[114:117]
	v_mfma_f32_16x16x128_f8f6f4 v[110:113], v[50:57], v[18:25], v[110:113]
	v_mfma_f32_16x16x128_f8f6f4 v[106:109], v[50:57], v[26:33], v[106:109]
	v_mfma_f32_16x16x128_f8f6f4 v[102:105], v[58:65], v[18:25], v[102:105]
	v_mfma_f32_16x16x128_f8f6f4 v[98:101], v[58:65], v[26:33], v[98:101]
	s_setprio 0
	; sched_barrier mask(0x00000000)
	s_barrier
	; sched_barrier mask(0x00000000)
	v_add_u32_e32 v34, 0x4000, v214
	v_lshrrev_b32_e32 v35, 4, v34
	v_bitop3_b32 v58, v35, v34, s14 bitop3:0x6c
	v_add_u32_e32 v34, 0x4040, v214
	v_lshrrev_b32_e32 v35, 4, v34
	v_bitop3_b32 v62, v35, v34, s14 bitop3:0x6c
	;;#ASMSTART
	ds_read_b128 v[34:37], v58 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[38:41], v62 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[42:45], v58 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[46:49], v62 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[50:53], v58 offset:0x1000

	;;#ASMEND
	s_add_u32 s21, s18, s8
	;;#ASMSTART
	ds_read_b128 v[54:57], v62 offset:0x1000

	;;#ASMEND
	s_addc_u32 s22, s19, s9
	v_or_b32_e32 v214, s24, v196
	;;#ASMSTART
	ds_read_b128 v[58:61], v58 offset:0x1800

	;;#ASMEND
	s_add_u32 s0, s21, 0x180
	v_readfirstlane_b32 s23, v214
	v_or_b32_e32 v215, 0x2000, v214
	;;#ASMSTART
	ds_read_b128 v[62:65], v62 offset:0x1800

	;;#ASMEND
	s_addc_u32 s1, s22, 0
	s_mov_b32 m0, s23
	v_readfirstlane_b32 s23, v215
	buffer_load_dwordx4 v194, s[0:3], 0 offen lds
	s_mov_b32 m0, s23
	s_nop 0
	buffer_load_dwordx4 v195, s[0:3], 0 offen lds
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(10)
	;;#ASMEND
	s_barrier
	; sched_barrier mask(0x00000000)
	s_setprio 1
	v_mfma_f32_16x16x128_f8f6f4 v[158:161], v[34:41], v[2:9], v[158:161]
	v_mfma_f32_16x16x128_f8f6f4 v[154:157], v[34:41], v[10:17], v[154:157]
	v_mfma_f32_16x16x128_f8f6f4 v[150:153], v[42:49], v[2:9], v[150:153]
	v_mfma_f32_16x16x128_f8f6f4 v[146:149], v[42:49], v[10:17], v[146:149]
	v_mfma_f32_16x16x128_f8f6f4 v[142:145], v[50:57], v[2:9], v[142:145]
	v_mfma_f32_16x16x128_f8f6f4 v[138:141], v[50:57], v[10:17], v[138:141]
	v_mfma_f32_16x16x128_f8f6f4 v[134:137], v[58:65], v[2:9], v[134:137]
	v_mfma_f32_16x16x128_f8f6f4 v[130:133], v[58:65], v[10:17], v[130:133]
	s_setprio 0
	; sched_barrier mask(0x00000000)
	s_barrier
	; sched_barrier mask(0x00000000)
	v_add3_u32 v2, v201, s20, v203
	v_lshrrev_b32_e32 v3, 4, v2
	v_bitop3_b32 v10, v3, v2, s14 bitop3:0x6c
	v_add_u32_e32 v2, 64, v2
	v_lshrrev_b32_e32 v3, 4, v2
	v_bitop3_b32 v14, v3, v2, s14 bitop3:0x6c
	;;#ASMSTART
	ds_read_b128 v[2:5], v10 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[6:9], v14 offset:0

	;;#ASMEND
	v_or_b32_e32 v215, 0x4000, v214
	;;#ASMSTART
	ds_read_b128 v[10:13], v10 offset:0x800

	;;#ASMEND
	s_add_u32 s0, s21, 0x100180
	v_readfirstlane_b32 s20, v215
	v_or_b32_e32 v214, 0x6000, v214
	;;#ASMSTART
	ds_read_b128 v[14:17], v14 offset:0x800

	;;#ASMEND
	s_addc_u32 s1, s22, 0
	s_mov_b32 m0, s20
	v_readfirstlane_b32 s20, v214
	buffer_load_dwordx4 v194, s[0:3], 0 offen lds
	s_mov_b32 m0, s20
	s_xor_b32 s15, s15, 1
	buffer_load_dwordx4 v195, s[0:3], 0 offen lds
	s_xor_b32 s13, s13, 1
	s_add_u32 s8, s8, 0x80
	s_addc_u32 s9, s9, 0
	s_cmpk_eq_i32 s8, 0x1e80
	s_cbranch_scc0 .LBB0_3
; %bb.4:
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	s_waitcnt vmcnt(10)
	;;#ASMEND
	s_barrier
	; sched_barrier mask(0x00000000)
	s_setprio 1
	v_mfma_f32_16x16x128_f8f6f4 v[190:193], v[34:41], v[18:25], v[190:193]
	v_mfma_f32_16x16x128_f8f6f4 v[38:41], v[34:41], v[26:33], v[186:189]
	v_mfma_f32_16x16x128_f8f6f4 v[182:185], v[42:49], v[18:25], v[182:185]
	v_mfma_f32_16x16x128_f8f6f4 v[46:49], v[42:49], v[26:33], v[178:181]
	v_mfma_f32_16x16x128_f8f6f4 v[174:177], v[50:57], v[18:25], v[174:177]
	v_mfma_f32_16x16x128_f8f6f4 v[54:57], v[50:57], v[26:33], v[170:173]
	v_mfma_f32_16x16x128_f8f6f4 v[18:21], v[58:65], v[18:25], v[166:169]
	v_mfma_f32_16x16x128_f8f6f4 v[26:29], v[58:65], v[26:33], v[162:165]
	s_setprio 0
	; sched_barrier mask(0x00000000)
	s_barrier
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	ds_read_b128 v[58:61], v199 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[62:65], v198 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[214:217], v199 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[218:221], v198 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[222:225], v199 offset:0x1000

	;;#ASMEND
	s_or_b32 s0, s12, 0x101f80
	;;#ASMSTART
	ds_read_b128 v[226:229], v198 offset:0x1000

	;;#ASMEND
	s_ashr_i32 s1, s0, 31
	;;#ASMSTART
	ds_read_b128 v[230:233], v199 offset:0x1800

	;;#ASMEND
	s_add_u32 s0, s6, s0
	v_readfirstlane_b32 s6, v208
	;;#ASMSTART
	ds_read_b128 v[234:237], v198 offset:0x1800

	;;#ASMEND
	s_addc_u32 s1, s7, s1
	s_mov_b32 s3, 0x110000
	s_mov_b32 s2, 0x100000
	s_mov_b32 m0, s6
	v_readfirstlane_b32 s6, v209
	buffer_load_dwordx4 v194, s[0:3], 0 offen lds
	s_mov_b32 m0, s6
	s_nop 0
	buffer_load_dwordx4 v195, s[0:3], 0 offen lds
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(10)
	;;#ASMEND
	s_barrier
	; sched_barrier mask(0x00000000)
	s_setprio 1
	v_mfma_f32_16x16x128_f8f6f4 v[162:165], v[58:65], v[2:9], v[94:97]
	v_mfma_f32_16x16x128_f8f6f4 v[86:89], v[214:221], v[2:9], v[86:89]
	v_mfma_f32_16x16x128_f8f6f4 v[166:169], v[58:65], v[10:17], v[90:93]
	v_mfma_f32_16x16x128_f8f6f4 v[170:173], v[214:221], v[10:17], v[82:85]
	v_mfma_f32_16x16x128_f8f6f4 v[178:181], v[222:229], v[2:9], v[78:81]
	v_mfma_f32_16x16x128_f8f6f4 v[186:189], v[222:229], v[10:17], v[74:77]
	v_mfma_f32_16x16x128_f8f6f4 v[194:197], v[230:237], v[2:9], v[70:73]
	v_mfma_f32_16x16x128_f8f6f4 v[198:201], v[230:237], v[10:17], v[66:69]
	s_setprio 0
	; sched_barrier mask(0x00000000)
	s_barrier
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	ds_read_b128 v[238:241], v207 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[242:245], v206 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[246:249], v207 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[250:253], v206 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(8)
	;;#ASMEND
	s_barrier
	; sched_barrier mask(0x00000000)
	s_setprio 1
	v_mfma_f32_16x16x128_f8f6f4 v[30:33], v[58:65], v[238:245], v[126:129]
	v_mfma_f32_16x16x128_f8f6f4 v[58:61], v[58:65], v[246:253], v[122:125]
	v_mfma_f32_16x16x128_f8f6f4 v[74:77], v[214:221], v[238:245], v[118:121]
	v_mfma_f32_16x16x128_f8f6f4 v[82:85], v[214:221], v[246:253], v[114:117]
	v_mfma_f32_16x16x128_f8f6f4 v[90:93], v[222:229], v[238:245], v[110:113]
	v_mfma_f32_16x16x128_f8f6f4 v[106:109], v[222:229], v[246:253], v[106:109]
	v_mfma_f32_16x16x128_f8f6f4 v[102:105], v[230:237], v[238:245], v[102:105]
	v_mfma_f32_16x16x128_f8f6f4 v[110:113], v[230:237], v[246:253], v[98:101]
	s_setprio 0
	; sched_barrier mask(0x00000000)
	s_barrier
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	ds_read_b128 v[114:117], v211 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[118:121], v210 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[214:217], v211 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[218:221], v210 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[222:225], v211 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[226:229], v210 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[230:233], v211 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[234:237], v210 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(6)
	;;#ASMEND
	s_barrier
	; sched_barrier mask(0x00000000)
	s_setprio 1
	v_mfma_f32_16x16x128_f8f6f4 v[22:25], v[114:121], v[2:9], v[158:161]
	v_mfma_f32_16x16x128_f8f6f4 v[34:37], v[114:121], v[10:17], v[154:157]
	v_mfma_f32_16x16x128_f8f6f4 v[42:45], v[214:221], v[2:9], v[150:153]
	v_mfma_f32_16x16x128_f8f6f4 v[50:53], v[214:221], v[10:17], v[146:149]
	v_mfma_f32_16x16x128_f8f6f4 v[62:65], v[222:229], v[2:9], v[142:145]
	v_mfma_f32_16x16x128_f8f6f4 v[66:69], v[222:229], v[10:17], v[138:141]
	v_mfma_f32_16x16x128_f8f6f4 v[78:81], v[230:237], v[2:9], v[134:137]
	v_mfma_f32_16x16x128_f8f6f4 v[94:97], v[230:237], v[10:17], v[130:133]
	s_setprio 0
	; sched_barrier mask(0x00000000)
	s_barrier
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	ds_read_b128 v[2:5], v213 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[6:9], v212 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[10:13], v213 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[14:17], v212 offset:0x800

	;;#ASMEND
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	s_waitcnt vmcnt(0)
	;;#ASMEND
	s_barrier
	; sched_barrier mask(0x00000000)
	s_setprio 1
	v_mfma_f32_16x16x128_f8f6f4 v[122:125], v[114:121], v[238:245], v[190:193]
	v_mfma_f32_16x16x128_f8f6f4 v[126:129], v[114:121], v[246:253], v[38:41]
	v_mfma_f32_16x16x128_f8f6f4 v[130:133], v[214:221], v[238:245], v[182:185]
	v_mfma_f32_16x16x128_f8f6f4 v[134:137], v[214:221], v[246:253], v[46:49]
	v_mfma_f32_16x16x128_f8f6f4 v[138:141], v[222:229], v[238:245], v[174:177]
	v_mfma_f32_16x16x128_f8f6f4 v[142:145], v[222:229], v[246:253], v[54:57]
	v_mfma_f32_16x16x128_f8f6f4 v[146:149], v[230:237], v[238:245], v[18:21]
	v_mfma_f32_16x16x128_f8f6f4 v[150:153], v[230:237], v[246:253], v[26:29]
	s_setprio 0
	; sched_barrier mask(0x00000000)
	s_barrier
	; sched_barrier mask(0x00000000)
	v_add_u32_e32 v182, v203, v204
	s_nop 2
	v_or_b32_e32 v18, 0x8000, v182
	v_lshrrev_b32_e32 v19, 4, v182
	s_movk_i32 s0, 0x70
	v_bitop3_b32 v18, v19, v18, s0 bitop3:0x6c
	v_add_u32_e32 v19, 0x8040, v182
	v_lshrrev_b32_e32 v20, 4, v19
	;;#ASMSTART
	ds_read_b128 v[206:209], v18 offset:0

	;;#ASMEND
	v_bitop3_b32 v19, v20, v19, s0 bitop3:0x6c
	;;#ASMSTART
	ds_read_b128 v[210:213], v19 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[214:217], v18 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[218:221], v19 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[222:225], v18 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[226:229], v19 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[230:233], v18 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[234:237], v19 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_barrier
	; sched_barrier mask(0x00000000)
	s_setprio 1
	v_mfma_f32_16x16x128_f8f6f4 v[162:165], v[206:213], v[2:9], v[162:165]
	v_mfma_f32_16x16x128_f8f6f4 v[158:161], v[206:213], v[10:17], v[166:169]
	v_mfma_f32_16x16x128_f8f6f4 v[154:157], v[214:221], v[2:9], v[86:89]
	v_mfma_f32_16x16x128_f8f6f4 v[118:121], v[214:221], v[10:17], v[170:173]
	v_mfma_f32_16x16x128_f8f6f4 v[114:117], v[222:229], v[2:9], v[178:181]
	v_mfma_f32_16x16x128_f8f6f4 v[70:73], v[222:229], v[10:17], v[186:189]
	v_mfma_f32_16x16x128_f8f6f4 v[54:57], v[230:237], v[2:9], v[194:197]
	v_mfma_f32_16x16x128_f8f6f4 v[26:29], v[230:237], v[10:17], v[198:201]
	s_setprio 0
	; sched_barrier mask(0x00000000)
	s_barrier
	; sched_barrier mask(0x00000000)
	v_add_u32_e32 v18, 0xc000, v205
	v_lshrrev_b32_e32 v19, 4, v18
	v_bitop3_b32 v18, v19, v18, s0 bitop3:0x6c
	v_add_u32_e32 v19, 0xc040, v205
	v_lshrrev_b32_e32 v20, 4, v19
	;;#ASMSTART
	ds_read_b128 v[166:169], v18 offset:0

	;;#ASMEND
	v_bitop3_b32 v19, v20, v19, s0 bitop3:0x6c
	;;#ASMSTART
	ds_read_b128 v[170:173], v19 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[174:177], v18 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[178:181], v19 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_barrier
	; sched_barrier mask(0x00000000)
	s_setprio 1
	v_mfma_f32_16x16x128_f8f6f4 v[86:89], v[206:213], v[166:173], v[30:33]
	v_mfma_f32_16x16x128_f8f6f4 v[98:101], v[206:213], v[174:181], v[58:61]
	v_mfma_f32_16x16x128_f8f6f4 v[74:77], v[214:221], v[166:173], v[74:77]
	v_mfma_f32_16x16x128_f8f6f4 v[58:61], v[214:221], v[174:181], v[82:85]
	v_mfma_f32_16x16x128_f8f6f4 v[46:49], v[222:229], v[166:173], v[90:93]
	v_mfma_f32_16x16x128_f8f6f4 v[38:41], v[222:229], v[174:181], v[106:109]
	v_mfma_f32_16x16x128_f8f6f4 v[30:33], v[230:237], v[166:173], v[102:105]
	v_mfma_f32_16x16x128_f8f6f4 v[18:21], v[230:237], v[174:181], v[110:113]
	s_setprio 0
	; sched_barrier mask(0x00000000)
	s_barrier
	; sched_barrier mask(0x00000000)
	s_nop 0
	v_add_u32_e32 v82, 0xc000, v182
	v_lshrrev_b32_e32 v83, 4, v82
	v_bitop3_b32 v82, v83, v82, s0 bitop3:0x6c
	v_add_u32_e32 v83, 0xc040, v182
	v_lshrrev_b32_e32 v84, 4, v83
	;;#ASMSTART
	ds_read_b128 v[182:185], v82 offset:0

	;;#ASMEND
	v_bitop3_b32 v83, v84, v83, s0 bitop3:0x6c
	;;#ASMSTART
	ds_read_b128 v[186:189], v83 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[190:193], v82 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[194:197], v83 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[204:207], v82 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[208:211], v83 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[212:215], v82 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[216:219], v83 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_barrier
	; sched_barrier mask(0x00000000)
	s_setprio 1
	v_mfma_f32_16x16x128_f8f6f4 v[110:113], v[182:189], v[2:9], v[22:25]
	v_mfma_f32_16x16x128_f8f6f4 v[102:105], v[182:189], v[10:17], v[34:37]
	v_mfma_f32_16x16x128_f8f6f4 v[106:109], v[190:197], v[2:9], v[42:45]
	v_mfma_f32_16x16x128_f8f6f4 v[82:85], v[190:197], v[10:17], v[50:53]
	v_mfma_f32_16x16x128_f8f6f4 v[90:93], v[204:211], v[2:9], v[62:65]
	v_mfma_f32_16x16x128_f8f6f4 v[66:69], v[204:211], v[10:17], v[66:69]
	v_mfma_f32_16x16x128_f8f6f4 v[78:81], v[212:219], v[2:9], v[78:81]
	v_mfma_f32_16x16x128_f8f6f4 v[62:65], v[212:219], v[10:17], v[94:97]
	v_mfma_f32_16x16x128_f8f6f4 v[50:53], v[182:189], v[166:173], v[122:125]
	v_mfma_f32_16x16x128_f8f6f4 v[42:45], v[182:189], v[174:181], v[126:129]
	v_mfma_f32_16x16x128_f8f6f4 v[34:37], v[190:197], v[166:173], v[130:133]
	v_mfma_f32_16x16x128_f8f6f4 v[22:25], v[190:197], v[174:181], v[134:137]
	v_mfma_f32_16x16x128_f8f6f4 v[14:17], v[204:211], v[166:173], v[138:141]
	v_mfma_f32_16x16x128_f8f6f4 v[10:13], v[204:211], v[174:181], v[142:145]
	v_mfma_f32_16x16x128_f8f6f4 v[6:9], v[212:219], v[166:173], v[146:149]
	v_mfma_f32_16x16x128_f8f6f4 v[2:5], v[212:219], v[174:181], v[150:153]
	s_setprio 0
	s_movk_i32 s0, 0x100
	v_cmp_gt_u32_e32 vcc, s0, v0
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_6
; %bb.5:
	s_barrier
.LBB0_6:
	s_or_b64 exec, exec, s[0:1]
	v_lshlrev_b32_e32 v94, 19, v202
	v_lshlrev_b32_e32 v1, 5, v1
	v_lshl_or_b32 v94, s10, 21, v94
	v_lshl_or_b32 v1, s11, 8, v1
	v_add_u32_e32 v140, v94, v1
	v_lshlrev_b32_e32 v1, 11, v0
	s_mov_b32 s0, 0x1800f
	v_bitop3_b32 v0, v1, s0, v0 bitop3:0xc8
	v_ashrrev_i32_e32 v141, 31, v140
	v_lshlrev_b32_e32 v0, 1, v0
	v_mov_b32_e32 v1, 0
	v_lshl_add_u64 v[142:143], v[140:141], 1, s[4:5]
	v_or_b32_e32 v94, 0x4000, v0
	v_mov_b32_e32 v95, v1
	v_lshl_add_u64 v[96:97], v[142:143], 0, v[94:95]
	global_store_short_d16_hi v[96:97], v163, off
	v_or_b32_e32 v96, 0x8000, v0
	v_mov_b32_e32 v97, v1
	v_lshl_add_u64 v[122:123], v[142:143], 0, v[96:97]
	global_store_short_d16_hi v[122:123], v164, off
	v_or_b32_e32 v122, 0xc000, v0
	v_mov_b32_e32 v123, v1
	v_lshl_add_u64 v[144:145], v[142:143], 0, v[0:1]
	v_lshl_add_u64 v[124:125], v[142:143], 0, v[122:123]
	global_store_short_d16_hi v[144:145], v162, off
	global_store_short_d16_hi v[124:125], v165, off
	global_store_short_d16_hi v[144:145], v158, off offset:32
	v_or_b32_e32 v124, 0x4020, v0
	v_mov_b32_e32 v125, v1
	v_lshl_add_u64 v[126:127], v[142:143], 0, v[124:125]
	global_store_short_d16_hi v[126:127], v159, off
	v_or_b32_e32 v126, 0x8020, v0
	v_mov_b32_e32 v127, v1
	v_lshl_add_u64 v[128:129], v[142:143], 0, v[126:127]
	global_store_short_d16_hi v[128:129], v160, off
	v_or_b32_e32 v128, 0xc020, v0
	v_mov_b32_e32 v129, v1
	v_lshl_add_u64 v[130:131], v[142:143], 0, v[128:129]
	global_store_short_d16_hi v[130:131], v161, off
	v_or_b32_e32 v130, 0x40000, v0
	v_mov_b32_e32 v131, v1
	v_lshl_add_u64 v[132:133], v[142:143], 0, v[130:131]
	global_store_short_d16_hi v[132:133], v154, off
	v_or_b32_e32 v132, 0x44000, v0
	v_mov_b32_e32 v133, v1
	v_lshl_add_u64 v[134:135], v[142:143], 0, v[132:133]
	global_store_short_d16_hi v[134:135], v155, off
	v_or_b32_e32 v134, 0x48000, v0
	v_mov_b32_e32 v135, v1
	v_lshl_add_u64 v[136:137], v[142:143], 0, v[134:135]
	global_store_short_d16_hi v[136:137], v156, off
	v_or_b32_e32 v136, 0x4c000, v0
	v_mov_b32_e32 v137, v1
	v_lshl_add_u64 v[138:139], v[142:143], 0, v[136:137]
	global_store_short_d16_hi v[138:139], v157, off
	v_or_b32_e32 v138, 0x40020, v0
	v_mov_b32_e32 v139, v1
	v_lshl_add_u64 v[146:147], v[142:143], 0, v[138:139]
	global_store_short_d16_hi v[146:147], v118, off
	v_or_b32_e32 v146, 0x44020, v0
	v_mov_b32_e32 v147, v1
	v_lshl_add_u64 v[148:149], v[142:143], 0, v[146:147]
	global_store_short_d16_hi v[148:149], v119, off
	v_or_b32_e32 v118, 0x48020, v0
	v_mov_b32_e32 v119, v1
	v_lshl_add_u64 v[148:149], v[142:143], 0, v[118:119]
	global_store_short_d16_hi v[148:149], v120, off
	v_or_b32_e32 v148, 0x4c020, v0
	v_mov_b32_e32 v149, v1
	v_lshl_add_u64 v[150:151], v[142:143], 0, v[148:149]
	global_store_short_d16_hi v[150:151], v121, off
	v_or_b32_e32 v120, 0x80000, v0
	v_mov_b32_e32 v121, v1
	v_lshl_add_u64 v[150:151], v[142:143], 0, v[120:121]
	global_store_short_d16_hi v[150:151], v114, off
	v_or_b32_e32 v150, 0x84000, v0
	v_mov_b32_e32 v151, v1
	v_lshl_add_u64 v[152:153], v[142:143], 0, v[150:151]
	global_store_short_d16_hi v[152:153], v115, off
	v_or_b32_e32 v114, 0x88000, v0
	v_mov_b32_e32 v115, v1
	v_lshl_add_u64 v[152:153], v[142:143], 0, v[114:115]
	global_store_short_d16_hi v[152:153], v116, off
	v_or_b32_e32 v152, 0x8c000, v0
	v_mov_b32_e32 v153, v1
	v_lshl_add_u64 v[154:155], v[142:143], 0, v[152:153]
	global_store_short_d16_hi v[154:155], v117, off
	v_or_b32_e32 v116, 0x80020, v0
	v_mov_b32_e32 v117, v1
	v_lshl_add_u64 v[154:155], v[142:143], 0, v[116:117]
	global_store_short_d16_hi v[154:155], v70, off
	v_or_b32_e32 v154, 0x84020, v0
	v_mov_b32_e32 v155, v1
	v_lshl_add_u64 v[156:157], v[142:143], 0, v[154:155]
	global_store_short_d16_hi v[156:157], v71, off
	v_or_b32_e32 v70, 0x88020, v0
	v_mov_b32_e32 v71, v1
	v_lshl_add_u64 v[156:157], v[142:143], 0, v[70:71]
	global_store_short_d16_hi v[156:157], v72, off
	v_or_b32_e32 v156, 0x8c020, v0
	v_mov_b32_e32 v157, v1
	v_lshl_add_u64 v[158:159], v[142:143], 0, v[156:157]
	global_store_short_d16_hi v[158:159], v73, off
	v_or_b32_e32 v72, 0xc0000, v0
	v_mov_b32_e32 v73, v1
	v_lshl_add_u64 v[158:159], v[142:143], 0, v[72:73]
	global_store_short_d16_hi v[158:159], v54, off
	v_or_b32_e32 v158, 0xc4000, v0
	v_mov_b32_e32 v159, v1
	v_lshl_add_u64 v[160:161], v[142:143], 0, v[158:159]
	global_store_short_d16_hi v[160:161], v55, off
	v_or_b32_e32 v54, 0xc8000, v0
	v_mov_b32_e32 v55, v1
	v_lshl_add_u64 v[160:161], v[142:143], 0, v[54:55]
	global_store_short_d16_hi v[160:161], v56, off
	v_or_b32_e32 v160, 0xcc000, v0
	v_mov_b32_e32 v161, v1
	v_lshl_add_u64 v[162:163], v[142:143], 0, v[160:161]
	global_store_short_d16_hi v[162:163], v57, off
	v_or_b32_e32 v56, 0xc0020, v0
	v_mov_b32_e32 v57, v1
	v_lshl_add_u64 v[162:163], v[142:143], 0, v[56:57]
	global_store_short_d16_hi v[162:163], v26, off
	v_or_b32_e32 v162, 0xc4020, v0
	v_mov_b32_e32 v163, v1
	v_lshl_add_u64 v[164:165], v[142:143], 0, v[162:163]
	global_store_short_d16_hi v[164:165], v27, off
	v_or_b32_e32 v26, 0xc8020, v0
	v_mov_b32_e32 v27, v1
	v_lshl_add_u64 v[164:165], v[142:143], 0, v[26:27]
	global_store_short_d16_hi v[164:165], v28, off
	v_or_b32_e32 v164, 0xcc020, v0
	v_mov_b32_e32 v165, v1
	v_lshl_add_u64 v[166:167], v[142:143], 0, v[164:165]
	s_mov_b64 s[0:1], 0x100
	global_store_short_d16_hi v[166:167], v29, off
	v_lshl_add_u64 v[28:29], v[142:143], 0, s[0:1]
	v_lshl_add_u64 v[142:143], v[28:29], 0, v[94:95]
	global_store_short_d16_hi v[144:145], v86, off offset:256
	global_store_short_d16_hi v[142:143], v87, off
	v_lshl_add_u64 v[86:87], v[28:29], 0, v[96:97]
	global_store_short_d16_hi v[86:87], v88, off
	v_lshl_add_u64 v[86:87], v[28:29], 0, v[122:123]
	global_store_short_d16_hi v[86:87], v89, off
	v_lshl_add_u64 v[86:87], v[28:29], 0, v[0:1]
	global_store_short_d16_hi v[86:87], v98, off offset:32
	v_lshl_add_u64 v[86:87], v[28:29], 0, v[124:125]
	global_store_short_d16_hi v[86:87], v99, off
	v_lshl_add_u64 v[86:87], v[28:29], 0, v[126:127]
	global_store_short_d16_hi v[86:87], v100, off
	v_lshl_add_u64 v[86:87], v[28:29], 0, v[128:129]
	global_store_short_d16_hi v[86:87], v101, off
	v_lshl_add_u64 v[86:87], v[28:29], 0, v[130:131]
	global_store_short_d16_hi v[86:87], v74, off
	v_lshl_add_u64 v[86:87], v[28:29], 0, v[132:133]
	global_store_short_d16_hi v[86:87], v75, off
	v_lshl_add_u64 v[74:75], v[28:29], 0, v[134:135]
	global_store_short_d16_hi v[74:75], v76, off
	v_lshl_add_u64 v[74:75], v[28:29], 0, v[136:137]
	global_store_short_d16_hi v[74:75], v77, off
	v_lshl_add_u64 v[74:75], v[28:29], 0, v[138:139]
	global_store_short_d16_hi v[74:75], v58, off
	v_lshl_add_u64 v[74:75], v[28:29], 0, v[146:147]
	global_store_short_d16_hi v[74:75], v59, off
	v_lshl_add_u64 v[58:59], v[28:29], 0, v[118:119]
	global_store_short_d16_hi v[58:59], v60, off
	v_lshl_add_u64 v[58:59], v[28:29], 0, v[148:149]
	global_store_short_d16_hi v[58:59], v61, off
	v_lshl_add_u64 v[58:59], v[28:29], 0, v[120:121]
	global_store_short_d16_hi v[58:59], v46, off
	v_lshl_add_u64 v[58:59], v[28:29], 0, v[150:151]
	global_store_short_d16_hi v[58:59], v47, off
	v_lshl_add_u64 v[46:47], v[28:29], 0, v[114:115]
	global_store_short_d16_hi v[46:47], v48, off
	v_lshl_add_u64 v[46:47], v[28:29], 0, v[152:153]
	global_store_short_d16_hi v[46:47], v49, off
	v_lshl_add_u64 v[46:47], v[28:29], 0, v[116:117]
	global_store_short_d16_hi v[46:47], v38, off
	v_lshl_add_u64 v[46:47], v[28:29], 0, v[154:155]
	global_store_short_d16_hi v[46:47], v39, off
	v_lshl_add_u64 v[38:39], v[28:29], 0, v[70:71]
	global_store_short_d16_hi v[38:39], v40, off
	v_lshl_add_u64 v[38:39], v[28:29], 0, v[156:157]
	global_store_short_d16_hi v[38:39], v41, off
	v_lshl_add_u64 v[38:39], v[28:29], 0, v[72:73]
	global_store_short_d16_hi v[38:39], v30, off
	v_lshl_add_u64 v[38:39], v[28:29], 0, v[158:159]
	global_store_short_d16_hi v[38:39], v31, off
	v_lshl_add_u64 v[30:31], v[28:29], 0, v[54:55]
	global_store_short_d16_hi v[30:31], v32, off
	v_lshl_add_u64 v[30:31], v[28:29], 0, v[160:161]
	global_store_short_d16_hi v[30:31], v33, off
	v_lshl_add_u64 v[30:31], v[28:29], 0, v[56:57]
	global_store_short_d16_hi v[30:31], v18, off
	v_lshl_add_u64 v[30:31], v[28:29], 0, v[162:163]
	global_store_short_d16_hi v[30:31], v19, off
	v_lshl_add_u64 v[18:19], v[28:29], 0, v[26:27]
	global_store_short_d16_hi v[18:19], v20, off
	v_lshl_add_u64 v[18:19], v[28:29], 0, v[164:165]
	global_store_short_d16_hi v[18:19], v21, off
	v_add_u32_e32 v18, 0x100000, v140
	v_ashrrev_i32_e32 v19, 31, v18
	v_lshl_add_u64 v[18:19], v[18:19], 1, s[4:5]
	v_lshl_add_u64 v[28:29], v[18:19], 0, v[94:95]
	global_store_short_d16_hi v[28:29], v111, off
	v_lshl_add_u64 v[28:29], v[18:19], 0, v[96:97]
	v_lshl_add_u64 v[20:21], v[18:19], 0, v[0:1]
	global_store_short_d16_hi v[28:29], v112, off
	v_lshl_add_u64 v[28:29], v[18:19], 0, v[122:123]
	global_store_short_d16_hi v[20:21], v110, off
	global_store_short_d16_hi v[28:29], v113, off
	global_store_short_d16_hi v[20:21], v102, off offset:32
	v_lshl_add_u64 v[28:29], v[18:19], 0, v[124:125]
	global_store_short_d16_hi v[28:29], v103, off
	v_lshl_add_u64 v[28:29], v[18:19], 0, v[126:127]
	global_store_short_d16_hi v[28:29], v104, off
	v_lshl_add_u64 v[28:29], v[18:19], 0, v[128:129]
	global_store_short_d16_hi v[28:29], v105, off
	v_lshl_add_u64 v[28:29], v[18:19], 0, v[130:131]
	global_store_short_d16_hi v[28:29], v106, off
	v_lshl_add_u64 v[28:29], v[18:19], 0, v[132:133]
	global_store_short_d16_hi v[28:29], v107, off
	v_lshl_add_u64 v[28:29], v[18:19], 0, v[134:135]
	global_store_short_d16_hi v[28:29], v108, off
	v_lshl_add_u64 v[28:29], v[18:19], 0, v[136:137]
	global_store_short_d16_hi v[28:29], v109, off
	v_lshl_add_u64 v[28:29], v[18:19], 0, v[138:139]
	global_store_short_d16_hi v[28:29], v82, off
	v_lshl_add_u64 v[28:29], v[18:19], 0, v[146:147]
	global_store_short_d16_hi v[28:29], v83, off
	v_lshl_add_u64 v[28:29], v[18:19], 0, v[118:119]
	global_store_short_d16_hi v[28:29], v84, off
	v_lshl_add_u64 v[28:29], v[18:19], 0, v[148:149]
	global_store_short_d16_hi v[28:29], v85, off
	v_lshl_add_u64 v[28:29], v[18:19], 0, v[120:121]
	global_store_short_d16_hi v[28:29], v90, off
	v_lshl_add_u64 v[28:29], v[18:19], 0, v[150:151]
	global_store_short_d16_hi v[28:29], v91, off
	v_lshl_add_u64 v[28:29], v[18:19], 0, v[114:115]
	global_store_short_d16_hi v[28:29], v92, off
	v_lshl_add_u64 v[28:29], v[18:19], 0, v[152:153]
	global_store_short_d16_hi v[28:29], v93, off
	v_lshl_add_u64 v[28:29], v[18:19], 0, v[116:117]
	global_store_short_d16_hi v[28:29], v66, off
	v_lshl_add_u64 v[28:29], v[18:19], 0, v[154:155]
	global_store_short_d16_hi v[28:29], v67, off
	v_lshl_add_u64 v[28:29], v[18:19], 0, v[70:71]
	global_store_short_d16_hi v[28:29], v68, off
	v_lshl_add_u64 v[28:29], v[18:19], 0, v[156:157]
	global_store_short_d16_hi v[28:29], v69, off
	v_lshl_add_u64 v[28:29], v[18:19], 0, v[72:73]
	global_store_short_d16_hi v[28:29], v78, off
	v_lshl_add_u64 v[28:29], v[18:19], 0, v[158:159]
	global_store_short_d16_hi v[28:29], v79, off
	v_lshl_add_u64 v[28:29], v[18:19], 0, v[54:55]
	global_store_short_d16_hi v[28:29], v80, off
	v_lshl_add_u64 v[28:29], v[18:19], 0, v[160:161]
	global_store_short_d16_hi v[28:29], v81, off
	v_lshl_add_u64 v[28:29], v[18:19], 0, v[56:57]
	global_store_short_d16_hi v[28:29], v62, off
	v_lshl_add_u64 v[28:29], v[18:19], 0, v[162:163]
	global_store_short_d16_hi v[28:29], v63, off
	v_lshl_add_u64 v[28:29], v[18:19], 0, v[26:27]
	global_store_short_d16_hi v[28:29], v64, off
	v_lshl_add_u64 v[28:29], v[18:19], 0, v[164:165]
	v_lshl_add_u64 v[18:19], v[18:19], 0, s[0:1]
	v_lshl_add_u64 v[0:1], v[18:19], 0, v[0:1]
	global_store_short_d16_hi v[0:1], v42, off offset:32
	v_lshl_add_u64 v[0:1], v[18:19], 0, v[124:125]
	global_store_short_d16_hi v[0:1], v43, off
	v_lshl_add_u64 v[0:1], v[18:19], 0, v[126:127]
	global_store_short_d16_hi v[0:1], v44, off
	v_lshl_add_u64 v[0:1], v[18:19], 0, v[128:129]
	global_store_short_d16_hi v[0:1], v45, off
	v_lshl_add_u64 v[0:1], v[18:19], 0, v[130:131]
	global_store_short_d16_hi v[0:1], v34, off
	v_lshl_add_u64 v[0:1], v[18:19], 0, v[132:133]
	global_store_short_d16_hi v[0:1], v35, off
	v_lshl_add_u64 v[0:1], v[18:19], 0, v[134:135]
	global_store_short_d16_hi v[0:1], v36, off
	v_lshl_add_u64 v[0:1], v[18:19], 0, v[136:137]
	global_store_short_d16_hi v[0:1], v37, off
	v_lshl_add_u64 v[0:1], v[18:19], 0, v[138:139]
	global_store_short_d16_hi v[0:1], v22, off
	v_lshl_add_u64 v[0:1], v[18:19], 0, v[146:147]
	global_store_short_d16_hi v[0:1], v23, off
	v_lshl_add_u64 v[0:1], v[18:19], 0, v[118:119]
	global_store_short_d16_hi v[0:1], v24, off
	v_lshl_add_u64 v[0:1], v[18:19], 0, v[148:149]
	global_store_short_d16_hi v[0:1], v25, off
	v_lshl_add_u64 v[0:1], v[18:19], 0, v[120:121]
	global_store_short_d16_hi v[0:1], v14, off
	v_lshl_add_u64 v[0:1], v[18:19], 0, v[150:151]
	global_store_short_d16_hi v[0:1], v15, off
	v_lshl_add_u64 v[0:1], v[18:19], 0, v[114:115]
	global_store_short_d16_hi v[0:1], v16, off
	v_lshl_add_u64 v[0:1], v[18:19], 0, v[152:153]
	global_store_short_d16_hi v[0:1], v17, off
	v_lshl_add_u64 v[0:1], v[18:19], 0, v[116:117]
	global_store_short_d16_hi v[0:1], v10, off
	v_lshl_add_u64 v[0:1], v[18:19], 0, v[154:155]
	global_store_short_d16_hi v[0:1], v11, off
	v_lshl_add_u64 v[0:1], v[18:19], 0, v[70:71]
	global_store_short_d16_hi v[0:1], v12, off
	v_lshl_add_u64 v[0:1], v[18:19], 0, v[156:157]
	global_store_short_d16_hi v[0:1], v13, off
	v_lshl_add_u64 v[0:1], v[18:19], 0, v[72:73]
	global_store_short_d16_hi v[0:1], v6, off
	v_lshl_add_u64 v[0:1], v[18:19], 0, v[158:159]
	global_store_short_d16_hi v[0:1], v7, off
	v_lshl_add_u64 v[0:1], v[18:19], 0, v[54:55]
	global_store_short_d16_hi v[0:1], v8, off
	v_lshl_add_u64 v[0:1], v[18:19], 0, v[160:161]
	global_store_short_d16_hi v[0:1], v9, off
	v_lshl_add_u64 v[0:1], v[18:19], 0, v[56:57]
	global_store_short_d16_hi v[20:21], v50, off offset:256
	v_lshl_add_u64 v[20:21], v[18:19], 0, v[94:95]
	global_store_short_d16_hi v[0:1], v2, off
	v_lshl_add_u64 v[0:1], v[18:19], 0, v[162:163]
	global_store_short_d16_hi v[20:21], v51, off
	v_lshl_add_u64 v[20:21], v[18:19], 0, v[96:97]
	global_store_short_d16_hi v[0:1], v3, off
	v_lshl_add_u64 v[0:1], v[18:19], 0, v[26:27]
	global_store_short_d16_hi v[20:21], v52, off
	v_lshl_add_u64 v[20:21], v[18:19], 0, v[122:123]
	global_store_short_d16_hi v[0:1], v4, off
	v_lshl_add_u64 v[0:1], v[18:19], 0, v[164:165]
	global_store_short_d16_hi v[28:29], v65, off
	global_store_short_d16_hi v[20:21], v53, off
	global_store_short_d16_hi v[0:1], v5, off
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z13matmul_deviceILi8192ELi8192ELi8192EEvN7kittens2glI14__hip_fp8_e4m3Li1ELi1EXT_EXT1_EJEEENS1_IS2_Li1ELi1EXT0_EXT1_EJEEENS1_I14__hip_bfloat16Li1ELi1EXT_EXT0_EJEEE
		.amdhsa_group_segment_fixed_size 131072
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 48
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_kernarg_preload_length 0
		.amdhsa_user_sgpr_kernarg_preload_offset 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 254
		.amdhsa_next_free_sgpr 96
		.amdhsa_accum_offset 256
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_fp16_overflow 0
		.amdhsa_tg_split 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.section	.text._Z13matmul_deviceILi8192ELi8192ELi8192EEvN7kittens2glI14__hip_fp8_e4m3Li1ELi1EXT_EXT1_EJEEENS1_IS2_Li1ELi1EXT0_EXT1_EJEEENS1_I14__hip_bfloat16Li1ELi1EXT_EXT0_EJEEE,"axG",@progbits,_Z13matmul_deviceILi8192ELi8192ELi8192EEvN7kittens2glI14__hip_fp8_e4m3Li1ELi1EXT_EXT1_EJEEENS1_IS2_Li1ELi1EXT0_EXT1_EJEEENS1_I14__hip_bfloat16Li1ELi1EXT_EXT0_EJEEE,comdat
.Lfunc_end0:
	.size	_Z13matmul_deviceILi8192ELi8192ELi8192EEvN7kittens2glI14__hip_fp8_e4m3Li1ELi1EXT_EXT1_EJEEENS1_IS2_Li1ELi1EXT0_EXT1_EJEEENS1_I14__hip_bfloat16Li1ELi1EXT_EXT0_EJEEE, .Lfunc_end0-_Z13matmul_deviceILi8192ELi8192ELi8192EEvN7kittens2glI14__hip_fp8_e4m3Li1ELi1EXT_EXT1_EJEEENS1_IS2_Li1ELi1EXT0_EXT1_EJEEENS1_I14__hip_bfloat16Li1ELi1EXT_EXT0_EJEEE
                                        ; -- End function
	.set _Z13matmul_deviceILi8192ELi8192ELi8192EEvN7kittens2glI14__hip_fp8_e4m3Li1ELi1EXT_EXT1_EJEEENS1_IS2_Li1ELi1EXT0_EXT1_EJEEENS1_I14__hip_bfloat16Li1ELi1EXT_EXT0_EJEEE.num_vgpr, 254
	.set _Z13matmul_deviceILi8192ELi8192ELi8192EEvN7kittens2glI14__hip_fp8_e4m3Li1ELi1EXT_EXT1_EJEEENS1_IS2_Li1ELi1EXT0_EXT1_EJEEENS1_I14__hip_bfloat16Li1ELi1EXT_EXT0_EJEEE.num_agpr, 0
	.set _Z13matmul_deviceILi8192ELi8192ELi8192EEvN7kittens2glI14__hip_fp8_e4m3Li1ELi1EXT_EXT1_EJEEENS1_IS2_Li1ELi1EXT0_EXT1_EJEEENS1_I14__hip_bfloat16Li1ELi1EXT_EXT0_EJEEE.numbered_sgpr, 25
	.set _Z13matmul_deviceILi8192ELi8192ELi8192EEvN7kittens2glI14__hip_fp8_e4m3Li1ELi1EXT_EXT1_EJEEENS1_IS2_Li1ELi1EXT0_EXT1_EJEEENS1_I14__hip_bfloat16Li1ELi1EXT_EXT0_EJEEE.private_seg_size, 0
	.set _Z13matmul_deviceILi8192ELi8192ELi8192EEvN7kittens2glI14__hip_fp8_e4m3Li1ELi1EXT_EXT1_EJEEENS1_IS2_Li1ELi1EXT0_EXT1_EJEEENS1_I14__hip_bfloat16Li1ELi1EXT_EXT0_EJEEE.uses_vcc, 1
	.set _Z13matmul_deviceILi8192ELi8192ELi8192EEvN7kittens2glI14__hip_fp8_e4m3Li1ELi1EXT_EXT1_EJEEENS1_IS2_Li1ELi1EXT0_EXT1_EJEEENS1_I14__hip_bfloat16Li1ELi1EXT_EXT0_EJEEE.uses_flat_scratch, 0
	.set _Z13matmul_deviceILi8192ELi8192ELi8192EEvN7kittens2glI14__hip_fp8_e4m3Li1ELi1EXT_EXT1_EJEEENS1_IS2_Li1ELi1EXT0_EXT1_EJEEENS1_I14__hip_bfloat16Li1ELi1EXT_EXT0_EJEEE.has_dyn_sized_stack, 0
	.set _Z13matmul_deviceILi8192ELi8192ELi8192EEvN7kittens2glI14__hip_fp8_e4m3Li1ELi1EXT_EXT1_EJEEENS1_IS2_Li1ELi1EXT0_EXT1_EJEEENS1_I14__hip_bfloat16Li1ELi1EXT_EXT0_EJEEE.has_recursion, 0
	.set _Z13matmul_deviceILi8192ELi8192ELi8192EEvN7kittens2glI14__hip_fp8_e4m3Li1ELi1EXT_EXT1_EJEEENS1_IS2_Li1ELi1EXT0_EXT1_EJEEENS1_I14__hip_bfloat16Li1ELi1EXT_EXT0_EJEEE.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 7628
; TotalNumSgprs: 31
; NumVgprs: 254
; NumAgprs: 0
; TotalNumVgprs: 254
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 131072 bytes/workgroup (compile time only)
; SGPRBlocks: 12
; VGPRBlocks: 31
; NumSGPRsForWavesPerEU: 102
; NumVGPRsForWavesPerEU: 254
; AccumOffset: 256
; Occupancy: 2
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 63
; COMPUTE_PGM_RSRC3_GFX90A:TG_SPLIT: 0
	.text
	.p2alignl 6, 3212836864
	.fill 256, 4, 3212836864
	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 0
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 0
	.text
	.type	__hip_cuid_e843e8b1c0f57c70,@object ; @__hip_cuid_e843e8b1c0f57c70
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_e843e8b1c0f57c70
__hip_cuid_e843e8b1c0f57c70:
	.byte	0                               ; 0x0
	.size	__hip_cuid_e843e8b1c0f57c70, 1

	.ident	"AMD clang version 20.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-7.0.0 25304 82aed4e69d70bef3c89c38a2ee85c8c41294dfc9)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_e843e8b1c0f57c70
	.amdgpu_metadata
---
amdhsa.kernels:
  - .agpr_count:     0
    .args:
      - .offset:         0
        .size:           16
        .value_kind:     by_value
      - .offset:         16
        .size:           16
        .value_kind:     by_value
      - .offset:         32
        .size:           16
        .value_kind:     by_value
    .group_segment_fixed_size: 131072
    .kernarg_segment_align: 8
    .kernarg_segment_size: 48
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 512
    .name:           _Z13matmul_deviceILi8192ELi8192ELi8192EEvN7kittens2glI14__hip_fp8_e4m3Li1ELi1EXT_EXT1_EJEEENS1_IS2_Li1ELi1EXT0_EXT1_EJEEENS1_I14__hip_bfloat16Li1ELi1EXT_EXT0_EJEEE
    .private_segment_fixed_size: 0
    .sgpr_count:     31
    .sgpr_spill_count: 0
    .symbol:         _Z13matmul_deviceILi8192ELi8192ELi8192EEvN7kittens2glI14__hip_fp8_e4m3Li1ELi1EXT_EXT1_EJEEENS1_IS2_Li1ELi1EXT0_EXT1_EJEEENS1_I14__hip_bfloat16Li1ELi1EXT_EXT0_EJEEE.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     254
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx950
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
