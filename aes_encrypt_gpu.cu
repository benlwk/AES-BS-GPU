/******************************************************************************
* Bitsliced implementations of AES-128 and AES-256 (encryption-only) in C using
* the barrel-shiftrows representation.
*
* See the paper at https://eprint.iacr.org/2020/1123.pdf for more details.
*
* @author 	Alexandre Adomnicai, Nanyang Technological University, Singapore
*			alexandre.adomnicai@ntu.edu.sg
*
* @date		August 2020
******************************************************************************/
#include "aes.h"
#include "kernel.h"
#include "internal-aes.h"
#include <stdio.h>
/******************************************************************************
* Packing routine to rearrange the 8 16-byte blocs (128 bytes in total) into 
* the barrel-shiftrows bitsliced representation:
* out[0] = b_0 b_32 b_64 b_96
* ...
* out[31] = b_31 b_63 b_95 b_127
******************************************************************************/
__device__ static void packing(uint32_t* out, const unsigned char* in) {
	uint32_t tmp;
	for(int i = 0; i < 8; i++) {
		out[i] 		= LE_LOAD_32(in + i*16);
		out[i+8] 	= LE_LOAD_32(in + i*16 + 4);
		out[i+16] 	= LE_LOAD_32(in + i*16 + 8);
		out[i+24] 	= LE_LOAD_32(in + i*16 + 12);
		SWAPMOVE(out[i], out[i+8], 0x00ff00ff, 8);
		SWAPMOVE(out[i+16], out[i+24], 0x00ff00ff, 8);
	}
	for(int i = 0; i < 16; i++)
		SWAPMOVE(out[i], out[i+16], 0x0000ffff, 16);
	for(int i = 0; i < 32; i+=8) {
		SWAPMOVE(out[i+1], out[i], 	0x55555555, 1);
		SWAPMOVE(out[i+3], out[i+2],0x55555555, 1);
		SWAPMOVE(out[i+5], out[i+4],0x55555555, 1);
		SWAPMOVE(out[i+7], out[i+6],0x55555555, 1);
		SWAPMOVE(out[i+2], out[i], 	0x33333333, 2);
		SWAPMOVE(out[i+3], out[i+1],0x33333333, 2);
		SWAPMOVE(out[i+6], out[i+4],0x33333333, 2);
		SWAPMOVE(out[i+7], out[i+5],0x33333333, 2);
		SWAPMOVE(out[i+4], out[i], 	0x0f0f0f0f, 4);
		SWAPMOVE(out[i+5], out[i+1],0x0f0f0f0f, 4);
		SWAPMOVE(out[i+6], out[i+2],0x0f0f0f0f, 4);
		SWAPMOVE(out[i+7], out[i+3],0x0f0f0f0f, 4);
	}
}

__device__ static void packing_word(uint32_t* out, const unsigned char* in) {
	uint32_t tmp;
	for(int i = 0; i < 8; i++) {
		out[i] 		= LE_LOAD_32_rev(in + i*16 + 0 );
		out[i+8] 	= LE_LOAD_32_rev(in + i*16 + 4);
		out[i+16] 	= LE_LOAD_32_rev(in + i*16 + 8);
		out[i+24] 	= LE_LOAD_32_rev(in + i*16 + 12);		
		SWAPMOVE(out[i], out[i+8], 0x00ff00ff, 8);
		SWAPMOVE(out[i+16], out[i+24], 0x00ff00ff, 8);
	}
	for(int i = 0; i < 16; i++)
		SWAPMOVE(out[i], out[i+16], 0x0000ffff, 16);
	for(int i = 0; i < 32; i+=8) {
		SWAPMOVE(out[i+1], out[i], 	0x55555555, 1);
		SWAPMOVE(out[i+3], out[i+2],0x55555555, 1);
		SWAPMOVE(out[i+5], out[i+4],0x55555555, 1);
		SWAPMOVE(out[i+7], out[i+6],0x55555555, 1);
		SWAPMOVE(out[i+2], out[i], 	0x33333333, 2);
		SWAPMOVE(out[i+3], out[i+1],0x33333333, 2);
		SWAPMOVE(out[i+6], out[i+4],0x33333333, 2);
		SWAPMOVE(out[i+7], out[i+5],0x33333333, 2);
		SWAPMOVE(out[i+4], out[i], 	0x0f0f0f0f, 4);
		SWAPMOVE(out[i+5], out[i+1],0x0f0f0f0f, 4);
		SWAPMOVE(out[i+6], out[i+2],0x0f0f0f0f, 4);
		SWAPMOVE(out[i+7], out[i+3],0x0f0f0f0f, 4);
	}
}


/******************************************************************************
* Unpacking routine to store the internal state in a 128-byte array.
******************************************************************************/
__device__ static void unpacking(unsigned char* out, uint32_t* in) {
	uint32_t tmp;
	for(int i = 0; i < 32; i+=8) {
		SWAPMOVE(in[i+1], in[i],	0x55555555, 1);
		SWAPMOVE(in[i+3], in[i+2],	0x55555555, 1);
		SWAPMOVE(in[i+5], in[i+4],	0x55555555, 1);
		SWAPMOVE(in[i+7], in[i+6],	0x55555555, 1);
		SWAPMOVE(in[i+2], in[i], 	0x33333333, 2);
		SWAPMOVE(in[i+3], in[i+1],	0x33333333, 2);
		SWAPMOVE(in[i+6], in[i+4],	0x33333333, 2);
		SWAPMOVE(in[i+7], in[i+5],	0x33333333, 2);
		SWAPMOVE(in[i+4], in[i], 	0x0f0f0f0f, 4);
		SWAPMOVE(in[i+5], in[i+1],	0x0f0f0f0f, 4);
		SWAPMOVE(in[i+6], in[i+2],	0x0f0f0f0f, 4);
		SWAPMOVE(in[i+7], in[i+3],	0x0f0f0f0f, 4);
	}
	for(int i = 0; i < 16; i++)
		SWAPMOVE(in[i], in[i+16], 	0x0000ffff, 16);
	for(int i = 0; i < 8; i++) {
		SWAPMOVE(in[i], in[i+8], 	0x00ff00ff, 8);
		SWAPMOVE(in[i+16], in[i+24],0x00ff00ff, 8);
		LE_STORE_32(out+i*16, 	in[i]);
		LE_STORE_32(out+i*16+4, in[i+8]);
		LE_STORE_32(out+i*16+8, in[i+16]);
		LE_STORE_32(out+i*16+12,in[i+24]);
	}
}

__device__ __forceinline__ void LE_STORE_32_rev2(uint32_t* out, uint32_t y)
{
	uint32_t tmp=0;
	// unsigned char x[4];
	// (x)[3] = (y) & 0xff; 									
	// (x)[2] = ((y) >> 8) & 0xff; 							
	// (x)[1] = ((y) >> 16) & 0xff; 							
	// (x)[0] = (y) >> 24;							
    // tmp =((((uint32_t)((x)[3])) << 24) | 						
    //  (((uint32_t)((x)[2])) << 16) | 						
    //  (((uint32_t)((x)[1])) << 8) | 							
    //   ((uint32_t)((x)[0])));
	// out[0] = tmp;
	//wklee, use + instead of |, could be better.
	tmp = ((y>>24)&0xff)+ // move byte 3 to byte 0
                    ((y<<8)&0xff0000)+ // move byte 1 to byte 2
                    ((y>>8)&0xff00)+ // move byte 2 to byte 1
                    ((y<<24)&0xff000000); // byte 0 to byte 3

	out[0] = tmp;
}


__device__ __forceinline__ void LE_STORE_32_rev3(uint32_t* out, uint32_t y)
{
	// uint32_t tmp=0;
	//wklee, use + instead of |, could be better.
	// tmp = ((y>>24)&0xff)+ // move byte 3 to byte 0
    //                 ((y<<8)&0xff0000)+ // move byte 1 to byte 2
    //                 ((y>>8)&0xff00)+ // move byte 2 to byte 1
    //                 ((y<<24)&0xff000000); // byte 0 to byte 3
	// out[0] = tmp;

    asm volatile ("{\n\t"        
        "prmt.b32 %0, %1, %2, 0x00004567;\n\t"                  
    "}"
    : "+r"(out[0]), "+r"(y), "+r"(y)) ;   
}

__device__ static void unpacking_word_prmt(uint32_t* volatile out, uint32_t* in) {
	uint32_t tmp;
	for(int i = 0; i < 32; i+=8) {
		SWAPMOVE(in[i+1], in[i],	0x55555555, 1);
		SWAPMOVE(in[i+3], in[i+2],	0x55555555, 1);
		SWAPMOVE(in[i+5], in[i+4],	0x55555555, 1);
		SWAPMOVE(in[i+7], in[i+6],	0x55555555, 1);
		SWAPMOVE(in[i+2], in[i], 	0x33333333, 2);
		SWAPMOVE(in[i+3], in[i+1],	0x33333333, 2);
		SWAPMOVE(in[i+6], in[i+4],	0x33333333, 2);
		SWAPMOVE(in[i+7], in[i+5],	0x33333333, 2);
		SWAPMOVE(in[i+4], in[i], 	0x0f0f0f0f, 4);
		SWAPMOVE(in[i+5], in[i+1],	0x0f0f0f0f, 4);
		SWAPMOVE(in[i+6], in[i+2],	0x0f0f0f0f, 4);
		SWAPMOVE(in[i+7], in[i+3],	0x0f0f0f0f, 4);
		// asm volatile ("{\n\t"
		// "shr.b32 %2, %0, 4;\n\t"
		// "lop3.b32 %2, %2, %1, 0x0f0f0f0f, 0x28;\n\t"	
		// "xor.b32 %1, %1, %2;\n\t"
		// "shl.b32 %2, %2, 4;\n\t"
		// "xor.b32 %0, %0, %2;\n\t"
		// "}"
		// : "+r"(in[i+4]), "+r"(in[i]), "+r"(tmp));
		// asm volatile ("{\n\t"
		// "shr.b32 %2, %0, 4;\n\t"
		// "lop3.b32 %2, %2, %1, 0x0f0f0f0f, 0x28;\n\t"	
		// "xor.b32 %1, %1, %2;\n\t"
		// "shl.b32 %2, %2, 4;\n\t"
		// "xor.b32 %0, %0, %2;\n\t"
		// "}"
		// : "+r"(in[i+5]), "+r"(in[i+1]), "+r"(tmp));
		// asm volatile ("{\n\t"
		// "shr.b32 %2, %0, 4;\n\t"
		// "lop3.b32 %2, %2, %1, 0x0f0f0f0f, 0x28;\n\t"	
		// "xor.b32 %1, %1, %2;\n\t"
		// "shl.b32 %2, %2, 4;\n\t"
		// "xor.b32 %0, %0, %2;\n\t"
		// "}"
		// : "+r"(in[i+6]), "+r"(in[i+2]), "+r"(tmp));			
		// asm volatile ("{\n\t"
		// "shr.b32 %2, %0, 4;\n\t"
		// "lop3.b32 %2, %2, %1, 0x0f0f0f0f, 0x28;\n\t"	
		// "xor.b32 %1, %1, %2;\n\t"
		// "shl.b32 %2, %2, 4;\n\t"
		// "xor.b32 %0, %0, %2;\n\t"
		// "}"
		// : "+r"(in[i+7]), "+r"(in[i+3]), "+r"(tmp));

	}
	for(int i = 0; i < 16; i++){
		// SWAPMOVE(in[i], in[i+16], 	0x0000ffff, 16);
		asm volatile ("{\n\t"        
	        "prmt.b32 %0, %1, %2, 0x00005410;\n\t"                  
	        "prmt.b32 %2, %1, %2, 0x00007632;\n\t"                  
	        "mov.b32 %1, %0;\n\t"          
	    "}"
	    : "+r"(tmp), "+r"(in[i]), "+r"(in[i+16])) ;   
		}
	for(int i = 0; i < 8; i++) {
		// SWAPMOVE(in[i], in[i+8], 	0x00ff00ff, 8);
		// SWAPMOVE(in[i+16], in[i+24],0x00ff00ff, 8);
	    asm volatile ("{\n\t"        
	        "prmt.b32 %0, %1, %2, 0x00006240;\n\t"                  
	        "prmt.b32 %2, %1, %2, 0x00007351;\n\t"                  
	        "mov.b32 %1, %0;\n\t"          
	        "prmt.b32 %0, %3, %4, 0x00006240;\n\t"                  
	        "prmt.b32 %4, %3, %4, 0x00007351;\n\t"                  
	        "mov.b32 %3, %0;\n\t"          	        
	    "}"
	    : "+r"(tmp), "+r"(in[i]), "+r"(in[i+8]), "+r"(in[i+16]), "+r"(in[i+24])) ;   
		
		LE_STORE_32_rev3(out+i*4*gridDim.x*blockDim.x, in[i]);
		LE_STORE_32_rev3(out+i*4*gridDim.x*blockDim.x + 1*gridDim.x*blockDim.x, in[i+8]);
		LE_STORE_32_rev3(out+i*4*gridDim.x*blockDim.x + 2*gridDim.x*blockDim.x, in[i+16]);
		LE_STORE_32_rev3(out+i*4*gridDim.x*blockDim.x + 3*gridDim.x*blockDim.x, in[i+24]);	
	}
}

__device__ static void unpacking_word(unsigned char* out, uint32_t* in) {
	uint32_t tmp;
	for(int i = 0; i < 32; i+=8) {
		SWAPMOVE(in[i+1], in[i],	0x55555555, 1);
		SWAPMOVE(in[i+3], in[i+2],	0x55555555, 1);
		SWAPMOVE(in[i+5], in[i+4],	0x55555555, 1);
		SWAPMOVE(in[i+7], in[i+6],	0x55555555, 1);
		SWAPMOVE(in[i+2], in[i], 	0x33333333, 2);
		SWAPMOVE(in[i+3], in[i+1],	0x33333333, 2);
		SWAPMOVE(in[i+6], in[i+4],	0x33333333, 2);
		SWAPMOVE(in[i+7], in[i+5],	0x33333333, 2);
		SWAPMOVE(in[i+4], in[i], 	0x0f0f0f0f, 4);
		SWAPMOVE(in[i+5], in[i+1],	0x0f0f0f0f, 4);
		SWAPMOVE(in[i+6], in[i+2],	0x0f0f0f0f, 4);
		SWAPMOVE(in[i+7], in[i+3],	0x0f0f0f0f, 4);
	}
	for(int i = 0; i < 16; i++){
		// SWAPMOVE(in[i], in[i+16], 	0x0000ffff, 16);
		asm volatile ("{\n\t"        
	        "prmt.b32 %0, %1, %2, 0x00005410;\n\t"                  
	        "prmt.b32 %2, %1, %2, 0x00007632;\n\t"                  
	        "mov.b32 %1, %0;\n\t"          
	    "}"
	    : "+r"(tmp), "+r"(in[i]), "+r"(in[i+16])) ;   
	}

	for(int i = 0; i < 8; i++) {
		// SWAPMOVE(in[i], in[i+8], 	0x00ff00ff, 8);
		// SWAPMOVE(in[i+16], in[i+24],0x00ff00ff, 8);
	    asm volatile ("{\n\t"        
	        "prmt.b32 %0, %1, %2, 0x00006240;\n\t"                  
	        "prmt.b32 %2, %1, %2, 0x00007351;\n\t"                  
	        "mov.b32 %1, %0;\n\t"          
	        "prmt.b32 %0, %3, %4, 0x00006240;\n\t"                  
	        "prmt.b32 %4, %3, %4, 0x00007351;\n\t"                  
	        "mov.b32 %3, %0;\n\t"          	        
	    "}"
	    : "+r"(tmp), "+r"(in[i]), "+r"(in[i+8]), "+r"(in[i+16]), "+r"(in[i+24])) ;   
		//wklee, the indices for out is count as bytes, not words.
		LE_STORE_32_rev(out+i*16*gridDim.x*blockDim.x, 	in[i]);
		LE_STORE_32_rev(out+i*16*gridDim.x*blockDim.x + 4*gridDim.x*blockDim.x, in[i+8]);
		LE_STORE_32_rev(out+i*16*gridDim.x*blockDim.x + 8*gridDim.x*blockDim.x, in[i+16]);
		LE_STORE_32_rev(out+i*16*gridDim.x*blockDim.x + 12*gridDim.x*blockDim.x,in[i+24]);		
	}
}


/******************************************************************************
* Bitsliced implementation of the AES Sbox based on Boyar, Peralta and Calik.
* See http://www.cs.yale.edu/homes/peralta/CircuitStuff/SLP_AES_113.txt
* Note that the 4 NOT (^= 0xffffffff) are moved to the key schedule.
* Updates only a quarter of the state (i.e. 256 bits) => need to be applied 4
* times per round when considering the barrel-shiftrows representation.
******************************************************************************/
__device__ static void sbox(uint32_t* state) {
	uint32_t t0, t1, t2, t3, t4, t5,
		t6, t7, t8, t9, t10, t11, t12,
		t13, t14, t15, t16, t17;
	t0			= state[3] ^ state[5];
	t1			= state[0] ^ state[6];
	t2			= t1 ^ t0;
	t3			= state[4] ^ t2;
	t4			= t3 ^ state[5];
	t5			= t2 & t4;
	t6			= t4 ^ state[7];
	t7			= t3 ^ state[1];
	t8			= state[0] ^ state[3]; 
	t9			= t7 ^ t8;
	t10			= t8 & t9;
	t11			= state[7] ^ t9; 
	t12			= state[0] ^ state[5];
	t13			= state[1] ^ state[2];
	t14			= t4 ^ t13;
	t15			= t14 ^ t9;
	t16			= t0 & t15;
	t17			= t16 ^ t10;
	state[1]	= t14 ^ t12; 
	state[2]	= t12 & t14;
	state[2] 	^= t10;
	state[4]	= t13 ^ t9;
	state[5]	= t1 ^ state[4];
	t3			= t1 & state[4];
	t10			= state[0] ^ state[4];
	t13 		^= state[7];
	state[3] 	^= t13; 
	t16			= state[3] & state[7];
	t16 		^= t5;
	t16 		^= state[2];
	state[1] 	^= t16;
	state[0] 	^= t13;
	t16			= state[0] & t11;
	t16 		^= t3;
	state[2] 	^= t16;
	state[2] 	^= t10;
	state[6] 	^= t13;
	t10			= state[6] & t13;
	t3 			^= t10;
	t3 			^= t17;
	state[5] 	^= t3;
	t3			= state[6] ^ t12;
	t10			= t3 & t6;
	t5 			^= t10;
	t5 			^= t7;
	t5 			^= t17;
	t7			= t5 & state[5];
	t10			= state[2] ^ t7;
	t7 			^= state[1];
	t5 			^= state[1];
	t16			= t5 & t10;
	state[1] 	^= t16;
	t17			= state[1] & state[0];
	t11			= state[1] & t11;
	t16			= state[5] ^ state[2];
	t7 			&= t16;
	t7 			^= state[2];
	t16			= t10 ^ t7;
	state[2] 	&= t16;
	t10 		^= state[2];
	t10 		&= state[1];
	t5 			^= t10;
	t10			= state[1] ^ t5;
	state[4] 	&= t10; 
	t11 		^= state[4];
	t1 			&= t10;
	state[6] 	&= t5; 
	t10			= t5 & t13;
	state[4] 	^= t10;
	state[5] 	^= t7;
	state[2] 	^= state[5];
	state[5]	= t5 ^ state[2];
	t5			= state[5] & t14;
	t10			= state[5] & t12;
	t12			= t7 ^ state[2];
	t4 			&= t12;
	t2 			&= t12;
	t3 			&= state[2]; 
	state[2] 	&= t6;
	state[2] 	^= t4;
	t13			= state[4] ^ state[2];
	state[3] 	&= t7;
	state[1] 	^= t7;
	state[5] 	^= state[1];
	t6			= state[5] & t15;
	state[4] 	^= t6; 
	t0 			&= state[5];
	state[5]	= state[1] & t9; 
	state[5] 	^= state[4];
	state[1] 	&= t8;
	t6			= state[1] ^ state[5];
	t0 			^= state[1];
	state[1]	= t3 ^ t0;
	t15			= state[1] ^ state[3];
	t2 			^= state[1];
	state[0]	= t2 ^ state[5];
	state[3]	= t2 ^ t13;
	state[1]	= state[3] ^ state[5];
	//state[1] 	^= 0xffffffff;
	t0 			^= state[6];
	state[5]	= t7 & state[7];
	t14			= t4 ^ state[5];
	state[6]	= t1 ^ t14;
	state[6] 	^= t5; 
	state[6] 	^= state[4];
	state[2]	= t17 ^ state[6];
	state[5]	= t15 ^ state[2];
	state[2] 	^= t6;
	state[2] 	^= t10;
	//state[2] 	^= 0xffffffff;
	t14 		^= t11;
	t0 			^= t14;
	state[6] 	^= t0;
	//state[6] 	^= 0xffffffff;
	state[7]	= t1 ^ t0;
	//state[7] 	^= 0xffffffff;
	state[4]	= t14 ^ state[3]; 
}

__device__ static void sbox_debug(uint32_t* state) {
	uint32_t t0, t1, t2, t3, t4, t5,
		t6, t7, t8, t9, t10, t11, t12,
		t13, t14, t15, t16, t17;
	t0			= state[3] ^ state[5];
	t1			= state[0] ^ state[6];
	t2			= t1 ^ t0;
	t3			= state[4] ^ t2;
	t4			= t3 ^ state[5];
	t5			= t2 & t4;
	t6			= t4 ^ state[7];
	t7			= t3 ^ state[1];
	t8			= state[0] ^ state[3]; 
	t9			= t7 ^ t8;
	t10			= t8 & t9;
	t11			= state[7] ^ t9; 
	t12			= state[0] ^ state[5];
	t13			= state[1] ^ state[2];
	t14			= t4 ^ t13;
	t15			= t14 ^ t9;
	t16			= t0 & t15;
	t17			= t16 ^ t10;
	state[1]	= t14 ^ t12; 
	state[2]	= t12 & t14;
	state[2] 	^= t10;
	state[4]	= t13 ^ t9;
	state[5]	= t1 ^ state[4];
	t3			= t1 & state[4];
	t10			= state[0] ^ state[4];
	t13 		^= state[7];
	state[3] 	^= t13; 
	t16			= state[3] & state[7];
	t16 		^= t5;
	t16 		^= state[2];
	state[1] 	^= t16;
	state[0] 	^= t13;
	t16			= state[0] & t11;
	t16 		^= t3;
	state[2] 	^= t16;
	state[2] 	^= t10;
	state[6] 	^= t13;
	t10			= state[6] & t13;
	t3 			^= t10;
	t3 			^= t17;
	state[5] 	^= t3;
	t3			= state[6] ^ t12;
	t10			= t3 & t6;
	t5 			^= t10;
	t5 			^= t7;
	t5 			^= t17;
	t7			= t5 & state[5];
	t10			= state[2] ^ t7;
	t7 			^= state[1];
	t5 			^= state[1];
	t16			= t5 & t10;
	state[1] 	^= t16;
	t17			= state[1] & state[0];
	t11			= state[1] & t11;
	t16			= state[5] ^ state[2];
	t7 			&= t16;
	t7 			^= state[2];
	t16			= t10 ^ t7;
	state[2] 	&= t16;
	t10 		^= state[2];
	t10 		&= state[1];
	t5 			^= t10;
	t10			= state[1] ^ t5;
	state[4] 	&= t10; 
	t11 		^= state[4];
	t1 			&= t10;
	state[6] 	&= t5; 
	t10			= t5 & t13;
	state[4] 	^= t10;
	state[5] 	^= t7;
	state[2] 	^= state[5];
	state[5]	= t5 ^ state[2];
	t5			= state[5] & t14;
	t10			= state[5] & t12;
	t12			= t7 ^ state[2];
	t4 			&= t12;
	t2 			&= t12;
	t3 			&= state[2]; 
	state[2] 	&= t6;
	state[2] 	^= t4;
	t13			= state[4] ^ state[2];
	state[3] 	&= t7;
	state[1] 	^= t7;
	state[5] 	^= state[1];
	t6			= state[5] & t15;
	state[4] 	^= t6; 
	t0 			&= state[5];
	state[5]	= state[1] & t9; 
	state[5] 	^= state[4];
	state[1] 	&= t8;
	t6			= state[1] ^ state[5];
	t0 			^= state[1];
	state[1]	= t3 ^ t0;
	t15			= state[1] ^ state[3];
	t2 			^= state[1];
	state[0]	= t2 ^ state[5];
	state[3]	= t2 ^ t13;
	state[1]	= state[3] ^ state[5];
	//state[1] 	^= 0xffffffff;
	t0 			^= state[6];
	state[5]	= t7 & state[7];
	t14			= t4 ^ state[5];
	state[6]	= t1 ^ t14;
	state[6] 	^= t5; 
	state[6] 	^= state[4];
	state[2]	= t17 ^ state[6];
	state[5]	= t15 ^ state[2];
	state[2] 	^= t6;
	state[2] 	^= t10;
	//state[2] 	^= 0xffffffff;
	t14 		^= t11;
	t0 			^= t14;
	state[6] 	^= t0;
	//state[6] 	^= 0xffffffff;
	state[7]	= t1 ^ t0;
	//state[7] 	^= 0xffffffff;
	state[4]	= t14 ^ state[3]; 
}
/******************************************************************************
* ShiftRows on the entire 1024-bit internal state.
******************************************************************************/
__device__ static void shiftrows(uint32_t* state) {
	uint32_t tmp;
	for(int i = 8; i < 16; i++) {		// shifts the 2nd row
		state[i] = ROR(state[i],8); 	// shifts the 2nd row
		// asm volatile ("{\n\t"
		// "shf.r.clamp.b32 %0, %0, %0, 8;\n\t"
		// "}"
		// : "+r"(state[i]));
		// asm volatile ("{\n\t"
		// 	"prmt.b32 %0, %1, %1, 0x00000321;\n\t"
		// 	"mov.b32 %1, %0;\n\t"
		// 	"}"
		// 	: "+r"(tmp), "+r"(state[i]));
	}
	for(int i = 16; i < 24; i++){
		state[i] = ROR(state[i],16); 	// shifts the 3rd row
		// asm volatile ("{\n\t"
		// 	"prmt.b32 %0, %1, %1, 0x00001032;\n\t"
		// 	"mov.b32 %1, %0;\n\t"
		// 	"}"
		// 	: "+r"(tmp), "+r"(state[i]));
	}
	for(int i = 24; i < 32; i++){ 		// shifts the 4th row
		state[i] = ROR(state[i],24); 	// shifts the 4th row
		// asm volatile ("{\n\t"
		// "shf.r.clamp.b32 %0, %0, %0, 24;\n\t"
		// "}"
		// : "+r"(state[i]));		
		// asm volatile ("{\n\t"
		// 	"prmt.b32 %0, %1, %1, 0x00002103;\n\t"
		// 	"mov.b32 %1, %0;\n\t"
		// 	"}"
		// 	: "+r"(state[i]), "+r"(state[i]));
	}
}

__device__ static void shiftrows_half(uint32_t* state) {
	// for(int i = 8; i < 16; i++) 		// shifts the 2nd row
	// 	state[i] = ROR(state[i],8); 	// shifts the 2nd row
	for(int i = 16; i < 24; i++) 		// shifts the 3rd row
		state[i] = ROR(state[i],16); 	// shifts the 3rd row
	for(int i = 24; i < 32; i++) 		// shifts the 4th row
		state[i] = ROR(state[i],24); 	// shifts the 4th row
}

/******************************************************************************
* MixColumns on the entire 1024-bit internal state.
******************************************************************************/
__device__ static void mixcolumns(uint32_t* state) {
	uint32_t tmp2_0, tmp2_1, tmp2_2, tmp2_3;
	uint32_t tmp, tmp_bis, tmp0_0, tmp0_1, tmp0_2, tmp0_3;
	uint32_t tmp1_0, tmp1_1, tmp1_2, tmp1_3;
	tmp2_0 = state[0] ^ state[8];
	tmp2_1 = state[8] ^ state[16];
	tmp2_2 = state[16] ^ state[24];
	tmp2_3 = state[24] ^ state[0];

	tmp0_0 = state[7] ^ state[15];
	tmp0_1 = state[15] ^ state[23];
	tmp0_2 = state[23] ^ state[31];
	tmp0_3 = state[31]^ state[7];
	tmp = state[7];
	state[7] = tmp2_0 ^ tmp0_2 ^ state[15];
	state[15] = tmp2_1 ^ tmp0_2 ^ tmp;
	tmp = state[23];
	state[23] = tmp2_2 ^ tmp0_0 ^ state[31];
	state[31] = tmp2_3 ^ tmp0_0 ^ tmp;
	tmp1_0 = state[6] ^ state[14];
	tmp1_1 = state[14] ^ state[22];
	tmp1_2 = state[22] ^ state[30];
	tmp1_3 = state[30] ^ state[6]; 
	tmp = state[6];
	state[6] = tmp0_0 ^ tmp2_0 ^ state[14] ^ tmp1_2;
	tmp_bis = state[14];
	state[14] =  tmp0_1 ^ tmp2_1 ^ tmp1_2 ^ tmp;
	tmp = state[22];
	state[22] = tmp0_2 ^ tmp2_2 ^ tmp1_3 ^ tmp_bis;
	state[30] =  tmp0_3 ^ tmp2_3 ^ tmp1_0 ^ tmp;
	tmp0_0 = state[5] ^ state[13];
	tmp0_1 = state[13] ^ state[21];
	tmp0_2 = state[21] ^ state[29];
	tmp0_3 = state[29]^ state[5];
	tmp = state[5];
	state[5] = tmp1_0 ^ tmp0_1 ^ state[29];
	tmp_bis = state[13];
	state[13] = tmp1_1 ^ tmp0_2 ^ tmp;
	tmp = state[21];
	state[21] =  tmp1_2 ^ tmp0_3 ^ tmp_bis;
	state[29] = tmp1_3 ^ tmp0_0 ^ tmp;
	tmp1_0 = state[4] ^ state[12];
	tmp1_1 = state[12] ^ state[20];
	tmp1_2 = state[20] ^ state[28];
	tmp1_3 = state[28] ^ state[4];
	tmp = state[4];
	state[4] = tmp0_0 ^ tmp2_0 ^ tmp1_1 ^ state[28];	
	tmp_bis = state[12];
	state[12] = tmp0_1 ^ tmp2_1 ^ tmp1_2 ^ tmp;
	tmp = state[20];
	state[20] = tmp0_2 ^ tmp2_2 ^ tmp1_3 ^ tmp_bis;
	state[28] = tmp0_3 ^ tmp2_3 ^ tmp1_0 ^ tmp;
	tmp0_0 = state[3] ^ state[11];
	tmp0_1 = state[11] ^ state[19];
	tmp0_2 = state[19] ^ state[27];
	tmp0_3 = state[27]^ state[3];
	tmp = state[3];
	state[3] = tmp1_0 ^ tmp2_0 ^ tmp0_1 ^ state[27];
	tmp_bis = state[11];
	state[11] = tmp1_1 ^ tmp2_1 ^ tmp0_2 ^ tmp;
	tmp = state[19];
	state[19] = tmp1_2 ^ tmp2_2 ^ tmp0_3 ^ tmp_bis;
	state[27] =  tmp1_3 ^ tmp2_3 ^ tmp0_0 ^ tmp;
	tmp1_0 = state[2] ^ state[10];
	tmp1_1 = state[10] ^ state[18];
	tmp1_2 = state[18] ^ state[26];
	tmp1_3 = state[26] ^ state[2];
	tmp = state[2];
	state[2] = tmp0_0 ^ tmp1_1 ^ state[26];
	tmp_bis = state[10];
	state[10] = tmp0_1 ^ tmp1_2 ^ tmp;
	tmp = state[18];
	state[18] = tmp0_2 ^ tmp1_3 ^ tmp_bis;
	state[26] = tmp0_3 ^ tmp1_0 ^ tmp;
	tmp0_0 = state[1] ^ state[9];
	tmp0_1 = state[9] ^ state[17];
	tmp0_2 = state[17] ^ state[25];
	tmp0_3 = state[25]^ state[1];
	tmp = state[1];
	state[1] = tmp1_0 ^ tmp0_1 ^ state[25];
	tmp_bis = state[9];
	state[9] = tmp1_1 ^ tmp0_2 ^ tmp;
	tmp = state[17];
	state[17] = tmp1_2 ^ tmp0_3 ^ tmp_bis;
	state[25] =  tmp1_3 ^ tmp0_0 ^ tmp;
	tmp = state[0];
	state[0] = tmp0_0 ^ tmp2_1 ^ state[24];
	tmp_bis = state[8];
	state[8] = tmp0_1 ^ tmp2_2 ^ tmp;
	tmp = state[16];
	state[16] = tmp0_2 ^ tmp2_3 ^ tmp_bis;
	state[24] = tmp0_3 ^ tmp2_0 ^ tmp;
}

__device__ static void mixcolumns_half(uint32_t* state) {
	uint32_t tmp2_0, tmp2_1, tmp2_2, tmp2_3;
	uint32_t tmp, tmp_bis, tmp0_0, tmp0_1, tmp0_2, tmp0_3;
	uint32_t tmp1_0, tmp1_1, tmp1_2, tmp1_3;
	tmp2_0 = state[0] ^ state[8];
	tmp2_1 = state[8] ^ state[16];
	tmp2_2 = state[16] ^ state[24];
	tmp2_3 = state[24] ^ state[0];

	tmp0_0 = state[7] ^ state[15];
	tmp0_1 = state[15] ^ state[23];
	tmp0_2 = state[23] ^ state[31];
	tmp0_3 = state[31]^ state[7];
	tmp = state[7];
	// state[7] = tmp2_0 ^ tmp0_2 ^ state[15];
	// state[15] = tmp2_1 ^ tmp0_2 ^ tmp;
	tmp = state[23];
	state[23] = tmp2_2 ^ tmp0_0 ^ state[31];
	state[31] = tmp2_3 ^ tmp0_0 ^ tmp;
	tmp1_0 = state[6] ^ state[14];
	tmp1_1 = state[14] ^ state[22];
	tmp1_2 = state[22] ^ state[30];
	tmp1_3 = state[30] ^ state[6]; 
	tmp = state[6];
	// state[6] = tmp0_0 ^ tmp2_0 ^ state[14] ^ tmp1_2;
	tmp_bis = state[14];
	// state[14] =  tmp0_1 ^ tmp2_1 ^ tmp1_2 ^ tmp;
	tmp = state[22];
	state[22] = tmp0_2 ^ tmp2_2 ^ tmp1_3 ^ tmp_bis;
	state[30] =  tmp0_3 ^ tmp2_3 ^ tmp1_0 ^ tmp;
	tmp0_0 = state[5] ^ state[13];
	tmp0_1 = state[13] ^ state[21];
	tmp0_2 = state[21] ^ state[29];
	tmp0_3 = state[29]^ state[5];
	tmp = state[5];
	// state[5] = tmp1_0 ^ tmp0_1 ^ state[29];
	tmp_bis = state[13];
	// state[13] = tmp1_1 ^ tmp0_2 ^ tmp;
	tmp = state[21];
	state[21] =  tmp1_2 ^ tmp0_3 ^ tmp_bis;
	state[29] = tmp1_3 ^ tmp0_0 ^ tmp;
	tmp1_0 = state[4] ^ state[12];
	tmp1_1 = state[12] ^ state[20];
	tmp1_2 = state[20] ^ state[28];
	tmp1_3 = state[28] ^ state[4];
	tmp = state[4];
	// state[4] = tmp0_0 ^ tmp2_0 ^ tmp1_1 ^ state[28];	
	tmp_bis = state[12];
	// state[12] = tmp0_1 ^ tmp2_1 ^ tmp1_2 ^ tmp;
	tmp = state[20];
	state[20] = tmp0_2 ^ tmp2_2 ^ tmp1_3 ^ tmp_bis;
	state[28] = tmp0_3 ^ tmp2_3 ^ tmp1_0 ^ tmp;
	tmp0_0 = state[3] ^ state[11];
	tmp0_1 = state[11] ^ state[19];
	tmp0_2 = state[19] ^ state[27];
	tmp0_3 = state[27]^ state[3];
	tmp = state[3];
	// state[3] = tmp1_0 ^ tmp2_0 ^ tmp0_1 ^ state[27];
	tmp_bis = state[11];
	// state[11] = tmp1_1 ^ tmp2_1 ^ tmp0_2 ^ tmp;
	tmp = state[19];
	state[19] = tmp1_2 ^ tmp2_2 ^ tmp0_3 ^ tmp_bis;
	state[27] =  tmp1_3 ^ tmp2_3 ^ tmp0_0 ^ tmp;
	tmp1_0 = state[2] ^ state[10];
	tmp1_1 = state[10] ^ state[18];
	tmp1_2 = state[18] ^ state[26];
	tmp1_3 = state[26] ^ state[2];
	tmp = state[2];
	// state[2] = tmp0_0 ^ tmp1_1 ^ state[26];
	tmp_bis = state[10];
	// state[10] = tmp0_1 ^ tmp1_2 ^ tmp;
	tmp = state[18];
	state[18] = tmp0_2 ^ tmp1_3 ^ tmp_bis;
	state[26] = tmp0_3 ^ tmp1_0 ^ tmp;
	tmp0_0 = state[1] ^ state[9];
	tmp0_1 = state[9] ^ state[17];
	tmp0_2 = state[17] ^ state[25];
	tmp0_3 = state[25]^ state[1];
	tmp = state[1];
	// state[1] = tmp1_0 ^ tmp0_1 ^ state[25];
	tmp_bis = state[9];
	// state[9] = tmp1_1 ^ tmp0_2 ^ tmp;
	tmp = state[17];
	state[17] = tmp1_2 ^ tmp0_3 ^ tmp_bis;
	state[25] =  tmp1_3 ^ tmp0_0 ^ tmp;
	tmp = state[0];
	// state[0] = tmp0_0 ^ tmp2_1 ^ state[24];
	tmp_bis = state[8];
	// state[8] = tmp0_1 ^ tmp2_2 ^ tmp;
	tmp = state[16];
	state[16] = tmp0_2 ^ tmp2_3 ^ tmp_bis;
	state[24] = tmp0_3 ^ tmp2_0 ^ tmp;
}

/******************************************************************************
* AddRoundKey on the entire 1024-bit internal state.
******************************************************************************/
__device__ static void ark(uint32_t* state, const uint32_t* rkey) {
	for(int i = 0; i < 32; i++)
		state[i] ^= rkey[i];
}

__device__ static void ark_half(uint32_t* state, const uint32_t* rkey) {
	for(int i = 16; i < 32; i++)
		state[i] ^= rkey[i];
}

/******************************************************************************
* Encryption of 8 128-bit blocks of data in parallel using AES-128 with the
* barrel-shiftrows representation.
* The round keys are assumed to be pre-computed.
******************************************************************************/
__global__ void aes128_encrypt_gpu(unsigned char* out, const unsigned char* in,	const uint32_t* rkeys) {
	uint32_t tid = blockDim.x*blockIdx.x + threadIdx.x;
	uint32_t state[32]; 	// 1024-bit state (8 blocks)
	// 1024-bit state (8 blocks)
	// __shared__ uint32_t rkeys[320];
	// uint32_t counter[32];
	uint32_t tmp;

	// if(tid<320)	rkeys[tid] = g_rkeys[tid];
	__syncthreads();
	// packing(state, in + tid); // From bytes to the barrel-shiftrows
	// packing(state, in + tid*128); 	
	state[3] = tid*8; 	state[7] = tid*8+1;
	state[11] = tid*8+2;	state[15] = tid*8+3;
	state[19] = tid*8+4;	state[23] = tid*8+5;
	state[27] = tid*8+6;	state[31] = tid*8+7;	        

	packing_word(state, (uint8_t*) state); 	
	// packing(state, (uint8_t*) counter); 
	for(int i = 0; i < 10; i++) {
		ark(state, rkeys+i*32);	// AddRoundKey on the entire state
		sbox(state); 			// S-box on the 1st quarter state
		sbox(state + 8); 		// S-box on the 2nd quarter state
		sbox(state + 16); 		// S-box on the 3rd quarter state
		sbox(state + 24); 		// S-box on the 4th quarter state
	    shiftrows(state); 		// ShiftRows on the entire state
	    if (i != 9) 			// No MixColumns in the last round
			mixcolumns(state);	// MixColumns on the entire state
	}	
	ark(state, rkeys+320); 		// AddRoundKey on the entire state
	// unpacking(out + tid*128, state); // From barrel-shiftrows to bytes
	// unpacking_word(out + tid*128, state);
	unpacking_word(out + 4*tid, state);
}

// __launch_bounds__(1024, 16)
__global__ void aes128_encrypt_gpu_repeat(unsigned char* out, const unsigned char* in,	const uint32_t* g_rkeys) {
	uint64_t tid = blockDim.x*blockIdx.x + threadIdx.x;
	uint32_t state[32] = {0}; 	// 1024-bit state (8 AES blocks)	
	// uint32_t counter[32] = {0};
	// uint32_t debug[32]={0};
	uint32_t ctr_repeat, idx;
	__shared__ uint32_t rkeys[352];
	if(threadIdx.x<352)	rkeys[threadIdx.x] = g_rkeys[threadIdx.x];
	__syncthreads();

	for(int j=0; j<REPEATBS; j++)
	{
		for(int k=0; k<5; k++) state[k] = 0;
		ctr_repeat = 8*blockDim.x*gridDim.x;
		idx = j*ctr_repeat + tid;
		state[31] = 0x55000000;		state[30] = 0x33000000;
		state[29] = 0x0F000000;		state[28] = (idx&1) * 0xFF000000;
		state[27] = ((idx>>1)&1) * 0xFF000000;
		state[26] = ((idx>>2)&1) * 0xFF000000;
		state[25] = ((idx>>3)&1) * 0xFF000000;
		state[24] = ((idx>>4)&1) * 0xFF000000;
		state[23] = ((idx>>5)&1) * 0xFF000000;
		state[22] = ((idx>>6)&1) * 0xFF000000;
		state[21] = ((idx>>7)&1) * 0xFF000000;
		state[20] = ((idx>>8)&1) * 0xFF000000;
		state[19] = ((idx>>9)&1) * 0xFF000000;
		state[18] = ((idx>>10)&1) * 0xFF000000;
		state[17] = ((idx>>11)&1) * 0xFF000000;
		state[16] = ((idx>>12)&1) * 0xFF000000;
		state[15] = ((idx>>13)&1) * 0xFF000000;
		state[14] = ((idx>>14)&1) * 0xFF000000;
		state[13] = ((idx>>15)&1) * 0xFF000000;
		state[12] = ((idx>>16)&1) * 0xFF000000;
		state[11] = ((idx>>17)&1) * 0xFF000000;
		state[10] = ((idx>>18)&1) * 0xFF000000; 
		state[9] = ((idx>>19)&1) * 0xFF000000;	
		state[8] = ((idx>>20)&1) * 0xFF000000;
		state[7] = ((idx>>21)&1) * 0xFF000000;
		state[6] = ((idx>>25)&1) * 0xFF000000;
		state[5] = ((idx>>26)&1) * 0xFF000000;
		// wklee, for a larger number of blocks.
		// state[4] = ((idx>>27)&1) * 0xFF000000;
		// state[3] = ((idx>>28)&1) * 0xFF000000;

		// counter[3] = j*ctr_repeat + tid*8; 	
		// counter[7] = j*ctr_repeat + tid*8+1;
		// counter[11] = j*ctr_repeat + tid*8+2;	
		// counter[15] = j*ctr_repeat + tid*8+3;
		// counter[19] = j*ctr_repeat + tid*8+4;	
		// counter[23] = j*ctr_repeat + tid*8+5;
		// counter[27] = j*ctr_repeat + tid*8+6;	
		// counter[31] = j*ctr_repeat + tid*8+7;	
		// packing_word(state, (uint8_t*) counter); 	
		
		ark(state, rkeys);	// AddRoundKey on the entire state
		sbox(state); // S-box on the 1st quarter state
		sbox(state + 8); // S-box on the 2nd quarter state		
		sbox(state + 16); 	// S-box on the 3rd quarter state
		sbox(state + 24); 	// S-box on the 4th quarter state
	    shiftrows(state); 	    
		mixcolumns(state);// MixColumns on the entire state

		ark(state, rkeys+1*32);	// AddRoundKey on the entire state
		sbox(state); // S-box on the 1st quarter state
		sbox(state + 8); // S-box on the 2nd quarter state
		sbox(state + 16); 	// S-box on the 3rd quarter state
		sbox(state + 24); 	// S-box on the 4th quarter state
	    shiftrows(state); 	// ShiftRows on the entire state
		mixcolumns(state);// MixColumns on the entire state		

		for(int i = 2; i < 7; i=i+3) {
			ark(state, rkeys+i*32);	// AddRoundKey on the entire state
			sbox(state); // S-box on the 1st quarter state
			sbox(state + 8); // S-box on the 2nd quarter state
			sbox(state + 16); 	// S-box on the 3rd quarter state
			sbox(state + 24); 	// S-box on the 4th quarter state
		    shiftrows(state); 	// ShiftRows on the entire state
			// if (i != 9) 
			mixcolumns(state);// MixColumns on the entire state

			ark(state, rkeys+(i+1)*32);	// AddRoundKey on the entire state
			sbox(state); // S-box on the 1st quarter state
			sbox(state + 8); // S-box on the 2nd quarter state
			sbox(state + 16); 	// S-box on the 3rd quarter state
			sbox(state + 24); 	// S-box on the 4th quarter state
		    shiftrows(state); 	// ShiftRows on the entire state
			mixcolumns(state);// MixColumns on the entire state		
			ark(state, rkeys+(i+2)*32);	// AddRoundKey on the entire state
			sbox(state); // S-box on the 1st quarter state
			sbox(state + 8); // S-box on the 2nd quarter state
			sbox(state + 16); 	// S-box on the 3rd quarter state
			sbox(state + 24); 	// S-box on the 4th quarter state
		    shiftrows(state); 	// ShiftRows on the entire state
		    mixcolumns(state);// MixColumns on the entire state	
		}	
		ark(state, rkeys+8*32);	// AddRoundKey on the entire state
		sbox(state); // S-box on the 1st quarter state
		sbox(state + 8); // S-box on the 2nd quarter state
		sbox(state + 16); 	// S-box on the 3rd quarter state
		sbox(state + 24); 	// S-box on the 4th quarter state
	    shiftrows(state); 	// ShiftRows on the entire state
	    mixcolumns(state);// MixColumns on the entire state	

		ark(state, rkeys+9*32);	// AddRoundKey on the entire state
		sbox(state); // S-box on the 1st quarter state
		sbox(state + 8); // S-box on the 2nd quarter state
		sbox(state + 16); 	// S-box on the 3rd quarter state
		sbox(state + 24); 	// S-box on the 4th quarter state
	    shiftrows(state); 	// ShiftRows on the entire state

		ark(state, rkeys+320); 	// AddRoundKey on the entire state
		unpacking_word(out + j*ctr_repeat*16 + 4*tid, state);
		//// unpacking_word(out + j*ctr_repeat*16 + tid*128, state);
	}
}


// __launch_bounds__(256, 16)
__global__ void aes128_encrypt_gpu_repeat_coalesced(uint32_t* out, const unsigned char* in,	const uint32_t* g_rkeys, uint32_t *nonce) {
	uint64_t tid = blockDim.x*blockIdx.x + threadIdx.x;
	uint32_t state[32] = {0}; 	// 1024-bit state (8 AES)	
	uint32_t ctr_repeat, idx;
	__shared__ uint32_t rkeys[352];
#if (threadSizeBS==64)
{
	rkeys[threadIdx.x] = g_rkeys[threadIdx.x];
	rkeys[threadIdx.x+64] = g_rkeys[threadIdx.x+64];	
	rkeys[threadIdx.x+128] = g_rkeys[threadIdx.x+128];	
	rkeys[threadIdx.x+192] = g_rkeys[threadIdx.x+192];	
	rkeys[threadIdx.x+256] = g_rkeys[threadIdx.x+256];	
	if(threadIdx.x<32)	
		rkeys[threadIdx.x+320] = g_rkeys[threadIdx.x+320];
}	
#elif(threadSizeBS==128)
{
	rkeys[threadIdx.x] = g_rkeys[threadIdx.x];
	rkeys[threadIdx.x+128] = g_rkeys[threadIdx.x+128];	
	if(threadIdx.x<96)	
		rkeys[threadIdx.x+256] = g_rkeys[threadIdx.x+256];
}	
#elif (threadSizeBS==256)
{
	rkeys[threadIdx.x] = g_rkeys[threadIdx.x];
	if(threadIdx.x<96)	
		rkeys[threadIdx.x+256] = g_rkeys[threadIdx.x+256];
}	
#elif (threadSizeBS==512)
{
	if(threadIdx.x<352)	
		rkeys[threadIdx.x] = g_rkeys[threadIdx.x];
}
#endif
	
	__syncthreads();
// #pragma unroll
	for(int j=0; j<REPEATBS; j++)
	{
		for(int k=0; k<5; k++) state[k] = 0;
		ctr_repeat = 8*blockDim.x*gridDim.x;// 8 AES per thread
		idx = j*ctr_repeat + tid;
		for (int i = 0; i < 5; ++i) state[i] = nonce[i];	
		state[31] = 0x55000000 | nonce[31];		
		state[30] = 0x33000000 | nonce[30];
		state[29] = 0x0F000000 | nonce[29];		
		state[28] = (idx&1) * 0xFF000000 | nonce[28];
		state[27] = ((idx>>1)&1) * 0xFF000000 | nonce[27];
		state[26] = ((idx>>2)&1) * 0xFF000000 | nonce[26];
		state[25] = ((idx>>3)&1) * 0xFF000000 | nonce[25];
		state[24] = ((idx>>4)&1) * 0xFF000000 | nonce[24];
		state[23] = ((idx>>5)&1) * 0xFF000000 | nonce[23];
		state[22] = ((idx>>6)&1) * 0xFF000000 | nonce[22];
		state[21] = ((idx>>7)&1) * 0xFF000000 | nonce[21];
		state[20] = ((idx>>8)&1) * 0xFF000000 | nonce[20];
		state[19] = ((idx>>9)&1) * 0xFF000000 | nonce[19];
		state[18] = ((idx>>10)&1) * 0xFF000000 | nonce[18];
		state[17] = ((idx>>11)&1) * 0xFF000000 | nonce[17];
		state[16] = ((idx>>12)&1) * 0xFF000000 | nonce[16];
		state[15] = ((idx>>13)&1) * 0xFF000000 | nonce[15];
		state[14] = ((idx>>14)&1) * 0xFF000000 | nonce[14];
		state[13] = ((idx>>15)&1) * 0xFF000000 | nonce[13];
		state[12] = ((idx>>16)&1) * 0xFF000000 | nonce[12];
		state[11] = ((idx>>17)&1) * 0xFF000000 | nonce[11];
		state[10] = ((idx>>18)&1) * 0xFF000000 | nonce[10]; 
		state[9] = ((idx>>19)&1) * 0xFF000000 | nonce[9];	
		state[8] = ((idx>>20)&1) * 0xFF000000 | nonce[8];
		state[7] = ((idx>>21)&1) * 0xFF000000 | nonce[7];
		state[6] = ((idx>>25)&1) * 0xFF000000 | nonce[6];
		state[5] = ((idx>>26)&1) * 0xFF000000 | nonce[5];
				
#pragma unroll
		for(int i = 0; i < 9; i++) {
			ark(state, rkeys+i*32);	// AddRoundKey on the entire state
			sbox(state); // S-box on the 1st quarter state
			sbox(state + 8); // S-box on the 2nd quarter state
			sbox(state + 16); 	// S-box on the 3rd quarter state
			sbox(state + 24); 	// S-box on the 4th quarter state
		    shiftrows(state); 	// ShiftRows on the entire state
			mixcolumns(state);// MixColumns on the entire state
		}	

		ark(state, rkeys+9*32);	// AddRoundKey on the entire state
		sbox(state); // S-box on the 1st quarter state
		sbox(state + 8); // S-box on the 2nd quarter state
		sbox(state + 16); 	// S-box on the 3rd quarter state
		sbox(state + 24); 	// S-box on the 4th quarter state
	    shiftrows(state); 	// ShiftRows on the entire state

		ark(state, rkeys+320); 	// AddRoundKey on the entire state
		unpacking_word_prmt(out + j*ctr_repeat*4 + 1*tid, state);
		//// unpacking_word(out + j*ctr_repeat*16 + tid*128, state);
	}
}


/******************************************************************************
* Encryption of 8 128-bit blocks of data in parallel using AES-256 with the
* barrel-shiftrows representation.
* The round keys are assumed to be pre-computed.
******************************************************************************/
__global__ void aes256_encrypt_gpu(unsigned char* out, const unsigned char* in,
				const uint32_t* rkeys) {
	uint32_t state[32]; 				// 1024-bit state (8 blocks in //)
	packing(state, in); 				// From bytes to the barrel-shiftrows
	for(int i = 0; i < 14; i++) {
		ark(state, rkeys+i*32); 		// AddRoundKey on the entire state
		sbox(state); 					// S-box on the 1st quarter state
		sbox(state + 8); 				// S-box on the 2nd quarter state
		sbox(state + 16); 				// S-box on the 3rd quarter state
		sbox(state + 24); 				// S-box on the 4th quarter state
	    shiftrows(state); 				// ShiftRows on the entire state
	    if (i != 13) 					// No MixColumns in the last round
			mixcolumns(state);		 	// MixColumns on the entire state
	}
	ark(state, rkeys+448); 				// AddRoundKey on the entire state
	unpacking(out, state); 				// From barrel-shiftrows to bytes
}
