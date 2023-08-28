
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <sys/types.h>
#include <time.h>//#include <sys/time.h>
#include <sys/stat.h>
#include <string.h>
#include <assert.h>

#define msgSize 		512*1024*1024	// size in word (4 bytes)
#define threadSize 		1024	// Minimum 256 Threads
#define threadSizeBS	64
#define REPEAT			32
#define REPEATBS		4
#define gridSize 		msgSize/threadSize/4 // Each thread encrypt one counter value, which is 16 bytes or 4 words.  
#define gridSizeBS 		msgSize/threadSizeBS/8/4 // Each thread encrypt one counter value, which is 8*16 bytes or 32 words.  
#define gridSizeBSSM	msgSize/threadSizeBSSM/8/4
#define ITERATION 		100	// Calculate the average time

// Choose only DEBUG or PROFILE
#define DEBUG 			// Print results
// #define PROFILE			// Remove all printf, only print the throughput TP.

void AESPrepareKey(char *dec_key, uint8_t *enc_key, unsigned int key_bits);
__global__ void encGPUshared(unsigned int *out, const unsigned int *roundkey, uint32_t* in);

