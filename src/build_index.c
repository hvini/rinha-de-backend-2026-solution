#include <immintrin.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define K 4096
#define D 14
#define BLOCK_SIZE 8

typedef struct {
  int16_t vec[D];
  uint8_t label;
  uint8_t padding[3];
} Record;

typedef struct {
  int16_t vec[16];
  int16_t min[16];
  int16_t max[16];
  uint32_t offset;
  uint32_t count;
} Centroid;

// SOA block of 8 records
typedef struct __attribute__((aligned(32))) {
  int16_t dims[D][BLOCK_SIZE];
  uint8_t labels[BLOCK_SIZE];
  uint8_t padding[24]; // Pad to 256 bytes (32 * 8)
} RecordBlock;

int main() {
  FILE *fp = fopen("resources/dataset_int16.bin", "rb");
  if (!fp) {
    perror("Missing dataset");
    return 1;
  }
  fseek(fp, 0, SEEK_END);
  long fsize = ftell(fp);
  fseek(fp, 0, SEEK_SET);
  int n_records = fsize / sizeof(Record);

  Record *data = malloc(fsize);
  if (!data) {
    perror("Malloc failed");
    return 1;
  }
  fread(data, 1, fsize, fp);
  fclose(fp);

  Centroid *centroids = calloc(K, sizeof(Centroid));
  for (int i = 0; i < K; i++) {
    int idx = (i * 34157 + 101) % n_records;
    for (int d = 0; d < D; d++) {
      centroids[i].vec[d] = data[idx].vec[d];
      centroids[i].min[d] = 32767;
      centroids[i].max[d] = -32768;
    }
  }

  int *assignments = malloc(n_records * sizeof(int));
  __m256i mask14 = _mm256_set_epi16(0, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                                    -1, -1, -1, -1, -1);

  int64_t (*sums)[D] = malloc(K * sizeof(*sums));
  uint32_t *counts = malloc(K * sizeof(uint32_t));

  printf("Starting K-Means (32 iterations)...\n");
  for (int iter = 0; iter < 32; iter++) {
    memset(sums, 0, K * sizeof(*sums));
    memset(counts, 0, K * sizeof(*counts));

    for (int i = 0; i < n_records; i++) {
      int best_c = 0;
      int best_dist = 2000000000;

      __m256i data_vec = _mm256_loadu_si256((__m256i *)&data[i]);
      data_vec = _mm256_and_si256(data_vec, mask14);

      for (int c = 0; c < K; c++) {
        __m256i cent_vec = _mm256_loadu_si256((__m256i *)centroids[c].vec);
        __m256i diff = _mm256_sub_epi16(data_vec, cent_vec);
        __m256i sq = _mm256_madd_epi16(diff, diff);

        __m128i sq128 = _mm_add_epi32(_mm256_castsi256_si128(sq),
                                      _mm256_extracti128_si256(sq, 1));
        sq128 = _mm_add_epi32(sq128, _mm_srli_si128(sq128, 8));
        sq128 = _mm_add_epi32(sq128, _mm_srli_si128(sq128, 4));
        int dist = _mm_cvtsi128_si32(sq128);

        if (dist < best_dist) {
          best_dist = dist;
          best_c = c;
        }
      }
      assignments[i] = best_c;
      counts[best_c]++;
      for (int d = 0; d < D; d++) {
        sums[best_c][d] += data[i].vec[d];
        if (iter == 31) { // Final iteration: calculate BBOX
          if (data[i].vec[d] < centroids[best_c].min[d]) centroids[best_c].min[d] = data[i].vec[d];
          if (data[i].vec[d] > centroids[best_c].max[d]) centroids[best_c].max[d] = data[i].vec[d];
        }
      }
    }

    if (iter < 31) {
        for (int c = 0; c < K; c++) {
            if (counts[c] > 0) {
                for (int d = 0; d < D; d++)
                    centroids[c].vec[d] = sums[c][d] / counts[c];
            }
        }
    }
    printf("Iteration %d done\n", iter + 1);
  }

  uint32_t current_offset = 0;
  for (int c = 0; c < K; c++) {
    centroids[c].offset = current_offset;
    centroids[c].count = counts[c];
    current_offset += (centroids[c].count + BLOCK_SIZE - 1) / BLOCK_SIZE;
  }

  int n_blocks = current_offset;
  RecordBlock *clustered_blocks = calloc(n_blocks, sizeof(RecordBlock));
  uint32_t *block_cursor = malloc(K * sizeof(uint32_t));
  uint32_t *lane_cursor = calloc(K, sizeof(uint32_t));
  for (int c = 0; c < K; c++)
    block_cursor[c] = centroids[c].offset;

  for (int i = 0; i < n_records; i++) {
    int c = assignments[i];
    uint32_t b = block_cursor[c];
    uint32_t l = lane_cursor[c];
    
    for (int d = 0; d < D; d++)
      clustered_blocks[b].dims[d][l] = data[i].vec[d];
    clustered_blocks[b].labels[l] = data[i].label;
    
    lane_cursor[c]++;
    if (lane_cursor[c] == BLOCK_SIZE) {
      lane_cursor[c] = 0;
      block_cursor[c]++;
    }
  }

  fp = fopen("resources/dataset_ivf_int16.bin", "wb");
  fwrite(clustered_blocks, sizeof(RecordBlock), n_blocks, fp);
  fclose(fp);

  fp = fopen("resources/centroids_int16.bin", "wb");
  fwrite(centroids, sizeof(Centroid), K, fp);
  fclose(fp);

  printf("Index built successfully: %d records, %d clusters, %d blocks.\n", n_records, K, n_blocks);

  free(data);
  free(assignments);
  free(clustered_blocks);
  free(block_cursor);
  free(lane_cursor);
  free(centroids);
  free(sums);
  free(counts);
  return 0;
}