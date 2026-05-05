#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <immintrin.h>

#define K 4096
#define D 14

typedef struct {
    int16_t vec[D];
    uint8_t label;
    uint8_t padding[3];
} Record;

typedef struct {
    int16_t vec[D];
    uint32_t offset;
    uint32_t count;
} Centroid;

int main() {
    FILE* fp = fopen("resources/dataset_int16.bin", "rb");
    if (!fp) { perror("Missing dataset"); return 1; }
    fseek(fp, 0, SEEK_END);
    long fsize = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    int n_records = fsize / sizeof(Record);
    
    // FIX 1: Standard malloc prevents NULL segfaults if size isn't a perfect multiple of 32
    Record* data = malloc(fsize);
    if (!data) { perror("Malloc failed"); return 1; }
    
    size_t bytes_read = fread(data, 1, fsize, fp);
    if (bytes_read != fsize) { printf("Warning: read %zu bytes, expected %ld\n", bytes_read, fsize); }
    fclose(fp);

    Centroid* centroids = calloc(K, sizeof(Centroid));
    for (int i = 0; i < K; i++) {
        int idx = (i * 34157 + 101) % n_records;
        memcpy(centroids[i].vec, data[idx].vec, D * sizeof(int16_t));
    }

    int* assignments = malloc(n_records * sizeof(int));
    __m256i mask14 = _mm256_set_epi16(0, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);

    // FIX 2: Move massive tracking arrays to the HEAP to prevent Docker Stack Overflow
    int64_t (*sums)[D] = malloc(K * sizeof(*sums));
    uint32_t *counts = malloc(K * sizeof(uint32_t));

    for(int iter = 0; iter < 8; iter++) {
        memset(sums, 0, K * sizeof(*sums));
        memset(counts, 0, K * sizeof(*counts));

        for (int i = 0; i < n_records; i++) {
            int best_c = 0;
            int best_dist = 2000000000;
            
            // FIX 3: loadu (unaligned) is completely immune to alignment segfaults
            __m256i data_vec = _mm256_loadu_si256((__m256i*)&data[i]);
            data_vec = _mm256_and_si256(data_vec, mask14);

            for (int c = 0; c < K; c++) {
                __m256i cent_vec = _mm256_loadu_si256((__m256i*)centroids[c].vec);
                cent_vec = _mm256_and_si256(cent_vec, mask14);

                __m256i diff = _mm256_sub_epi16(data_vec, cent_vec);
                __m256i sq = _mm256_madd_epi16(diff, diff);
                
                __m128i sq128 = _mm_add_epi32(_mm256_castsi256_si128(sq), _mm256_extracti128_si256(sq, 1));
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
            for (int d = 0; d < D; d++) sums[best_c][d] += data[i].vec[d];
        }

        for (int c = 0; c < K; c++) {
            if (counts[c] > 0) {
                for (int d = 0; d < D; d++) centroids[c].vec[d] = sums[c][d] / counts[c];
            }
        }
        if (iter == 7) { 
            for (int c = 0; c < K; c++) centroids[c].count = counts[c];
        }
    }

    uint32_t current_offset = 0;
    for (int c = 0; c < K; c++) {
        centroids[c].offset = current_offset;
        current_offset += centroids[c].count;
    }

    Record* clustered_data = malloc(fsize);
    uint32_t* current_pos = malloc(K * sizeof(uint32_t));
    for (int c = 0; c < K; c++) current_pos[c] = centroids[c].offset;

    for (int i = 0; i < n_records; i++) {
        int c = assignments[i];
        clustered_data[current_pos[c]++] = data[i];
    }

    fp = fopen("resources/dataset_ivf_int16.bin", "wb");
    fwrite(clustered_data, sizeof(Record), n_records, fp);
    fclose(fp);

    fp = fopen("resources/centroids_int16.bin", "wb");
    fwrite(centroids, sizeof(Centroid), K, fp);
    fclose(fp);

    free(data); free(assignments); free(clustered_data); free(current_pos); free(centroids);
    free(sums); free(counts);
    return 0;
}