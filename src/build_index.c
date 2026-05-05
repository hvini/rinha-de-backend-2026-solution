#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#define K 1024
#define D 14

typedef struct {
    uint8_t vec[D];
    uint8_t label;
    uint8_t padding;
} Record;

typedef struct {
    uint8_t vec[D];
    uint32_t offset;
    uint32_t count;
} Centroid;

int main() {
    FILE* fp = fopen("resources/dataset_uint8.bin", "rb");
    if (!fp) { perror("Missing dataset"); return 1; }
    fseek(fp, 0, SEEK_END);
    long fsize = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    int n_records = fsize / sizeof(Record);
    Record* data = malloc(fsize);
    fread(data, 1, fsize, fp);
    fclose(fp);

    Centroid* centroids = calloc(K, sizeof(Centroid));
    
    // Pick initial centroids (strided to ensure variety)
    for (int i = 0; i < K; i++) {
        int idx = (i * 34157 + 101) % n_records;
        memcpy(centroids[i].vec, data[idx].vec, D);
    }

    // 2 iterations of Lloyd's Algorithm (K-Means)
    int* assignments = malloc(n_records * sizeof(int));
    for(int iter = 0; iter < 2; iter++) {
        uint64_t sums[K][D];
        uint32_t counts[K];
        memset(sums, 0, sizeof(sums));
        memset(counts, 0, sizeof(counts));

        for (int i = 0; i < n_records; i++) {
            int best_c = 0;
            int best_dist = 2000000000;
            for (int c = 0; c < K; c++) {
                int dist = 0;
                for (int d = 0; d < D; d++) {
                    int diff = (int)data[i].vec[d] - (int)centroids[c].vec[d];
                    dist += diff * diff;
                }
                if (dist < best_dist) {
                    best_dist = dist;
                    best_c = c;
                }
            }
            assignments[i] = best_c;
            counts[best_c]++;
            for (int d = 0; d < D; d++) {
                sums[best_c][d] += data[i].vec[d];
            }
        }

        // Update centroids
        for (int c = 0; c < K; c++) {
            if (counts[c] > 0) {
                for (int d = 0; d < D; d++) centroids[c].vec[d] = sums[c][d] / counts[c];
            }
        }
        if (iter == 1) { // Final pass
            for (int c = 0; c < K; c++) centroids[c].count = counts[c];
        }
    }

    // Calculate offsets and shuffle dataset into clusters
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

    fp = fopen("resources/dataset_ivf.bin", "wb");
    fwrite(clustered_data, sizeof(Record), n_records, fp);
    fclose(fp);

    fp = fopen("resources/centroids.bin", "wb");
    fwrite(centroids, sizeof(Centroid), K, fp);
    fclose(fp);

    printf("IVF Index built successfully.\n");
    return 0;
}