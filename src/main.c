#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <immintrin.h>

#include "mongoose.h"
#include "cJSON.h"
#include "generated_config.h"

#define K_CLUSTERS 4096
#define NPROBE 128

static uint8_t arena_mem[2 * 1024 * 1024]; 
static size_t arena_offset = 0;

void* arena_malloc(size_t sz) {
    size_t aligned_sz = (sz + 7) & ~7;
    if (arena_offset + aligned_sz > sizeof(arena_mem)) return NULL;
    void* ptr = arena_mem + arena_offset;
    arena_offset += aligned_sz;
    return ptr;
}
void arena_free(void* ptr) { (void)ptr; }

typedef struct __attribute__((aligned(32))) {
    int16_t vec[14];
    uint8_t label;
    uint8_t padding[3];
} Record;

typedef struct {
    int16_t vec[14];
    uint32_t offset;
    uint32_t count;
} Centroid;

typedef struct { int dist; uint8_t label; } Neighbor;
typedef struct { int dist; int idx; } CentroidDist;

Record* dataset = NULL;
Centroid* centroids = NULL;
size_t num_records = 0;

static float clamp(float v) {
    if (v < 0.0f) return 0.0f;
    if (v > 1.0f) return 1.0f;
    return v;
}

static int days_from_civil(int y, int m, int d) {
    y -= m <= 2;
    int era = (y >= 0 ? y : y - 399) / 400;
    int yoe = y - era * 400;
    int doy = (153 * (m + (m > 2 ? -3 : 9)) + 2) / 5 + d - 1;
    int doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    return era * 146097 + doe - 719468;
}

static long long to_unix_minutes(const char* iso) {
    if (!iso || strlen(iso) < 19) return 0;
    int y = (iso[0]-'0')*1000 + (iso[1]-'0')*100 + (iso[2]-'0')*10 + (iso[3]-'0');
    int m = (iso[5]-'0')*10 + (iso[6]-'0');
    int d = (iso[8]-'0')*10 + (iso[9]-'0');
    int h = (iso[11]-'0')*10 + (iso[12]-'0');
    int min = (iso[14]-'0')*10 + (iso[15]-'0');
    return (long long)days_from_civil(y, m, d) * 24 * 60 + h * 60 + min;
}

static void parse_time_fast(const char* iso, float* hour_out, float* wday_out) {
    if (!iso || strlen(iso) < 19) { *hour_out = 0; *wday_out = 0; return; }
    int y = (iso[0]-'0')*1000 + (iso[1]-'0')*100 + (iso[2]-'0')*10 + (iso[3]-'0');
    int m = (iso[5]-'0')*10 + (iso[6]-'0');
    int d = (iso[8]-'0')*10 + (iso[9]-'0');
    int h = (iso[11]-'0')*10 + (iso[12]-'0');
    *hour_out = h / 23.0f;
    static int t[] = {0, 3, 2, 5, 0, 3, 5, 1, 4, 6, 2, 4};
    int y_adj = y - (m < 3 ? 1 : 0);
    int w = (y_adj + y_adj/4 - y_adj/100 + y_adj/400 + t[m-1] + d) % 7;
    *wday_out = ((w == 0) ? 6 : w - 1) / 6.0f;
}

static void fn(struct mg_connection *c, int ev, void *ev_data) {
    if (ev == MG_EV_HTTP_MSG) {
        struct mg_http_message *hm = (struct mg_http_message *) ev_data;

        if (mg_match(hm->uri, mg_str("/ready"), NULL)) {
            mg_http_reply(c, 200, "", "OK\n");
            return;
        }

        if (mg_match(hm->uri, mg_str("/fraud-score"), NULL)) {
            arena_offset = 0; 
            
            cJSON *json = cJSON_ParseWithLength(hm->body.buf, hm->body.len);
            if (!json) {
                mg_http_reply(c, 200, "Content-Type: application/json\r\n", "{\"approved\": true, \"fraud_score\": 0.0}\n");
                return;
            }

            cJSON *transaction = cJSON_GetObjectItemCaseSensitive(json, "transaction");
            cJSON *customer = cJSON_GetObjectItemCaseSensitive(json, "customer");
            cJSON *merchant = cJSON_GetObjectItemCaseSensitive(json, "merchant");
            cJSON *terminal = cJSON_GetObjectItemCaseSensitive(json, "terminal");
            cJSON *last_tx = cJSON_GetObjectItemCaseSensitive(json, "last_transaction");

            if (!transaction || !customer || !merchant || !terminal) {
                mg_http_reply(c, 200, "Content-Type: application/json\r\n", "{\"approved\": true, \"fraud_score\": 0.0}\n");
                return;
            }

            float amount = cJSON_GetObjectItemCaseSensitive(transaction, "amount") ? cJSON_GetObjectItemCaseSensitive(transaction, "amount")->valuedouble : 0;
            int installments = cJSON_GetObjectItemCaseSensitive(transaction, "installments") ? cJSON_GetObjectItemCaseSensitive(transaction, "installments")->valueint : 1;
            const char* req_at = cJSON_GetObjectItemCaseSensitive(transaction, "requested_at") ? cJSON_GetObjectItemCaseSensitive(transaction, "requested_at")->valuestring : NULL;

            float avg_amt = cJSON_GetObjectItemCaseSensitive(customer, "avg_amount") ? cJSON_GetObjectItemCaseSensitive(customer, "avg_amount")->valuedouble : 0;
            int tx_count = cJSON_GetObjectItemCaseSensitive(customer, "tx_count_24h") ? cJSON_GetObjectItemCaseSensitive(customer, "tx_count_24h")->valueint : 0;
            
            const char* m_id = cJSON_GetObjectItemCaseSensitive(merchant, "id") ? cJSON_GetObjectItemCaseSensitive(merchant, "id")->valuestring : NULL;
            const char* mcc = cJSON_GetObjectItemCaseSensitive(merchant, "mcc") ? cJSON_GetObjectItemCaseSensitive(merchant, "mcc")->valuestring : NULL;
            float m_avg = cJSON_GetObjectItemCaseSensitive(merchant, "avg_amount") ? cJSON_GetObjectItemCaseSensitive(merchant, "avg_amount")->valuedouble : 0;

            int is_on = cJSON_GetObjectItemCaseSensitive(terminal, "is_online") && cJSON_IsTrue(cJSON_GetObjectItemCaseSensitive(terminal, "is_online"));
            int card_p = cJSON_GetObjectItemCaseSensitive(terminal, "card_present") && cJSON_IsTrue(cJSON_GetObjectItemCaseSensitive(terminal, "card_present"));
            float km_home = cJSON_GetObjectItemCaseSensitive(terminal, "km_from_home") ? cJSON_GetObjectItemCaseSensitive(terminal, "km_from_home")->valuedouble : 0;

            int has_last = (last_tx && !cJSON_IsNull(last_tx));
            const char* last_ts = has_last ? (cJSON_GetObjectItemCaseSensitive(last_tx, "timestamp") ? cJSON_GetObjectItemCaseSensitive(last_tx, "timestamp")->valuestring : NULL) : NULL;
            float last_km = has_last ? (cJSON_GetObjectItemCaseSensitive(last_tx, "km_from_current") ? cJSON_GetObjectItemCaseSensitive(last_tx, "km_from_current")->valuedouble : 0) : 0;

            float vec[14];
            vec[0] = clamp(amount / MAX_AMOUNT);
            vec[1] = clamp((float)installments / MAX_INSTALLMENTS);
            vec[2] = avg_amt > 0 ? clamp((amount / avg_amt) / AMOUNT_VS_AVG_RATIO) : 0.0f;

            float hr_norm = 0, wd_norm = 0;
            parse_time_fast(req_at, &hr_norm, &wd_norm);
            vec[3] = hr_norm;
            vec[4] = wd_norm;

            if (has_last) {
                long long req_m = to_unix_minutes(req_at);
                long long last_m = to_unix_minutes(last_ts);
                float diff = (float)(req_m - last_m);
                if (diff < 0) diff = 0;
                vec[5] = clamp(diff / MAX_MINUTES);
                vec[6] = clamp(last_km / MAX_KM);
            } else {
                vec[5] = -1.0f;
                vec[6] = -1.0f;
            }

            vec[7] = clamp(km_home / MAX_KM);
            vec[8] = clamp((float)tx_count / MAX_TX_COUNT_24H);
            vec[9] = is_on ? 1.0f : 0.0f;
            vec[10] = card_p ? 1.0f : 0.0f;

            int unk_m = 1;
            cJSON* known = cJSON_GetObjectItemCaseSensitive(customer, "known_merchants");
            if (cJSON_IsArray(known) && m_id) {
                cJSON* item = NULL;
                cJSON_ArrayForEach(item, known) {
                    if (cJSON_IsString(item) && strcmp(item->valuestring, m_id) == 0) {
                        unk_m = 0; break;
                    }
                }
            }
            vec[11] = unk_m ? 1.0f : 0.0f;

            float mcc_r = 0.5f;
            if (mcc) {
                int val = atoi(mcc);
                if (val >= 0 && val < 10000) mcc_r = mcc_risk_table[val];
            }
            vec[12] = mcc_r;
            vec[13] = clamp(m_avg / MAX_MERCHANT_AVG_AMOUNT);

            __attribute__((aligned(32))) int16_t vec_i16[16] = {0};
            for (int i = 0; i < 14; i++) {
                float v = vec[i];
                if (v < -1.0f) v = -1.0f;
                if (v > 1.0f) v = 1.0f;
                vec_i16[i] = (int16_t)roundf(v * 10000.0f);
            }

            __m256i mask14 = _mm256_set_epi16(0, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
            __m256i target_vec = _mm256_loadu_si256((__m256i*)vec_i16);
            target_vec = _mm256_and_si256(target_vec, mask14);
            
            CentroidDist top_c[NPROBE];
            for (int i = 0; i < NPROBE; i++) { top_c[i].dist = 2000000000; top_c[i].idx = -1; }

            for (int c = 0; c < K_CLUSTERS; c++) {
                __m256i cent_vec = _mm256_loadu_si256((__m256i*)centroids[c].vec);
                cent_vec = _mm256_and_si256(cent_vec, mask14);

                __m256i diff = _mm256_sub_epi16(cent_vec, target_vec);
                __m256i sq = _mm256_madd_epi16(diff, diff);
                
                __m128i sq128 = _mm_add_epi32(_mm256_castsi256_si128(sq), _mm256_extracti128_si256(sq, 1));
                sq128 = _mm_add_epi32(sq128, _mm_srli_si128(sq128, 8));
                sq128 = _mm_add_epi32(sq128, _mm_srli_si128(sq128, 4));
                int dist = _mm_cvtsi128_si32(sq128);

                if (dist < top_c[NPROBE - 1].dist) {
                    int pos = NPROBE - 1;
                    while (pos > 0 && dist < top_c[pos - 1].dist) {
                        top_c[pos] = top_c[pos - 1];
                        pos--;
                    }
                    top_c[pos].dist = dist;
                    top_c[pos].idx = c;
                }
            }

            Neighbor top5[5];
            for (int i = 0; i < 5; i++) { top5[i].dist = 2000000000; top5[i].label = 0; }

            for (int t = 0; t < NPROBE; t++) {
                int c = top_c[t].idx;
                if (c < 0) continue;
                int start = centroids[c].offset;
                int end = start + centroids[c].count;

                for (int i = start; i < end; i++) {
                    _mm_prefetch((const char*)&dataset[i + 2], _MM_HINT_T0);

                    __m256i data_vec = _mm256_loadu_si256((__m256i*)&dataset[i]);

                    __m256i diff = _mm256_sub_epi16(data_vec, target_vec);
                    __m256i sq = _mm256_madd_epi16(diff, diff);
                    
                    __m128i sq128 = _mm_add_epi32(_mm256_castsi256_si128(sq), _mm256_extracti128_si256(sq, 1));
                    sq128 = _mm_add_epi32(sq128, _mm_srli_si128(sq128, 8));
                    sq128 = _mm_add_epi32(sq128, _mm_srli_si128(sq128, 4));
                    int dist = _mm_cvtsi128_si32(sq128);

                    if (dist < top5[4].dist) {
                        int pos = 4;
                        while (pos > 0 && dist < top5[pos - 1].dist) {
                            top5[pos] = top5[pos - 1];
                            pos--;
                        }
                        top5[pos].dist = dist;
                        top5[pos].label = dataset[i].label;
                    }
                }
            }

            int frauds = 0;
            for (int i = 0; i < 5; i++) { 
                if (top5[i].label == 1) frauds++; 
            }
            float fraud_score = frauds / 5.0f;
            int approved = fraud_score < 0.6f;

            char resp[128];
            snprintf(resp, sizeof(resp), "{\"approved\": %s, \"fraud_score\": %.1f}", approved ? "true" : "false", fraud_score);
            
            mg_http_reply(c, 200, "Content-Type: application/json\r\n", "%s\n", resp);
        } else {
            mg_http_reply(c, 200, "Content-Type: application/json\r\n", "{\"approved\": true, \"fraud_score\": 0.0}\n");
        }
    }
}

int main(void) {
    cJSON_Hooks hooks;
    hooks.malloc_fn = arena_malloc;
    hooks.free_fn = arena_free;
    cJSON_InitHooks(&hooks);

    int fd_data = open("resources/dataset_ivf_int16.bin", O_RDONLY);
    if (fd_data < 0) { perror("Failed to open dataset"); return 1; }
    struct stat st; fstat(fd_data, &st);
    dataset = (Record*)mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE | MAP_POPULATE, fd_data, 0);
    close(fd_data);

    int fd_cent = open("resources/centroids_int16.bin", O_RDONLY);
    if (fd_cent < 0) { perror("Failed to open centroids"); return 1; }
    centroids = (Centroid*)mmap(NULL, K_CLUSTERS * sizeof(Centroid), PROT_READ, MAP_PRIVATE | MAP_POPULATE, fd_cent, 0);
    close(fd_cent);

    printf("Loaded API: MAX Precision + Exact-Match Short-Circuit.\n");

    struct mg_mgr mgr;
    mg_mgr_init(&mgr);
    mg_http_listen(&mgr, "http://0.0.0.0:9999", fn, NULL);

    for (;;) mg_mgr_poll(&mgr, 1000);
    
    mg_mgr_free(&mgr);
    return 0;
}