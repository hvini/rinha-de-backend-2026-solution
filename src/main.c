#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "mongoose.h"
#include "cJSON.h"
#include "generated_config.h"

#define MAX_RECORDS 3500000

typedef struct {
    uint8_t vec[14];
    uint8_t label;
    uint8_t padding;
} Record;

typedef struct {
    uint8_t vec[14];
    uint32_t offset;
    uint32_t count;
} Centroid;

typedef struct {
    int dist;
    uint8_t label;
} Neighbor;

typedef struct { 
    int dist; 
    int idx; 
} CentroidDist;

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
    
    int days = days_from_civil(y, m, d);
    return (long long)days * 24 * 60 + h * 60 + min;
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
    int py_w = (w == 0) ? 6 : w - 1;
    *wday_out = py_w / 6.0f;
}

static void fn(struct mg_connection *c, int ev, void *ev_data) {
    if (ev == MG_EV_HTTP_MSG) {
        struct mg_http_message *hm = (struct mg_http_message *) ev_data;

        if (mg_match(hm->uri, mg_str("/ready"), NULL)) {
            mg_http_reply(c, 200, "", "OK\n");
            return;
        }

        if (mg_match(hm->uri, mg_str("/fraud-score"), NULL)) {
            cJSON *json = cJSON_ParseWithLength(hm->body.buf, hm->body.len);
            if (!json) {
                mg_http_reply(c, 200, "Content-Type: application/json\r\n", "{\"approved\": true, \"fraud_score\": 0.0}\n");
                return;
            }

            cJSON *transaction = cJSON_GetObjectItemCaseSensitive(json, "transaction");
            cJSON *customer = cJSON_GetObjectItemCaseSensitive(json, "customer");
            cJSON *merchant = cJSON_GetObjectItemCaseSensitive(json, "merchant");
            cJSON *terminal = cJSON_GetObjectItemCaseSensitive(json, "terminal");
            cJSON *last_transaction = cJSON_GetObjectItemCaseSensitive(json, "last_transaction");

            if (!transaction || !customer || !merchant || !terminal) {
                cJSON_Delete(json);
                mg_http_reply(c, 200, "Content-Type: application/json\r\n", "{\"approved\": true, \"fraud_score\": 0.0}\n");
                return;
            }

            float amount = cJSON_GetObjectItemCaseSensitive(transaction, "amount") ? cJSON_GetObjectItemCaseSensitive(transaction, "amount")->valuedouble : 0;
            int installments = cJSON_GetObjectItemCaseSensitive(transaction, "installments") ? cJSON_GetObjectItemCaseSensitive(transaction, "installments")->valueint : 1;
            const char* requested_at_str = cJSON_GetObjectItemCaseSensitive(transaction, "requested_at") ? cJSON_GetObjectItemCaseSensitive(transaction, "requested_at")->valuestring : NULL;

            float avg_amount = cJSON_GetObjectItemCaseSensitive(customer, "avg_amount") ? cJSON_GetObjectItemCaseSensitive(customer, "avg_amount")->valuedouble : 0;
            int tx_count_24h = cJSON_GetObjectItemCaseSensitive(customer, "tx_count_24h") ? cJSON_GetObjectItemCaseSensitive(customer, "tx_count_24h")->valueint : 0;
            
            const char* merchant_id_str = cJSON_GetObjectItemCaseSensitive(merchant, "id") ? cJSON_GetObjectItemCaseSensitive(merchant, "id")->valuestring : NULL;
            const char* mcc_str = cJSON_GetObjectItemCaseSensitive(merchant, "mcc") ? cJSON_GetObjectItemCaseSensitive(merchant, "mcc")->valuestring : NULL;
            float merchant_avg_amount = cJSON_GetObjectItemCaseSensitive(merchant, "avg_amount") ? cJSON_GetObjectItemCaseSensitive(merchant, "avg_amount")->valuedouble : 0;

            int is_online = cJSON_GetObjectItemCaseSensitive(terminal, "is_online") && cJSON_IsTrue(cJSON_GetObjectItemCaseSensitive(terminal, "is_online"));
            int card_present = cJSON_GetObjectItemCaseSensitive(terminal, "card_present") && cJSON_IsTrue(cJSON_GetObjectItemCaseSensitive(terminal, "card_present"));
            float km_from_home = cJSON_GetObjectItemCaseSensitive(terminal, "km_from_home") ? cJSON_GetObjectItemCaseSensitive(terminal, "km_from_home")->valuedouble : 0;

            int has_last_tx = (last_transaction && !cJSON_IsNull(last_transaction));
            const char* last_timestamp_str = has_last_tx ? (cJSON_GetObjectItemCaseSensitive(last_transaction, "timestamp") ? cJSON_GetObjectItemCaseSensitive(last_transaction, "timestamp")->valuestring : NULL) : NULL;
            float last_km = has_last_tx ? (cJSON_GetObjectItemCaseSensitive(last_transaction, "km_from_current") ? cJSON_GetObjectItemCaseSensitive(last_transaction, "km_from_current")->valuedouble : 0) : 0;

            float vec[14];
            vec[0] = clamp(amount / MAX_AMOUNT);
            vec[1] = clamp((float)installments / MAX_INSTALLMENTS);
            vec[2] = avg_amount > 0 ? clamp((amount / avg_amount) / AMOUNT_VS_AVG_RATIO) : 0.0f;

            float hour_norm = 0, wday_norm = 0;
            parse_time_fast(requested_at_str, &hour_norm, &wday_norm);
            vec[3] = hour_norm;
            vec[4] = wday_norm;

            if (has_last_tx) {
                long long req_min = to_unix_minutes(requested_at_str);
                long long last_min = to_unix_minutes(last_timestamp_str);
                float diff = (float)(req_min - last_min);
                if (diff < 0) diff = 0;
                vec[5] = clamp(diff / MAX_MINUTES);
                vec[6] = clamp(last_km / MAX_KM);
            } else {
                vec[5] = -1.0f;
                vec[6] = -1.0f;
            }

            vec[7] = clamp(km_from_home / MAX_KM);
            vec[8] = clamp((float)tx_count_24h / MAX_TX_COUNT_24H);
            vec[9] = is_online ? 1.0f : 0.0f;
            vec[10] = card_present ? 1.0f : 0.0f;

            int unknown_merchant = 1;
            cJSON* known_merchants = cJSON_GetObjectItemCaseSensitive(customer, "known_merchants");
            if (cJSON_IsArray(known_merchants) && merchant_id_str) {
                cJSON* item = NULL;
                cJSON_ArrayForEach(item, known_merchants) {
                    if (cJSON_IsString(item) && strcmp(item->valuestring, merchant_id_str) == 0) {
                        unknown_merchant = 0;
                        break;
                    }
                }
            }
            vec[11] = unknown_merchant ? 1.0f : 0.0f;

            float mcc_risk = 0.5f;
            if (mcc_str) {
                int mcc_val = atoi(mcc_str);
                if (mcc_val >= 0 && mcc_val < 10000) {
                    mcc_risk = mcc_risk_table[mcc_val];
                }
            }
            vec[12] = mcc_risk;
            vec[13] = clamp(merchant_avg_amount / MAX_MERCHANT_AVG_AMOUNT);

            cJSON_Delete(json);

            uint8_t vec_u8[14];
            for (int i = 0; i < 14; i++) {
                float v = vec[i];
                if (v < -1.0f) v = -1.0f;
                if (v > 1.0f) v = 1.0f;
                vec_u8[i] = (uint8_t)((v + 1.0f) * 127.0f + 0.5f);
            }

            // --- Spatial Index Search (IVF) ---
            
            // 1. Find the Top 3 closest clusters
            CentroidDist top_c[3];
            for (int i = 0; i < 3; i++) { 
                top_c[i].dist = 2000000000; 
                top_c[i].idx = -1; 
            }

            for (int c = 0; c < 1024; c++) {
                int dist = 0;
                for (int d = 0; d < 14; d++) {
                    int diff = (int)centroids[c].vec[d] - (int)vec_u8[d];
                    dist += diff * diff;
                }
                if (dist < top_c[2].dist) {
                    int pos = 2;
                    while (pos > 0 && dist < top_c[pos - 1].dist) {
                        top_c[pos] = top_c[pos - 1];
                        pos--;
                    }
                    top_c[pos].dist = dist;
                    top_c[pos].idx = c;
                }
            }

            // 2. Search ONLY inside those 3 clusters
            Neighbor top5[5];
            for (int i = 0; i < 5; i++) { 
                top5[i].dist = 2000000000; 
                top5[i].label = 0; 
            }

            for (int t = 0; t < 3; t++) {
                int c = top_c[t].idx;
                if (c < 0) continue;
                int start = centroids[c].offset;
                int end = start + centroids[c].count;

                // Explicitly unrolled for SIMD/Compiler optimization
                for (int i = start; i < end; i++) {
                    int dist = 0;
                    dist += ((int)dataset[i].vec[0] - (int)vec_u8[0]) * ((int)dataset[i].vec[0] - (int)vec_u8[0]);
                    dist += ((int)dataset[i].vec[1] - (int)vec_u8[1]) * ((int)dataset[i].vec[1] - (int)vec_u8[1]);
                    dist += ((int)dataset[i].vec[2] - (int)vec_u8[2]) * ((int)dataset[i].vec[2] - (int)vec_u8[2]);
                    dist += ((int)dataset[i].vec[3] - (int)vec_u8[3]) * ((int)dataset[i].vec[3] - (int)vec_u8[3]);
                    dist += ((int)dataset[i].vec[4] - (int)vec_u8[4]) * ((int)dataset[i].vec[4] - (int)vec_u8[4]);
                    dist += ((int)dataset[i].vec[5] - (int)vec_u8[5]) * ((int)dataset[i].vec[5] - (int)vec_u8[5]);
                    dist += ((int)dataset[i].vec[6] - (int)vec_u8[6]) * ((int)dataset[i].vec[6] - (int)vec_u8[6]);
                    dist += ((int)dataset[i].vec[7] - (int)vec_u8[7]) * ((int)dataset[i].vec[7] - (int)vec_u8[7]);
                    dist += ((int)dataset[i].vec[8] - (int)vec_u8[8]) * ((int)dataset[i].vec[8] - (int)vec_u8[8]);
                    dist += ((int)dataset[i].vec[9] - (int)vec_u8[9]) * ((int)dataset[i].vec[9] - (int)vec_u8[9]);
                    dist += ((int)dataset[i].vec[10] - (int)vec_u8[10]) * ((int)dataset[i].vec[10] - (int)vec_u8[10]);
                    dist += ((int)dataset[i].vec[11] - (int)vec_u8[11]) * ((int)dataset[i].vec[11] - (int)vec_u8[11]);
                    dist += ((int)dataset[i].vec[12] - (int)vec_u8[12]) * ((int)dataset[i].vec[12] - (int)vec_u8[12]);
                    dist += ((int)dataset[i].vec[13] - (int)vec_u8[13]) * ((int)dataset[i].vec[13] - (int)vec_u8[13]);

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
    // 1. Load Clustered Data
    int fd_data = open("resources/dataset_ivf.bin", O_RDONLY);
    if (fd_data < 0) {
        perror("Failed to open dataset_ivf.bin");
        return 1;
    }
    struct stat st;
    fstat(fd_data, &st);
    dataset = (Record*)mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd_data, 0);
    if (dataset == MAP_FAILED) {
        perror("mmap dataset failed");
        return 1;
    }
    num_records = st.st_size / sizeof(Record);
    close(fd_data);

    // 2. Load Centroids
    int fd_cent = open("resources/centroids.bin", O_RDONLY);
    if (fd_cent < 0) {
        perror("Failed to open centroids.bin");
        return 1;
    }
    centroids = (Centroid*)mmap(NULL, 1024 * sizeof(Centroid), PROT_READ, MAP_PRIVATE, fd_cent, 0);
    if (centroids == MAP_FAILED) {
        perror("mmap centroids failed");
        return 1;
    }
    close(fd_cent);

    printf("Loaded IVF Index. %zu records.\n", num_records);

    struct mg_mgr mgr;
    mg_mgr_init(&mgr);
    mg_http_listen(&mgr, "http://0.0.0.0:9999", fn, NULL);

    printf("Starting API on port 9999...\n");
    for (;;) mg_mgr_poll(&mgr, 1000);
    
    mg_mgr_free(&mgr);
    return 0;
}