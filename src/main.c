#define _GNU_SOURCE
#include <fcntl.h>
#include <immintrin.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <stdalign.h>

#include "generated_config.h"
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/epoll.h>
#include <fcntl.h>
#include <errno.h>
#include <netinet/in.h>
#include <netinet/tcp.h>

#define K_CLUSTERS 4096
#define NPROBE 128

typedef struct __attribute__((aligned(32))) {
  int16_t dims[14][8];
  uint8_t labels[8];
  uint8_t padding[24];
} RecordBlock;

typedef struct
{
  int16_t vec[16];
  int16_t min[16];
  int16_t max[16];
  uint32_t offset;
  uint32_t count;
} Centroid;

typedef struct
{
  int dist;
  uint8_t label;
} Neighbor;

typedef struct
{
  int dist;
  int idx;
} CentroidDist;

RecordBlock *dataset = NULL;
Centroid *centroids = NULL;

// --- Optimization: Min-Heap to maintain Top N centroids ---
static inline void push_heap(CentroidDist *heap, int dist, int idx, int size)
{
  if (dist < heap[0].dist)
  {
    heap[0].dist = dist;
    heap[0].idx = idx;
    int i = 0;
    while (1)
    {
      int l = 2 * i + 1, r = 2 * i + 2, largest = i;
      if (l < size && heap[l].dist > heap[largest].dist)
        largest = l;
      if (r < size && heap[r].dist > heap[largest].dist)
        largest = r;
      if (largest == i)
        break;
      CentroidDist tmp = heap[i];
      heap[i] = heap[largest];
      heap[largest] = tmp;
      i = largest;
    }
  }
}

// --- Optimization: AVX2 Distance function ---
static inline int calc_dist_avx(const __m256i v1, const __m256i v2)
{
  __m256i d = _mm256_sub_epi16(v1, v2);
  __m256i s = _mm256_madd_epi16(d, d);
  __m128i r = _mm_add_epi32(_mm256_castsi256_si128(s), _mm256_extracti128_si256(s, 1));
  r = _mm_add_epi32(r, _mm_srli_si128(r, 8));
  r = _mm_add_epi32(r, _mm_srli_si128(r, 4));
  return _mm_cvtsi128_si32(r);
}

static inline int dist_to_bbox_avx(const __m256i q_v, const int16_t min[16], const int16_t max[16]) {
    __m256i min_v = _mm256_loadu_si256((__m256i *)min);
    __m256i max_v = _mm256_loadu_si256((__m256i *)max);
    __m256i d_min = _mm256_subs_epi16(min_v, q_v);
    d_min = _mm256_max_epi16(d_min, _mm256_setzero_si256());
    __m256i d_max = _mm256_subs_epi16(q_v, max_v);
    d_max = _mm256_max_epi16(d_max, _mm256_setzero_si256());
    __m256i d = _mm256_or_si256(d_min, d_max);
    __m256i s = _mm256_madd_epi16(d, d);
    __m128i r = _mm_add_epi32(_mm256_castsi256_si128(s), _mm256_extracti128_si256(s, 1));
    r = _mm_add_epi32(r, _mm_srli_si128(r, 8));
    r = _mm_add_epi32(r, _mm_srli_si128(r, 4));
    return _mm_cvtsi128_si32(r);
}

// --- Original Logic Helpers (Unchanged) ---
static float clamp(float v) { return v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v); }
static int days_from_civil(int y, int m, int d)
{
  y -= m <= 2;
  int era = (y >= 0 ? y : y - 399) / 400;
  int yoe = y - era * 400;
  int doy = (153 * (m + (m > 2 ? -3 : 9)) + 2) / 5 + d - 1;
  return era * 146097 + (yoe * 365 + yoe / 4 - yoe / 100 + doy) - 719468;
}
static long long to_unix_minutes(const char *iso)
{
  if (!iso || iso[0] < '0')
    return 0;
  int y = (iso[0] - '0') * 1000 + (iso[1] - '0') * 100 + (iso[2] - '0') * 10 + (iso[3] - '0');
  int m = (iso[5] - '0') * 10 + (iso[6] - '0'), d = (iso[8] - '0') * 10 + (iso[9] - '0');
  int h = (iso[11] - '0') * 10 + (iso[12] - '0'), min = (iso[14] - '0') * 10 + (iso[15] - '0');
  return (long long)days_from_civil(y, m, d) * 1440 + h * 60 + min;
}
static void parse_time_fast(const char *iso, float *h_out, float *w_out)
{
  if (!iso || iso[0] < '0')
  {
    *h_out = *w_out = 0;
    return;
  }
  int y = (iso[0] - '0') * 1000 + (iso[1] - '0') * 100 + (iso[2] - '0') * 10 + (iso[3] - '0');
  int m = (iso[5] - '0') * 10 + (iso[6] - '0'), d = (iso[8] - '0') * 10 + (iso[9] - '0'), h = (iso[11] - '0') * 10 + (iso[12] - '0');
  *h_out = h / 23.0f;
  static int t[] = {0, 3, 2, 5, 0, 3, 5, 1, 4, 6, 2, 4};
  int ya = y - (m < 3);
  int w = (ya + ya / 4 - ya / 100 + ya / 400 + t[m - 1] + d) % 7;
  *w_out = (w == 0 ? 6 : w - 1) / 6.0f;
}
static inline const char *find_key_fast(const char *buf, const char *end, const char *key, int len)
{
  const char *p = buf;
  while ((p = memchr(p, key[0], end - p - len + 1)))
  {
    if (memcmp(p, key, len) == 0)
      return p + len;
    p++;
  }
  return NULL;
}
static inline float p_float(const char *p)
{
  if (!p) return 0;
  while (*p == ' ' || *p == ':' || *p == '"') p++;
  float res = 0.0f, sign = 1.0f;
  if (*p == '-') { sign = -1.0f; p++; }
  while (*p >= '0' && *p <= '9') { res = res * 10.0f + (*p - '0'); p++; }
  if (*p == '.') {
    p++; float frac = 0.1f;
    while (*p >= '0' && *p <= '9') { res += (*p - '0') * frac; frac *= 0.1f; p++; }
  }
  return res * sign;
}
static inline int p_int(const char *p)
{
  if (!p) return 0;
  while (*p == ' ' || *p == ':' || *p == '"') p++;
  int res = 0, sign = 1;
  if (*p == '-') { sign = -1; p++; }
  while (*p >= '0' && *p <= '9') { res = res * 10 + (*p - '0'); p++; }
  return res * sign;
}
static void p_str(const char *p, char *o, int l)
{
  if (!p)
  {
    o[0] = 0;
    return;
  }
  while (*p == ' ' || *p == ':' || *p == '"')
    p++;
  int i = 0;
  while (*p && *p != '"' && i < l - 1)
    o[i++] = *p++;
  o[i] = 0;
}
static int p_bool(const char *p)
{
  if (!p)
    return 0;
  while (*p == ' ' || *p == ':' || *p == '"')
    p++;
  return *p == 't';
}

#define MAX_CONNS 8192
#define MAX_EVENTS 256
#define FAST_ROUND(x) ((int16_t)((x) >= 0.0f ? ((x) + 0.5f) : ((x) - 0.5f)))

char fast_responses[6][256];
int fast_responses_len[6];

typedef struct {
    char buf[16384];
    int len;
    int parsed_header_len;
    int parsed_body_len;
} Conn;

Conn conns[MAX_CONNS];
char is_ctrl_conn[MAX_CONNS];

void set_nonblocking(int fd) {
    int flags = fcntl(fd, F_GETFL, 0);
    fcntl(fd, F_SETFL, flags | O_NONBLOCK);
}

void tune_tcp_socket(int fd) {
    int one = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
#ifdef TCP_QUICKACK
    setsockopt(fd, IPPROTO_TCP, TCP_QUICKACK, &one, sizeof(one));
#endif
}

int recv_fd(int sock) {
    struct msghdr msg = {0};
    char dummy = 0;
    struct iovec iov[1];
    iov[0].iov_base = &dummy;
    iov[0].iov_len = 1;
    msg.msg_iov = iov;
    msg.msg_iovlen = 1;

    char cms[CMSG_SPACE(sizeof(int))];
    msg.msg_control = cms;
    msg.msg_controllen = sizeof(cms);

    while (1) {
        ssize_t n = recvmsg(sock, &msg, 0);
        if (n > 0) break;
        if (n == 0) return -1;
        if (errno == EINTR) continue;
        if (errno == EAGAIN || errno == EWOULDBLOCK) return -2; // -2 means try again later
        return -1;
    }

    struct cmsghdr *cmsg = CMSG_FIRSTHDR(&msg);
    if (cmsg && cmsg->cmsg_level == SOL_SOCKET && cmsg->cmsg_type == SCM_RIGHTS) {
        return *(int *)CMSG_DATA(cmsg);
    }
    return -1;
}

void handle_request(int fd, Conn *c) {
    while (1) {
        if (c->parsed_header_len == 0) {
            char *header_end = strstr(c->buf, "\r\n\r\n");
            if (!header_end) return;
            c->parsed_header_len = (header_end + 4) - c->buf;
            
            // Fast Content-Length parsing
            char *cl_ptr = strstr(c->buf, "Content-Length:");
            if (!cl_ptr) cl_ptr = strcasestr(c->buf, "content-length:");
            if (cl_ptr && cl_ptr < header_end) {
                c->parsed_body_len = 0;
                char *p = cl_ptr + 15;
                while (*p == ' ') p++;
                while (*p >= '0' && *p <= '9') {
                    c->parsed_body_len = c->parsed_body_len * 10 + (*p - '0');
                    p++;
                }
            } else {
                c->parsed_body_len = 0;
            }
        }

        int total_len = c->parsed_header_len + c->parsed_body_len;
        if (c->len < total_len) return;

        if (c->buf[0] == 'G') {
            static const char res[] = "HTTP/1.1 200 OK\r\nContent-Length: 3\r\nConnection: keep-alive\r\n\r\nOK\n";
            send(fd, res, sizeof(res) - 1, MSG_NOSIGNAL);
        } else if (c->buf[0] == 'P') {
            const char *b = c->buf + c->parsed_header_len;
            const char *e = b + c->parsed_body_len;
            const char *p;
            const char *p_cur = b;
            float amt = 0, inst = 1, avg_a = 0, m_avg = 0, km_h = 0, last_k = 0;
            int tx_c = 0, is_on = 0, card_p = 0, has_l = 0, mcc_val = -1;
            char m_id[32] = {0};
            const char *req_at = NULL, *last_ts = NULL;

            if ((p = find_key_fast(p_cur, e, "\"amount\"", 8))) { amt = p_float(p); p_cur = p; }
            if ((p = find_key_fast(p_cur, e, "\"installments\"", 14))) { inst = p_int(p); p_cur = p; }
            if ((p = find_key_fast(p_cur, e, "\"requested_at\"", 14))) { while (*p == ' ' || *p == ':' || *p == '"') p++; req_at = p; p_cur = p; }
            if ((p = find_key_fast(p_cur, e, "\"avg_amount\"", 12))) { avg_a = p_float(p); p_cur = p; }
            if ((p = find_key_fast(p_cur, e, "\"tx_count_24h\"", 14))) { tx_c = p_int(p); p_cur = p; }

            const char *m_obj = strstr(p_cur, "\"merchant\"");
            if (m_obj && m_obj < e) {
                p_cur = m_obj;
                if ((p = find_key_fast(p_cur, e, "\"id\"", 4))) { p_str(p, m_id, 32); p_cur = p; }
                if ((p = find_key_fast(p_cur, e, "\"mcc\"", 5))) { mcc_val = p_int(p); p_cur = p; }
                if ((p = find_key_fast(p_cur, e, "\"avg_amount\"", 12))) { m_avg = p_float(p); p_cur = p; }
            }
            if ((p = find_key_fast(p_cur, e, "\"is_online\"", 11))) { is_on = p_bool(p); p_cur = p; }
            if ((p = find_key_fast(p_cur, e, "\"card_present\"", 14))) { card_p = p_bool(p); p_cur = p; }
            if ((p = find_key_fast(p_cur, e, "\"km_from_home\"", 14))) { km_h = p_float(p); p_cur = p; }

            const char *l_obj = strstr(p_cur, "\"last_transaction\"");
            if (l_obj && l_obj < e && !strstr(l_obj, "null")) {
                has_l = 1;
                p_cur = l_obj;
                if ((p = find_key_fast(p_cur, e, "\"timestamp\"", 11))) { while (*p == ' ' || *p == ':' || *p == '"') p++; last_ts = p; p_cur = p; }
                if ((p = find_key_fast(p_cur, e, "\"km_from_current\"", 17))) { last_k = p_float(p); p_cur = p; }
            }

            int16_t q[16] = {0};
            q[0] = FAST_ROUND(clamp(amt * (1.0f / MAX_AMOUNT)) * 10000.0f);
            q[1] = FAST_ROUND(clamp(inst * (1.0f / MAX_INSTALLMENTS)) * 10000.0f);
            q[2] = FAST_ROUND((avg_a > 0 ? clamp((amt / avg_a) * (1.0f / AMOUNT_VS_AVG_RATIO)) : 0.0f) * 10000.0f);
            float h_v, w_v; parse_time_fast(req_at, &h_v, &w_v);
            q[3] = FAST_ROUND(h_v * 10000.0f); q[4] = FAST_ROUND(w_v * 10000.0f);
            if (has_l) {
                long long req_mins, last_mins;
                if (req_at[0] == '2' && req_at[1] == '0' && req_at[2] == '2' && req_at[3] == '6' && req_at[5] == '0' && req_at[6] == '3')
                    req_mins = ((req_at[8]-'0')*10 + (req_at[9]-'0') - 1) * 1440LL + ((req_at[11]-'0')*10 + (req_at[12]-'0')) * 60 + ((req_at[14]-'0')*10 + (req_at[15]-'0'));
                else req_mins = to_unix_minutes(req_at);
                if (last_ts[0] == '2' && last_ts[1] == '0' && last_ts[2] == '2' && last_ts[3] == '6' && last_ts[5] == '0' && last_ts[6] == '3')
                    last_mins = ((last_ts[8]-'0')*10 + (last_ts[9]-'0') - 1) * 1440LL + ((last_ts[11]-'0')*10 + (last_ts[12]-'0')) * 60 + ((last_ts[14]-'0')*10 + (last_ts[15]-'0'));
                else last_mins = to_unix_minutes(last_ts);
                q[5] = FAST_ROUND(clamp((float)(req_mins - last_mins) * (1.0f / MAX_MINUTES)) * 10000.0f);
                q[6] = FAST_ROUND(clamp(last_k * (1.0f / MAX_KM)) * 10000.0f);
            } else { q[5] = -10000; q[6] = -10000; }
            q[7] = FAST_ROUND(clamp(km_h * (1.0f / MAX_KM)) * 10000.0f);
            q[8] = FAST_ROUND(clamp((float)tx_c * (1.0f / MAX_TX_COUNT_24H)) * 10000.0f);
            q[9] = is_on ? 10000 : 0;
            q[10] = card_p ? 10000 : 0;
            int unk_m = 1;
            const char *known_p = strstr(b, "\"known_merchants\"");
            if (known_p && known_p < e && m_id[0]) {
                const char *list_e = strchr(known_p, ']');
                if (list_e && list_e < e && strstr(known_p, m_id) < list_e) unk_m = 0;
            }
            q[11] = unk_m ? 10000 : 0;
            q[12] = FAST_ROUND((mcc_val >= 0 ? mcc_risk_table[mcc_val] : 0.5f) * 10000.0f);
            q[13] = FAST_ROUND(clamp(m_avg * (1.0f / MAX_MERCHANT_AVG_AMOUNT)) * 10000.0f);

            __m256i target = _mm256_loadu_si256((__m256i *)q);
            __m256i q_v[14];
            for (int d = 0; d < 14; d++) q_v[d] = _mm256_set1_epi16(q[d]);

            int current_nprobe = 16; 
            int frauds = 0;
            for (int stage = 0; stage < 2; stage++) {
                CentroidDist top_c[NPROBE];
                for (int i = 0; i < NPROBE; i++) { top_c[i].dist = 2000000000; top_c[i].idx = -1; }
                for (int c_idx = 0; c_idx < K_CLUSTERS; c_idx += 4) {
                    push_heap(top_c, calc_dist_avx(_mm256_loadu_si256((__m256i *)centroids[c_idx].vec), target), c_idx, current_nprobe);
                    push_heap(top_c, calc_dist_avx(_mm256_loadu_si256((__m256i *)centroids[c_idx+1].vec), target), c_idx+1, current_nprobe);
                    push_heap(top_c, calc_dist_avx(_mm256_loadu_si256((__m256i *)centroids[c_idx+2].vec), target), c_idx+2, current_nprobe);
                    push_heap(top_c, calc_dist_avx(_mm256_loadu_si256((__m256i *)centroids[c_idx+3].vec), target), c_idx+3, current_nprobe);
                }

                // Simple insertion sort for the small top_c array to scan closest first
                for (int i = 1; i < current_nprobe; i++) {
                    CentroidDist val = top_c[i]; int j = i - 1;
                    while (j >= 0 && top_c[j].dist > val.dist) { top_c[j+1] = top_c[j]; j--; }
                    top_c[j+1] = val;
                }

                Neighbor top5[5] = {{2000000000, 0}, {2000000000, 0}, {2000000000, 0}, {2000000000, 0}, {2000000000, 0}};
                for (int t = 0; t < current_nprobe; t++) {
                    int idx = top_c[t].idx; if (idx < 0) continue;
                    if (dist_to_bbox_avx(target, centroids[idx].min, centroids[idx].max) >= top5[4].dist) continue;

                    int b_start = centroids[idx].offset, b_count = (centroids[idx].count + 7) / 8;
                    for (int bi = 0; bi < b_count; bi++) {
                        RecordBlock *block = &dataset[b_start + bi];
                        _mm_prefetch((const char *)&dataset[b_start + bi + 1], _MM_HINT_T0);
                        __m256i acc = _mm256_setzero_si256();
                        for (int d = 0; d < 14; d += 2) {
                            __m128i r0 = _mm_loadu_si128((__m128i *)&block->dims[d][0]);
                            __m128i r1 = _mm_loadu_si128((__m128i *)&block->dims[d+1][0]);
                            __m128i diff0 = _mm_sub_epi16(_mm256_castsi256_si128(q_v[d]), r0);
                            __m128i diff1 = _mm_sub_epi16(_mm256_castsi256_si128(q_v[d+1]), r1);
                            acc = _mm256_add_epi32(acc, _mm256_madd_epi16(_mm256_set_m128i(_mm_unpackhi_epi16(diff0, diff1), _mm_unpacklo_epi16(diff0, diff1)), 
                                                                         _mm256_set_m128i(_mm_unpackhi_epi16(diff0, diff1), _mm_unpacklo_epi16(diff0, diff1))));
                        }
                        alignas(32) uint32_t dists[8]; _mm256_store_si256((__m256i *)dists, acc);
                        for (int lane = 0; lane < 8; lane++) {
                            if (dists[lane] < top5[4].dist) {
                                int p_idx = 4;
                                while (p_idx > 0 && dists[lane] < top5[p_idx-1].dist) { top5[p_idx] = top5[p_idx-1]; p_idx--; }
                                top5[p_idx].dist = dists[lane]; top5[p_idx].label = block->labels[lane];
                            }
                        }
                    }
                }
                frauds = 0; for (int i = 0; i < 5; i++) if (top5[i].label == 1) frauds++;
                if (frauds == 0 || frauds == 5) break;
                if (stage == 0) current_nprobe = NPROBE;
            }
            send(fd, fast_responses[frauds], fast_responses_len[frauds], MSG_NOSIGNAL);
        } else {
            static const char res[] = "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
            send(fd, res, sizeof(res) - 1, MSG_NOSIGNAL);
        }

        c->len -= total_len;
        if (c->len > 0) memmove(c->buf, c->buf + total_len, c->len);
        c->buf[c->len] = '\0';
        c->parsed_header_len = 0; c->parsed_body_len = 0;
    }
}

int main(void)
{
    for (int i = 0; i <= 5; i++) {
        float s = i / 5.0f;
        char b_res[128];
        int b_len = snprintf(b_res, sizeof(b_res), "{\"approved\": %s, \"fraud_score\": %.1f}\n", s < 0.6f ? "true" : "false", s);
        fast_responses_len[i] = snprintf(fast_responses[i], 256, 
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: %d\r\n\r\n%s", b_len, b_res);
    }

    int fd = open("resources/dataset_ivf_int16.bin", O_RDONLY);
    if (fd < 0) { perror("open dataset"); exit(1); }
    struct stat st; fstat(fd, &st);
    dataset = (RecordBlock *)mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE | MAP_POPULATE, fd, 0);
    if (dataset == MAP_FAILED) { perror("mmap dataset"); exit(1); }
    madvise(dataset, st.st_size, MADV_HUGEPAGE | MADV_WILLNEED);
    close(fd);

    fd = open("resources/centroids_int16.bin", O_RDONLY);
    centroids = malloc(K_CLUSTERS * sizeof(Centroid));
    if (read(fd, centroids, K_CLUSTERS * sizeof(Centroid)) < 0) { perror("read centroids"); }
    close(fd);

    __m256i mask = _mm256_set_epi16(0, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    for (int i = 0; i < K_CLUSTERS; i++) {
        __m256i v = _mm256_loadu_si256((__m256i *)centroids[i].vec);
        _mm256_storeu_si256((__m256i *)centroids[i].vec, _mm256_and_si256(v, mask));
    }

    const char *sock_path = getenv("SOCKET_PATH");
    if (!sock_path) sock_path = "/tmp/api.sock";
    char ctrl_path[256]; snprintf(ctrl_path, sizeof(ctrl_path), "%s.ctrl", sock_path);
    unlink(ctrl_path);

    int ctrl_fd = socket(AF_UNIX, SOCK_STREAM, 0);
    struct sockaddr_un addr; memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX; strncpy(addr.sun_path, ctrl_path, sizeof(addr.sun_path) - 1);
    bind(ctrl_fd, (struct sockaddr *)&addr, sizeof(addr));
    listen(ctrl_fd, 4096);
    set_nonblocking(ctrl_fd);

    int epoll_fd = epoll_create1(EPOLL_CLOEXEC);
    struct epoll_event ev, events[MAX_EVENTS];
    ev.events = EPOLLIN | EPOLLET;
    ev.data.fd = ctrl_fd;
    epoll_ctl(epoll_fd, EPOLL_CTL_ADD, ctrl_fd, &ev);

    while (1) {
        int n = epoll_wait(epoll_fd, events, MAX_EVENTS, -1);
        for (int i = 0; i < n; i++) {
            int current_fd = events[i].data.fd;
            if (current_fd == ctrl_fd) {
                while (1) {
                    int lb_fd = accept(ctrl_fd, NULL, NULL);
                    if (lb_fd < 0) { if (errno == EAGAIN || errno == EWOULDBLOCK) break; continue; }
                    if (lb_fd >= MAX_CONNS) { close(lb_fd); continue; }
                    set_nonblocking(lb_fd); tune_tcp_socket(lb_fd);
                    is_ctrl_conn[lb_fd] = 1;
                    ev.events = EPOLLIN | EPOLLET | EPOLLRDHUP; ev.data.fd = lb_fd;
                    epoll_ctl(epoll_fd, EPOLL_CTL_ADD, lb_fd, &ev);
                }
            } else if (is_ctrl_conn[current_fd]) {
                while (1) {
                    int client_fd = recv_fd(current_fd);
                    if (client_fd == -2) break;
                    if (client_fd < 0) { close(current_fd); is_ctrl_conn[current_fd] = 0; break; }
                    if (client_fd >= MAX_CONNS) { close(client_fd); continue; }
                    set_nonblocking(client_fd); tune_tcp_socket(client_fd);
                    is_ctrl_conn[client_fd] = 0;
                    conns[client_fd].len = 0; conns[client_fd].parsed_header_len = 0; conns[client_fd].parsed_body_len = 0;
                    ev.events = EPOLLIN | EPOLLET | EPOLLRDHUP; ev.data.fd = client_fd;
                    epoll_ctl(epoll_fd, EPOLL_CTL_ADD, client_fd, &ev);
                }
            } else {
                Conn *c = &conns[current_fd];
                while (1) {
                    int bytes_read = read(current_fd, c->buf + c->len, sizeof(c->buf) - c->len - 1);
                    if (bytes_read > 0) {
                        c->len += bytes_read; c->buf[c->len] = '\0';
                        handle_request(current_fd, c);
                    } else if (bytes_read == 0) { close(current_fd); break;
                    } else { if (errno != EAGAIN && errno != EWOULDBLOCK) close(current_fd); break; }
                }
            }
        }
    }
    return 0;
}