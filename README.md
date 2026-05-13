# Rinha de Backend 2026 - C SOA/IVF Implementation

This repository contains a C-based solution for the Rinha de Backend 2026 competition. The competition consists of a fraud detection system using array similarities (vectors). The implementation uses deep hardware optimization, including AVX2-accelerated vector search, custom Block-SOA data layouts, and a zero-copy networking stack.

## Technical Architecture

### 0. Load Balancing
The solution utilizes the **[SoNoForevis](https://github.com/jairoblatt/SoNoForevis)** load balancer, specifically designed for extreme low-latency handoffs.
- **FD Passing**: Unlike traditional reverse proxies that copy data between sockets, SoNoForevis uses `SCM_RIGHTS` to pass the client's TCP file descriptor directly to the C workers via Unix Domain Sockets.
- **Zero Overhead**: Once the `fd` is passed, the API worker communicates directly with the client, eliminating the load balancer from the data path and reducing tail latency.

### 1. Data Layout: Block-Transposed SOA
To maximize SIMD (Single Instruction, Multiple Data) efficiency, the dataset is stored in a **Structure of Arrays (SOA)** format within 256-byte blocks.
- **Parallel Processing**: Each block contains 8 records. By transposing the data, we load 8 features into a single AVX2 register.
- **Throughput**: Distance calculations are performed on 8 records simultaneously using `_mm256_madd_epi16`, octupling the search speed compared to standard record-based layouts.

### 2. Search Algorithm: Adaptive IVF
We implement a two-stage Inverted File Index (IVF) search to balance speed and accuracy:
- **Fast Path (16 Probes)**: Initially scans only the 16 closest clusters.
- **Early Exit**: if the top 5 neighbors are unanimously fraud (5/5) or legit (0/5), the server returns immediately.
- **Full Path (128 Probes)**: If the result is ambiguous, the search expands to 128 clusters to ensure precision.

### 3. Geometric Pruning: Bounding Boxes (BBOX)
Each cluster in the index is assigned a 14-dimensional bounding box (`min`/`max`).
- **Pruning**: Before scanning a cluster's records, we calculate the minimum possible distance to its bounding box.
- **Efficiency**: If the BBOX distance is greater than the current 5th-best neighbor, the entire cluster is skipped, significantly reducing CPU cycles.

### 4. Optimized Event Loop
The server uses a custom `epoll` loop designed for high concurrency:
- **Edge-Triggered (EPOLLET)**: Minimizes system call overhead by reducing the number of events returned by the kernel.
- **Zero-Copy Parsing**: HTTP requests are parsed in-place within the read buffer. No memory allocations (`malloc`) occur during the request lifecycle.
- **UDS Descriptor Passing**: Integrates with the `SoNoForevis` load balancer using Unix Domain Sockets and `SCM_RIGHTS` for zero-overhead connection handoffs.

### 5. Kernel & System Tuning
- **Memory Mapping**: The dataset is `mmap`'d with `MAP_POPULATE` and `MADV_HUGEPAGE` to eliminate page faults and reduce TLB pressure.
- **TCP Stack**: Fine-tuned using `TCP_NODELAY` and `TCP_QUICKACK` to eliminate Nagle-induced latencies.

## API Specification

### POST `/fraud-score`
Analyzes a transaction for fraud risk.
- **Request**: JSON payload with transaction details (amount, installments, merchant info, etc.).
- **Response**: JSON payload with `approved` (boolean) and `fraud_score` (0.0 to 1.0).

## Usage

### Local Build
Ensure you have an x86_64-v3 compatible CPU (Haswell or newer).
```bash
# Generate configuration and index
python3 preprocess.py
gcc -O3 -march=x86-64-v3 -ffast-math -flto -o build_index src/build_index.c -lm
./build_index

# Build and run the API
gcc -O3 -march=x86-64-v3 -ffast-math -flto -o api src/main.c -lm
./api
```

### Docker Deployment
```bash
docker-compose up --build
```

## Build Requirements
- **Compiler**: GCC 11+
- **Flags**: `-O3 -march=x86-64-v3 -ffast-math -flto -mavx2`
- **Architecture**: Requires x86_64-v3 (Haswell or newer) for AVX2 support.

## Project Structure
- `src/main.c`: The core API server and search engine.
- `src/build_index.c`: Tooling to cluster data and generate the SOA-transposed index.
- `preprocess.py`: Data normalization and configuration generation.
