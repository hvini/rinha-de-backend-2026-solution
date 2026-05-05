FROM ubuntu:24.04 AS builder

RUN apt-get update && apt-get install -y gcc make python3

WORKDIR /app
COPY . /app

# Run preprocessing (Generates resources/dataset_int16.bin)
RUN python3 preprocess.py

# Build the spatial index in C (Generates resources/dataset_ivf_int16.bin & centroids_int16.bin)
RUN gcc -O3 -march=x86-64-v3 -ffast-math -flto -o build_index src/build_index.c && ./build_index

# Compile API
RUN gcc -O3 -march=x86-64-v3 -ffast-math -flto -DMG_ENABLE_EPOLL=1 -DMG_ENABLE_LOG=0 -o api src/main.c src/mongoose.c src/cJSON.c -lm

FROM ubuntu:24.04
WORKDIR /app
COPY --from=builder /app/api /app/api

COPY --from=builder /app/resources/dataset_ivf_int16.bin /app/resources/dataset_ivf_int16.bin
COPY --from=builder /app/resources/centroids_int16.bin /app/resources/centroids_int16.bin

EXPOSE 9999
CMD ["/app/api"]