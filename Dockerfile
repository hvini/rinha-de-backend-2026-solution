FROM ubuntu:24.04 AS builder

RUN apt-get update && apt-get install -y gcc make python3

WORKDIR /app
COPY . /app

# Run preprocessing
RUN python3 preprocess.py

# Build the spatial index in C (Executes K-means clustering)
RUN gcc -O3 -march=x86-64-v3 -ffast-math -flto -o build_index src/build_index.c && ./build_index

# Compile API
RUN gcc -O3 -march=x86-64-v3 -ffast-math -flto -DMG_ENABLE_EPOLL=1 -DMG_ENABLE_LOG=0 -o api src/main.c src/mongoose.c src/cJSON.c -lm

FROM ubuntu:24.04
WORKDIR /app
COPY --from=builder /app/api /app/api
# Map the NEW clustered files
COPY --from=builder /app/resources/dataset_ivf.bin /app/resources/dataset_ivf.bin
COPY --from=builder /app/resources/centroids.bin /app/resources/centroids.bin

EXPOSE 9999
CMD ["/app/api"]