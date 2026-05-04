FROM ubuntu:24.04 AS builder

RUN apt-get update && apt-get install -y gcc make python3

WORKDIR /app
COPY . /app

# Run preprocessing to generate dataset_uint8.bin
RUN python3 preprocess.py

# Compile API with AVX2 and fast math for optimal distance calculation
RUN gcc -O3 -march=x86-64-v3 -ffast-math -flto -o api src/main.c src/mongoose.c src/cJSON.c -lm

FROM ubuntu:24.04
WORKDIR /app
COPY --from=builder /app/api /app/api
COPY --from=builder /app/resources/dataset_uint8.bin /app/resources/dataset_uint8.bin

EXPOSE 9999
CMD ["/app/api"]
