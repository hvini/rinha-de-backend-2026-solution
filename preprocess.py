import json
import gzip
import sys
import random

def main():
    print("Loading normalization.json...")
    with open('resources/normalization.json', 'r') as f:
        norm = json.load(f)

    print("Loading mcc_risk.json...")
    with open('resources/mcc_risk.json', 'r') as f:
        mcc = json.load(f)

    print("Generating src/generated_config.h...")
    with open('src/generated_config.h', 'w') as f:
        f.write("#ifndef GENERATED_CONFIG_H\n")
        f.write("#define GENERATED_CONFIG_H\n\n")
        f.write(f"#define MAX_AMOUNT {norm['max_amount']}.0f\n")
        f.write(f"#define MAX_INSTALLMENTS {norm['max_installments']}.0f\n")
        f.write(f"#define AMOUNT_VS_AVG_RATIO {norm['amount_vs_avg_ratio']}.0f\n")
        f.write(f"#define MAX_MINUTES {norm['max_minutes']}.0f\n")
        f.write(f"#define MAX_KM {norm['max_km']}.0f\n")
        f.write(f"#define MAX_TX_COUNT_24H {norm['max_tx_count_24h']}.0f\n")
        f.write(f"#define MAX_MERCHANT_AVG_AMOUNT {norm['max_merchant_avg_amount']}.0f\n\n")
        
        f.write("static const float mcc_risk_table[10000] = {\n")
        for i in range(10000):
            mcc_code = f"{i:04d}"
            risk = mcc.get(mcc_code, 0.5)
            f.write(f"    {risk}f,")
            if i % 10 == 9:
                f.write("\n")
        f.write("};\n\n")
        f.write("#endif\n")

    print("Generating dataset_uint8.bin from references.json.gz...")
    with gzip.open('resources/references.json.gz', 'rt') as f:
        data = json.load(f)

    print("Quantizing and building KD-Tree in-memory...")
    
    # 1. Load and quantize all records
    records = []
    for record in data:
        vec = record['vector']
        label = 1 if record['label'] == 'fraud' else 0
        
        buf = bytearray(16)
        for i, v in enumerate(vec):
            if v < -1.0: v = -1.0
            if v > 1.0: v = 1.0
            val = int(round((v + 1.0) * 127.0))
            buf[i] = val
        buf[14] = label
        buf[15] = 0
        records.append(buf)
        
    print("Sorting dataset by dimension 0 for early stopping...")
    records.sort(key=lambda x: x[0])
    
    print("Writing 1D-sorted array to dataset_uint8.bin...")
    with open('resources/dataset_uint8.bin', 'wb') as out_f:
        for r in records:
            out_f.write(r)
            
    print("Done generating 1D-sorted dataset_uint8.bin!")

if __name__ == '__main__':
    main()
