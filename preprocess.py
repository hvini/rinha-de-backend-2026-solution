import json
import gzip
import struct

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
            if i % 10 == 9: f.write("\n")
        f.write("};\n\n")
        f.write("#endif\n")

    print("Generating Max-Precision 16-Bit dataset...")
    with gzip.open('resources/references.json.gz', 'rt') as f:
        data = json.load(f)
    
    records = []
    for record in data:
        vec = record['vector']
        label = 1 if record['label'] == 'fraud' else 0
        
        buf = bytearray()
        for v in vec:
            if v < -1.0: v = -1.0
            if v > 1.0: v = 1.0
            val = int(round(v * 10000.0))
            buf += struct.pack('<h', val)
            
        buf += struct.pack('<B', label)
        buf += b'\x00\x00\x00' # 3 bytes padding to reach exactly 32 bytes
        records.append(buf)
        
    with open('resources/dataset_int16.bin', 'wb') as out_f:
        for r in records: out_f.write(r)
            
    print("Done! Output optimized for MAX AVX2 bounds.")

if __name__ == '__main__':
    main()