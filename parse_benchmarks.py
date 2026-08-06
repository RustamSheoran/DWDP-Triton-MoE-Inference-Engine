import json
import glob
import os

results = []
for filepath in glob.glob("benchmark-results/**/report.json", recursive=True):
    with open(filepath, 'r') as f:
        try:
            data = json.load(f)
            
            # Extract basic info
            env = data.get("environment", {})
            gpu_info = env.get("gpu", {}).get("name", "Unknown")
            
            metadata = data.get("metadata", {})
            model_id = metadata.get("model_id", "Unknown")
            run_name = metadata.get("run_name", "Unknown")
            date = metadata.get("timestamp", "Unknown")
            
            config = data.get("config", {})
            batch_size = config.get("batch_size", "Unknown")
            seq_len = config.get("sequence_length", "Unknown")
            precision = config.get("dtype", "Unknown")
            
            # Find comparison table info
            summary = data.get("summary", {})
            metrics = data.get("metrics", {})
            
            # Some older/different benchmarks might have different formats
            dwdp_results = metrics.get("dwdp", {})
            hf_results = metrics.get("baseline", {})
            
            # Extract latency, tok/s, VRAM, throughput
            # Example format if we don't know it yet, let's print the keys
            dwdp_latency = dwdp_results.get("latency_ms", "Unknown")
            hf_latency = hf_results.get("latency_ms", "Unknown")
            
            improvement = summary.get("speedup", "Unknown")
            
            results.append({
                "dir": os.path.dirname(filepath),
                "gpu": gpu_info,
                "model": model_id,
                "precision": precision,
                "batch_size": batch_size,
                "seq_len": seq_len,
                "dwdp_latency": dwdp_latency,
                "hf_latency": hf_latency,
                "improvement": improvement,
                "date": date
            })
        except Exception as e:
            print(f"Error parsing {filepath}: {e}")

print("dir,gpu,model,precision,batch,seq,dwdp_lat,hf_lat,speedup,date")
for r in results:
    print(f"{r['dir']},{r['gpu']},{r['model']},{r['precision']},{r['batch_size']},{r['seq_len']},{r['dwdp_latency']},{r['hf_latency']},{r['improvement']},{r['date']}")

