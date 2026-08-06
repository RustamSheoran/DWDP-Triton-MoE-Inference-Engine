import json
import glob
import os

results = []
for filepath in glob.glob("benchmark-results/**/report.json", recursive=True):
    with open(filepath, 'r') as f:
        try:
            data = json.load(f)
            
            env = data.get("environment", {})
            gpu = env.get("gpu_model", "Unknown")
            
            config = data.get("config", {})
            model = config.get("model_name", "Unknown")
            batch = config.get("batch_size", "Unknown")
            seq = config.get("sequence_length", "Unknown")
            precision = env.get("precision", "Unknown")
            date = env.get("benchmark_timestamp", "Unknown")
            
            perf = data.get("performance", {})
            dwdp = perf.get("dwdp", {})
            hf = perf.get("huggingface", {})
            
            d_ttft = dwdp.get("ttft_ms", "Unknown")
            d_dec = dwdp.get("decode_latency_ms", "Unknown")
            d_pref = dwdp.get("prefill_latency_ms", "Unknown")
            d_tps = dwdp.get("tokens_per_second", "Unknown")
            d_vram = dwdp.get("memory", {}).get("peak_gpu_memory_bytes", "Unknown")
            d_lat = dwdp.get("total_runtime_ms", "Unknown")
            
            h_ttft = hf.get("ttft_ms", "Unknown")
            h_dec = hf.get("decode_latency_ms", "Unknown")
            h_pref = hf.get("prefill_latency_ms", "Unknown")
            h_tps = hf.get("tokens_per_second", "Unknown")
            h_vram = hf.get("memory", {}).get("peak_gpu_memory_bytes", "Unknown")
            h_lat = hf.get("total_runtime_ms", "Unknown")
            
            obs = data.get("observations", [])
            improvement = "Unknown"
            if len(obs) > 0:
                improvement = " ".join(obs)
            
            results.append({
                "dir": os.path.basename(os.path.dirname(filepath)),
                "gpu": gpu,
                "model": model,
                "precision": precision,
                "batch_size": batch,
                "seq_len": seq,
                "d_ttft": f"{d_ttft:.2f}" if isinstance(d_ttft, float) else d_ttft,
                "d_dec": f"{d_dec:.2f}" if isinstance(d_dec, float) else d_dec,
                "d_pref": f"{d_pref:.2f}" if isinstance(d_pref, float) else d_pref,
                "d_tps": f"{d_tps:.2f}" if isinstance(d_tps, float) else d_tps,
                "d_vram": d_vram,
                "d_lat": f"{d_lat:.2f}" if isinstance(d_lat, float) else d_lat,
                "h_ttft": f"{h_ttft:.2f}" if isinstance(h_ttft, float) else h_ttft,
                "h_dec": f"{h_dec:.2f}" if isinstance(h_dec, float) else h_dec,
                "h_pref": f"{h_pref:.2f}" if isinstance(h_pref, float) else h_pref,
                "h_tps": f"{h_tps:.2f}" if isinstance(h_tps, float) else h_tps,
                "h_vram": h_vram,
                "h_lat": f"{h_lat:.2f}" if isinstance(h_lat, float) else h_lat,
                "date": date,
                "obs": improvement
            })
        except Exception as e:
            print(f"Error parsing {filepath}: {e}")

print(f"{'dir':<40} | {'gpu':<10} | {'batch':<5} | {'seq':<5} | {'d_lat':<8} | {'h_lat':<8} | {'d_tps':<6} | {'h_tps':<6} | {'d_vram':<12} | {'h_vram':<12} | {'date'}")
for r in sorted(results, key=lambda x: x['dir']):
    print(f"{r['dir'][:38]:<40} | {r['gpu']:<10} | {r['batch_size']:<5} | {r['seq_len']:<5} | {r['d_lat']:<8} | {r['h_lat']:<8} | {r['d_tps']:<6} | {r['h_tps']:<6} | {r['d_vram']:<12} | {r['h_vram']:<12} | {r['date']}")

