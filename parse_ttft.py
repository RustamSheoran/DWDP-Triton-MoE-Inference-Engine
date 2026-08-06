import json
import glob
import os

results = []
for filepath in glob.glob("benchmark-results/**/report.json", recursive=True):
    with open(filepath, 'r') as f:
        try:
            data = json.load(f)
            perf = data.get("performance", {})
            dwdp = perf.get("dwdp", {})
            hf = perf.get("huggingface", {})
            
            d_ttft = dwdp.get("ttft_ms")
            h_ttft = hf.get("ttft_ms")
            
            d_pref = dwdp.get("prefill_latency_ms")
            h_pref = hf.get("prefill_latency_ms")
            
            if d_ttft and h_ttft:
                ttft_impr = (h_ttft - d_ttft) / h_ttft * 100
            else:
                ttft_impr = 0
                
            if d_pref and h_pref:
                pref_impr = (h_pref - d_pref) / h_pref * 100
            else:
                pref_impr = 0
                
            results.append({
                "dir": os.path.basename(os.path.dirname(filepath)),
                "d_ttft": d_ttft, "h_ttft": h_ttft, "ttft_impr": ttft_impr,
                "d_pref": d_pref, "h_pref": h_pref, "pref_impr": pref_impr
            })
        except Exception as e:
            pass

print(f"{'dir':<40} | {'d_ttft':<10} | {'h_ttft':<10} | {'impr%':<10} | {'d_pref':<10} | {'h_pref':<10} | {'impr%':<10}")
for r in sorted(results, key=lambda x: x['dir']):
    print(f"{r['dir'][:38]:<40} | {str(r['d_ttft'])[:10]:<10} | {str(r['h_ttft'])[:10]:<10} | {r['ttft_impr']:.2f}     | {str(r['d_pref'])[:10]:<10} | {str(r['h_pref'])[:10]:<10} | {r['pref_impr']:.2f}")

