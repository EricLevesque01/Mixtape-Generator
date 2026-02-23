import json

r1 = [json.loads(l) for l in open('eval_results_run1.jsonl') if l.strip()]
r2 = [json.loads(l) for l in open('eval_results.jsonl') if l.strip()]

# Remove timing field
for r in r1 + r2:
    r.pop('processing_time_ms', None)

print(f"DETERMINISTIC (excl timing): {r1 == r2}")
for a, b in zip(r1, r2):
    match = a == b
    print(f"  {a['case_id']}: score={a['final_score']} tracks={a['track_count']} dur={a['playlist_duration_s']} match={match}")
    if not match:
        for k in a:
            if a[k] != b.get(k):
                print(f"    DIFF {k}: {a[k]} != {b.get(k)}")
