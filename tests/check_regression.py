import sys
import json

def check_tokens(baseline, current):
    # Both are lists of dicts representing different presets. We will just check the first ('small')
    b_session = baseline[0]["session"]
    c_session = current[0]["session"]
    
    b_warm = b_session["warm_total"]
    c_warm = c_session["warm_total"]
    
    print(f"Tokens baseline warm total: ${b_warm:.4f}")
    print(f"Tokens current warm total:  ${c_warm:.4f}")
    
    if c_warm > b_warm * 1.01: # More than 1% cost increase
        print("ERROR: Token costs increased by > 1%!")
        sys.exit(1)
    print("Token costs OK.")

def check_retrieval(baseline, current):
    # Both are lists of instance dicts
    # We want to check aggregate Hit@10 and Recall@10

    def instance_keys(data):
        return [(r.get("repo"), r.get("instance_id")) for r in data]

    if instance_keys(baseline) != instance_keys(current):
        print("ERROR: Retrieval benchmark slice does not match baseline.")
        print(f"Baseline instances: {instance_keys(baseline)}")
        print(f"Current instances:  {instance_keys(current)}")
        sys.exit(1)
    
    def avg_metric(data, key, k):
        k_str = str(k)
        if not data: return 0.0
        return sum(r["metrics"][key][k_str] for r in data) / len(data)

    b_hit = avg_metric(baseline, "hit_at_k", 10)
    c_hit = avg_metric(current, "hit_at_k", 10)
    
    b_recall = avg_metric(baseline, "recall_at_k", 10)
    c_recall = avg_metric(current, "recall_at_k", 10)
    
    print(f"Retrieval baseline Hit@10: {b_hit:.3f}, Recall@10: {b_recall:.3f}")
    print(f"Retrieval current Hit@10:  {c_hit:.3f}, Recall@10: {c_recall:.3f}")
    
    if c_hit < b_hit - 0.02 or c_recall < b_recall - 0.02:
        print("ERROR: Retrieval quality dropped by > 2%!")
        sys.exit(1)
        
    print("Retrieval quality OK.")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python check_regression.py <type> <baseline.json> <current.json>")
        sys.exit(1)
        
    check_type, base_file, curr_file = sys.argv[1:4]
    
    def load_json_forgiving(filepath):
        with open(filepath, "rb") as f:
            raw = f.read()
        if raw.startswith(b'\xff\xfe') or raw.startswith(b'\xfe\xff'):
            return json.loads(raw.decode("utf-16"))
        return json.loads(raw.decode("utf-8"))
        
    baseline = load_json_forgiving(base_file)
    current = load_json_forgiving(curr_file)
        
    if check_type == "tokens":
        check_tokens(baseline, current)
    elif check_type == "retrieval":
        check_retrieval(baseline, current)
    else:
        print(f"Unknown type: {check_type}")
        sys.exit(1)
