import requests
import os

BASE_URL = "http://127.0.0.1:5000"
K = 10

test_pairs = [
    {
        "text": "A man with gauges and glasses is wearing a Blitz hat .", 
        "image": "1007129816_e794419615.jpg"
    },
    {
        "text": "Dog with orange ball at feet , stands on shore shaking off water", 
        "image": "1012212859_01547e3f17.jpg"
    },
    {
        "text": "A man in a brown shirt and dark shorts plays on the beach with his two black dogs .", 
        "image": "1075867198_27ca2e7efe.jpg"
    }
]

# CHANGE THIS if your flickr images are in a differently named folder!
FLICKR_FOLDER_NAME = "Flickr8k_Dataset" 

def calculate_metrics(ranks):
    # FIXED: Prevent ZeroDivisionError if ranks list is empty
    if not ranks:
        return 0.0, 0.0
        
    ap_sum = 0.0
    hits = 0
    for r in ranks:
        if r != -1 and r <= K:
            hits += 1
            ap_sum += (1.0 / r)
            
    recall = (hits / len(ranks)) * 100
    map_score = (ap_sum / len(ranks)) * 100
    return recall, map_score

print("==================================================")
print(" MULTIMODAL EVALUATION - IEEE REPORT GENERATOR")
print("==================================================\n")

print(">>> RUNNING TEXT-TO-IMAGE EVALUATION...")
t2i_ranks = []
for idx, pair in enumerate(test_pairs, 1):
    text, target_img = pair["text"], pair["image"]
    try:
        res = requests.post(f"{BASE_URL}/search_text", json={"query": text, "k": K}).json()
        
        # Check for the filename
        rank = next((i + 1 for i, r in enumerate(res.get("results", [])) if target_img in r["filename"]), -1)
        t2i_ranks.append(rank)
        print(f"  [Query {idx}] Rank: {f'#{rank}' if rank != -1 else 'Not in Top 10'} | Text: {text[:40]}...")
        
        # DEBUG: If it fails, print what the engine ACTUALLY returned as #1
        if rank == -1 and res.get("results"):
            print(f"      -> [Debug] The #1 result was actually: {res['results'][0]['filename']}")
            
    except Exception as e:
        print(f"  [Error] Is Flask running? {e}")

t2i_recall, t2i_map = calculate_metrics(t2i_ranks)


print("\n>>> RUNNING IMAGE-TO-TEXT EVALUATION...")
i2t_ranks = []
for idx, pair in enumerate(test_pairs, 1):
    text, target_img = pair["text"], pair["image"]
    img_path = os.path.join(FLICKR_FOLDER_NAME, target_img)
    
    if not os.path.exists(img_path):
        print(f"  [Error] Cannot find image locally at: {img_path}")
        continue
        
    try:
        with open(img_path, "rb") as img_file:
            res = requests.post(f"{BASE_URL}/search_image_to_text", files={"query_image": img_file}, data={"k": K}).json()
            rank = next((i + 1 for i, r in enumerate(res.get("results", [])) if text.strip().lower() in r["caption"].strip().lower()), -1)
            i2t_ranks.append(rank)
            print(f"  [Query {idx}] Rank: {f'#{rank}' if rank != -1 else 'Not in Top 10'} | Image: {target_img}")
    except Exception as e:
        print(f"  [Error] API call failed: {e}")

i2t_recall, i2t_map = calculate_metrics(i2t_ranks)

print("\n==================================================")
print(" FORMATTED RESULTS FOR IEEE REPORT TABLE")
print("==================================================")
print(f"Metric       | Text-to-Image (T2I) | Image-to-Text (I2T)")
print(f"-------------|---------------------|--------------------")
print(f"Recall@10    | {t2i_recall:17.1f}% | {i2t_recall:17.1f}%")
print(f"mAP          | {t2i_map:17.1f}% | {i2t_map:17.1f}%")
print("==================================================")