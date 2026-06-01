import numpy as np
import cv2
cv = cv2
import os
import time
import operator
from skimage import img_as_ubyte
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.transform import resize
from scipy.spatial.distance import euclidean


# --- Fonctions de distance / similarité ---

def euclidian_distance(x,y):
    return np.sqrt(np.sum((x-y)**2))

def manhattan_distance(x,y):
    return np.sum(np.abs(x-y))

def chebyshev_distance(x,y):
    return np.max(np.abs(x-y))

def minkowski_distance(x,y,p):
    return np.sum(np.abs(x-y)**p)**(1/p)

def cosine_similarity(x,y):
    return np.dot(x,y)/(np.linalg.norm(x)*np.linalg.norm(y))

def chi_squared_distance(x,y):
    return np.sum((x-y)**2/(x+y+1e-10))

def intersection_distance(x,y):
    return 1 - np.sum(np.minimum(x,y)) / (np.sum(x) + 1e-10)

def bhattacharyya_distance(x,y):
    x_norm = np.abs(x) / (np.sum(np.abs(x)) + 1e-10)
    y_norm = np.abs(y) / (np.sum(np.abs(y)) + 1e-10)
    coefficient = np.sum(np.sqrt(x_norm * y_norm))
    return -np.log(coefficient + 1e-10)

def bhatta(x,y):
    return bhattacharyya_distance(x,y)

def chiSquareDistance(x, y):
    return np.sum((x - y)**2 / (x + y + 1e-10))

def bruteForceMatching(des1, des2, is_binary=False):
    if is_binary:
        bf = cv.BFMatcher(cv.NORM_HAMMING)
        des1 = np.uint8(np.round(des1))
        des2 = np.uint8(np.round(des2))
    else:
        bf = cv.BFMatcher(cv.NORM_L2)
    matches = bf.knnMatch(des1, des2, k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append([m])
    return len(good)

def flann(des1, des2, is_binary=False):
    if is_binary:
        index_params = dict(algorithm=6,
                            table_number=6,
                            key_size=12,
                            multi_probe_level=1)
        des1 = np.uint8(np.round(des1))
        des2 = np.uint8(np.round(des2))
    else:
        index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    matcher = cv.FlannBasedMatcher(index_params, search_params)
    matches = matcher.knnMatch(des1, des2, k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append([m])
    return len(good)


# --- Descripteurs ---

def generateHistogramme_HSV(image):
    # Pixel counts are integers — return int32 so np.savetxt uses '%d' (writes "256" not "2.56e+02")
    start_time = time.time()
    img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    histH = cv2.calcHist([img],[0],None,[180],[0,180])
    histS = cv2.calcHist([img],[1],None,[256],[0,256])
    histV = cv2.calcHist([img],[2],None,[256],[0,256])
    feature = np.concatenate((histH, np.concatenate((histS,histV),axis=None)),axis=None).astype(np.int32)
    print(f"Hist HSV terminé en {time.time()-start_time}s")
    return feature

def generateHistogramme_Color(image):
    # Pixel counts are integers — return int32 so np.savetxt uses '%d'
    start_time = time.time()
    histB = cv2.calcHist([image],[0],None,[256],[0,256])
    histG = cv2.calcHist([image],[1],None,[256],[0,256])
    histR = cv2.calcHist([image],[2],None,[256],[0,256])
    feature = np.concatenate((histB, np.concatenate((histG,histR),axis=None)),axis=None).astype(np.int32)
    print(f"Hist Couleur terminé en {time.time()-start_time}s")
    return feature

def generateSIFT(image):
    # nfeatures=200 caps keypoints (200×128 max vs ~500×128 default)
    # Returns uint8 array — SIFT bins are integers 0-255, so no precision is lost
    # and np.savetxt writes "128" instead of "1.280000000000000000e+02", ~6x smaller files
    start_time = time.time()
    sift = cv2.SIFT_create(nfeatures=200)
    key_point1, descrip1 = sift.detectAndCompute(image, None)
    if descrip1 is not None:
        descrip1 = descrip1.astype(np.uint8)
    print(f"SIFT terminé en {time.time()-start_time}s")
    return descrip1

def generateORB(image):
    # nfeatures=200 matches SIFT cap; uint8 cast ensures '%d' format in make_descriptor_as_files
    start_time = time.time()
    orb = cv2.ORB_create(nfeatures=200)
    key_point1, descrip1 = orb.detectAndCompute(image, None)
    if descrip1 is not None:
        descrip1 = descrip1.astype(np.uint8)
    print(f"ORB terminé en {time.time()-start_time}s")
    return descrip1

def generateGLCM(image):
    start_time = time.time()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = img_as_ubyte(gray)
    distances = [1, 2]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    glcmMatrix = graycomatrix(gray, distances=distances, angles=angles, normed=True)
    glcmProperties1 = graycoprops(glcmMatrix, "contrast").ravel()
    glcmProperties2 = graycoprops(glcmMatrix, "dissimilarity").ravel()
    glcmProperties3 = graycoprops(glcmMatrix, "homogeneity").ravel()
    glcmProperties4 = graycoprops(glcmMatrix, "energy").ravel()
    glcmProperties5 = graycoprops(glcmMatrix, "correlation").ravel()
    glcmProperties6 = graycoprops(glcmMatrix, "ASM").ravel()
    feature = np.array([
        glcmProperties1, glcmProperties2, glcmProperties3,
        glcmProperties4, glcmProperties5, glcmProperties6
    ]).ravel()
    print(f"GLCM terminé en {time.time()-start_time}s")
    return feature

def generateLBP(image):
    start_time = time.time()
    points = 8
    radius = 1
    method = 'default'
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (350, 350))
    fullLBPmatrix = local_binary_pattern(img, points, radius, method)
    histograms, edges = np.histogram(fullLBPmatrix.ravel(), bins=10, range=(0, 2**points))
    histograms = histograms.astype("float")
    histograms /= histograms.sum() + 1e-6
    print(f"LBP terminé en {time.time()-start_time}s")
    return histograms

def generateHOG(image):
    start_time = time.time()
    cellSize = (25, 25)
    blockSize = (50, 50)
    blockStride = (25, 25)
    nBins = 9
    winSize = (350, 350)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, winSize)
    hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nBins)
    vect_features = hog.compute(img)
    print(f"HOG terminé en {time.time()-start_time}s")
    return vect_features

def generateCLIP(image):
    start_time = time.time()
    from clip_model import get_embedder
    embedder = get_embedder()
    feature = embedder.encode_image(image)
    print(f"CLIP terminé en {time.time()-start_time}s")
    return feature

# singleton ViT
_vit_model = None
_vit_processor = None

def _load_vit():
    global _vit_model, _vit_processor
    if _vit_model is None:
        import torch
        from transformers import ViTModel, ViTImageProcessor
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _vit_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
        _vit_model = ViTModel.from_pretrained("google/vit-base-patch16-224").to(device)

        weights_dir = os.environ.get(
            "MIR_WEIGHTS_PATH",
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights")
        )
        finetuned_path = os.path.join(weights_dir, "vit_finetuned.pth")
        if os.path.exists(finetuned_path):
            print(f"[ViT] Chargement poids fine-tunés: {finetuned_path}")
            state_dict = torch.load(finetuned_path, map_location=device, weights_only=True)
            result = _vit_model.load_state_dict(state_dict, strict=False)
            if result.missing_keys:
                print(f"[ViT] {len(result.missing_keys)} clé(s) manquante(s)")
            print(f"[ViT] Modèle chargé sur {device}.")
        else:
            print(f"[ViT] Pas de poids fine-tunés, utilisation du pretrained sur {device}.")

        _vit_model.eval()
    return _vit_model, _vit_processor

def generateViT(image):
    """Extraction vecteur ViT 768-dim normalisé L2. Entrée: image BGR (numpy)."""
    start_time = time.time()
    import torch
    from PIL import Image as PILImage

    model, processor = _load_vit()
    device = next(model.parameters()).device

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_img = PILImage.fromarray(rgb)

    inputs = processor(images=pil_img, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)

    with torch.no_grad():
        outputs = model(pixel_values=pixel_values)
        patch_embeddings = outputs.last_hidden_state[:, 1:, :]
        mean_embedding = patch_embeddings.mean(dim=1)

    mean_embedding = mean_embedding / mean_embedding.norm(dim=-1, keepdim=True)
    feature = mean_embedding.cpu().numpy().flatten().astype(np.float32)

    print(f"ViT terminé en {time.time()-start_time:.3f}s")
    return feature

# singleton ResNet50
_resnet_model = None

def _load_resnet():
    global _resnet_model
    if _resnet_model is None:
        import torch
        import torch.nn as nn
        from torchvision import models
        device = "cuda" if torch.cuda.is_available() else "cpu"
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        weights_dir = os.environ.get(
            "MIR_WEIGHTS_PATH",
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights")
        )
        finetuned_path = os.path.join(weights_dir, "resnet_finetuned.pth")
        if os.path.exists(finetuned_path):
            print(f"[ResNet50] Chargement poids fine-tunés: {finetuned_path}")
            state_dict = torch.load(finetuned_path, map_location=device, weights_only=True)
            resnet.load_state_dict(state_dict, strict=False)
            print(f"[ResNet50] Modèle chargé sur {device}.")
        else:
            print(f"[ResNet50] Pas de poids fine-tunés, utilisation du pretrained sur {device}.")

        _resnet_model = nn.Sequential(*list(resnet.children())[:-1]).to(device)
        _resnet_model.eval()
    return _resnet_model

def generateResNet(image):
    """Extraction vecteur ResNet50 2048-dim normalisé L2. Entrée: image BGR (numpy)."""
    start_time = time.time()
    import torch
    from torchvision import transforms
    from PIL import Image as PILImage

    model = _load_resnet()
    device = next(model.parameters()).device

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_img = PILImage.fromarray(rgb)

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    input_tensor = preprocess(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        features = model(input_tensor)

    features = features.squeeze()
    features = features / features.norm()
    feature = features.cpu().numpy().astype(np.float32)

    print(f"ResNet50 terminé en {time.time()-start_time:.3f}s")
    return feature

def extractReqFeatures(fileName, algo_choice):
    print(algo_choice)
    if fileName:
        img = cv2.imread(fileName)
        resized_img = resize(img, (128*4, 64*4))

        if algo_choice == 1:
            start_time = time.time()
            histB = cv2.calcHist([img],[0],None,[256],[0,256])
            histG = cv2.calcHist([img],[1],None,[256],[0,256])
            histR = cv2.calcHist([img],[2],None,[256],[0,256])
            vect_features = np.concatenate((histB, np.concatenate((histG,histR),axis=None)),axis=None)
            print(f"Hist couleur requête terminé en {time.time()-start_time}s")
        elif algo_choice == 2:
            start_time = time.time()
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            histH = cv2.calcHist([hsv],[0],None,[180],[0,180])
            histS = cv2.calcHist([hsv],[1],None,[256],[0,256])
            histV = cv2.calcHist([hsv],[2],None,[256],[0,256])
            vect_features = np.concatenate((histH, np.concatenate((histS,histV),axis=None)),axis=None)
            print(f"HSV requête terminé en {time.time()-start_time}s")
        elif algo_choice == 3:
            start_time = time.time()
            sift = cv2.SIFT_create()
            kps, vect_features = sift.detectAndCompute(img, None)
            print(f"SIFT requête terminé en {time.time()-start_time}s")
        elif algo_choice == 4:
            start_time = time.time()
            orb = cv2.ORB_create()
            key_point1, vect_features = orb.detectAndCompute(img, None)
            print(f"ORB requête terminé en {time.time()-start_time}s")
        elif algo_choice == 5:
            vect_features = generateGLCM(img)
        elif algo_choice == 6:
            vect_features = generateLBP(img)
        elif algo_choice == 7:
            vect_features = generateHOG(img)

        np.savetxt("Methode_"+str(algo_choice)+"_query.txt", vect_features)
        print("sauvegardé")
        return vect_features

def distance_f(l1, l2, distanceName, is_binary=False):
    if distanceName == "Euclidienne":
        distance = euclidean(l1, l2)
    elif distanceName in ["Correlation", "Chi carre", "Intersection", "Bhattacharyya"]:
        if distanceName == "Correlation":
            distance = cv2.compareHist(np.float32(l1), np.float32(l2), cv2.HISTCMP_CORREL)
        elif distanceName == "Chi carre":
            distance = chiSquareDistance(l1, l2)
        elif distanceName == "Intersection":
            distance = cv2.compareHist(np.float32(l1), np.float32(l2), cv2.HISTCMP_INTERSECT)
        elif distanceName == "Bhattacharyya":
            distance = bhatta(l1, l2)
    elif distanceName == "Brute force":
        distance = bruteForceMatching(l1, l2, is_binary=is_binary)
    elif distanceName == "Flann":
        distance = flann(l1, l2, is_binary=is_binary)
    elif distanceName == "Cosine Similarity":
        distance = cosine_similarity(l1, l2)
    return distance

def getkNeighbors(lfeatures, req, k, distanceName):
    start_time = time.time()
    ldistances = []
    for i in range(len(lfeatures)):
        dist = distance_f(req, lfeatures[i][1], distanceName)
        ldistances.append((lfeatures[i][0], lfeatures[i][1], dist))
    if distanceName in ["Correlation", "Brute force", "Flann", "Cosine Similarity"]:
        order = True
    else:
        order = False
    ldistances.sort(key=operator.itemgetter(2), reverse=order)

    lneighbors = []
    for i in range(k):
        lneighbors.append(ldistances[i])
    print(f"kNN terminé en {time.time()-start_time}s")
    return lneighbors

def concatenate_multiple_descriptors(list_of_features):
    flattened_features = [np.array(f).ravel() for f in list_of_features]
    combined_feature = np.concatenate(flattened_features, axis=None)
    return combined_feature


def make_descriptor_as_files(dataset_path, descriptor_func, descriptor_name):
    """Applique un descripteur aux images dont le 1er chiffre est pair et sauvegarde en .txt."""
    if not os.path.exists("descriptors"):
        os.makedirs("descriptors")

    target_dir = os.path.join("descriptors", descriptor_name)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    for filename in os.listdir(dataset_path):
        if filename[0].isdigit() and int(filename[0]) % 2 == 0:
            img_path = os.path.join(dataset_path, filename)
            img = cv.imread(img_path)
            if img is not None:
                try:
                    feature = descriptor_func(img)
                    num_image = os.path.splitext(filename)[0]
                    save_path = os.path.join(target_dir, f"{num_image}_{descriptor_name}.txt")
                    # integers (histograms, SIFT, ORB) → '%d';  floats → '%.8f' (vs default %.18e)
                    if np.issubdtype(np.asarray(feature).dtype, np.integer):
                        np.savetxt(save_path, feature, fmt='%d')
                    else:
                        np.savetxt(save_path, feature, fmt='%.8f')
                except Exception as e:
                    print(f"Erreur {filename}: {e}")

    print(f"Indexation {descriptor_name} pour images paires terminée!")
