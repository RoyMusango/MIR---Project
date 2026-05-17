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
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image


#COMPARATORS
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
    return np.sum((x-y)**2/(x+y))

def intersection_distance(x,y):
    return 1 - np.sum(np.minimum(x,y)) / (np.sum(x) + 1e-10)

def bhattacharyya_distance(x,y):
    coefficient = np.sum(np.sqrt(np.abs(x) * np.abs(y)))
    return -np.log(coefficient + 1e-10)

def bhatta(x,y):
    return bhattacharyya_distance(x,y)

def chiSquareDistance(x, y):
    return np.sum((x - y)**2 / (x + y + 1e-10))

def bruteForceMatching(des1, des2):
    bf = cv.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append([m])
    return len(good)

def flann(des1, des2):
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.7*n.distance:
            good.append([m])
    return len(good)



#DESCRIPTORS

def generateHistogramme_HSV(image):
    start_time = time.time()
    img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    histH = cv2.calcHist([img],[0],None,[180],[0,180])
    histS = cv2.calcHist([img],[1],None,[256],[0,256])
    histV = cv2.calcHist([img],[2],None,[256],[0,256])
    feature = np.concatenate((histH, np.concatenate((histS,histV),axis=None)),axis=None)

    print(f"indexation Hist HSV finished in {time.time()-start_time} seconds !!!!")
    return feature
        
def generateHistogramme_Color(image):
    start_time = time.time()
    histB = cv2.calcHist([image],[0],None,[256],[0,256])
    histG = cv2.calcHist([image],[1],None,[256],[0,256])
    histR = cv2.calcHist([image],[2],None,[256],[0,256])
    feature = np.concatenate((histB, np.concatenate((histG,histR),axis=None)),axis=None)

    print(f"indexation Hist Couleur finished in {time.time()-start_time} seconds !!!!")
    return feature

def generateSIFT(image):
    start_time = time.time()
    sift = cv2.SIFT_create()  
    key_point1,descrip1 = sift.detectAndCompute(image,None)

    print(f"Indexation SIFT finished in {time.time()-start_time} seconds !!!!")
    return descrip1


def generateORB(image):
    start_time = time.time()
    orb = cv2.ORB_create()
    key_point1,descrip1 = orb.detectAndCompute(image,None)
    
    print(f"indexation ORB finished in {time.time()-start_time} seconds !!!!")
    return descrip1

def generateGLCM(image):
    start_time = time.time()
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = img_as_ubyte(gray)
    distances = [1, 2]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    # Compute GLCM matrix
    glcmMatrix = graycomatrix(
        gray,
        distances=distances,
        angles=angles,
        normed=True
    )
    # Extract GLCM properties
    glcmProperties1 = graycoprops(glcmMatrix, "contrast").ravel()
    glcmProperties2 = graycoprops(glcmMatrix, "dissimilarity").ravel()
    glcmProperties3 = graycoprops(glcmMatrix, "homogeneity").ravel()
    glcmProperties4 = graycoprops(glcmMatrix, "energy").ravel()
    glcmProperties5 = graycoprops(glcmMatrix, "correlation").ravel()
    glcmProperties6 = graycoprops(glcmMatrix, "ASM").ravel()
    # Feature vector
    feature = np.array([
        glcmProperties1,
        glcmProperties2,
        glcmProperties3,
        glcmProperties4,
        glcmProperties5,
        glcmProperties6
    ]).ravel()
    print(f"GLCM feature extraction finished in {time.time()-start_time} seconds !!!!")
    return feature

def generateLBP(image):
    start_time = time.time()
    points = 8
    radius = 1
    method= 'default' 
    #subSize = (70, 70)
    # Convert image to grayscale
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (350, 350))
    # Compute LBP matrix
    fullLBPmatrix = local_binary_pattern(img, points, radius, method)
    """
    histograms = np.array([], dtype=np.float32)
    # Compute histograms over sub-regions
    for k in range(int(fullLBPmatrix.shape[0] / subSize[0])):
        for j in range(int(fullLBPmatrix.shape[1] / subSize[1])):
            subVector = fullLBPmatrix[
                k*subSize[0]:(k+1)*subSize[0],
                j*subSize[1]:(j+1)*subSize[1]
            ].ravel()
            subHist, edges = np.histogram(
                subVector,
                bins=int(2**points),
                range=(0, 2**points)
            )
            histograms = np.concatenate(
                (histograms, subHist),
                axis=None
            )"""
    histograms, edges = np.histogram(fullLBPmatrix.ravel(), bins=10, range=(0, 2**points))
    histograms=histograms.astype("float")
    histograms /= histograms.sum() + 1e-6  
    print(f"LBP feature extraction finished in {time.time()-start_time} seconds !!!!")
    return histograms

def generateHOG(image):
    start_time = time.time()
    # HOG parameters
    cellSize = (25, 25)
    blockSize = (50, 50)
    blockStride = (25, 25)
    nBins = 9
    winSize = (350, 350)
    # Convert image to grayscale
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, winSize)
    # Create HOG descriptor
    hog = cv2.HOGDescriptor(
        winSize,
        blockSize,
        blockStride,
        cellSize,
        nBins
    )
    # Compute HOG features
    vect_features = hog.compute(img)
    print(f"HOG feature extraction finished in {time.time()-start_time} seconds !!!!")
    return vect_features
	
def generateCLIP(image):
    start_time = time.time()
    from clip_model import get_embedder
    embedder = get_embedder()
    feature = embedder.encode_image(image)
    print(f"CLIP feature extraction finished in {time.time()-start_time} seconds !!!!")
    return feature

_resnet_model = None

def _get_resnet():
    global _resnet_model
    if _resnet_model is None:
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        model = torch.nn.Sequential(*list(model.children())[:-1])
        model.eval()
        _resnet_model = model
    return _resnet_model

_resnet_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

def generateResNet50(image):
    start_time = time.time()
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    x = _resnet_transform(pil_img).unsqueeze(0)
    with torch.no_grad():
        feat = _get_resnet()(x).squeeze().numpy()
    feat = feat / (np.linalg.norm(feat) + 1e-10)
    print(f"ResNet50 feature extraction finished in {time.time()-start_time} seconds !!!!")
    return feat.astype(np.float32)

def extractReqFeatures(fileName,algo_choice):  
    print(algo_choice)
    if fileName : 
        img = cv2.imread(fileName)
        resized_img = resize(img, (128*4, 64*4))
            
        if algo_choice==1: #Couleurs
            start_time = time.time()
            histB = cv2.calcHist([img],[0],None,[256],[0,256])
            histG = cv2.calcHist([img],[1],None,[256],[0,256])
            histR = cv2.calcHist([img],[2],None,[256],[0,256])
            vect_features = np.concatenate((histB, np.concatenate((histG,histR),axis=None)),axis=None)
            print(f"Color Histo for query image finished in {time.time()-start_time} seconds !!!!")
        elif algo_choice==2: # Histo HSV
            start_time = time.time()
            hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
            histH = cv2.calcHist([hsv],[0],None,[180],[0,180])
            histS = cv2.calcHist([hsv],[1],None,[256],[0,256])
            histV = cv2.calcHist([hsv],[2],None,[256],[0,256])
            vect_features = np.concatenate((histH, np.concatenate((histS,histV),axis=None)),axis=None)
            print(f"HSV for query image finished in {time.time()-start_time} seconds !!!!")
        elif algo_choice==3: #SIFT
            start_time = time.time()
            sift = cv2.SIFT_create() #cv2.xfeatures2d.SIFT_create() pour py < 3.4 
            # Find the key point
            kps , vect_features = sift.detectAndCompute(img,None)
            print(f"SIFT for query image finished in {time.time()-start_time} seconds !!!!")
        elif algo_choice==4: #ORB
            start_time = time.time()
            orb = cv2.ORB_create()
            # finding key points and descriptors of both images using detectAndCompute() function
            key_point1,vect_features = orb.detectAndCompute(img,None)
            print(f"ORB for query image finished in {time.time()-start_time} seconds !!!!")
        elif algo_choice==5: #GLCM
            vect_features = generateGLCM(img)

        elif algo_choice==6: #LBP
            vect_features = generateLBP(img)

        elif algo_choice==7: #HOG
            vect_features = generateHOG(img)
        
        
        np.savetxt("Methode_"+str(algo_choice)+"_query.txt" ,vect_features)
        print("saved")
        #print("vect_features", vect_features)
        return vect_features

def distance_f(l1,l2,distanceName):
    if distanceName=="Euclidienne":
        distance = euclidean(l1, l2)
    elif distanceName in ["Correlation","Chi carre","Intersection","Bhattacharyya"]:
        if distanceName=="Correlation":
            methode=cv2.HISTCMP_CORREL
            distance = cv2.compareHist(np.float32(l1), np.float32(l2), methode)
        elif distanceName=="Chi carre":
            distance = chiSquareDistance(l1, l2)
        elif distanceName=="Intersection":
            methode=cv2.HISTCMP_INTERSECT
            distance = cv2.compareHist(np.float32(l1), np.float32(l2), methode)
        elif distanceName=="Bhattacharyya":
            distance = bhatta(l1, l2)
    elif distanceName=="Brute force":
        distance = bruteForceMatching(l1, l2)
    elif distanceName=="Flann":
        distance= flann(l1, l2)
    elif distanceName=="Cosine Similarity":
        distance = cosine_similarity(l1, l2)
    return distance

def getkNeighbors(lfeatures, req, k,distanceName) : 
    start_time = time.time()
    ldistances = [] 
    for i in range(len(lfeatures)): 
        dist = distance_f(req, lfeatures[i][1],distanceName)
        ldistances.append((lfeatures[i][0], lfeatures[i][1], dist)) 
    if distanceName in ["Correlation","Brute force","Flann","Cosine Similarity"]:
        order=True
    else:
        order=False
    ldistances.sort(key=operator.itemgetter(2),reverse=order) 

    lneighbors = [] 
    for i in range(k): 
        lneighbors.append(ldistances[i]) 
    print(f"getkNeighbors finished in {time.time()-start_time} seconds !!!!")
    return lneighbors

def concatenate_multiple_descriptors(list_of_features):
    flattened_features = [np.array(f).ravel() for f in list_of_features]
    combined_feature = np.concatenate(flattened_features, axis=None)
    
    return combined_feature


def make_descriptor_as_files(dataset_path, descriptor_func, descriptor_name):
    """
    Apply a descriptor function to images whose first digit is even.
    Save the descriptors in 'descriptors/<descriptor_name>/filename.txt'.
    """
    # Ensure base descriptors directory exists
    if not os.path.exists("descriptors"):
        os.makedirs("descriptors")
    
    # Ensure subdirectory for this descriptor exists
    target_dir = os.path.join("descriptors", descriptor_name)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        
    for filename in os.listdir(dataset_path):
        # Check if first digit is even
        if filename[0].isdigit() and int(filename[0]) % 2 == 0:
            img_path = os.path.join(dataset_path, filename)
            img = cv.imread(img_path)
            if img is not None:
                # Apply descriptor
                # Some functions take image, some take path.
                # We assume descriptor_func takes an image array.
                try:
                    feature = descriptor_func(img)
                    
                    # Save as txt
                    num_image = os.path.splitext(filename)[0]
                    save_path = os.path.join(target_dir, f"{num_image}_{descriptor_name}.txt")
                    np.savetxt(save_path, feature)
                except Exception as e:
                    print(f"Error processing {filename}: {e}")
                    
    print(f"Indexation {descriptor_name} for even-prefixed images finished!")