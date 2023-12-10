# import os
# import cv2
# import numpy as np
# import pandas as pd
# from skimage.feature import hog
# from skimage import feature

# # Đường dẫn đến các thư mục chứa ảnh
# folders = ["../DDSM/train_calc_ben", "../DDSM/train_mass_ben", "../DDSM/train_calc_mal", "../DDSM/train_mass_mal"]

# # Tạo DataFrame để lưu trữ đặc trưng
# # data = pd.DataFrame(columns=["Image Path"] + ["HOG_" + str(i) for i in range(1, 3781)] + ["Area", "AspectRatio", "MeanHOG", "StdHOG", "NumPeaks", "NumValleys"])
# data_list=[]
# # Hàm để trích xuất đặc trưng HOG từ ảnh
# def extract_hog_features(image_path):
#     img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#     # Resize ảnh để đảm bảo kích thước cố định (nếu cần)
#     img = cv2.resize(img, (64, 64))

#     # Tính toán đặc trưng HOG
#     hog_features, hog_image = hog(img, orientations=8, pixels_per_cell=(8, 8), cells_per_block=(1, 1), block_norm='L2-Hys', visualize=True)
    
#     return hog_features, hog_image

# # Hàm để tính các đặc trưng khác dựa vào đặc trưng HOG
# def calculate_additional_features(hog_features):
#     area = len(hog_features)
#     aspect_ratio = len(hog_features) / 64.0  # 64 là kích thước ảnh đã resize
#     mean_hog = np.mean(hog_features)
#     std_hog = np.std(hog_features)
    
#     # Đếm số lượng đỉnh và đáy trong biểu đồ HOG
#     num_peaks = len(np.where(np.diff(np.sign(np.diff(hog_features))) > 0)[0]) + 1
#     num_valleys = len(np.where(np.diff(np.sign(np.diff(hog_features))) < 0)[0]) + 1
    
#     return area, aspect_ratio, mean_hog, std_hog, num_peaks, num_valleys

# # Duyệt qua các thư mục và ảnh
# for folder in folders:
#     folder_path = os.path.join(folder)
#     for filename in os.listdir(folder_path):
#         if filename.endswith(".jpg") or filename.endswith(".png"):
#             image_path = os.path.join(folder_path, filename)
#             hog_features, _ = extract_hog_features(image_path)
#             additional_features = calculate_additional_features(hog_features)
#             features = [image_path] + hog_features.tolist() + list(additional_features)
#             data = data.append(pd.Series(features, index=data.columns), ignore_index=True)

# # Lưu DataFrame vào Excel
# excel_path = "../DDSM/BC.xlsx"
# data.to_excel(excel_path, index=False)

import os
import cv2
import numpy as np
import pandas as pd
from skimage.feature import hog

# Đường dẫn đến các thư mục chứa ảnh
folders = ["../DDSM/train_calc_ben", "../DDSM/train_mass_ben", "../DDSM/train_calc_mal", "../DDSM/train_mass_mal"]

# Tạo list để lưu trữ dữ liệu
data_list = []

# Hàm để trích xuất đặc trưng HOG từ ảnh
def extract_hog_features(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # Resize ảnh để đảm bảo kích thước cố định (nếu cần)
    img = cv2.resize(img, (64, 64))

    # Tính toán đặc trưng HOG
    hog_features, _ = hog(img, orientations=8, pixels_per_cell=(8, 8), cells_per_block=(1, 1), block_norm='L2-Hys', visualize=True)
    
    return hog_features

# Hàm để tính các đặc trưng khác dựa vào đặc trưng HOG
def calculate_additional_features(hog_features):
    area = len(hog_features)
    aspect_ratio = len(hog_features) / 64.0  # 64 là kích thước ảnh đã resize
    mean_hog = np.mean(hog_features)
    std_hog = np.std(hog_features)
    
    # Đếm số lượng đỉnh và đáy trong biểu đồ HOG
    num_peaks = len(np.where(np.diff(np.sign(np.diff(hog_features))) > 0)[0]) + 1
    num_valleys = len(np.where(np.diff(np.sign(np.diff(hog_features))) < 0)[0]) + 1
    
    return area, aspect_ratio, mean_hog, std_hog, num_peaks, num_valleys

# Duyệt qua các thư mục và ảnh
for folder in folders:
    folder_path = os.path.join(folder)
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(folder_path, filename)
            hog_features = extract_hog_features(image_path)
            additional_features = calculate_additional_features(hog_features)
            features = {"Image Path": image_path, **{"HOG_" + str(i): val for i, val in enumerate(hog_features, start=1)}, "Area": additional_features[0], "AspectRatio": additional_features[1], "MeanHOG": additional_features[2], "StdHOG": additional_features[3], "NumPeaks": additional_features[4], "NumValleys": additional_features[5]}
            data_list.append(features)

# Tạo DataFrame từ list của dictionaries
data = pd.DataFrame(data_list)

# Lưu DataFrame vào Excel
excel_path = "../DDSM/BC.xlsx"
data.to_excel(excel_path, index=False)
