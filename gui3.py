import numpy as np 
import cv2 
from keras import models
import os
# Khai báo biến đầu ra
Output = ['car', 'truck']
# Load model 
modeltest = models.load_model("F:\Mon_Hoc\coder\python\DetaiANN\Model") 
path = "F:\Mon_Hoc\coder\python\car-vs-truck\ktramodule\\test1"
# Đọc ảnh từ hệ thống
# for category in os.listdir(DATADIR):
#     path = os.path.join(DATADIR,category)
for file in os.listdir(path):  
        curr_file = os.path.join(path,file)
        img = cv2.imread(curr_file) 
        img1 = img
        # Chuẩn hóa ảnh đúng với kích thước qui định
        img = np.asarray(img) 
        img = cv2.resize(img, (64, 64)) 
        img = img.reshape(1, 64, 64, 3) 
        # Đánh giá ảnh
        predictions = modeltest.predict(img) 
        # Lấy giá trị lớn nhất trong 2 giá trị dự đoán được
        probabilityValue = np.amax(predictions) 
        print(probabilityValue) 
        # Chỉ nhận diện khi giá trị dự đoán lớn hơn 0.7 (70%)
        if probabilityValue > 0.7: 
        # In giá trị dự đoán ra console
            print('Dự đoán: ' + str(round(probabilityValue * 100, 2)) + '%') 
            print(Output[np.argmax(predictions)]) 
            # Ghi kết quả dự đoán và giá trị dự đoán lên ảnh
            cv2.putText(img1, Output[np.argmax(predictions)] + ' ' + 
        str(round(probabilityValue * 100, 2)) + '%', (0, 70), cv2.FONT_HERSHEY_PLAIN, 2, (0, 100, 255), 2) 
        # Hiển thị ảnh ra màn hình
        cv2.imshow('img', img1) 
        # Đợi đến khi nhấn nút bất kỳ
        cv2.waitKey(0)
        
