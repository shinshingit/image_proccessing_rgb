import cv2
import os

# 読み込みファイルパス
img_path = 'ColorAnalysis_TargetImages\colorball_blue_original.png'

# ディレクトリの取得(ファイル名の除外)
dir = os.path.dirname(img_path)
print(f' dir :{dir}')
#　ファイル名抽出 + 拡張子除去
img_filename = os.path.basename(img_path)
img_filename = os.path.splitext(img_filename)[0]
print(f'img_filename:{img_filename}')



# 切り出しの一辺の長さ
side_len = 400




# 画像読み込み
img_ori = cv2.imread(img_path, cv2.IMREAD_COLOR)
print(f'img_ori.shape:{img_ori.shape}')

# ROI切り出し
img_roi = img_ori[0:side_len, 0:side_len]
print(f'img_roi.shape:{img_roi.shape}')

# ROI画像保存
roi_filename = img_filename + '_roi.png'
cv2.imwrite(dir+'/'+roi_filename, img_roi)


# 画像表示
cv2.imshow('img_ori', img_ori)
cv2.imshow('img_roi', img_roi)

cv2.waitKey(0)
