import os
import cv2
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

import warnings
warnings.simplefilter('ignore', FutureWarning)

# <変更する変数>画像パス
img_path = 'ColorAnalysis_TargetImages\colorball_blue_original_roi.png'
#　ファイル名抽出 + 拡張子除去
img_filename = os.path.basename(img_path)
img_filename = os.path.splitext(img_filename)[0]


# <変更する変数>抽出領域サイズ
pickupsize = 200
# <変更する変数>スライド回数(1方向当たりの)
slidecnt = 20

# 結果群を格納する配列(平均値、中央値、最頻値)
rgb_aveVals = []
rgb_medianVals = []
rgb_modVals = []
total_ave_rgbArray = np.empty
total_median_rgbArray = np.empty
total_mod_rgbArray = np.empty

# グラフタイトル
charttitle = img_filename + '   PickSize:'+str(pickupsize)


def main():
    # 画像読み込み
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)

    # 画像サイズ取得
    h,w = img.shape[:2]

    # 抽出領域のスライド幅の算出
    stride = (int)((w-pickupsize)/slidecnt)
    print(f'stride :{stride}   type(stride):{type(stride)}')


    # for分で400通り(20×20)の切り出し条件で行う
    cnt = 0
    for y in range(slidecnt):
        for x in range(slidecnt):
            cnt=cnt+1
            #print(f'◆loop count:{cnt} =========================')
            
            # 画像抽出(ROI[top:bottom, left:right]・・・タテ・ヨコにstride分ずつ移動する)
            pickup_img = img[stride*y : stride*y+pickupsize , stride*x : stride*x+pickupsize]
            #print(f'pickup_img({y},{x})  :{pickup_img.shape}')

            # 抽出画像を分析関数へ渡す
            rgb_analysis(cnt, pickup_img)



    # 分析結果の集計 (各パラメータ(+各色)の標準偏差・変動係数(coefficient of variation)を算出 ) --------
    # 平均値の変動係数
    rslt_ave_stddev = [np.std(total_ave_rgbArray[0]),np.std(total_ave_rgbArray[1]),np.std(total_ave_rgbArray[2])]
    rslt_ave_ave = [np.mean(total_ave_rgbArray[0]),np.mean(total_ave_rgbArray[1]),np.mean(total_ave_rgbArray[2])]
    rslt_ave_coeff = [round(rslt_ave_stddev[0]/rslt_ave_ave[0],3), round(rslt_ave_stddev[1]/rslt_ave_ave[1],3), round(rslt_ave_stddev[2]/rslt_ave_ave[2],3)]
    print(f' >> rslt_ave_coeff:{rslt_ave_coeff}')
    # 中央値の変動係数
    rslt_med_stddev = [np.std(total_median_rgbArray[0]),np.std(total_median_rgbArray[1]),np.std(total_median_rgbArray[2])]
    rslt_med_ave = [np.mean(total_median_rgbArray[0]),np.mean(total_median_rgbArray[1]),np.mean(total_median_rgbArray[2])]
    rslt_med_coeff = [round(rslt_med_stddev[0]/rslt_med_ave[0],3), round(rslt_med_stddev[1]/rslt_med_ave[1],3), round(rslt_med_stddev[2]/rslt_med_ave[2],3)]
    print(f' >> rslt_med_coeff:{rslt_med_coeff}')




    # グラフの描画
    plt.plot(total_ave_rgbArray[0], label='R(ave):'+str(rslt_ave_coeff[0]),color='r')
    plt.plot(total_ave_rgbArray[1], label='G(ave):'+str(rslt_ave_coeff[1]),color='g')
    plt.plot(total_ave_rgbArray[2], label='B(ave):'+str(rslt_ave_coeff[2]),color='b')
    plt.plot(total_median_rgbArray[0], label='R(med):'+str(rslt_med_coeff[0]),color='r',linestyle='--')
    plt.plot(total_median_rgbArray[1], label='G(med)'+str(rslt_med_coeff[1]),color='g',linestyle='--')
    plt.plot(total_median_rgbArray[2], label='B(med)'+str(rslt_med_coeff[2]),color='b',linestyle='--')
    # plt.plot(total_mod_rgbArray[0], label='R(mod)',color='r',linestyle=':')
    # plt.plot(total_mod_rgbArray[1], label='G(mod)',color='g',linestyle=':')
    # plt.plot(total_mod_rgbArray[2], label='B(mod)',color='b',linestyle=':')
    plt.title(charttitle)
    plt.legend()
    plt.show()


    exit()



    

    cv2.waitKey(0)


def rgb_analysis(cnt, img):
    #print('rgb_analysis()')

    #R,G,Bの単色画像を作成
    img_b,img_g,img_r = cv2.split(img)

    # 平均値の算出 (round()は小数点以下の桁数丸め込み)
    ave_r = round( np.mean(img_r), 2)
    ave_g = round( np.mean(img_g), 2)
    ave_b = round( np.mean(img_b), 2)
    # print('●Average ------------------')
    # print(f'Red   Ave: {ave_r}')
    # print(f'Green Ave: {ave_g}')
    # print(f'Blue  Ave: {ave_b}')

    # 中央値の算出
    med_r = round( np.median(img_r), 2)
    med_g = round( np.median(img_g), 2)
    med_b = round( np.median(img_b), 2)
    # print('●Median  ------------------')
    # print(f'Red   Med: {med_r}')
    # print(f'Red   Med: {med_g}')
    # print(f'Red   Med: {med_b}')

    # 最頻値の算出(最頻値は英語「mode」)
    mod_r_val, mod_r_cnt = stats.mode(img_r, axis=None)
    mod_g_val, mod_g_cnt = stats.mode(img_g, axis=None)
    mod_b_val, mod_b_cnt = stats.mode(img_b, axis=None)
    # print('●Mode    ------------------')
    # print(f'Red   Mod  Val: {mod_r_val}     count:{mod_r_cnt}')
    # print(f'Green Mod  Val: {mod_g_val}     count:{mod_g_cnt}')
    # print(f'Blue  Mod  Val: {mod_b_val}     count:{mod_b_cnt}')


    # 配列に格納 ------------------
    #平均値
    ave = [ave_r,ave_g,ave_b]
    rgb_aveVals.append(ave)
    #中央値
    med = [med_r,med_g,med_b]
    rgb_medianVals.append(med)
    #最頻値
    mod = [mod_r_val[0],mod_g_val[0],mod_b_val[0]]
    rgb_modVals.append(mod)


    if cnt >= slidecnt*slidecnt:
        print(f'--len(rgb_aveVals):{len(rgb_aveVals)}')
        # print(rgb_aveVals)

        # 天地(np.arrayに変換　→　転置(T))-----------------------
        #平均値
        ndarry = np.array(rgb_aveVals)
        global total_ave_rgbArray
        total_ave_rgbArray = ndarry.T
        # print(f'--len(nparry):{len(nparry.T)}')
        print(f'--len(total_ave_rgbArray):{len(total_ave_rgbArray)}')

        #中央値
        ndarry = np.array(rgb_medianVals)
        global total_median_rgbArray
        total_median_rgbArray = ndarry.T
        #print(f'--len(total_median_rgbArray):{len(total_median_rgbArray)}')
        #最頻値
        ndarry = np.array(rgb_modVals)
        global total_mod_rgbArray
        total_mod_rgbArray = ndarry.T
        #print(f'--len(total_ave_rgbArray):{len(total_mod_rgbArray)}')


        



if __name__ == "__main__":
    main()