
import numpy as np
import cv2
import matplotlib.pyplot as plt
from itertools import combinations    
import os
from os import listdir

def cal_similarity_score(matches, keypint_image1, keypint_image2):
    num_matches = len(matches)
    similarity_score = num_matches / min(len(keypint_image1), len(keypint_image2))
    return similarity_score


def cross_check(matches_img1, matches_img2):
    matches_dict_img1 = {match.queryIdx: match for match in matches_img1}
    cross_check_matches_lst = []
    for match in matches_img2:
        if (match.trainIdx in matches_dict_img1):
            cross_match_pxl = matches_dict_img1[match.trainIdx]
            if (cross_match_pxl.trainIdx == match.queryIdx):
                cross_check_matches_lst.append(cross_match_pxl)

    return cross_check_matches_lst


def filter_matches_ratio_test(matches):
    filtered_matches = []
    for match in matches:
        if (match[0].distance < 0.7 * match[1].distance):
            filtered_matches.append(match[0])
    return filtered_matches


def filter_matches_distance_tst(matches):

    filtered_matches_lst = []
    distances_tst = [match_pair.distance for match_pair in matches]
    distance_thresholdd = np.median(distances_tst)

    for match_ in matches:
        match = match_
        if match.distance < distance_thresholdd:
            filtered_matches_lst.append(match)

    return filtered_matches_lst


def ransac(mtsh,kp1_1,kp2_2):
  src_pts_ran = np.float32([kp1_1[m.queryIdx].pt for m in mtsh]).reshape(-1, 1, 2)
  dst_pts_ran = np.float32([kp2_2[m.trainIdx].pt for m in mtsh]).reshape(-1, 1, 2)
  M, mask = cv2.findHomography(src_pts_ran, dst_pts_ran, cv2.RANSAC, 20)
  matchesMask_ran = mask.ravel().tolist()

  return matchesMask_ran


    
data=r'C:\Users\Hazem\Desktop\4th\vision\tasks\task2\assignment data'

for i in range(1,10):
   
    img_1 = os.path.join(data, f'image{i}a.jpeg')
    img_2 = os.path.join(data, f'image{i}b.jpeg')

    img1 = cv2.imread(img_1, 0)
    img2 = cv2.imread(img_2, 0)

    if i ==4:
      
        file_paths =[
        r"C:\Users\Hazem\Desktop\4th\vision\tasks\task2\assignment data\image4c.png",
        r"C:\Users\Hazem\Desktop\4th\vision\tasks\task2\assignment data\image4a.jpeg",
        r"C:\Users\Hazem\Desktop\4th\vision\tasks\task2\assignment data\image4b.jpeg" ]

        comb = combinations(file_paths, 2)

        for n, j in list(comb):
                
                img1 = cv2.imread(n, 0)  
                img2 = cv2.imread(j, 0)  
                          
                sift_feat = cv2.SIFT_create()
                kp1_1, des1_ = sift_feat.detectAndCompute(img1, None)
                kp2_2, des2_ = sift_feat.detectAndCompute(img2, None)
                bf = cv2.BFMatcher()

                matches = bf.knnMatch(des1_, des2_, k=2)

                filtered_matches_ratio = filter_matches_ratio_test(matches)
                filtered_matches_ = filter_matches_distance_tst(filtered_matches_ratio)

                matchesMask_ran=ransac(filtered_matches_,kp1_1,kp2_2)

                similarity_score= cal_similarity_score(matchesMask_ran,kp1_1,kp2_2)

                img4 = cv2.drawMatches(img1, kp1_1, img2, kp2_2,
                                        filtered_matches_, None, matchColor=(0, 255, 0), flags=2)

                plt.imshow(img4, 'ocean')
                plt.show()

                print("Similarity Score:", similarity_score)
                if similarity_score > 0.1:
                    print("The 2 Images are similar")
                else:
                    print("The 2 Images are not similar")


    else:  
        sift_feat = cv2.SIFT_create()
        kp1_1, des1_ = sift_feat.detectAndCompute(img1, None)
        kp2_2, des2_ = sift_feat.detectAndCompute(img2, None)
        bf = cv2.BFMatcher()

        matches = bf.knnMatch(des1_, des2_, k=2)

        filtered_matches_ratio = filter_matches_ratio_test(matches)
        filtered_matches_ = filter_matches_distance_tst(filtered_matches_ratio)

        matchesMask_ran=ransac(filtered_matches_,kp1_1,kp2_2)

        similarity_score= cal_similarity_score(matchesMask_ran,kp1_1,kp2_2)

        img4 = cv2.drawMatches(img1, kp1_1, img2, kp2_2,
                                filtered_matches_, None, matchColor=(0, 255, 0), flags=2)


        plt.imshow(img4, 'ocean')
        plt.show()

        print("Similarity Score:", similarity_score)
        if similarity_score > 0.1:
            print("The 2 Images are similar")
        else:
            print("The 2 Images are not similar")


