import numpy as np
import matplotlib.pyplot as plt
import shapely.geometry as SG


FAR_list = []; FRR_list = []
for thres in range(-100,100,1):      # Generate [-1,1] with interval 0.01
    thres = thres/100
    total_match = 0; total_unmatch = 0; FA = 0; FR = 0
    for pair in open("./tisv_model/cos_similarity.txt"):  # load evaluation pairs 
        value = float(pair.split()[0])
        flag = pair.split()[1]  # label of pair
        
        if flag == 'unmatch':
            total_unmatch += 1
            if value > thres:
                FA += 1
        elif flag == 'match':
            total_match += 1
            if value < thres:
                FR += 1

    FAR = FA/total_unmatch      # the FAR under current thres
    FAR_list.append(FAR)
    FRR = FR/total_match        # the FRR under current thres
    FRR_list.append(FRR)

line1 = SG.LineString(list(zip(FAR_list,FRR_list)))
line2 = SG.LineString([(0,0),(1,1)])
coords = np.array(line2.intersection(line1))
print(coords)

plt.plot(FAR_list,FRR_list,'r')     # plot ROC
plt.plot([0,1],[0,1],'b')
plt.show()

